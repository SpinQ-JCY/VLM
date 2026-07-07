"""VLM v1：SigLIP2 视觉编码 + 2 层 MLP 投影 + Qwen3-1.7B 文本生成。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, SiglipVisionModel
from transformers import logging as transformers_logging
from transformers.modeling_outputs import CausalLMOutputWithPast

ROOT = Path(__file__).resolve().parents[2]
IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
DEFAULT_SYSTEM = "你是一个乐于助人的助手，可以根据图片内容回答问题。"
MAX_TEXT_TOKENS = 128  # 与 MiniLlava 一致：整段文本（含 system / 问题 / answer / EOS）截断上限


@contextmanager
def suppress_load_report():
    """保留权重加载进度条，仅抑制 transformers 的 LOAD REPORT（UNEXPECTED keys 等 warning）。"""
    old_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(old_verbosity)


def build_train_prompt_text(system: str, question: str, im_end: str) -> str:
    """拼推理/训练用的 prompt 前缀（含 system、user、<image>、assistant 头）。"""
    return (
        f"<|im_start|>system\n{system}\n{im_end}\n"
        f"<|im_start|>user\n{IMAGE_TOKEN}\n{question}\n{im_end}\n"
        f"<|im_start|>assistant\n"
    )


def build_train_text(system: str, question: str, answer: str, im_end: str) -> str:
    """在 prompt 前缀后拼接 answer，供训练 tokenize。"""
    return build_train_prompt_text(system, question, im_end) + answer + im_end


def encode_train_sample(
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    max_text_tokens: int = MAX_TEXT_TOKENS,
    system: str = DEFAULT_SYSTEM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """tokenize 单条 QA，labels 仅在 assistant answer 段非 -100。"""
    im_end = tokenizer.eos_token
    full_ids = tokenizer.encode(
        build_train_text(system, question, answer, im_end),
        max_length=max_text_tokens,
        truncation=True,
        add_special_tokens=False,
    )
    marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    start = len(full_ids)
    for i in range(len(full_ids) - len(marker), -1, -1):
        if full_ids[i : i + len(marker)] == marker:
            start = i + len(marker)
            break

    # 例：共 43 token（id 0–42），start=36（0–35 是 prompt，36–42 是 answer+EOS）
    #     input_ids = [0, 1, 2, ..., 35, 36, 37, 38, 39, 40, 41, 42]
    #     labels    = [-100, -100, ..., -100, 36, 37, 38, 39, 40, 41, 42]  后 7 个算 loss
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[start:] = input_ids[start:]
    return input_ids, labels


@dataclass
class VLM_v1_Config:
    siglip_path: str = str(ROOT / "models" / "siglip2-so400m-patch16-384")
    qwen_path: str = str(ROOT / "models" / "Qwen3-1.7B")
    vision_hidden: int = 1152
    llm_hidden: int = 2048
    num_image_tokens: int = 576  # 384 / 16 = 24 → 24×24
    freeze_vision: bool = True
    freeze_llm: bool = True


class VLM_v1_Projector(nn.Module):
    """SigLIP patch 特征 → Qwen 语义 token 空间（2 层全连接）。"""

    def __init__(self, vision_hidden: int, llm_hidden: int):
        """初始化 1152→2048→2048 两层 MLP。"""
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden, llm_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_hidden, llm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 SigLIP patch 特征映射到 Qwen 隐层维度。"""
        return self.fc2(self.act(self.fc1(x)))


class VLM_v1_Model(nn.Module):
    """
    数据流（单条样本 B=1；训练 max_seq_len=704 = 576 图 + 128 文本）。

    用例（demo 图 + InstructFT 权重实测）：
        question = "请用一句话描述这张图片。"
        answer   = "客厅里，一位女子正在厨房里。"

        prompt 字符串（训练时 answer 接在后面，truncate 至 128 token）：
            <|im_start|>system
            {system}
            
            <|im_start|>user
            <image>
            {question}
            
            <|im_start|>assistant
            {answer}

        ① 视觉路径
            pixel_values                 (1, 3, 384, 384)
              → SigLIP Vision            (1, 576, 1152)
              → Projector                (1, 576, 2048)

        ② 文本 tokenize（T_text=46，含 1 个 <image> 占位 id）
            input_ids                    (46,)
              → embed_tokens             (46, 2048)

        ③ 在 pos=22 处展开 <image>
            prefix  text[:22]            (22, 2048)    system + user 段至 <image> 前
            image   576 个视觉 token      (576, 2048)  替换 1 个 <image> id
            suffix  text[23:]            (23, 2048)    问题 + assistant 头 + answer
            ────────────────────────────────────────
            merged  cat 沿 seq 维         (621, 2048)  621 = 46 - 1 + 576

        ④ 送入 Qwen3 LLM（训练 batch 形状加 B 维）
            inputs_embeds                (1, 621, 2048)
            labels                       (1, 621)       前 611 为 -100，后 10 个 answer 算 loss
              → Causal LM                (1, 621, 151670)

        推理（仅 prompt、无 answer）时 T_text=36 → T_seq=611，generate 再追加 T_new 个 token。
    """

    def __init__(self):
        """加载 SigLIP、Qwen、Projector，并冻结 vision/llm。"""
        super().__init__()
        self.config = VLM_v1_Config()
        self._tokenizer: AutoTokenizer | None = None

        with suppress_load_report():
            self.vision = SiglipVisionModel.from_pretrained(
                self.config.siglip_path,
                torch_dtype=torch.bfloat16,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.qwen_path,
                torch_dtype=torch.bfloat16,
            )
        self.projector = VLM_v1_Projector(self.config.vision_hidden, self.config.llm_hidden)
        self.projector.to(dtype=self.llm.dtype)

        if self.config.freeze_vision:
            self.vision.requires_grad_(False)
            self.vision.eval()
        if self.config.freeze_llm:
            self.llm.requires_grad_(False)
            self.llm.eval()

    @property
    def device(self) -> torch.device:
        """模型所在设备；用法 model.device（属性，非 model.device()）。"""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """模型参数默认 dtype（如 bfloat16）。"""
        return next(self.parameters()).dtype

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:  # pixel_values (B, 3, 384, 384)
        """SigLIP 编码图片并投影为 LLM 视觉 token。"""
        vision_out = self.vision(pixel_values=pixel_values).last_hidden_state  # (B, 576, 1152)
        return self.projector(vision_out)  # (B, 576, 2048)

    def _build_prompt(self, question: str, system: str = DEFAULT_SYSTEM) -> str:
        """构造单轮问答的 chat prompt 字符串。"""
        return build_train_prompt_text(system, question, self._tokenizer.eos_token)

    def _image_token_pos(self, ids: torch.Tensor) -> int:  # ids (T_text,)
        """返回 input_ids 中 <image> 占位符的下标。"""
        image_token_id = self._tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
        if len(pos) != 1:
            raise ValueError(f"每条样本应包含 1 个 {IMAGE_TOKEN}，实际 {len(pos)} 个")
        return pos.item()

    def _pad_batch(self, tensors: list[torch.Tensor], pad_value: float | int) -> torch.Tensor:
        """将变长序列列表 padding 为等长 batch 张量。"""
        max_len = max(t.size(0) for t in tensors)
        out = torch.full(  # (B, T_seq_max, D)；1D 时 D 为空，即 (B, T_seq_max)
            (len(tensors), max_len, *tensors[0].shape[1:]),
            pad_value,
            device=self.device,
            dtype=tensors[0].dtype,
        )
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,  # (B, T_text)，T_text ≤ 128
        image_embeds: torch.Tensor,  # (B, 576, 2048)
        labels: Optional[torch.Tensor] = None,  # (B, T_text)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 <image> 展开为 576 视觉 token，拼出 inputs_embeds / mask / labels。"""
        embeds_list, masks_list, labels_list = [], [], []

        for i in range(input_ids.size(0)):
            ids = input_ids[i]  # (T_text,)
            pos = self._image_token_pos(ids)

            text_embeds = self.llm.get_input_embeddings()(ids)  # (T_text, 2048)
            img = image_embeds[i].to(text_embeds.dtype)  # (576, 2048)
            merged_embeds = torch.cat([text_embeds[:pos], img, text_embeds[pos + 1 :]], dim=0)  # (T_seq, 2048)，T_seq=T_text-1+576
            embeds_list.append(merged_embeds)
            masks_list.append(torch.ones(merged_embeds.size(0), device=self.device, dtype=torch.long))  # (T_seq,)

            lab = labels[i] if labels is not None else ids  # (T_text,)
            img_lab = torch.full((self.config.num_image_tokens,), IGNORE_INDEX, device=lab.device, dtype=lab.dtype)  # (576,)
            merged_labels = torch.cat([lab[:pos], img_lab, lab[pos + 1 :]], dim=0)  # (T_seq,)
            labels_list.append(merged_labels)

        inputs_embeds = self._pad_batch(embeds_list, 0.0)  # (B, T_seq, 2048)
        attention_mask = self._pad_batch(masks_list, 0).squeeze(-1)  # (B, T_seq)
        merged_labels = self._pad_batch(labels_list, IGNORE_INDEX).squeeze(-1)  # (B, T_seq)
        merged_labels[attention_mask == 0] = IGNORE_INDEX
        return inputs_embeds, attention_mask, merged_labels

    def bind_tokenizer(self, tokenizer: AutoTokenizer) -> None:
        """绑定 tokenizer 并向词表注册 <image> 特殊 token。"""
        self._tokenizer = tokenizer
        if IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            self.llm.resize_token_embeddings(len(tokenizer))

    def train(self, mode: bool = True):
        """进入训练模式，但保持冻结的 vision/llm 为 eval。"""
        super().train(mode)
        if self.config.freeze_vision:
            self.vision.eval()
        if self.config.freeze_llm:
            self.llm.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T_text)
        pixel_values: torch.Tensor,  # (B, 3, 384, 384)
        labels: Optional[torch.Tensor] = None,  # (B, T_text)
    ) -> CausalLMOutputWithPast:
        """训练前向：图文拼接后送入 Qwen，返回含 loss 的输出。"""
        if self._tokenizer is None:
            raise RuntimeError("请先调用 bind_tokenizer()")

        image_embeds = self.encode_images(pixel_values)  # (B, 576, 2048)
        # (B, T_seq, 2048), (B, T_seq), (B, T_seq)；labels 由 (B, T_text) 扩图对齐
        inputs_embeds, attention_mask, labels = self._prepare_inputs(
            input_ids, image_embeds, labels
        )  
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)  # logits (B, T_seq, vocab)

    @torch.no_grad()
    def generate(
        self,
        tokenizer: AutoTokenizer,
        pixel_values: torch.Tensor,  # (1, 3, 384, 384)
        question: str,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """看图问答推理，自回归生成 answer 文本。"""
        self.bind_tokenizer(tokenizer)
        prompt = self._build_prompt(question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)  # (1, T_text)
        image_embeds = self.encode_images(pixel_values)
        inputs_embeds, attention_mask, _ = self._prepare_inputs(input_ids, image_embeds)

        prompt_len = inputs_embeds.size(1)  # T_seq，例：推理 T_text=36 → T_seq=611
        out_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            # do_sample：False=贪心（每步取最高概率，temperature/top_p/top_k 无效）；
            #            True=按概率采样（可通过 **kwargs 传 temperature/top_p/top_k）
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )  # (1, T_seq + T_new)
        new_ids = out_ids[0, prompt_len:] if out_ids.size(1) > prompt_len else out_ids[0]  # (T_new,)
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def load_VLM_v1(device: str = "cuda") -> tuple[VLM_v1_Model, AutoTokenizer]:
    """加载 VLM v1 模型与 tokenizer。"""
    with suppress_load_report():
        model = VLM_v1_Model().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model.config.qwen_path)
    model.bind_tokenizer(tokenizer)
    return model, tokenizer


def load_VLM_v1_image_processor():
    """SigLIP 图像预处理（与 vision tower 配套）。"""
    cfg = VLM_v1_Config()
    with suppress_load_report():
        return AutoProcessor.from_pretrained(cfg.siglip_path)


if __name__ == "__main__":
    import sys
    from PIL import Image

    sys.path.insert(0, str(ROOT))

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 GPU，请确认 CUDA 可用")

    instructft_checkpoint = ROOT / "checkpoints/instructft/projector.pt"
    model, tokenizer = load_VLM_v1(device=device)
    if instructft_checkpoint.is_file():
        model.projector.load_state_dict(
            torch.load(instructft_checkpoint, map_location=device, weights_only=True)
        )
        print(f"已加载 InstructFT projector → {instructft_checkpoint.relative_to(ROOT)}")
    else:
        print(f"警告: InstructFT checkpoint 不存在 → {instructft_checkpoint}")

    processor = load_VLM_v1_image_processor()
    model.eval()

    image_path = ROOT / "models" / "demo" / "000000000139.jpg"
    question = "请用一句话描述这张图片。"
    pixel_values = processor(
        images=Image.open(image_path).convert("RGB"),
        return_tensors="pt",
    ).pixel_values.to(device, dtype=torch.float32)

    # 1. 视觉编码 → 语义 token
    with torch.no_grad():
        img_tokens = model.encode_images(pixel_values)
    print(f"[1] image tokens shape: {tuple(img_tokens.shape)}")  # (1, 576, 2048)

    # 2. generate（InstructFT 推理）
    answer = model.generate(tokenizer, pixel_values, question, max_new_tokens=64)
    print(f"[2] answer: {answer}")
