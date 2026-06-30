"""VLM v1：SigLIP2 视觉编码 + 2 层 MLP 投影 + Qwen3-1.7B 文本生成。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, SiglipVisionModel
from transformers.modeling_outputs import CausalLMOutputWithPast

ROOT = Path(__file__).resolve().parents[2]
IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100


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
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden, llm_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_hidden, llm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VLM_v1_Model(nn.Module):
    """
    数据流（单条样本 B=1；训练 max_seq_len=704 = 576 图 + 128 文本）。

    举例：question="这张图里有什么？"，answer="图片中有一只猫和一张沙发。"

        prompt 字符串（训练时拼接 answer 在后面）：
            <|im_start|>user
            <image>
            这张图里有什么？
            
            <|im_start|>assistant
            图片中有一只猫和一张沙发。

        ① 视觉路径
            pixel_values                 (1, 3, 384, 384)
              → SigLIP Vision            (1, 576, 1152)
              → Projector                (1, 576, 2048)

        ② 文本 tokenize（假设 T_text=28，其中 1 个是 <image> 占位 id）
            input_ids                    (28,)
              → embed_tokens             (28, 2048)

        ③ 在 pos=3 处展开 <image>（假设 prefix 共 3 个 token：user 段开头）
            prefix  text[:3]             (3, 2048)    <|im_start|> user \\n
            image   576 个视觉 token      (576, 2048)  替换掉 1 个 <image> id
            suffix  text[4:]             (24, 2048)   问题 + im_end + assistant 头 + answer
            ────────────────────────────────────────
            merged  cat 沿 seq 维         (603, 2048)  603 = 3 + 576 + 24 = 28 - 1 + 576

        ④ batch padding 后送入 Qwen3 LLM
            inputs_embeds                (B, 603, 2048)
              → Causal LM                (B, 603, vocab)   只对 answer 部分算 loss
    """

    def __init__(self, config: VLM_v1_Config | None = None):
        super().__init__()
        self.config = config or VLM_v1_Config()
        self._tokenizer: AutoTokenizer | None = None

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
        if self.config.freeze_llm:
            self.llm.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """提取并投影图片语义 token：(B, num_image_tokens, llm_hidden)。"""
        vision_out = self.vision(pixel_values=pixel_values).last_hidden_state
        return self.projector(vision_out)

    def _build_prompt(self, question: str) -> str:
        im_end = self._tokenizer.eos_token
        return (
            f"<|im_start|>user\n{IMAGE_TOKEN}\n{question}\n"
            f"{im_end}\n"
            f"<|im_start|>assistant\n"
        )

    def _image_token_pos(self, ids: torch.Tensor) -> int:
        image_token_id = self._tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
        if len(pos) != 1:
            raise ValueError(f"每条样本应包含 1 个 {IMAGE_TOKEN}，实际 {len(pos)} 个")
        return pos.item()

    def _pad_batch(self, tensors: list[torch.Tensor], pad_value: float | int) -> torch.Tensor:
        max_len = max(t.size(0) for t in tensors)
        out = torch.full(
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
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将 <image> 占位符展开为 num_image_tokens 个视觉 token。"""
        embeds_list, masks_list, labels_list = [], [], []

        for i in range(input_ids.size(0)):
            ids = input_ids[i]
            pos = self._image_token_pos(ids)

            text_embeds = self.llm.model.embed_tokens(ids)
            img = image_embeds[i].to(text_embeds.dtype)
            merged_embeds = torch.cat([text_embeds[:pos], img, text_embeds[pos + 1 :]], dim=0)
            embeds_list.append(merged_embeds)
            masks_list.append(torch.ones(merged_embeds.size(0), device=self.device, dtype=torch.long))

            lab = labels[i] if labels is not None else ids
            img_lab = torch.full((self.config.num_image_tokens,), IGNORE_INDEX, device=lab.device, dtype=lab.dtype)
            merged_labels = torch.cat([lab[:pos], img_lab, lab[pos + 1 :]], dim=0)
            labels_list.append(merged_labels)

        inputs_embeds = self._pad_batch(embeds_list, 0.0)
        attention_mask = self._pad_batch(masks_list, 0).squeeze(-1)
        merged_labels = self._pad_batch(labels_list, IGNORE_INDEX).squeeze(-1)
        merged_labels[attention_mask == 0] = IGNORE_INDEX
        return inputs_embeds, attention_mask, merged_labels

    def bind_tokenizer(self, tokenizer: AutoTokenizer) -> None:
        """注册 <image> 占位符（需在 tokenize 前调用一次）。"""
        self._tokenizer = tokenizer
        if IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            self.llm.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        if self._tokenizer is None:
            raise RuntimeError("请先调用 bind_tokenizer()")

        image_embeds = self.encode_images(pixel_values)
        inputs_embeds, attention_mask, labels = self._prepare_inputs(input_ids, image_embeds, labels)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        tokenizer: AutoTokenizer,
        pixel_values: torch.Tensor,
        question: str,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        self.bind_tokenizer(tokenizer)
        prompt = self._build_prompt(question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        image_embeds = self.encode_images(pixel_values)
        inputs_embeds, attention_mask, _ = self._prepare_inputs(input_ids, image_embeds)

        prompt_len = inputs_embeds.size(1)
        out_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )
        new_ids = out_ids[0, prompt_len:] if out_ids.size(1) > prompt_len else out_ids[0]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def load_VLM_v1(config: VLM_v1_Config | None = None, device: str = "cuda") -> tuple[VLM_v1_Model, AutoTokenizer]:
    """加载 VLM v1 模型与 tokenizer。"""
    cfg = config or VLM_v1_Config()
    model = VLM_v1_Model(cfg).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_path)
    model.bind_tokenizer(tokenizer)
    return model, tokenizer


def load_VLM_v1_image_processor():
    """SigLIP 图像预处理（与 vision tower 配套）。"""
    return AutoProcessor.from_pretrained(VLM_v1_Config().siglip_path)


if __name__ == "__main__":
    from PIL import Image

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 GPU，请确认 CUDA 可用")

    model, tokenizer = load_VLM_v1(device=device)

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

    # 2. forward（训练路径）
    prompt = model._build_prompt(question)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=pixel_values)
    print(f"[2] forward loss: {out.loss.item():.4f}")

    # 3. generate（推理路径）
    answer = model.generate(tokenizer, pixel_values, question, max_new_tokens=64)
    print(f"[3] answer: {answer}")
    print("OK")
