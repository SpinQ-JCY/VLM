"""VLM v1 + Qwen LoRA：InstructFT 阶段联合训练 Projector 与 LLM adapter。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

from models.vlms.VLM_v1_model import VLM_v1_Model, load_VLM_v1

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "v_proj", "o_proj")


class VLM_v1_LoraModel(VLM_v1_Model):
    def __init__(self, lora_rank: int = LORA_RANK, lora_alpha: int = LORA_ALPHA):
        super().__init__()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
            target_modules=list(LORA_TARGET_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.vision.eval()
        self.projector.train(mode)
        self.llm.train(mode)
        return self


def load_for_train(
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    device: str = "cuda",
) -> tuple[VLM_v1_LoraModel, AutoTokenizer]:
    model = VLM_v1_LoraModel(lora_rank=lora_rank, lora_alpha=lora_alpha).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.config.qwen_path)
    model.bind_tokenizer(tokenizer)
    return model, tokenizer


def load_for_inference(
    projector: Path | str,
    lora_dir: Path | str,
    device: str = "cuda",
) -> tuple[VLM_v1_Model, AutoTokenizer]:
    model, tokenizer = load_VLM_v1(device=device)
    state = torch.load(Path(projector), map_location=device, weights_only=True)
    model.projector.load_state_dict(state)
    model.llm = PeftModel.from_pretrained(model.llm, str(lora_dir))
    model.eval()
    return model, tokenizer
