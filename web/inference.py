"""VLM v1 Web 推理封装。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
sys.path.insert(0, str(ROOT))

from models.vlms.VLM_v1_model import load_VLM_v1, load_VLM_v1_image_processor  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints/VLM_v1_sft/projector.pt"


class VLM_v1_Predictor:
    def __init__(
        self,
        checkpoint: Path | str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("需要 GPU")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.checkpoint = Path(checkpoint)

        self.model, self.tokenizer = load_VLM_v1(device=device)
        if self.checkpoint.is_file():
            state = torch.load(self.checkpoint, map_location=device, weights_only=True)
            self.model.projector.load_state_dict(state)
        self.model.eval()
        self.processor = load_VLM_v1_image_processor()

    @torch.no_grad()
    def predict(self, image_path: str | Path, question: str) -> str:
        pv = self.processor(
            images=Image.open(image_path).convert("RGB"),
            return_tensors="pt",
        ).pixel_values.to(self.device, dtype=torch.float32)
        return self.model.generate(
            self.tokenizer,
            pv,
            question.strip(),
            max_new_tokens=self.max_new_tokens,
        )

    def device_name(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return f"cuda · {torch.cuda.get_device_name(0)}"
        return self.device
