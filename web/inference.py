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

DEFAULT_CHECKPOINT = ROOT / "checkpoints/instructft/projector.pt"


def resolve_project_path(path: Path | str, root: Path = ROOT) -> Path:
    """解析权重/LoRA 路径：支持相对项目根（checkpoints/...）或相对当前目录（../checkpoints/...）。"""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    candidates: list[Path] = []
    for base in (Path.cwd(), root):
        candidate = (base / p).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return (root / p).resolve() if ".." not in p.parts else (Path.cwd() / p).resolve()


class VLM_v1_Predictor:
    def __init__(
        self,
        checkpoint: Path | str = DEFAULT_CHECKPOINT,
        lora_dir: Path | str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("需要 GPU")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.checkpoint = resolve_project_path(checkpoint)

        lora_path = resolve_project_path(lora_dir) if lora_dir is not None else None
        use_lora = lora_path is not None or self.checkpoint.parent.name.endswith("_lora")

        if use_lora:
            from models.vlms.VLM_v1_lora import load_for_inference, resolve_lora_dir  # noqa: E402

            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"projector 不存在 → {self.checkpoint}")
            self.lora_dir = resolve_lora_dir(self.checkpoint, lora_path)
            self.model, self.tokenizer = load_for_inference(
                self.checkpoint, self.lora_dir, device=device
            )
        else:
            self.lora_dir = None
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

    def checkpoint_rel(self) -> str:
        try:
            return self.checkpoint.relative_to(ROOT).as_posix()
        except ValueError:
            return str(self.checkpoint)

    def lora_dir_rel(self) -> str | None:
        if self.lora_dir is None:
            return None
        try:
            return self.lora_dir.relative_to(ROOT).as_posix()
        except ValueError:
            return str(self.lora_dir)
