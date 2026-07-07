"""Step 6b：InstructFT + LoRA 推理测试"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vlms.VLM_v1_lora import load_for_inference, resolve_lora_dir  # noqa: E402
from models.vlms.VLM_v1_model import load_VLM_v1_image_processor  # noqa: E402

DEFAULT_PROJECTOR = ROOT / "checkpoints/instructft_lora/projector.pt"
DEFAULT_LORA = ROOT / "checkpoints/instructft_lora/lora"

TEST_QUESTIONS = ["请简要描述图片主要内容"]
TEST_IMAGES = [
    "data/test_imgs/img1.jpg",
    "data/test_imgs/img2.jpg",
]


def main():
    parser = argparse.ArgumentParser(description="Step 6b：InstructFT + LoRA 推理测试")
    parser.add_argument("--projector", type=Path, default=DEFAULT_PROJECTOR)
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    projector = (ROOT / args.projector).resolve()
    lora_dir = resolve_lora_dir(projector, args.lora_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 GPU")

    model, tokenizer = load_for_inference(projector, lora_dir, device="cuda")
    processor = load_VLM_v1_image_processor()

    for rel_path in TEST_IMAGES:
        path = ROOT / rel_path
        if not path.is_file():
            print(f"跳过: {rel_path}")
            continue
        pv = processor(Image.open(path).convert("RGB"), return_tensors="pt").pixel_values
        pv = pv.to("cuda", dtype=torch.float32)
        print(f"\n{rel_path}")
        for q in TEST_QUESTIONS:
            with torch.no_grad():
                ans = model.generate(tokenizer, pv, q, max_new_tokens=args.max_new_tokens)
            print(f"  Q: {q}\n  A: {ans}")


if __name__ == "__main__":
    main()
