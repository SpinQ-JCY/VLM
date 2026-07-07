"""Step 6：VLM v1 推理测试"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vlms.VLM_v1_model import load_VLM_v1, load_VLM_v1_image_processor  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints/instructft/projector.pt"

TEST_QUESTIONS = [
    "请简要描述图片主要内容",
]

TEST_IMAGES = [
    "data/COCO2014/val2014/COCO_val2014_000000000139.jpg",
    "data/COCO2014/val2014/COCO_val2014_000000000285.jpg",
    "data/COCO2014/val2014/COCO_val2014_000000000632.jpg",
]


def main():
    parser = argparse.ArgumentParser(description="Step 6：VLM v1 推理测试")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 GPU")

    model, tokenizer = load_VLM_v1(device="cuda")
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if ckpt.is_file():
        model.projector.load_state_dict(
            torch.load(ckpt, map_location="cuda", weights_only=True)
        )
        print(f"已加载 → {ckpt.relative_to(ROOT)}")
    else:
        print(f"警告: checkpoint 不存在 → {ckpt}")

    model.eval()
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
