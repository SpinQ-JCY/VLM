"""VLM v1 推理测试：对列表中每张图依次问 6 类问题的第一种问法。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vlms.VLM_v1_model import load_VLM_v1, load_VLM_v1_image_processor  # noqa: E402
from utils.question_templates import CATEGORY_NAMES, QUESTION_TEMPLATES  # noqa: E402

# 每次运行会测列表中的全部图片；可按需增删
TEST_IMAGES: list[str] = [
    "data/COCO2017/val2017/000000000632.jpg",
]


def get_test_questions() -> list[tuple[int, str, str]]:
    """6 类问题，每类取模板列表中的第一个。"""
    return [(cat, CATEGORY_NAMES[cat], QUESTION_TEMPLATES[cat][0]) for cat in sorted(QUESTION_TEMPLATES)]


def load_model(checkpoint: Path | None, device: str):
    model, tokenizer = load_VLM_v1(device=device)
    if checkpoint is not None:
        if not checkpoint.is_file():
            print(f"警告: checkpoint 不存在，使用未训练的 projector → {checkpoint}")
        else:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.projector.load_state_dict(state)
            print(f"已加载 projector: {checkpoint}")
    model.eval()
    return model, tokenizer


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 GPU")

    device = "cuda"
    images = TEST_IMAGES
    questions = get_test_questions()

    model, tokenizer = load_model(args.checkpoint, device)
    processor = load_VLM_v1_image_processor()

    print(f"图片数: {len(images)} | 每图问题数: {len(questions)}")
    print("-" * 60)

    for img_idx, rel_path in enumerate(images, 1):
        image_path = ROOT / rel_path
        if not image_path.is_file():
            print(f"[{img_idx}/{len(images)}] 跳过（文件不存在）: {rel_path}")
            continue

        pixel_values = processor(
            images=Image.open(image_path).convert("RGB"),
            return_tensors="pt",
        ).pixel_values.to(device, dtype=torch.float32)

        print(f"[{img_idx}/{len(images)}] {rel_path}")
        for cat, cat_name, question in questions:
            with torch.no_grad():
                answer = model.generate(
                    tokenizer,
                    pixel_values,
                    question,
                    max_new_tokens=args.max_new_tokens,
                )
            print(f"  [{cat}] {cat_name}")
            print(f"    Q: {question}")
            print(f"    A: {answer}")
        print("-" * 60)

    print("OK")


def main():
    parser = argparse.ArgumentParser(description="VLM v1 推理测试")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/VLM_v1/projector.pt",
        help="projector 权重；传 none 表示不加载（随机初始化）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.checkpoint is not None and str(args.checkpoint).lower() == "none":
        args.checkpoint = None
    run(args)


if __name__ == "__main__":
    main()
