"""
Step 8.0：从验证集随机抽样图片，写入固定清单 JSON（仅需运行一次）。

后续 Step 8.1 / 8.2 / 8.3 默认从该 JSON 加载图片，不再重新抽样。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from step8_common import (
    DEFAULT_NUM_EVAL_IMAGES,
    DEFAULT_VAL_DIR,
    ROOT,
    STEP8_0_OUTPUT,
    load_json,
    save_json,
)


def sample_image_paths(
    image_dir: Path,
    num_images: int,
    seed: int,
) -> list[str]:
    all_images = sorted(image_dir.glob("*.jpg"))
    if not all_images:
        return []
    rng = random.Random(seed)
    if num_images > 0 and num_images < len(all_images):
        picked = sorted(rng.sample(all_images, num_images))
    else:
        picked = all_images
    return [p.relative_to(ROOT).as_posix() for p in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8.0：随机抽样测评图片清单（仅运行一次）")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output", type=Path, default=STEP8_0_OUTPUT)
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_EVAL_IMAGES,
        help="随机抽样张数，0=全部",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的图片清单（默认已存在则跳过）",
    )
    args = parser.parse_args()

    if args.output.is_file() and not args.force:
        data = load_json(args.output)
        images = data.get("images", []) if isinstance(data, dict) else []
        print(
            f"图片清单已存在 → {args.output.relative_to(ROOT)}（{len(images)} 张），跳过抽样。"
            f"如需重抽请加 --force",
            flush=True,
        )
        return

    if not args.image_dir.is_dir():
        sys.exit(f"图片目录不存在: {args.image_dir}")

    images = sample_image_paths(args.image_dir, args.num_images, args.seed)
    if not images:
        sys.exit(f"未找到图片: {args.image_dir}")

    payload = {
        "meta": {
            "seed": args.seed,
            "num_images": len(images),
            "image_dir": str(args.image_dir.relative_to(ROOT)),
        },
        "images": images,
    }
    save_json(args.output, payload)
    print(
        f"已抽样 {len(images)} 张 → {args.output.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
