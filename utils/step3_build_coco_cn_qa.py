"""Step 3.1：语义对齐数据 — 将 COCO-CN human-written caption 转为 QA JSON。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTION_FILE = (
    ROOT / "data" / "COCO2014" / "coco-cn-version1805v1.1" / "imageid.human-written-caption.txt"
)
DEFAULT_OUTPUT = ROOT / "data" / "qa" / "coco_cn_qa.json"

QUESTION = "请简要描述图片主要内容"

def image_id_to_path(image_id: str) -> Path:
    """COCO_train2014_xxx / COCO_val2014_xxx → 相对项目根的图片路径。"""
    if image_id.startswith("COCO_train2014_"):
        subdir = "train2014"
    elif image_id.startswith("COCO_val2014_"):
        subdir = "val2014"
    else:
        raise ValueError(f"未知 image_id 前缀: {image_id}")
    return Path("data") / "COCO2014" / subdir / f"{image_id}.jpg"


def load_captions(caption_file: Path, only_first: bool = True) -> list[tuple[str, str]]:
    """解析 caption 文件，返回 (image_id, caption) 列表。"""
    records: list[tuple[str, str]] = []
    with caption_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, caption = line.split("\t", 1)
            if only_first and not key.endswith("#0"):
                continue
            image_id = key.split("#", 1)[0]
            records.append((image_id, caption.strip()))
    return records


def build_qa_records(
    caption_file: Path,
    only_first: bool = True,
    check_images: bool = False,
) -> list[dict]:
    records = []
    missing = 0
    for image_id, caption in load_captions(caption_file, only_first=only_first):
        rel_path = image_id_to_path(image_id).as_posix()
        if check_images and not (ROOT / rel_path).is_file():
            missing += 1
            continue
        records.append({
            "image": rel_path,
            "question": QUESTION,
            "answer": caption,
        })
    if check_images and missing:
        print(f"跳过 {missing} 条（图片尚未下载）", file=sys.stderr)
    records.sort(key=lambda r: r["image"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3.1：COCO-CN caption → 语义对齐 QA JSON")
    parser.add_argument("--caption-file", type=Path, default=DEFAULT_CAPTION_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all-captions",
        action="store_true",
        help="保留 #1…#n 多条描述（默认仅 #0）",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="仅保留本地已有图片的样本",
    )
    args = parser.parse_args()

    if not args.caption_file.is_file():
        sys.exit(f"未找到 caption 文件: {args.caption_file}")

    records = build_qa_records(
        args.caption_file,
        only_first=not args.all_captions,
        check_images=args.check_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {len(records)} 条 → {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
