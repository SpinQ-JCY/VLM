"""
Step 8.1：由 Qwen3.5-9B 对 Step 8.0 清单中的图片看图生成基准问答。

图片列表从 step8_0_images.json 加载，不在此步重新抽样。

每张图 2 条：
  - scene：描述图片主要内容（对应 align 语义对齐测评）
  - detail：针对图片细节的自拟问答（对应 sft 指令微调测评）

依赖：vLLM 服务（默认 localhost:8033）
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from step8_common import (
    DEFAULT_VLLM_HOST,
    DEFAULT_VLLM_MODEL,
    MAX_DETAIL_ANSWER_CHARS,
    MAX_SCENE_ANSWER_CHARS,
    QA_PER_IMAGE,
    QA_TYPE_DETAIL,
    QA_TYPE_SCENE,
    ROOT,
    SCENE_QUESTIONS,
    STEP8_0_OUTPUT,
    STEP8_1_OUTPUT,
    build_benchmark_user_prompt,
    call_vllm_vision,
    load_image_set,
    load_json,
    make_qa_id,
    parse_json_response,
    save_json,
    truncate_cn,
)

STEP8_1_SYSTEM = "你是视觉问答数据标注助手。根据图片生成问答；回答必须简洁直接，不要解释。"


def generate_for_image(
    image_path: Path,
    host: str,
    model: str,
    max_retries: int = 3,
) -> list[dict]:
    rel_path = image_path.relative_to(ROOT).as_posix()
    question1 = random.choice(SCENE_QUESTIONS)
    user_prompt = build_benchmark_user_prompt(question1)
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            raw = call_vllm_vision(
                host, model, STEP8_1_SYSTEM, user_prompt, image_path, max_tokens=256
            )
            data = parse_json_response(raw)
            for key in ("question2", "answer1", "answer2"):
                if key not in data or not str(data[key]).strip():
                    raise ValueError(f"缺少或为空: {key}")

            return [
                {
                    "qa_id": make_qa_id(rel_path, QA_TYPE_SCENE),
                    "image": rel_path,
                    "type": QA_TYPE_SCENE,
                    "question": question1,
                    "reference_answer": truncate_cn(
                        str(data["answer1"]), MAX_SCENE_ANSWER_CHARS
                    ),
                },
                {
                    "qa_id": make_qa_id(rel_path, QA_TYPE_DETAIL),
                    "image": rel_path,
                    "type": QA_TYPE_DETAIL,
                    "question": str(data["question2"]).strip(),
                    "reference_answer": truncate_cn(
                        str(data["answer2"]), MAX_DETAIL_ANSWER_CHARS
                    ),
                },
            ]
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ValueError,
        ) as e:
            last_err = e
            print(f"[重试 {attempt + 1}/{max_retries}] {rel_path}: {e}", flush=True)

    raise RuntimeError(f"生成失败: {rel_path}") from last_err


def load_done_images(records: list[dict]) -> set[str]:
    by_image: dict[str, int] = defaultdict(int)
    for r in records:
        by_image[r["image"]] += 1
    return {img for img, n in by_image.items() if n >= QA_PER_IMAGE}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8.1：Qwen3.5 生成测评基准问答")
    parser.add_argument(
        "--images-json",
        type=Path,
        default=STEP8_0_OUTPUT,
        help="Step 8.0 图片清单（默认 eval/outputs/step8_0_images.json）",
    )
    parser.add_argument("--output", type=Path, default=STEP8_1_OUTPUT)
    parser.add_argument("--host", default=DEFAULT_VLLM_HOST)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10, help="每完成 N 张图写入一次")
    parser.add_argument("--seed", type=int, default=42, help="随机选描述类问题的种子")
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        image_meta, image_rels = load_image_set(args.images_json)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))

    image_paths = [ROOT / rel for rel in image_rels]

    raw = load_json(args.output)
    if isinstance(raw, dict) and "items" in raw:
        records = raw["items"]
    elif isinstance(raw, list):
        records = raw
    else:
        records = []
    done_images = load_done_images(records)
    pending = [p for p in image_paths if p.relative_to(ROOT).as_posix() not in done_images]

    meta = {
        "images_json": str(args.images_json.relative_to(ROOT)),
        "num_images": len(image_rels),
        "num_images_done": len(done_images),
        "question_seed": args.seed,
        "vllm_model": args.model,
        **image_meta,
    }
    print(
        f"图片清单 {len(image_rels)} 张 ← {args.images_json.relative_to(ROOT)} | "
        f"已完成 {len(done_images)} 张 | 待处理 {len(pending)} 张 | "
        f"已有 {len(records)} 条 → {args.output.relative_to(ROOT)}",
        flush=True,
    )
    if not pending:
        save_json(args.output, {"meta": meta, "items": records})
        print("无需处理，已全部完成。", flush=True)
        return

    lock = Lock()
    completed = 0
    total = len(pending)

    def write_output() -> None:
        save_json(args.output, {"meta": meta, "items": records})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_for_image, path, args.host, args.model): path
            for path in pending
        }
        for future in as_completed(futures):
            path = futures[future]
            rel = path.relative_to(ROOT).as_posix()
            try:
                new_items = future.result()
            except Exception as e:
                print(f"[失败] {rel}: {e}", flush=True)
                continue

            with lock:
                records.extend(new_items)
                completed += 1
                meta["num_images_done"] = len(load_done_images(records))
                if completed % 10 == 0 or completed == total:
                    print(
                        f"[{completed}/{total}] {rel} | 累计 {len(records)} 条",
                        flush=True,
                    )
                if completed % args.save_every == 0 or completed == total:
                    write_output()

    meta["num_images_done"] = len(load_done_images(records))
    save_json(args.output, {"meta": meta, "items": records})
    print(f"完成 → {args.output.relative_to(ROOT)}（共 {len(records)} 条）", flush=True)


if __name__ == "__main__":
    main()
