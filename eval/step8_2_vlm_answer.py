"""
Step 8.2：训练好的 VLM v1 逐条回答 Step 8.1 基准问题（每题单独推理一次）。

每条记录带上 type（scene / detail），便于 Step 8.3 分维度汇总 Align 与 InstructFT 得分。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ROOT))

from step8_common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    STEP8_1_OUTPUT,
    checkpoint_weight_tag,
    load_json,
    project_rel_path,
    resolve_project_path,
    save_json,
    step8_2_output_path,
)
from models.vlms.VLM_v1_model import load_VLM_v1, load_VLM_v1_image_processor  # noqa: E402


def load_benchmark_items(path: Path) -> tuple[dict, list[dict]]:
    data = load_json(path)
    if isinstance(data, dict) and "items" in data:
        return data.get("meta", {}), data["items"]
    if isinstance(data, list):
        return {}, data
    raise ValueError(f"无法解析 Step 8.1 输出: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8.2：VLM v1 逐条回答测评问题")
    parser.add_argument("--input", type=Path, default=STEP8_1_OUTPUT)
    parser.add_argument("--output", type=Path, default=None, help="默认随 --checkpoint 自动加权重后缀")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    checkpoint = resolve_project_path(args.checkpoint)
    args.checkpoint = checkpoint
    if args.output is None:
        args.output = step8_2_output_path(checkpoint)
    elif not args.output.is_absolute():
        args.output = (ROOT / args.output).resolve()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("需要 GPU")

    bench_meta, items = load_benchmark_items(args.input)
    if not items:
        sys.exit(f"Step 8.1 数据为空: {args.input}")

    existing = load_json(args.output)
    if isinstance(existing, dict) and "items" in existing:
        done_ids = {r["qa_id"] for r in existing["items"] if r.get("model_answer") is not None}
        results = existing["items"]
    else:
        done_ids = set()
        results = []

    pending = [it for it in items if it["qa_id"] not in done_ids]
    ckpt_rel = project_rel_path(checkpoint)
    ckpt_tag = checkpoint_weight_tag(checkpoint)

    def output_meta() -> dict:
        return {
            **bench_meta,
            "checkpoint": str(ckpt_rel),
            "checkpoint_tag": ckpt_tag,
            "max_new_tokens": args.max_new_tokens,
        }

    print(
        f"基准 {len(items)} 条 | 已完成 {len(done_ids)} 条 | 待推理 {len(pending)} 条 | "
        f"checkpoint → {ckpt_rel} | 输出 → {project_rel_path(args.output)}",
        flush=True,
    )

    if not pending and results:
        print("无需处理，已全部完成。", flush=True)
        return

    if not results:
        results = [
            {
                "qa_id": it["qa_id"],
                "image": it["image"],
                "type": it["type"],
                "question": it["question"],
                "reference_answer": it["reference_answer"],
                "model_answer": None,
                "latency_s": None,
            }
            for it in items
        ]
    else:
        by_id = {r["qa_id"]: r for r in results}
        for it in items:
            if it["qa_id"] not in by_id:
                by_id[it["qa_id"]] = {
                    "qa_id": it["qa_id"],
                    "image": it["image"],
                    "type": it["type"],
                    "question": it["question"],
                    "reference_answer": it["reference_answer"],
                    "model_answer": None,
                    "latency_s": None,
                }
        results = [by_id[it["qa_id"]] for it in items]

    model, tokenizer = load_VLM_v1(device=args.device)
    if checkpoint.is_file():
        model.projector.load_state_dict(
            torch.load(checkpoint, map_location=args.device, weights_only=True)
        )
        print(f"已加载 → {ckpt_rel}", flush=True)
    else:
        print(f"警告: checkpoint 不存在 → {checkpoint}", flush=True)
    model.eval()
    processor = load_VLM_v1_image_processor()

    by_id = {r["qa_id"]: r for r in results}
    completed = 0

    for it in tqdm(pending, desc="VLM 推理", unit="qa"):
        image_path = ROOT / it["image"]
        if not image_path.is_file():
            print(f"[跳过] 图片不存在: {it['image']}", flush=True)
            continue

        pv = processor(
            images=Image.open(image_path).convert("RGB"),
            return_tensors="pt",
        ).pixel_values.to(args.device, dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            answer = model.generate(
                tokenizer,
                pv,
                it["question"],
                max_new_tokens=args.max_new_tokens,
            )
        latency = round(time.perf_counter() - t0, 3)

        rec = by_id[it["qa_id"]]
        rec["model_answer"] = answer
        rec["latency_s"] = latency
        completed += 1

        if completed % args.save_every == 0 or completed == len(pending):
            save_json(args.output, {"meta": output_meta(), "items": results})

    save_json(args.output, {"meta": output_meta(), "items": results})
    print(f"完成 → {project_rel_path(args.output)}（共 {len(results)} 条）", flush=True)


if __name__ == "__main__":
    main()
