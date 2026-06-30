"""统计 VLM 训练样本合并后的 LLM 序列长度分布。"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TOKEN = "<image>"
NUM_IMAGE_TOKENS = 576


def main(num_images: int = 100):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-1.7B")
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})

    records = json.loads((ROOT / "data/qa/coco_val_qa.json").read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for r in records:
        by_image[r["image"]].append(r)
    images = sorted(by_image)[:num_images]
    subset = [r for img in images for r in by_image[img]]

    def lengths(question, answer):
        im_end = tokenizer.eos_token
        prompt = (
            f"<|im_start|>user\n{IMAGE_TOKEN}\n{question}\n"
            f"{im_end}\n"
            f"<|im_start|>assistant\n"
        )
        full = prompt + answer
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        text_len = len(full_ids)
        merged_len = text_len - 1 + NUM_IMAGE_TOKENS
        return {
            "prompt_text": len(prompt_ids),
            "answer_text": len(answer_ids),
            "full_text": text_len,
            "merged_llm": merged_len,
        }

    stats = [lengths(r["question"], r["answer"]) for r in subset]

    def summarize(name, vals):
        vals = sorted(vals)
        n = len(vals)
        pct = lambda p: vals[int((n - 1) * p)]
        print(f"\n{name} (n={n})")
        print(
            f"  min={vals[0]}  p50={pct(0.5):.0f}  p90={pct(0.9):.0f}  "
            f"p95={pct(0.95):.0f}  p99={pct(0.99):.0f}  max={vals[-1]}"
        )
        print(f"  mean={statistics.mean(vals):.1f}  stdev={statistics.stdev(vals):.1f}")

    print(f"图片数: {len(images)}")
    print(f"QA 条数: {len(subset)}")

    summarize("文本 token（prompt+answer，含 1 个 <image>）", [s["full_text"] for s in stats])
    summarize("prompt 文本 token", [s["prompt_text"] for s in stats])
    summarize("answer 文本 token", [s["answer_text"] for s in stats])
    summarize("合并后 LLM 序列长度（576 图 token + 文本）", [s["merged_llm"] for s in stats])

    by_cat = defaultdict(list)
    for r, s in zip(subset, stats):
        by_cat[r["category_name"]].append(s["merged_llm"])
    print("\n按问题类别 merged_llm（p50 / max）:")
    for cat in sorted(by_cat, key=lambda c: statistics.median(by_cat[c])):
        v = by_cat[cat]
        print(f"  {cat:8s}  p50={statistics.median(v):.0f}  max={max(v)}")

    merged = sorted(s["merged_llm"] for s in stats)
    print("\n截断上限 vs 超出比例:")
    for cap in [704, 768, 896, 1024, 1280, 1536, 2048]:
        over = sum(1 for x in merged if x > cap)
        print(f"  cap={cap:4d}  超出 {over:3d}/{len(merged)} ({100 * over / len(merged):.1f}%)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n)
