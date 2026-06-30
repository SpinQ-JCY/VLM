"""用 vLLM (Qwen3.5-9B) 为 COCO 图片批量生成视觉问答 JSON。"""

import argparse
import base64
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from question_templates import CATEGORY_NAMES, QUESTION_TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "data" / "COCO2017" / "val2017"
DEFAULT_OUTPUT = ROOT / "data" / "qa" / "coco_val_qa.json"

SYSTEM_PROMPT = (
    "请根据图片内容简要回答问题。回答问题时，直接给出答案，不用解释。如果涉及具体目标，需要在答案中明确其名称。"
)


def encode_image(image_path: Path) -> tuple[str, str]:
    """读取图片并转为 base64。"""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
    return mime, base64.b64encode(image_path.read_bytes()).decode()


def ask_vllm(mime: str, b64: str, question: str, host: str, model: str) -> str:
    """调用 vLLM 多模态 API 获取回答。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        msg = json.load(resp)["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning") or "").strip()


def build_tasks(image_paths: list[Path]) -> list[dict]:
    """每张图 6 类问题，每类随机选 1 种问法。"""
    tasks = []
    for image_path in image_paths:
        rel_path = image_path.relative_to(ROOT).as_posix()
        mime, b64 = encode_image(image_path)
        for category_id, questions in QUESTION_TEMPLATES.items():
            tasks.append({
                "image": rel_path,
                "category": category_id,
                "category_name": CATEGORY_NAMES[category_id],
                "question": random.choice(questions),
                "mime": mime,
                "b64": b64,
            })
    return tasks


def run_batch(tasks: list[dict], host: str, model: str, workers: int) -> list[dict]:
    """并发执行一批问答任务。"""
    records = []
    lock = Lock()
    done = 0
    total = len(tasks)

    def work(task: dict) -> dict:
        answer = ask_vllm(task["mime"], task["b64"], task["question"], host, model)
        return {
            "image": task["image"],
            "category": task["category"],
            "category_name": task["category_name"],
            "question": task["question"],
            "answer": answer,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            with lock:
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"[{done}/{total}] {record['image']} | cat={record['category']}", flush=True)
    return records


def save(output: Path, records: list[dict]) -> None:
    """写入 JSON 文件。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda r: (r["image"], r["category"]))
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 COCO 视觉问答 JSON")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=0, help="0 表示全部")
    parser.add_argument("--batch-size", type=int, default=50, help="每批图片数，批完成后写入")
    parser.add_argument("--host", default="http://localhost:8033")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    images = sorted(args.image_dir.glob("*.jpg"))
    if args.num_images > 0:
        images = images[: args.num_images]
    if not images:
        sys.exit(f"未找到图片: {args.image_dir}")

    records: list[dict] = []
    batch_total = (len(images) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(images), args.batch_size):
        batch = images[i : i + args.batch_size]
        batch_idx = i // args.batch_size + 1
        print(f"\n>>> 批次 {batch_idx}/{batch_total}：{len(batch)} 张", flush=True)

        records.extend(run_batch(build_tasks(batch), args.host, args.model, args.workers))
        save(args.output, records)
        print(f">>> 已写入 {len(records)} 条", flush=True)

    print(f"完成 → {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
