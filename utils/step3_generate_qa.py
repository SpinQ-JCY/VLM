"""Step 3.2：指令微调数据 — vLLM 看图生成 2 条问答（1 描述场景 + 1 针对性问题）。"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1] 

DEFAULT_IMAGE_DIR = ROOT / "data" / "COCO2014" / "train2014"
DEFAULT_OUTPUT = ROOT / "data" / "qa" / "coco_train_qa_qwen3.5.json"

# 描述类问题共 10 种问法，每张图随机选 1 条
SCENE_QUESTIONS = [
    "请描述图片主要内容",
    "请简要描述这张图片",
    "这张图片展示了什么？",
    "用一句话概括图片中的场景",
    "请描述图片中的主要场景和内容",
    "图片里发生了什么？",
    "请说说这张图片拍的是什么",
    "概括一下这张图片的核心内容",
    "请描述你在这张图片里看到的主要画面",
    "这张图片的主要内容是什么？",
]

MAX_SCENE_ANSWER_CHARS = 30   # 第 1 问：描绘场景
MAX_DETAIL_ANSWER_CHARS = 15  # 第 2 问：具体问题
QA_PER_IMAGE = 2

SYSTEM_PROMPT = (
    "你是视觉问答数据标注助手。根据图片生成问答；回答必须简洁直接，不要解释。"
)


def build_user_prompt(question1: str) -> str:
    return f"""请仔细观察图片，完成以下任务，仅返回 JSON，不要其他文字。

示例（格式参考；question2 / answer1 / answer2 须根据当前图片重新生成，不要照抄）：
若第一个问题是「请描述图片主要内容」，可返回：
{{
  "question2": "图中的狗是什么颜色的？",
  "answer1": "草地上，一个男孩正在和一只棕白相间的狗玩耍",
  "answer2": "狗的颜色是棕白色。"
}}
说明：answer1 描绘整体场景（≤{MAX_SCENE_ANSWER_CHARS} 字）；question2 问具体细节；answer2 极简作答（≤{MAX_DETAIL_ANSWER_CHARS} 字）。

本次第一个问题为「{question1}」，请返回：
{{
  "question2": "针对本图的一个具体问题（如物体颜色、位置、动作、天气、数量、室内室外等）",
  "answer1": "对「{question1}」的回答",
  "answer2": "对 question2 的回答"
}}
要求：
- question2 必须与图中内容相关，不要出通用题
- answer1 不超过 {MAX_SCENE_ANSWER_CHARS} 个汉字；answer2 不超过 {MAX_DETAIL_ANSWER_CHARS} 个汉字"""


def encode_image(image_path: Path) -> tuple[str, str]:
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
    return mime, base64.b64encode(image_path.read_bytes()).decode()


def truncate_cn(text: str, max_len: int) -> str:
    text = text.strip().replace("\n", "")
    return text[:max_len] if len(text) > max_len else text


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    data = json.loads(text)
    for key in ("question2", "answer1", "answer2"):
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"缺少或为空: {key}")
    return data


def call_vllm(mime: str, b64: str, host: str, model: str, user_text: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ],
        "temperature": 0.4,
        "top_p": 0.8,
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


def generate_for_image(
    image_path: Path,
    host: str,
    model: str,
    max_retries: int = 3,
) -> list[dict]:
    rel_path = image_path.relative_to(ROOT).as_posix()
    mime, b64 = encode_image(image_path)
    question1 = random.choice(SCENE_QUESTIONS)
    user_prompt = build_user_prompt(question1)
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            raw = call_vllm(mime, b64, host, model, user_prompt)
            data = parse_json_response(raw)
            return [
                {
                    "image": rel_path,
                    "question": question1,
                    "answer": truncate_cn(str(data["answer1"]), MAX_SCENE_ANSWER_CHARS),
                },
                {
                    "image": rel_path,
                    "question": str(data["question2"]).strip(),
                    "answer": truncate_cn(str(data["answer2"]), MAX_DETAIL_ANSWER_CHARS),
                },
            ]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"[重试 {attempt + 1}/{max_retries}] {rel_path}: {e}", flush=True)

    raise RuntimeError(f"生成失败: {rel_path}") from last_err


def load_records(output: Path) -> tuple[list[dict], set[str]]:
    """加载已有记录；仅保留已完成 2 条问答的图片，丢弃中断产生的残缺条目。"""
    if not output.is_file():
        return [], set()
    records = json.loads(output.read_text(encoding="utf-8"))
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_image[r["image"]].append(r)
    done = {img for img, items in by_image.items() if len(items) >= QA_PER_IMAGE}
    clean = [r for r in records if r["image"] in done]
    return clean, done


def save_records(output: Path, records: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda r: (r["image"], r["question"]))
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3.2：vLLM 看图生成指令微调问答 JSON")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=0, help="0 表示全部")
    parser.add_argument("--host", default="http://localhost:8033")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--workers", type=int, default=10, help="并发处理的图片数")
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="每完成 N 条问答写入一次 JSON（默认 100）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机选描述类问题的种子")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    images = sorted(args.image_dir.glob("*.jpg"))
    if args.num_images > 0:
        images = images[: args.num_images]
    if not images:
        sys.exit(f"未找到图片: {args.image_dir}")

    records, done_images = load_records(args.output)
    pending = [p for p in images if p.relative_to(ROOT).as_posix() not in done_images]
    print(
        f"图片总数 {len(images)} | 已完成 {len(done_images)} 张 | 待处理 {len(pending)} 张 | "
        f"已有问答 {len(records)} 条 → {args.output.relative_to(ROOT)}",
        flush=True,
    )
    if not pending:
        print("无需处理，已全部完成。", flush=True)
        return

    lock = Lock()
    qa_since_save = 0
    completed_images = 0
    total_pending = len(pending)

    def flush(force: bool = False) -> None:
        nonlocal qa_since_save
        with lock:
            if not force and qa_since_save < args.save_every:
                return
            save_records(args.output, records)
            print(
                f">>> 已写入 {len(records)} 条（+{qa_since_save} 条待落盘已保存）",
                flush=True,
            )
            qa_since_save = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_for_image, path, args.host, args.model): path for path in pending
        }
        for future in as_completed(futures):
            path = futures[future]
            rel = path.relative_to(ROOT).as_posix()
            try:
                new_records = future.result()
            except Exception as e:
                print(f"[失败] {rel}: {e}", flush=True)
                continue

            with lock:
                records.extend(new_records)
                qa_since_save += len(new_records)
                completed_images += 1
                if completed_images % 10 == 0 or completed_images == total_pending:
                    print(
                        f"[{completed_images}/{total_pending}] {rel} | 累计 {len(records)} 条",
                        flush=True,
                    )
            flush()

    flush(force=True)
    print(f"完成 → {args.output.relative_to(ROOT)}（共 {len(records)} 条）", flush=True)


if __name__ == "__main__":
    main()
