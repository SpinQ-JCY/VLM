"""
Step 8.3：Qwen3.5-9B 作为裁判，结合原图、问题、参考答案与模型回答，按严格标准多维打分。

五维度评分，每维 0–20 分，单题满分 100。
scene 类型汇总为 align（语义对齐）得分；detail 类型汇总为 sft（指令微调）得分。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from step8_common import (
    DEFAULT_VLLM_HOST,
    DEFAULT_VLLM_MODEL,
    QA_TYPE_DETAIL,
    QA_TYPE_SCENE,
    ROOT,
    SCORE_DIM_MAX,
    SCORE_DIM_MIN,
    SCORE_TOTAL_MAX,
    call_vllm_vision,
    default_step8_2_output,
    load_json,
    parse_json_response,
    project_rel_path,
    resolve_project_path,
    save_json,
    step8_3_output_path,
)

SCENE_DIMS = (
    "main_content_accuracy",
    "key_element_coverage",
    "reference_semantic_match",
    "factual_faithfulness",
    "expression_quality",
)

DETAIL_DIMS = (
    "factual_correctness",
    "question_focus",
    "reference_semantic_match",
    "factual_faithfulness",
    "conciseness",
)

DIM_LABELS = {
    "main_content_accuracy": "主要语义准确性",
    "key_element_coverage": "关键元素覆盖",
    "reference_semantic_match": "参考答案语义一致",
    "factual_faithfulness": "事实忠实（无幻觉）",
    "expression_quality": "表达质量",
    "factual_correctness": "事实正确性",
    "question_focus": "答所问",
    "conciseness": "简洁规范",
}

JUDGE_SYSTEM = (
    "你是严格、客观的视觉问答评测裁判。"
    "必须仅依据图片可见内容评判，不得臆测画面外信息。"
    "须对照参考答案，但参考答案不是唯一标准答案；语义等价可给高分。"
    "打分从严：有明显错误、漏答、幻觉或答非所问须明显扣分。"
    "各维度独立评分，仅返回 JSON。"
)

SCORE_ANCHOR = """【通用分档】（各维度均适用，须给出 0–20 的整数）
- 18–20：几乎无瑕疵，明显优于一般水平
- 14–17：基本正确，仅有轻微遗漏或措辞问题
- 10–13：部分正确，存在明显遗漏、含糊或不严谨
- 5–9：大部分不正确，或仅有少量正确信息
- 0–4：完全错误、严重幻觉、与图无关或拒答/空答"""

SCENE_RUBRIC = f"""【题型】scene — 描述图片主要内容（测评语义对齐 / align）

{SCORE_ANCHOR}

【五维评分标准】（每项 0–20 整数）

1. main_content_accuracy（主要语义准确性）
   - 20：准确说出画面核心主题（何处、何人/何物、在做什么）
   - 10：只说对一部分主体，或场景判断含糊
   - 0：主体完全错误，或将图中不存在的内容当作主体

2. key_element_coverage（关键元素覆盖）
   - 20：覆盖图中 2 个以上关键可见元素（人/物/动作/环境等）
   - 10：只提到 1 个关键元素，或遗漏重要动作/环境
   - 0：几乎未描述任何可辨识元素

3. reference_semantic_match（参考答案语义一致）
   - 20：与参考答案在核心语义上等价或高度一致
   - 10：有部分一致，但遗漏参考答案中的关键信息
   - 0：与参考答案语义冲突或几乎无关

4. factual_faithfulness（事实忠实 / 无幻觉）
   - 20：所有陈述均可在图中找到依据，无任何编造
   - 10：存在 1 处轻微夸大或不确定描述
   - 0：出现明显编造对象、颜色、数量、动作或场景

5. expression_quality（表达质量）
   - 20：中文自然通顺，单句/短句描述清晰
   - 10：能理解但啰嗦、语序别扭或用词不当
   - 0：难以理解、逻辑混乱或严重语法问题

【总分】total = 五维之和（0–100）"""

DETAIL_RUBRIC = f"""【题型】detail — 回答针对图片细节的具体问题（测评指令微调 / sft）

{SCORE_ANCHOR}

【五维评分标准】（每项 0–20 整数）

1. factual_correctness（事实正确性）
   - 20：答案事实与图中完全一致，可直接验证
   - 10：大方向对，但细节有误（如颜色、数量、位置偏差）
   - 0：核心事实错误

2. question_focus（答所问）
   - 20：正面、直接回答所问问题，无跑题
   - 10：部分回答所问，夹杂无关描述
   - 0：答非所问或仅复述场景描述

3. reference_semantic_match（参考答案语义一致）
   - 20：与参考答案在语义上等价（允许措辞不同）
   - 10：部分一致，但遗漏参考答案要点
   - 0：与参考答案冲突或无关

4. factual_faithfulness（事实忠实 / 无幻觉）
   - 20：答案每个事实点均可在图中验证
   - 10：有 1 处无法从图中确认的说法
   - 0：编造图中不存在的信息

5. conciseness（简洁规范）
   - 20：简短直接，无废话，符合细节问答长度预期（通常 ≤15 字）
   - 10：偏长或有多余铺垫，但信息可用
   - 0：冗长混乱，或空泛套话

【总分】total = 五维之和（0–100）"""


def _json_template(keys: tuple[str, ...]) -> str:
    lines = [f'  "{k}": 0,' for k in keys]
    lines.append('  "total": 0,')
    lines.append('  "brief_reason": "逐维简述主要扣分点，不超过 80 字"')
    return "{\n" + "\n".join(lines) + "\n}"


def build_judge_prompt(
    qa_type: str,
    question: str,
    reference_answer: str,
    model_answer: str,
) -> str:
    rubric = SCENE_RUBRIC if qa_type == QA_TYPE_SCENE else DETAIL_RUBRIC
    keys = SCENE_DIMS if qa_type == QA_TYPE_SCENE else DETAIL_DIMS
    return f"""请作为裁判评测「待测模型」的回答质量。必须先观察图片，再对照参考答案。

{rubric}

【问题】{question}
【参考答案（Qwen3.5 生成）】{reference_answer}
【待测模型回答】{model_answer}

仅返回 JSON（不要 markdown 代码块）：
{_json_template(keys)}

要求：
- 五个维度必须均为 0–20 的整数
- total 必须严格等于五维分数之和（0–100）
- 从严给分，不要默认给高分"""


def normalize_scores(data: dict, qa_type: str) -> dict:
    keys = SCENE_DIMS if qa_type == QA_TYPE_SCENE else DETAIL_DIMS
    scores: dict[str, int] = {}
    for k in keys:
        raw = data.get(k)
        if raw is None:
            raise ValueError(f"缺少维度: {k}")
        scores[k] = max(SCORE_DIM_MIN, min(SCORE_DIM_MAX, int(round(float(raw)))))
    total = sum(scores.values())
    return {
        **scores,
        "total": total,
        "brief_reason": str(data.get("brief_reason", "")).strip(),
    }


def judge_one(
    item: dict,
    host: str,
    model: str,
    max_retries: int = 3,
) -> dict:
    image_path = ROOT / item["image"]
    if not image_path.is_file():
        raise FileNotFoundError(item["image"])
    if not item.get("model_answer"):
        raise ValueError("model_answer 为空")

    prompt = build_judge_prompt(
        item["type"],
        item["question"],
        item["reference_answer"],
        item["model_answer"],
    )
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            raw = call_vllm_vision(
                host,
                model,
                JUDGE_SYSTEM,
                prompt,
                image_path,
                temperature=0.1,
                max_tokens=768,
            )
            data = parse_json_response(raw)
            scores = normalize_scores(data, item["type"])
            return {
                "qa_id": item["qa_id"],
                "image": item["image"],
                "type": item["type"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "model_answer": item["model_answer"],
                "scores": scores,
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ValueError,
            KeyError,
        ) as e:
            last_err = e
            print(f"[重试 {attempt + 1}/{max_retries}] {item['qa_id']}: {e}", flush=True)

    raise RuntimeError(f"评分失败: {item['qa_id']}") from last_err


def _mean_dim(subset: list[dict], dim: str) -> float:
    vals = [r["scores"][dim] for r in subset if dim in r.get("scores", {})]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def summarize(items: list[dict]) -> dict:
    def agg(qa_type: str, dims: tuple[str, ...]) -> dict:
        subset = [r for r in items if r.get("type") == qa_type and r.get("scores")]
        if not subset:
            return {"count": 0}
        totals = [r["scores"]["total"] for r in subset]
        out = {
            "count": len(subset),
            "mean_total": round(sum(totals) / len(totals), 2),
            "max_total": SCORE_TOTAL_MAX,
            "mean_pct": round(100 * sum(totals) / len(totals) / SCORE_TOTAL_MAX, 2),
            "mean_by_dim": {d: _mean_dim(subset, d) for d in dims},
            "dim_labels": {d: DIM_LABELS[d] for d in dims},
        }
        return out

    scene = agg(QA_TYPE_SCENE, SCENE_DIMS)
    detail = agg(QA_TYPE_DETAIL, DETAIL_DIMS)
    all_totals = [r["scores"]["total"] for r in items if r.get("scores")]
    return {
        "score_scale": f"每维 {SCORE_DIM_MIN}–{SCORE_DIM_MAX}，单题满分 {SCORE_TOTAL_MAX}",
        "align_scene": scene,
        "sft_detail": detail,
        "overall": {
            "count": len(all_totals),
            "mean_total": round(sum(all_totals) / len(all_totals), 2) if all_totals else 0,
            "max_total": SCORE_TOTAL_MAX,
            "mean_pct": round(100 * sum(all_totals) / len(all_totals) / SCORE_TOTAL_MAX, 2)
            if all_totals
            else 0,
        },
    }


def load_vlm_answers(path: Path) -> tuple[dict, list[dict]]:
    data = load_json(path)
    if not isinstance(data, dict) or "items" not in data:
        raise ValueError(f"无法解析 Step 8.2 输出: {path}")
    return data.get("meta", {}), data["items"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8.3：Qwen3.5 裁判打分")
    parser.add_argument("--input", type=Path, default=None, help="Step 8.2 输出（默认 sft 权重对应文件）")
    parser.add_argument("--output", type=Path, default=None, help="默认由 --input 推导为 step8_3_scores_*.json")
    parser.add_argument("--host", default=DEFAULT_VLLM_HOST)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    if args.input is None:
        args.input = default_step8_2_output()
    else:
        args.input = resolve_project_path(args.input)
    if args.output is None:
        args.output = step8_3_output_path(args.input)
    elif not args.output.is_absolute():
        args.output = resolve_project_path(args.output)

    print(
        f"输入 → {project_rel_path(args.input)} | 输出 → {project_rel_path(args.output)}",
        flush=True,
    )

    answer_meta, items = load_vlm_answers(args.input)
    items = [it for it in items if it.get("model_answer")]
    if not items:
        sys.exit(f"无可评分条目（请先运行 step8_2）: {args.input}")

    existing = load_json(args.output)
    if isinstance(existing, dict) and "items" in existing:
        done_ids = {r["qa_id"] for r in existing["items"] if r.get("scores")}
        results = existing["items"]
    else:
        done_ids = set()
        results = []

    by_id = {r["qa_id"]: r for r in results}
    pending = [it for it in items if it["qa_id"] not in done_ids]

    print(
        f"待评分 {len(items)} 条 | 已完成 {len(done_ids)} 条 | 剩余 {len(pending)} 条 | "
        f"满分 {SCORE_TOTAL_MAX}",
        flush=True,
    )
    if not pending and results:
        summary = summarize(results)
        save_json(args.output, {"meta": {**answer_meta, "summary": summary}, "items": results})
        print("无需处理，已全部完成。", flush=True)
        _print_summary(summary)
        return

    lock = Lock()
    completed = 0
    total = len(pending)

    def write_output() -> None:
        summary = summarize(results)
        save_json(
            args.output,
            {"meta": {**answer_meta, "summary": summary}, "items": results},
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, it, args.host, args.model): it for it in pending}
        for future in as_completed(futures):
            it = futures[future]
            try:
                scored = future.result()
            except Exception as e:
                print(f"[失败] {it['qa_id']}: {e}", flush=True)
                continue

            with lock:
                by_id[scored["qa_id"]] = scored
                results = [by_id[it2["qa_id"]] for it2 in items if it2["qa_id"] in by_id]
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(
                        f"[{completed}/{total}] {scored['qa_id']} "
                        f"total={scored['scores']['total']}/{SCORE_TOTAL_MAX}",
                        flush=True,
                    )
                if completed % args.save_every == 0 or completed == total:
                    write_output()

    summary = summarize(results)
    save_json(args.output, {"meta": {**answer_meta, "summary": summary}, "items": results})
    print(f"完成 → {args.output.relative_to(ROOT)}", flush=True)
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print(f"\n========== Step 8 测评汇总（{summary['score_scale']}）==========", flush=True)

    def _print_block(title: str, block: dict) -> None:
        if not block.get("count"):
            return
        print(
            f"\n{title}: n={block['count']}  "
            f"均分 {block['mean_total']}/{block['max_total']} ({block['mean_pct']}%)",
            flush=True,
        )
        for dim, mean in block.get("mean_by_dim", {}).items():
            label = block.get("dim_labels", {}).get(dim, dim)
            print(f"  · {label}: {mean}/{SCORE_DIM_MAX}", flush=True)

    _print_block("align (scene)", summary["align_scene"])
    _print_block("sft (detail)", summary["sft_detail"])
    o = summary["overall"]
    if o.get("count"):
        print(
            f"\noverall: n={o['count']}  "
            f"均分 {o['mean_total']}/{o['max_total']} ({o['mean_pct']}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
