"""Step 8 模型评估：共享工具（vLLM 调用、图片编码、JSON 解析、断点续跑）。"""

from __future__ import annotations

import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = EVAL_DIR / "outputs"
DEFAULT_VAL_DIR = ROOT / "data" / "COCO2014" / "val2014"
DEFAULT_VLLM_HOST = "http://localhost:8033"
DEFAULT_VLLM_MODEL = "models/Qwen3.5-9B"

QA_TYPE_SCENE = "scene"    # 描述主要内容 → Align 语义对齐测评
QA_TYPE_DETAIL = "detail"  # 针对性细节问答 → InstructFT 指令微调测评

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

MAX_SCENE_ANSWER_CHARS = 30
MAX_DETAIL_ANSWER_CHARS = 15
QA_PER_IMAGE = 2
DEFAULT_NUM_EVAL_IMAGES = 50

# Step 8.3 评分：五维度，每维 0–20，单题满分 100
SCORE_DIM_MIN = 0
SCORE_DIM_MAX = 20
SCORE_DIM_COUNT = 5
SCORE_TOTAL_MAX = SCORE_DIM_MAX * SCORE_DIM_COUNT

# Step 8 各子步默认输出
STEP8_0_OUTPUT = DEFAULT_OUTPUT_DIR / "step8_0_images.json"
STEP8_1_OUTPUT = DEFAULT_OUTPUT_DIR / "step8_1_benchmark.json"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/instructft/projector.pt"
STEP8_2_PREFIX = "step8_2_vlm_answers"
STEP8_3_PREFIX = "step8_3_scores"


def step8_2_output_path(checkpoint: Path | str) -> Path:
    """Step 8.2 输出路径，带权重后缀便于区分不同 checkpoint。"""
    tag = checkpoint_weight_tag(checkpoint)
    return DEFAULT_OUTPUT_DIR / f"{STEP8_2_PREFIX}_{tag}.json"


def step8_3_output_path(step8_2_path: Path | str) -> Path:
    """由 Step 8.2 输出路径推导 Step 8.3 评分结果路径。

    step8_2_vlm_answers_instructft.json → step8_3_scores_instructft.json
    """
    p = Path(step8_2_path)
    stem = p.stem
    if stem == STEP8_2_PREFIX:
        new_stem = STEP8_3_PREFIX
    elif stem.startswith(f"{STEP8_2_PREFIX}_"):
        new_stem = f"{STEP8_3_PREFIX}_{stem[len(STEP8_2_PREFIX) + 1:]}"
    else:
        new_stem = f"{STEP8_3_PREFIX}_{stem}"
    return p.with_name(f"{new_stem}{p.suffix}")


def default_step8_2_output() -> Path:
    return step8_2_output_path(DEFAULT_CHECKPOINT)


def checkpoint_weight_tag(checkpoint: Path | str) -> str:
    """从权重路径提取标签。

    checkpoints/instructft/projector.pt              → instructft
    checkpoints/instructft/projector_step_3000.pt    → instructft_projector_step_3000
    """
    p = Path(checkpoint)
    if p.suffix != ".pt":
        return p.stem
    if p.stem == "projector":
        return p.parent.name
    if p.stem.startswith("projector_"):
        return f"{p.parent.name}_{p.stem}"
    return f"{p.parent.name}_{p.stem}"


def resolve_project_path(path: Path | str, root: Path = ROOT) -> Path:
    """将相对路径解析为项目根下的绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def project_rel_path(path: Path | str, root: Path = ROOT) -> str:
    """转为相对项目根的路径字符串，便于写入 meta / 日志。"""
    p = resolve_project_path(path, root)
    return p.relative_to(root.resolve()).as_posix()


def encode_image(image_path: Path) -> tuple[str, str]:
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    return mime, base64.b64encode(image_path.read_bytes()).decode()


def truncate_cn(text: str, max_len: int) -> str:
    text = text.strip().replace("\n", "")
    return text[:max_len] if len(text) > max_len else text


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def call_vllm(
    host: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.4,
    top_p: float = 0.8,
    max_tokens: int = 256,
    timeout: int = 600,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        msg = json.load(resp)["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning") or "").strip()


def call_vllm_vision(
    host: str,
    model: str,
    system: str,
    user_text: str,
    image_path: Path,
    **kwargs: Any,
) -> str:
    mime, b64 = encode_image(image_path)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        },
    ]
    return call_vllm(host, model, messages, **kwargs)


def load_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_image_set(path: Path = STEP8_0_OUTPUT) -> tuple[dict, list[str]]:
    """加载 Step 8.0 图片清单。返回 (meta, 图片相对路径列表)。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"图片清单不存在: {path.relative_to(ROOT)}，请先运行 eval/step8_0_sample_images.py"
        )
    data = load_json(path)
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError(f"无法解析图片清单: {path}")
    images = [str(p) for p in data["images"]]
    if not images:
        raise ValueError(f"图片清单为空: {path}")
    return data.get("meta", {}), images


def make_qa_id(image_rel: str, qa_type: str) -> str:
    stem = Path(image_rel).stem
    return f"{stem}_{qa_type}"


def build_benchmark_user_prompt(question1: str) -> str:
    """与 utils/step3_generate_qa.py 一致的看图出题 prompt。"""
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
