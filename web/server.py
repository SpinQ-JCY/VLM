#!/usr/bin/env python3
"""VLM v1 Web 问答服务：上传图片后逐条提问（无对话上下文）。"""

from __future__ import annotations

import argparse
import shutil
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from inference import DEFAULT_CHECKPOINT, VLM_v1_Predictor

WEB_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = WEB_DIR / ".sessions"

app = FastAPI(title="VLM v1")
_lock = threading.Lock()
_predictor: VLM_v1_Predictor | None = None
ACCESS_URL: str | None = None


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


@dataclass
class Session:
    image_path: Path
    created_at: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}


class AskResponse(BaseModel):
    answer: str
    latency_s: float


def _save_upload(upload: UploadFile, session_id: str) -> Path:
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    dest = SESSIONS_DIR / session_id / f"image{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _get_session(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if session is None or not session.image_path.is_file():
        raise HTTPException(404, "会话不存在或图片已失效，请重新上传")
    return session


@app.on_event("startup")
def startup():
    SESSIONS_DIR.mkdir(exist_ok=True)
    global _predictor
    if _predictor is None:
        _predictor = VLM_v1_Predictor()


@app.get("/health")
def health():
    if _predictor is None:
        raise HTTPException(503, "model not loaded")
    return {
        "status": "ok",
        "device": _predictor.device_name(),
        "checkpoint": str(_predictor.checkpoint),
        "access_url": ACCESS_URL,
    }


@app.post("/api/upload")
async def upload_image(image: UploadFile = File(...), session_id: str = Form("")):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")
    sid = session_id.strip() or str(uuid.uuid4())
    path = _save_upload(image, sid)
    _sessions[sid] = Session(image_path=path)
    return {"session_id": sid, "preview_url": f"/api/session/{sid}/image"}


@app.get("/api/session/{session_id}/image")
def session_image(session_id: str):
    return FileResponse(_get_session(session_id).image_path)


@app.post("/api/ask", response_model=AskResponse)
async def ask(session_id: str = Form(...), question: str = Form(...)):
    if _predictor is None:
        raise HTTPException(503, "model not loaded")
    if not question.strip():
        raise HTTPException(400, "问题不能为空")
    session = _get_session(session_id.strip())
    with _lock:
        t0 = time.perf_counter()
        answer = _predictor.predict(session.image_path, question.strip())
        latency = time.perf_counter() - t0
    return AskResponse(answer=answer, latency_s=round(latency, 3))


@app.get("/")
def index():
    path = WEB_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(404, "index.html 不存在")
    return FileResponse(path)


def main():
    parser = argparse.ArgumentParser(description="VLM v1 Web 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    global _predictor, ACCESS_URL
    _predictor = VLM_v1_Predictor(
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
    )
    ACCESS_URL = f"http://{_lan_ip()}:{args.port}/"

    print(f"内网访问: {ACCESS_URL}")
    print(f"本机访问: http://127.0.0.1:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
