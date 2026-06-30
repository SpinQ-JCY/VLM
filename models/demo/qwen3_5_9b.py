"""Qwen3.5-9B：vLLM 图片对话（需先启动 vllm serve）"""

import base64
import json
import urllib.request
from pathlib import Path

image = Path(__file__).parent / "000000000139.jpg"
b64 = base64.b64encode(image.read_bytes()).decode()

payload = {
    "model": "models/Qwen3.5-9B",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述这张图片"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
    "temperature": 0.7,   # 随机性：越高越发散，越低越稳定（非推理模式推荐 0.7）
    "top_p": 0.8,         # 核采样：只从累计概率前 80% 的 token 里选（非推理模式推荐 0.8）
    "top_k": 20,          # 每步最多考虑概率最高的 20 个 token
    "max_tokens": 2048,
    "chat_template_kwargs": {"enable_thinking": False},
}

resp = urllib.request.urlopen(urllib.request.Request(
    "http://localhost:8033/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
))
print(json.load(resp)["choices"][0]["message"]["content"])
