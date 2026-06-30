"""Qwen3-1.7B：Transformers 文本对话"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path(__file__).parent.parent / "Qwen3-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto").to(device)

messages = [{"role": "user", "content": "用一句话介绍你自己。"}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)

# print(text)

inputs = tokenizer([text], return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.8, top_k=20)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
