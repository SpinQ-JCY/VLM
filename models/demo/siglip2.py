"""SigLIP2：Transformers 零样本分类"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

demo = Path(__file__).parent
model_dir = demo.parent / "siglip2-so400m-patch16-384"
labels = ["a living room", "a kitchen", "outdoor street", "a bedroom"]
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(model_dir, torch_dtype="auto").to(device).eval()
processor = AutoProcessor.from_pretrained(model_dir)
image = Image.open(demo / "000000000139.jpg").convert("RGB")

inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    logits = model(**inputs).logits_per_image[0]
    feat = model.get_image_features(**processor(images=[image], return_tensors="pt").to(device))

for label, score in zip(labels, torch.softmax(logits, dim=-1).tolist()):
    print(f"{score:.4f}  {label}")


print(f"pooler_output: {tuple(feat.pooler_output.shape)}")           # 全局向量 (1, 1152)
print(f"last_hidden_state: {tuple(feat.last_hidden_state.shape)}")  # patch 序列 (1, 576, 1152)
# print(model)
