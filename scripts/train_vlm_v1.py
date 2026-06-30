"""
VLM v1 训练脚本（SFT）

Loss 说明（与 GPT/Qwen 等通用 LM 相同）：
  - 本质：因果语言模型的 Cross-Entropy（next-token prediction）
  - 公式：对每个有效 token 位置 t，用 logits[t] 预测 labels[t+1]
  - SFT 做法：prompt（user + 图片 + 问题 + assistant 开头）labels 设为 -100，只对 answer 算 loss
  - 其它常见 loss（DPO、对比学习等）属于对齐/偏好阶段，不在本脚本范围
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vlms.VLM_v1_model import (  # noqa: E402
    IGNORE_INDEX,
    IMAGE_TOKEN,
    VLM_v1_Config,
    load_VLM_v1,
    load_VLM_v1_image_processor,
)

NUM_IMAGE_TOKENS = VLM_v1_Config().num_image_tokens
DEFAULT_MAX_SEQ_LEN = NUM_IMAGE_TOKENS + 128  # 576 图 token + 128 文本 token


def encode_sample(
    tokenizer,
    question: str,
    answer: str,
    max_text_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 input_ids / labels：仅 answer 部分参与 loss（标准 instruction tuning）。"""
    im_end = tokenizer.eos_token
    prompt = (
        f"<|im_start|>user\n{IMAGE_TOKEN}\n{question}\n"
        f"{im_end}\n"
        f"<|im_start|>assistant\n"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    # token 序列含 1 个 <image> 占位符，合并后展开为 NUM_IMAGE_TOKENS
    max_ids_len = max_text_tokens + 1
    if len(prompt_ids) >= max_ids_len:
        full_ids = prompt_ids[:max_ids_len]
    else:
        budget = max_ids_len - len(prompt_ids)
        full_ids = prompt_ids + answer_ids[:budget]

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = torch.full((len(full_ids),), IGNORE_INDEX, dtype=torch.long)
    labels[len(prompt_ids) :] = torch.tensor(full_ids[len(prompt_ids) :], dtype=torch.long)
    return input_ids, labels


class QADataset(Dataset):
    def __init__(self, qa_path: Path, processor, max_samples: int = 0):
        records = json.loads(qa_path.read_text(encoding="utf-8"))
        if max_samples > 0:
            records = records[:max_samples]
        self.records = records
        self.processor = processor
        self.root = ROOT

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        image = Image.open(self.root / r["image"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        return {
            "pixel_values": pixel_values,
            "question": r["question"],
            "answer": r["answer"],
        }


def collate_fn(batch, tokenizer, max_text_tokens: int):
    input_ids_list, labels_list, pixels = [], [], []
    max_ids_len = max_text_tokens + 1
    for item in batch:
        ids, labs = encode_sample(
            tokenizer, item["question"], item["answer"], max_text_tokens=max_text_tokens
        )
        input_ids_list.append(ids)
        labels_list.append(labs)
        pixels.append(item["pixel_values"])

    max_len = min(max(x.size(0) for x in input_ids_list), max_ids_len)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : ids.size(0)] = ids
        labels[i, : labs.size(0)] = labs

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": torch.stack(pixels),
    }


def save_projector(model, save_dir: Path, step: int | None = None) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / (f"projector_step_{step}.pt" if step is not None else "projector.pt")
    torch.save(model.projector.state_dict(), path)
    return path


def train(args):
    device = "cuda"
    max_text_tokens = args.max_seq_len - NUM_IMAGE_TOKENS
    if max_text_tokens < 1:
        raise ValueError(f"--max-seq-len 须大于 {NUM_IMAGE_TOKENS}（图像 token 数）")

    model, tokenizer = load_VLM_v1(device=device)
    processor = load_VLM_v1_image_processor()
    model.train()

    # 只训练 projector（vision / llm 已在 VLM_v1_Config 中冻结）
    trainable = [p for p in model.projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    dataset = QADataset(ROOT / args.qa, processor, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_text_tokens),
    )

    print(
        f"样本数: {len(dataset)} | 可训练参数: {sum(p.numel() for p in trainable):,} | "
        f"max_seq_len={args.max_seq_len} ({NUM_IMAGE_TOKENS} 图 + {max_text_tokens} 文本) | "
        f"save_every={args.save_every}"
    )

    save_dir = ROOT / args.output
    global_step = 0

    for epoch in range(args.epochs):
        total_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = out.loss / args.grad_accum
            loss.backward()

            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if args.save_every > 0 and global_step % args.save_every == 0:
                    path = save_projector(model, save_dir, step=global_step)
                    pbar.write(f"[step {global_step}] 已保存 → {path}")

            total_loss += out.loss.item()
            n += 1
            pbar.set_postfix(loss=f"{out.loss.item():.4f}", step=global_step)

        print(f"Epoch {epoch + 1} avg loss: {total_loss / max(n, 1):.4f}")

    path = save_projector(model, save_dir)
    print(f"已保存 projector → {path}")


def main():
    parser = argparse.ArgumentParser(description="VLM v1 SFT 训练")
    parser.add_argument("--qa", default="data/qa/coco_val_qa.json")
    parser.add_argument("--output", default="checkpoints/VLM_v1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示全部")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"合并后 LLM 最大序列长度，默认 {DEFAULT_MAX_SEQ_LEN}（576 图 + 128 文本）",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="每 N 次 optimizer step 保存一次 checkpoint（0 表示仅训练结束保存）",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
