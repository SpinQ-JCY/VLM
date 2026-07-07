"""
Step 5b：InstructFT + LoRA

在 Align projector 基础上，用 InstructFT 数据联合训练 Projector 与 Qwen attention LoRA。
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

from models.vlms.VLM_v1_lora import LORA_ALPHA, LORA_RANK, load_for_train  # noqa: E402
from models.vlms.VLM_v1_model import (  # noqa: E402
    IGNORE_INDEX,
    MAX_TEXT_TOKENS,
    VLM_v1_Config,
    encode_train_sample,
    load_VLM_v1_image_processor,
)
from utils.step5_train_logger import TrainLogger, log_dir_for_output  # noqa: E402

NUM_IMAGE_TOKENS = VLM_v1_Config().num_image_tokens
DEFAULT_MAX_SEQ_LEN = NUM_IMAGE_TOKENS + MAX_TEXT_TOKENS

DEFAULT_QA = "data/qa/coco_train_qa_qwen3.5.json"
DEFAULT_OUTPUT = "checkpoints/instructft_lora"
DEFAULT_INIT_PROJECTOR = "checkpoints/semantic_align/projector.pt"


class QADataset(Dataset):
    def __init__(
        self,
        qa_path: Path,
        processor,
        tokenizer,
        max_text_tokens: int = MAX_TEXT_TOKENS,
        max_samples: int = 0,
    ):
        records = json.loads(qa_path.read_text(encoding="utf-8"))
        if max_samples > 0:
            records = records[:max_samples]
        self.records = records
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.root = ROOT

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        image = Image.open(self.root / r["image"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        input_ids, labels = encode_train_sample(
            self.tokenizer,
            r["question"],
            r["answer"],
            max_text_tokens=self.max_text_tokens,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


def collate_fn(batch, tokenizer, max_text_tokens: int):
    input_ids = torch.full(
        (len(batch), max_text_tokens), tokenizer.pad_token_id, dtype=torch.long
    )
    labels = torch.full((len(batch), max_text_tokens), IGNORE_INDEX, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
    }


def save_checkpoint(model, save_dir: Path, step: int | None = None) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    projector_path = save_dir / (
        f"projector_step_{step}.pt" if step is not None else "projector.pt"
    )
    lora_dir = save_dir / (f"lora_step_{step}" if step is not None else "lora")
    torch.save(model.projector.state_dict(), projector_path)
    model.llm.save_pretrained(lora_dir)
    return projector_path, lora_dir


def load_projector(model, checkpoint: Path, device: str) -> None:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.projector.load_state_dict(state)
    print(f"已加载 init projector → {checkpoint}")


def train(args):
    device = "cuda"
    max_text_tokens = args.max_seq_len - NUM_IMAGE_TOKENS
    if max_text_tokens < 1:
        raise ValueError(f"--max-seq-len 须大于 {NUM_IMAGE_TOKENS}（图像 token 数）")

    model, tokenizer = load_for_train(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
    )
    load_projector(model, ROOT / args.init_projector, device)
    processor = load_VLM_v1_image_processor()
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    qa_path = ROOT / args.qa
    dataset = QADataset(qa_path, processor, tokenizer, max_text_tokens, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_text_tokens),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print(
        f"InstructFT + LoRA | 数据: {qa_path.relative_to(ROOT)} | 样本数: {len(dataset)} | "
        f"可训练参数: {sum(p.numel() for p in trainable):,} | "
        f"LoRA rank={args.lora_rank} alpha={args.lora_alpha} | "
        f"max_seq_len={args.max_seq_len} | save_every={args.save_every}",
        flush=True,
    )

    save_dir = ROOT / args.output
    logger = TrainLogger(log_dir_for_output(save_dir, ROOT), log_interval=args.log_interval)
    global_step = 0

    for epoch in range(args.epochs):
        total_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
            loss = out.loss / args.grad_accum
            loss.backward()

            loss_val = out.loss.item()
            logger.record_batch(loss_val, epoch=epoch + 1, global_step=global_step)

            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if args.save_every > 0 and global_step % args.save_every == 0:
                    proj_path, lora_dir = save_checkpoint(model, save_dir, step=global_step)
                    logger.save_plot()
                    pbar.write(
                        f"[step {global_step}] projector → {proj_path} | lora → {lora_dir}"
                    )

            total_loss += loss_val
            n += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)

        print(f"Epoch {epoch + 1} avg loss: {total_loss / max(n, 1):.4f}")

    proj_path, lora_dir = save_checkpoint(model, save_dir)
    logger.finalize(epoch=args.epochs, global_step=global_step)
    print(f"已保存 projector → {proj_path}")
    print(f"已保存 lora → {lora_dir}")
    print(f"训练日志 → {logger.log_file}")
    print(f"损失曲线 → {logger.loss_plot}")


def main():
    parser = argparse.ArgumentParser(description="Step 5b：InstructFT + LoRA 联合训练")
    parser.add_argument("--qa", default=DEFAULT_QA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--init-projector", default=DEFAULT_INIT_PROJECTOR)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示全部")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
