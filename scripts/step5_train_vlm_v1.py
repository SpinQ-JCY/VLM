"""
Step 5：VLM v1 训练（两阶段）

  - Align 语义对齐：COCO-CN 中文描述 → 对齐 SigLIP 视觉特征与 Qwen 语义空间
  - InstructFT 指令微调：多类视觉问答 → 学会按问题类型回答

Loss：因果 LM Cross-Entropy；prompt labels 置 -100，只对 answer 算 loss。
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
    MAX_TEXT_TOKENS,
    VLM_v1_Config,
    encode_train_sample,
    load_VLM_v1,
    load_VLM_v1_image_processor,
)
from utils.step5_train_logger import TrainLogger, log_dir_for_output  # noqa: E402

NUM_IMAGE_TOKENS = VLM_v1_Config().num_image_tokens
DEFAULT_MAX_SEQ_LEN = NUM_IMAGE_TOKENS + MAX_TEXT_TOKENS

STAGE_LABELS = {
    "semantic_align": "Align 语义对齐",
    "instructft": "InstructFT 指令微调",
}

STAGE_DEFAULTS = {
    "semantic_align": {
        "qa": "data/qa/coco_cn_qa.json",
        "output": "checkpoints/semantic_align",
        "init_checkpoint": None,
    },
    "instructft": {
        "qa": "data/qa/coco_train_qa_qwen3.5.json",
        "output": "checkpoints/instructft",
        "init_checkpoint": "checkpoints/semantic_align/projector.pt",
    },
}


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


def save_projector(model, save_dir: Path, step: int | None = None) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / (f"projector_step_{step}.pt" if step is not None else "projector.pt")
    torch.save(model.projector.state_dict(), path)
    return path


def load_projector(model, checkpoint: Path, device: str) -> None:
    if not checkpoint.is_file():
        print(f"警告: init checkpoint 不存在，从随机 projector 开始 → {checkpoint}")
        return
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.projector.load_state_dict(state)
    print(f"已加载 init checkpoint → {checkpoint}")


def train(args):
    device = "cuda"
    max_text_tokens = args.max_seq_len - NUM_IMAGE_TOKENS
    if max_text_tokens < 1:
        raise ValueError(f"--max-seq-len 须大于 {NUM_IMAGE_TOKENS}（图像 token 数）")

    model, tokenizer = load_VLM_v1(device=device)
    if args.init_checkpoint is not None:
        load_projector(model, ROOT / args.init_checkpoint, device)
    processor = load_VLM_v1_image_processor()
    model.train()

    trainable = [p for p in model.projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    qa_path = ROOT / args.qa
    dataset = QADataset(qa_path, processor, tokenizer, max_text_tokens, args.max_samples)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "collate_fn": lambda b: collate_fn(b, tokenizer, max_text_tokens),
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    stage_label = STAGE_LABELS[args.stage]
    print(
        f"阶段: {stage_label} ({args.stage}) | 数据: {qa_path.relative_to(ROOT)} | "
        f"样本数: {len(dataset)} | 可训练参数: {sum(p.numel() for p in trainable):,} | "
        f"max_seq_len={args.max_seq_len} ({NUM_IMAGE_TOKENS} 图 + {max_text_tokens} 文本) | "
        f"save_every={args.save_every}",
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
                    path = save_projector(model, save_dir, step=global_step)
                    logger.save_plot()
                    pbar.write(
                        f"[step {global_step}] 已保存 → {path} | 曲线 → {logger.loss_plot}"
                    )

            total_loss += loss_val
            n += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)

        print(f"Epoch {epoch + 1} avg loss: {total_loss / max(n, 1):.4f}")

    path = save_projector(model, save_dir)
    logger.finalize(epoch=args.epochs, global_step=global_step)
    print(f"已保存 projector → {path}")
    print(f"训练日志 → {logger.log_file}")
    print(f"损失曲线 → {logger.loss_plot}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 5：VLM v1 训练（Align 语义对齐 / InstructFT 指令微调）"
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_DEFAULTS),
        default="semantic_align",
        help="semantic_align=Align 语义对齐；instructft=InstructFT 指令微调",
    )
    parser.add_argument("--qa", default=None, help="QA JSON 路径（默认随 --stage 选择）")
    parser.add_argument("--output", default=None, help="checkpoint 输出目录（默认随 --stage 选择）")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="训练前加载的 projector 权重；InstructFT 阶段默认 Align 阶段 projector.pt",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
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
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="每 N 个 batch 记录一次平均 loss 到 logs/<run>/train.log",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 并行加载进程数（tokenize + 读图在 worker 中执行）",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forward 使用 bf16 autocast（默认开启）",
    )
    args = parser.parse_args()

    defaults = STAGE_DEFAULTS[args.stage]
    if args.qa is None:
        args.qa = defaults["qa"]
    if args.output is None:
        args.output = defaults["output"]
    if args.init_checkpoint is None:
        args.init_checkpoint = defaults["init_checkpoint"]

    train(args)


if __name__ == "__main__":
    main()
