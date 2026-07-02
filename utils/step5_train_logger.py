"""Step 5：训练过程日志与损失曲线。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class TrainLogger:
    """每 N batch 写入 train.log；保存 checkpoint 时更新 loss.png（始终覆盖同一张）。"""

    def __init__(self, log_dir: Path, log_interval: int = 100):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "train.log"
        self.loss_plot = log_dir / "loss.png"
        self.log_interval = log_interval
        self.global_batch = 0
        self.recent_losses: list[float] = []
        self.history: list[tuple[int, float]] = []  # (batch, avg_loss)

    def record_batch(self, loss: float, epoch: int, global_step: int) -> None:
        self.global_batch += 1
        self.recent_losses.append(loss)
        if self.global_batch % self.log_interval != 0:
            return
        window = self.recent_losses[-self.log_interval :]
        avg_loss = sum(window) / len(window)
        self.history.append((self.global_batch, avg_loss))
        self._append_log(epoch, global_step, avg_loss)

    def save_plot(self) -> None:
        if not self.history:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        batches, losses = zip(*self.history)
        plt.figure(figsize=(9, 4))
        plt.plot(batches, losses, linewidth=1.2)
        plt.xlabel("Batch")
        plt.ylabel(f"Avg Loss (per {self.log_interval} batches)")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.loss_plot, dpi=120)
        plt.close()

    def finalize(self, epoch: int, global_step: int) -> None:
        """训练结束时补记未满 interval 的剩余 batch，并更新曲线。"""
        remainder = len(self.recent_losses) % self.log_interval
        if remainder:
            window = self.recent_losses[-remainder:]
            avg_loss = sum(window) / len(window)
            self.history.append((self.global_batch, avg_loss))
            self._append_log(epoch, global_step, avg_loss, partial=True)
        self.save_plot()

    def _append_log(self, epoch: int, global_step: int, avg_loss: float, partial: bool = False) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag = "partial" if partial else "interval"
        line = (
            f"{ts}\tevent={tag}\tepoch={epoch}\tbatch={self.global_batch}\t"
            f"global_step={global_step}\tloss={avg_loss:.6f}\n"
        )
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line)


def log_dir_for_output(output_dir: Path, root: Path) -> Path:
    """checkpoints/VLM_v1_align → logs/VLM_v1_align"""
    return root / "logs" / output_dir.name
