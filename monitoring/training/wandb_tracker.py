"""W&B experiment tracking สำหรับ training.

Free tier: unlimited personal projects, 5GB storage.
Track: reward, sharpe, drawdown, loss, GPU usage.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WandbTracker:
    """Wrapper สำหรับ W&B tracking."""

    def __init__(
        self,
        project: str = "garic",
        run_name: str | None = None,
        config: dict | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._run = None

        if not enabled:
            logger.info("W&B tracking disabled")
            return

        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            logger.info(f"W&B initialized: {project}/{run_name}")
        except ImportError:
            logger.warning("wandb not installed — tracking disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"W&B init failed: {e} — tracking disabled")
            self.enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to W&B."""
        if not self.enabled:
            return
        self._wandb.log(metrics, step=step)

    def log_training_step(
        self,
        step: int,
        reward: float,
        policy_loss: float,
        value_loss: float,
        sharpe: float = 0.0,
        max_drawdown: float = 0.0,
    ):
        """Log standard training metrics."""
        self.log({
            "train/reward": reward,
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
            "train/sharpe": sharpe,
            "train/max_drawdown": max_drawdown,
        }, step=step)

    def log_eval(self, metrics: dict, step: int | None = None):
        """Log evaluation metrics."""
        self.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

    def log_gpu(self):
        """Log GPU utilization."""
        if not self.enabled:
            return
        try:
            import torch
            if torch.cuda.is_available():
                self.log({
                    "system/gpu_memory_gb": torch.cuda.memory_allocated() / 1e9,
                    "system/gpu_utilization": torch.cuda.utilization(),
                })
        except Exception:
            pass

    def finish(self):
        if self._run:
            self._run.finish()
