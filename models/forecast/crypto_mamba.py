"""CryptoMamba — Mamba SSM สำหรับ crypto price forecasting.

Pure PyTorch implementation ของ simplified Mamba architecture.
ไม่ต้องพึ่ง mamba-ssm CUDA kernel → รันได้ทุก GPU/CPU.
O(n) linear complexity (vs O(n²) Transformer).

RTX 2060 6GB: Fine-tune ได้ด้วย d_model=64, n_layers=4 (~0.5M params)
Cloud GPU: เพิ่ม d_model=256, n_layers=8 (~8M params)

References:
  - CryptoMamba (arXiv:2501.01010, IEEE ICBC 2025)
  - Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)
"""

import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.forecast.base import BaseForecaster


# =============================================================================
# Mamba SSM Block (Pure PyTorch — no custom CUDA kernel)
# =============================================================================

if TORCH_AVAILABLE:

    class SelectiveSSM(nn.Module):
        """Simplified Selective State Space Model (core of Mamba).

        Key: B, C, Δ (delta) are input-dependent (selective) → content-aware filtering.
        """

        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state

            # Input projection: x → (B, C, Δ)
            self.proj_bc = nn.Linear(d_model, d_state * 2, bias=False)
            self.proj_delta = nn.Linear(d_model, d_model, bias=False)

            # Learnable A matrix (diagonal, negative for stability)
            self.A_log = nn.Parameter(
                torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            )

            # 1D convolution for local context
            self.conv = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)

            # Output projection
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (batch, seq_len, d_model) → (batch, seq_len, d_model)"""
            batch, seq_len, d = x.shape

            # 1D conv for local context
            x_conv = self.conv(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            x_conv = torch.nn.functional.silu(x_conv)

            # Input-dependent B, C
            bc = self.proj_bc(x_conv)
            B = bc[:, :, :self.d_state]
            C = bc[:, :, self.d_state:]

            # Input-dependent delta (discretization step)
            delta = torch.nn.functional.softplus(self.proj_delta(x_conv))

            # Discretize A
            A = -torch.exp(self.A_log)  # (d_state,) — negative for stability

            # Sequential scan (simplified — no parallel scan for clarity)
            # For RTX 2060 testing, seq_len is small enough
            h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
            outputs = []

            for t in range(seq_len):
                dt = delta[:, t, :].unsqueeze(-1)     # (batch, d_model, 1)
                Bt = B[:, t, :].unsqueeze(1)           # (batch, 1, d_state)
                Ct = C[:, t, :].unsqueeze(1)           # (batch, 1, d_state)
                xt = x_conv[:, t, :].unsqueeze(-1)     # (batch, d_model, 1)

                # State update: h = A_bar * h + B_bar * x
                A_bar = torch.exp(dt * A.view(1, 1, -1))  # (batch, d_model, d_state)
                B_bar = dt * Bt                             # (batch, d_model, d_state)
                h = A_bar * h + B_bar * xt

                # Output: y = C * h
                y = (Ct * h).sum(dim=-1)  # (batch, d_model)
                outputs.append(y)

            output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
            return self.out_proj(output)

    class MambaBlock(nn.Module):
        """Single Mamba block = SSM + skip connection + norm."""

        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.ssm = SelectiveSSM(d_model, d_state, d_conv)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.ssm(self.norm(x))

    class CryptoMambaModel(nn.Module):
        """Full CryptoMamba model for price forecasting.

        Architecture:
        Input → Linear embed → N × MambaBlock → Linear → Forecast
        """

        def __init__(
            self,
            input_dim: int = 1,
            d_model: int = 64,
            d_state: int = 16,
            n_layers: int = 4,
            horizon: int = 12,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.horizon = horizon

            self.embed = nn.Linear(input_dim, d_model)
            self.blocks = nn.ModuleList([
                MambaBlock(d_model, d_state) for _ in range(n_layers)
            ])
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, horizon)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (batch, seq_len, input_dim) → (batch, horizon)"""
            h = self.embed(x)
            for block in self.blocks:
                h = self.dropout(block(h))
            # Use last hidden state for forecast
            return self.head(h[:, -1, :])

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Forecaster wrapper (implements BaseForecaster)
# =============================================================================

class CryptoMambaForecaster(BaseForecaster):
    """CryptoMamba forecaster — fine-tunable on historical data.

    Light mode (RTX 2060): d_model=64, n_layers=4 (~0.5M params, ~0.1GB VRAM)
    Full mode (Cloud GPU): d_model=256, n_layers=8 (~8M params, ~2GB VRAM)
    """

    def __init__(
        self,
        context_len: int = 128,
        horizon: int = 12,
        d_model: int = 64,
        n_layers: int = 4,
        device: str = "cuda",
        checkpoint_path: str | None = None,
    ):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed — CryptoMamba unavailable")
            self._available = False
            return

        self.context_len = context_len
        self.horizon = horizon
        self._device = device if torch.cuda.is_available() else "cpu"
        self._available = True

        self._model = CryptoMambaModel(
            input_dim=1,
            d_model=d_model,
            n_layers=n_layers,
            horizon=horizon,
        ).to(self._device)

        n_params = self._model.count_parameters()
        logger.info(f"CryptoMamba: {n_params:,} params, device={self._device}")

        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

    @property
    def available(self) -> bool:
        return self._available

    def predict(
        self,
        price_series: np.ndarray,
        horizon: int = 12,
    ) -> tuple[np.ndarray, float]:
        if not self._available:
            from models.forecast.naive import NaiveForecaster
            return NaiveForecaster().predict(price_series, horizon)

        self._model.eval()

        # Normalize: log returns
        prices = price_series[-self.context_len:].astype(np.float64)
        if len(prices) < 2:
            return np.full(horizon, prices[-1], dtype=np.float32), 1.0

        log_returns = np.diff(np.log(prices))
        mean_r = log_returns.mean()
        std_r = log_returns.std()
        if std_r < 1e-10:
            std_r = 1.0
        normalized = ((log_returns - mean_r) / std_r).astype(np.float32)

        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x = x.to(self._device)

        with torch.no_grad():
            pred = self._model(x)  # (1, horizon)
        pred_np = pred.cpu().numpy()[0]

        # Denormalize: convert back to prices
        denorm_returns = pred_np * std_r + mean_r
        last_price = prices[-1]
        forecast = last_price * np.exp(np.cumsum(denorm_returns[:horizon]))
        forecast = forecast.astype(np.float32)

        # Uncertainty: std of predictions / mean
        uncertainty = float(std_r * np.sqrt(horizon))

        return forecast, uncertainty

    def fine_tune(
        self,
        price_series: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_samples: int = 50_000,
        save_path: str = "checkpoints/crypto_mamba.pt",
        val_ratio: float = 0.2,
        patience: int = 5,
        min_improvement: float = 0.0,
        window_stride: int = 1,
        eval_batch_size: int = 1024,
        use_amp: bool | None = None,
    ) -> dict:
        """Fine-tune on historical data with validation + early stopping.

        Returns quality metrics including directional accuracy and RMSE vs naive.
        """
        if not self._available:
            return {"error": "PyTorch not available"}

        # Create sliding window dataset from log returns
        log_returns = np.diff(np.log(price_series.astype(np.float64)))

        ctx = self.context_len
        h = self.horizon
        stride = max(int(window_stride), 1)
        start_idx = np.arange(0, len(log_returns) - ctx - h + 1, stride, dtype=np.int64)
        n_total = len(start_idx)
        if n_total < batch_size:
            return {"error": f"Not enough data: {n_total} < {batch_size}"}

        n_samples = min(n_total, max_samples)
        sample_idx = start_idx[-n_samples:]
        logger.info(
            "Fine-tune: using %s / %s windows (most recent, stride=%s)",
            f"{n_samples:,}",
            f"{n_total:,}",
            stride,
        )

        # Build raw windows (per-window normalization — matches predict_batch inference)
        X_raw = np.array([log_returns[i:i + ctx] for i in sample_idx], dtype=np.float32)
        Y_raw = np.array([log_returns[i + ctx:i + ctx + h] for i in sample_idx], dtype=np.float32)

        # Per-window normalization: each window uses its OWN mean/std
        # This matches predict_batch exactly — no train/inference mismatch
        win_means = X_raw.mean(axis=1, keepdims=True)   # (n_samples, 1)
        win_stds = np.maximum(X_raw.std(axis=1, keepdims=True), 1e-10)
        X = ((X_raw - win_means) / win_stds).astype(np.float32)
        Y = ((Y_raw - win_means) / win_stds).astype(np.float32)  # same scale as X

        # Train/validation split (time-ordered, no shuffle)
        n_val = max(int(n_samples * val_ratio), batch_size)
        n_train = n_samples - n_val
        X_train, X_val = X[:n_train], X[n_train:]
        Y_train, Y_val = Y[:n_train], Y[n_train:]

        # Keep raw stats for accuracy evaluation
        val_means = win_means[n_train:]
        val_stds = win_stds[n_train:]

        X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        Y_t = torch.tensor(Y_train, dtype=torch.float32)
        X_v = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
        Y_v = torch.tensor(Y_val, dtype=torch.float32)

        logger.info(f"Fine-tune split: train={n_train:,} val={n_val:,}")

        self._model.train()
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        amp_enabled = bool((self._device != "cpu") and (use_amp if use_amp is not None else True))
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # --- Train ---
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            indices = torch.randperm(n_train)
            for start in range(0, n_train - batch_size + 1, batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = X_t[batch_idx].to(self._device, non_blocking=True)
                yb = Y_t[batch_idx].to(self._device, non_blocking=True)
                optimizer.zero_grad()
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else nullcontext()
                with autocast_ctx:
                    pred = self._model(xb)
                    loss = torch.nn.functional.mse_loss(pred, yb)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train)

            # --- Validate ---
            self._model.eval()
            val_loss_total = 0.0
            val_seen = 0
            with torch.no_grad():
                for start in range(0, n_val, eval_batch_size):
                    end = min(start + eval_batch_size, n_val)
                    xb = X_v[start:end].to(self._device, non_blocking=True)
                    yb = Y_v[start:end].to(self._device, non_blocking=True)
                    autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else nullcontext()
                    with autocast_ctx:
                        val_pred = self._model(xb)
                        batch_loss = torch.nn.functional.mse_loss(val_pred, yb, reduction="sum")
                    val_loss_total += float(batch_loss.item())
                    val_seen += (end - start) * yb.shape[1]
            val_loss = val_loss_total / max(val_seen, 1)
            val_losses.append(val_loss)

            logger.info(
                f"CryptoMamba epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train:.6f} val_loss={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Early stopping on validation loss
            if val_loss < (best_val_loss - float(min_improvement)):
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                    break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

        # --- Evaluate accuracy on validation set ---
        quality = self._evaluate_accuracy(
            X_val,
            Y_val,
            val_stds,
            val_means,
            batch_size=eval_batch_size,
            use_amp=amp_enabled,
        )
        logger.info(
            "CryptoMamba accuracy: dir_acc=%.1f%% rmse_ratio=%.3f (vs naive) quality=%s",
            quality["directional_accuracy"] * 100,
            quality["rmse_ratio_vs_naive"],
            "GOOD" if quality["better_than_naive"] else "BAD (worse than naive)",
        )

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), save_path)
        logger.info(f"Saved: {save_path}")

        del X_t, Y_t, X_v, Y_v, optimizer, scheduler, scaler
        if self._device != "cpu":
            torch.cuda.empty_cache()

        return {
            "final_loss": train_losses[-1],
            "best_val_loss": best_val_loss,
            "n_train": n_train,
            "n_val": n_val,
            "n_epochs_actual": len(train_losses),
            "save_path": save_path,
            **quality,
        }

    def _evaluate_accuracy(
        self,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        val_stds: np.ndarray,
        val_means: np.ndarray,
        batch_size: int = 1024,
        use_amp: bool = False,
    ) -> dict:
        """Evaluate forecast quality on validation data.

        Returns directional accuracy and RMSE ratio vs naive (last-value) baseline.
        Uses per-window denormalization matching predict_batch behavior.
        """
        self._model.eval()
        pred_chunks = []
        with torch.no_grad():
            for start in range(0, len(X_val), batch_size):
                end = min(start + batch_size, len(X_val))
                x_t = torch.tensor(X_val[start:end], dtype=torch.float32).unsqueeze(-1).to(self._device)
                autocast_ctx = torch.cuda.amp.autocast(enabled=bool(use_amp and self._device != "cpu")) if self._device != "cpu" else nullcontext()
                with autocast_ctx:
                    pred_batch = self._model(x_t)
                pred_chunks.append(pred_batch.cpu().numpy())
        pred_norm = np.concatenate(pred_chunks, axis=0)

        # Denormalize: per-window stats (matches how training data was normalized)
        pred_returns = pred_norm * val_stds + val_means
        true_returns = Y_val * val_stds + val_means

        # Naive baseline: predict zero return (last price = next price)
        naive_returns = np.zeros_like(true_returns)

        # RMSE
        rmse_model = float(np.sqrt(np.mean((pred_returns - true_returns) ** 2)))
        rmse_naive = float(np.sqrt(np.mean((naive_returns - true_returns) ** 2)))
        rmse_ratio = rmse_model / max(rmse_naive, 1e-10)

        # Directional accuracy: does the model predict the right direction?
        pred_dir = np.sign(pred_returns[:, 0])
        true_dir = np.sign(true_returns[:, 0])
        n_correct = int(np.sum(pred_dir == true_dir))
        n_total = len(true_dir)
        dir_acc = n_correct / max(n_total, 1)

        # Multi-step directional accuracy (avg across all horizon steps)
        all_dir_acc = float(np.mean(np.sign(pred_returns) == np.sign(true_returns)))

        return {
            "directional_accuracy": float(dir_acc),
            "multi_step_dir_accuracy": all_dir_acc,
            "rmse_model": rmse_model,
            "rmse_naive": rmse_naive,
            "rmse_ratio_vs_naive": float(rmse_ratio),
            "better_than_naive": rmse_ratio < 1.0 and dir_acc > 0.50,
            "n_val_samples": n_total,
        }

    def predict_batch(
        self,
        windows: np.ndarray,
        horizon: int = 12,
        batch_size: int = 2048,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch prediction for multiple price windows at once.

        windows: (n_windows, context_len) — raw prices per window
        Returns: (forecasts (n_windows, horizon), uncertainties (n_windows,))
        """
        if not self._available:
            from models.forecast.naive import NaiveForecaster
            naive = NaiveForecaster()
            forecasts = []
            uncs = []
            for w in windows:
                f, u = naive.predict(w, horizon)
                forecasts.append(f)
                uncs.append(u)
            return np.array(forecasts), np.array(uncs)

        self._model.eval()
        n = len(windows)
        h = min(horizon, self.horizon)

        # Normalize per-window: log returns
        log_prices = np.log(np.maximum(windows.astype(np.float64), 1e-10))
        log_returns = np.diff(log_prices, axis=1)  # (n, context_len - 1)

        means = log_returns.mean(axis=1, keepdims=True)
        stds = log_returns.std(axis=1, keepdims=True)
        stds = np.maximum(stds, 1e-10)
        normalized = ((log_returns - means) / stds).astype(np.float32)

        all_preds = []
        # Sub-batch to avoid OOM on very large inputs
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.tensor(normalized[start:end]).unsqueeze(-1).to(self._device)
            with torch.no_grad():
                pred = self._model(x)
            all_preds.append(pred.cpu().numpy())

        pred_np = np.concatenate(all_preds, axis=0)  # (n, model_horizon)

        # Denormalize
        denorm = pred_np[:, :h] * stds + means
        last_prices = windows[:, -1:]
        forecasts = last_prices * np.exp(np.cumsum(denorm, axis=1))

        uncertainties = (stds[:, 0] * np.sqrt(h)).astype(np.float32)

        return forecasts.astype(np.float32), uncertainties

    def release_gpu(self):
        """Move model to CPU and clear GPU memory."""
        if not self._available:
            return
        self._model.cpu()
        self._device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CryptoMamba released GPU memory")

    def name(self) -> str:
        if self._available:
            n = self._model.count_parameters()
            return f"CryptoMamba-{n // 1000}K"
        return "CryptoMamba (unavailable)"
