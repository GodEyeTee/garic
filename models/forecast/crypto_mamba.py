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
    ) -> dict:
        """Fine-tune on historical data.

        สร้าง sliding window dataset จาก price_series (ใช้ข้อมูลล่าสุด max_samples windows).
        """
        if not self._available:
            return {"error": "PyTorch not available"}

        # Create sliding window dataset
        log_returns = np.diff(np.log(price_series.astype(np.float64)))
        mean_r = log_returns.mean()
        std_r = max(log_returns.std(), 1e-10)
        normalized = ((log_returns - mean_r) / std_r).astype(np.float32)

        ctx = self.context_len
        h = self.horizon
        n_total = len(normalized) - ctx - h + 1
        if n_total < batch_size:
            return {"error": f"Not enough data: {n_total} < {batch_size}"}

        # Use most recent data only (recent data is more relevant + faster training)
        n_samples = min(n_total, max_samples)
        offset = n_total - n_samples
        logger.info(f"Fine-tune: using {n_samples:,} / {n_total:,} windows (most recent)")

        X = np.array([normalized[offset + i:offset + i + ctx] for i in range(n_samples)])
        Y = np.array([normalized[offset + i + ctx:offset + i + ctx + h] for i in range(n_samples)])

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self._device)
        Y_t = torch.tensor(Y, dtype=torch.float32).to(self._device)

        self._model.train()
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(n_samples)
            for start in range(0, n_samples - batch_size + 1, batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = X_t[batch_idx]
                yb = Y_t[batch_idx]

                pred = self._model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            logger.info(f"CryptoMamba epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), save_path)
        logger.info(f"Saved: {save_path}")

        # Clear training tensors from GPU
        del X_t, Y_t, optimizer, scheduler
        if self._device != "cpu":
            torch.cuda.empty_cache()

        self._model.eval()
        return {
            "final_loss": losses[-1],
            "n_samples": n_samples,
            "n_epochs": epochs,
            "save_path": save_path,
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
