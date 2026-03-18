"""Mixture of Experts router — เลือก expert ตาม market regime.

Top-2 gating (ไม่ใช่ top-1) ให้ผลดีกว่า + noise สำหรับ load balancing.
Expert แต่ละตัวเชี่ยวชาญ regime ต่างกัน.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RegimeDetector:
    """ตรวจจับ market regime จาก features.

    ใช้ z-score normalization → threshold ทำงานได้กับทุก timeframe (1m, 15m, 1h, 1d)
    โดย normalize avg_return และ volatility ด้วย long-run std ของ returns

    Regimes:
    0: Trending Bull    - high returns, low vol
    1: Trending Bear    - negative returns, low vol
    2: Ranging          - near-zero returns, low vol
    3: High Volatility  - high vol, unclear direction
    4: Mean Reversion   - oscillating returns
    5: Momentum         - strong directional move
    """

    N_REGIMES = 6

    def __init__(self, vol_window: int = 20, return_window: int = 10):
        self.vol_window = vol_window
        self.return_window = return_window

    def detect(self, returns: np.ndarray, volatility: float) -> np.ndarray:
        """Return soft regime probabilities, shape (N_REGIMES,).

        Normalize returns/vol to z-scores ก่อนเปรียบเทียบ threshold
        ทำให้ทำงานได้กับทุก timeframe (1m ถึง 1d)
        """
        avg_return = returns[-self.return_window:].mean() if len(returns) >= self.return_window else 0.0

        # Normalize ด้วย long-run std เพื่อให้ threshold-agnostic ต่อ timeframe
        long_run_std = returns.std() if len(returns) > 50 else 1e-8
        long_run_std = max(long_run_std, 1e-10)  # prevent division by zero

        # return_z: normalize avg by std of the mean = sigma/sqrt(N)
        # ทำให้ return_z ~ N(0,1) ภายใต้ null hypothesis
        n_window = min(self.return_window, len(returns))
        se_mean = long_run_std / np.sqrt(max(n_window, 1))
        return_z = avg_return / se_mean  # unitless: กี่ SE จาก 0

        vol_z = volatility / long_run_std  # unitless: vol เทียบกับ long-run

        # Simple rule-based soft assignment (จะถูกแทนที่ด้วย learned gating)
        # threshold เป็น z-score → ทำงานกับทุก timeframe
        scores = np.zeros(self.N_REGIMES, dtype=np.float32)

        if return_z > 1.0 and vol_z < 1.2:
            scores[0] = 2.0  # bull: return > 1σ, vol ปกติ
        elif return_z < -1.0 and vol_z < 1.2:
            scores[1] = 2.0  # bear: return < -1σ, vol ปกติ
        elif abs(return_z) < 0.5 and vol_z < 0.8:
            scores[2] = 2.0  # ranging: return ใกล้ 0, vol ต่ำ
        elif vol_z > 2.0:
            scores[3] = 2.0  # high vol: vol > 2x long-run
        elif vol_z > 1.0 and abs(return_z) < 0.5:
            scores[4] = 1.5  # mean reversion: vol สูง, ไม่มี direction
        elif abs(return_z) > 0.5:
            scores[5] = 1.5  # momentum: มี direction ชัด
        else:
            scores[2] = 1.0  # default: ranging (แต่ score ต่ำกว่า)

        # Add noise for exploration (load balancing)
        scores += np.random.randn(self.N_REGIMES) * 0.1

        # Softmax
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        return probs


class MoERouter:
    """Mixture of Experts — route input to top-K experts.

    ใช้ top-2 gating ตาม research (ดีกว่า top-1).
    """

    def __init__(self, n_experts: int = 6, top_k: int = 2):
        self.n_experts = n_experts
        self.top_k = top_k
        self.regime_detector = RegimeDetector()

    def route(
        self,
        returns: np.ndarray,
        volatility: float,
    ) -> list[tuple[int, float]]:
        """Return top-K (expert_index, weight) pairs.

        ใช้สำหรับ weighted combination ของ expert outputs.
        """
        probs = self.regime_detector.detect(returns, volatility)

        # Top-K selection
        top_indices = np.argsort(probs)[-self.top_k:][::-1]
        top_weights = probs[top_indices]

        # Renormalize weights
        total = top_weights.sum()
        if total > 0:
            top_weights = top_weights / total

        return [(int(idx), float(w)) for idx, w in zip(top_indices, top_weights)]

    def combine_expert_outputs(
        self,
        expert_outputs: dict[int, np.ndarray],
        routing: list[tuple[int, float]],
    ) -> np.ndarray:
        """Weighted combination ของ expert outputs."""
        result = None
        for expert_idx, weight in routing:
            if expert_idx in expert_outputs:
                output = expert_outputs[expert_idx] * weight
                result = output if result is None else result + output
        return result
