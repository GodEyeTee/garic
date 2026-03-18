"""Trading engine — รวม WebSocket + Model + Risk + Orders.

*** กฎสำคัญ ***
1. Action เฉพาะ candle close — ห้าม action ระหว่างแท่งเทียน
2. Feature vector ต้องเหมือน training 100%
3. Risk check ก่อนส่ง order ทุกครั้ง
4. Drift detection ทุก check_interval
5. Paper mode ก่อน live เสมอ

Usage:
  python -m execution.live.trading_engine --mode paper --symbol BTCUSDT
"""

import logging
import time
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.adapters.live import LiveAdapter
from execution.live.order_manager import OrderManager
from monitoring.live.dashboard import DashboardData
from monitoring.live.drift_detector import DriftDetector, AlertManager
from risk.manager import RiskManager

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine — paper และ live."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe_seconds: int = 900,  # 15m
        paper_mode: bool = True,
        model_path: str | None = None,
        api_key: str = "",
        api_secret: str = "",
        telegram_token: str = "",
        telegram_chat_id: str = "",
        max_drawdown: float = 0.15,
        daily_loss_limit: float = 0.03,
        max_open_positions: int = 5,
    ):
        self.symbol = symbol
        self.paper_mode = paper_mode
        self._running = False

        # Components
        self.adapter = LiveAdapter(
            timeframe_seconds=timeframe_seconds,
            lookback=60,
        )
        self.order_manager = OrderManager(
            paper_mode=paper_mode,
            api_key=api_key,
            secret=api_secret,
        )
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager(telegram_token, telegram_chat_id)
        self.dashboard_data = DashboardData()
        self.risk_manager = RiskManager(
            max_drawdown=max_drawdown,
            daily_loss_limit=daily_loss_limit,
            max_open_positions=max_open_positions,
        )

        # Model (load if path provided)
        self._model = None
        if model_path and Path(model_path).exists():
            self._load_model(model_path)

        # State
        self._last_action = 0.0
        self._candle_count = 0
        self._paused = False
        self._flat_steps = 0
        self._pos_steps = 0
        self._entry_price = 0.0
        self._trade_returns: list[float] = []
        self._last_reset_day: str | None = None
        self._last_equity: float | None = None
        self._peak_equity: float | None = None

    def _load_model(self, path: str):
        """Load trained RL model."""
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(path)
            logger.info(f"Model loaded: {path}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")

    def _maybe_reset_daily(self, timestamp_ms: int):
        day_key = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        if self._last_reset_day is None:
            self._last_reset_day = day_key
            return
        if day_key != self._last_reset_day:
            self.risk_manager.reset_daily()
            self._last_reset_day = day_key
            logger.info("RiskManager daily state reset for UTC day %s", day_key)

    def _estimate_equity(self, price: float) -> float:
        balance = self._estimate_equity(price)
        if self.paper_mode:
            return float(balance + (self.order_manager.get_position(self.symbol) * price))
        return float(balance)

    def _update_risk_state(self, timestamp_ms: int, price: float):
        self._maybe_reset_daily(timestamp_ms)
        equity = self._estimate_equity(price)
        if self._last_equity is None:
            self._last_equity = equity
            self._peak_equity = equity
        else:
            self.risk_manager.update_pnl(equity - self._last_equity)
            self._last_equity = equity
            self._peak_equity = max(self._peak_equity or equity, equity)
        return equity

    def _current_drawdown(self, equity: float) -> float:
        peak = max(self._peak_equity or equity, equity, 1e-9)
        return max((peak - equity) / peak, 0.0)

    def _trade_stats(self) -> tuple[float, float, float]:
        if not self._trade_returns:
            return 0.5, 0.02, 0.01
        wins = [r for r in self._trade_returns if r > 0]
        losses = [-r for r in self._trade_returns if r < 0]
        win_rate = len(wins) / max(len(self._trade_returns), 1)
        avg_win = float(np.mean(wins)) if wins else 0.02
        avg_loss = float(np.mean(losses)) if losses else 0.01
        return float(win_rate), avg_win, avg_loss

    def _estimate_atr_pct(self, features) -> float:
        ohlcv = np.asarray(features.ohlcv, dtype=np.float64)
        if ohlcv.ndim != 2 or ohlcv.shape[0] == 0:
            return 0.02
        recent = ohlcv[-min(len(ohlcv), 14):]
        closes = np.maximum(recent[:, 3], 1e-9)
        return float(np.mean((recent[:, 1] - recent[:, 2]) / closes))

    def _evaluate_risk(self, direction: float, features, price: float, equity: float):
        win_rate, avg_win, avg_loss = self._trade_stats()
        uncertainty = float(getattr(features, "forecast_uncertainty", 0.0))
        model_confidence = 0.5 if uncertainty <= 0 else float(np.clip(1.0 - uncertainty, 0.1, 1.0))
        current_drawdown = self._current_drawdown(equity)
        decision = self.risk_manager.evaluate(
            symbol=self.symbol,
            direction=direction,
            equity=equity,
            model_confidence=model_confidence,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            atr_value=self._estimate_atr_pct(features),
            current_drawdown=current_drawdown,
            is_major=self.symbol.upper() in {"BTCUSDT", "ETHUSDT"},
        )
        if not decision.approved:
            logger.warning("Risk rejected trade: %s", decision.reject_reason)
        return decision

    def _record_closed_trade(self, direction: float, price: float):
        if direction == 0 or self._entry_price <= 0:
            return
        trade_return = direction * ((price / self._entry_price) - 1.0)
        self._trade_returns.append(float(trade_return))
        self._trade_returns = self._trade_returns[-200:]

    def on_trade(self, price: float, volume: float, timestamp_ms: int):
        """Callback จาก WebSocket — accumulate แต่ไม่ action."""
        self.adapter.on_trade(price, volume, timestamp_ms)

        # *** ตรวจว่าแท่งเทียนปิดแล้วหรือยัง ***
        if self.adapter.is_candle_closed():
            self._on_candle_close(timestamp_ms)

    def on_orderbook(self, best_bid: float, best_ask: float):
        """Callback จาก WebSocket — accumulate spread."""
        self.adapter.on_orderbook(best_bid, best_ask)

    def _on_candle_close(self, timestamp_ms: int):
        """*** เกิดเฉพาะเมื่อแท่งเทียนปิด — ที่นี่คือจุดตัดสินใจ ***"""
        self._candle_count += 1

        if self._paused:
            logger.info(f"Candle {self._candle_count}: PAUSED — skip action")
            return

        try:
            # 1. Get feature vector (same as training)
            features = self.adapter.get_feature_vector(self.symbol, timestamp_ms)
            feature_array = features.to_array()

        except ValueError as e:
            logger.info(f"Warmup: {e}")
            return

        price = float(features.ohlcv[-1, 3])
        equity = self._update_risk_state(timestamp_ms, price)

        # 2. Model prediction
        action = self._predict_action(feature_array)

        # 3. Drift detection
        self.drift_detector.update(feature_array, action, 0.0)
        if self.drift_detector.should_check():
            drift_result = self.drift_detector.check_all()
            if drift_result["should_pause"]:
                self._paused = True
                self.alert_manager.send_alert(
                    f"Trading PAUSED: {drift_result['details']}",
                    level="critical",
                )
                return
            elif drift_result["should_retrain"]:
                self.alert_manager.send_alert(
                    f"Retrain recommended: {drift_result['details']}",
                    level="warning",
                )

        # 4. Execute (ถ้า action เปลี่ยน)
        if abs(action - self._last_action) > 0.05:  # minimum change threshold
            self._execute_action(action, features, timestamp_ms, equity)
            self._last_action = action

        # 5. Update dashboard
        balance = self._estimate_equity(price)
        self.dashboard_data.add_record(
            timestamp=timestamp_ms,
            balance=balance,
            position=self.order_manager.get_position(self.symbol),
            pnl=0.0,
            action=action,
            confidence=features.forecast_uncertainty,
            symbol=self.symbol,
        )

        if self._candle_count % 100 == 0:
            self.dashboard_data.save()
            metrics = self.dashboard_data.get_metrics()
            logger.info(f"Candle {self._candle_count}: balance=${balance:.2f}, metrics={metrics}")

    def _predict_action(self, feature_array: np.ndarray) -> float:
        """Get discrete action from model and map it to [-1, 0, 1]."""
        if self._model is None:
            return 0.0  # no model = flat

        current_position = self.order_manager.get_position(self.symbol)
        position_state = float(np.clip(np.sign(current_position), -1.0, 1.0))
        if position_state == 0.0:
            self._flat_steps += 1
            self._pos_steps = 0
        else:
            self._pos_steps += 1
            self._flat_steps = 0

        obs = np.concatenate([
            feature_array,
            np.array([
                position_state,
                0.0,
                self._flat_steps / 100.0,
                self._pos_steps / 100.0,
            ], dtype=np.float32),
        ])
        action, _ = self._model.predict(obs, deterministic=True)
        action_idx = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        if action_idx == 0:
            return -1.0
        if action_idx == 2:
            return 1.0
        return 0.0

    def _execute_action(self, action: float, features, timestamp_ms: int, equity: float):
        """Execute trade based on action with risk approval."""
        price = float(features.ohlcv[-1, 3])
        if price <= 0:
            return

        current_qty = float(self.order_manager.get_position(self.symbol))
        current_direction = float(np.sign(current_qty))
        desired_direction = float(np.sign(action))

        self.risk_manager.set_position(self.symbol, abs(current_qty) * price)

        if desired_direction == 0.0:
            if current_qty == 0.0:
                return
            self._record_closed_trade(current_direction, price)
            side = "sell" if current_qty > 0 else "buy"
            self.order_manager.place_order(
                symbol=self.symbol,
                side=side,
                amount=abs(current_qty),
                current_price=price,
            )
            self._entry_price = 0.0
            self.risk_manager.set_position(self.symbol, 0.0)
            self._last_equity = self._estimate_equity(price)
            self._peak_equity = max(self._peak_equity or self._last_equity, self._last_equity)
            return

        if current_direction == desired_direction and current_qty != 0.0:
            self.risk_manager.set_position(self.symbol, abs(current_qty) * price)
            return

        if current_direction != 0.0 and current_direction != desired_direction:
            self._record_closed_trade(current_direction, price)
            close_side = "sell" if current_qty > 0 else "buy"
            self.order_manager.place_order(
                symbol=self.symbol,
                side=close_side,
                amount=abs(current_qty),
                current_price=price,
            )
            current_qty = 0.0
            current_direction = 0.0
            self._entry_price = 0.0
            self.risk_manager.set_position(self.symbol, 0.0)
            equity = self._estimate_equity(price)

        decision = self._evaluate_risk(desired_direction, features, price, equity)
        if not decision.approved or decision.size <= 0:
            return

        quantity = float(decision.size / price)
        side = "buy" if desired_direction > 0 else "sell"
        result = self.order_manager.place_order(
            symbol=self.symbol,
            side=side,
            amount=quantity,
            current_price=price,
        )
        if result.status == "filled":
            self._entry_price = price
            self.risk_manager.set_position(self.symbol, decision.size)
            self._last_equity = self._estimate_equity(price)
            self._peak_equity = max(self._peak_equity or self._last_equity, self._last_equity)

    def start(self):
        """Start trading engine with WebSocket."""
        from execution.live.websocket_client import BinanceFuturesWS

        self._running = True
        mode = "PAPER" if self.paper_mode else "LIVE"
        logger.info(f"=== GARIC Trading Engine [{mode}] — {self.symbol} ===")

        # Graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down...")
            self._running = False
            self.dashboard_data.save()
            ws.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        ws = BinanceFuturesWS(
            symbol=self.symbol,
            on_trade=self.on_trade,
            on_depth=self.on_orderbook,
        )
        ws.start()

        # Keep alive
        while self._running:
            time.sleep(1)

    def resume(self):
        """Resume trading after pause."""
        self._paused = False
        logger.info("Trading resumed")
        self.alert_manager.send_alert("Trading RESUMED", level="warning")


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="GARIC Trading Engine")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", type=int, default=900, help="Candle timeframe in seconds")
    parser.add_argument("--model", default=None, help="Path to trained RL model")
    parser.add_argument("--api-key", default="", help="Exchange API key (live mode)")
    parser.add_argument("--api-secret", default="", help="Exchange API secret (live mode)")
    parser.add_argument("--telegram-token", default="", help="Telegram bot token for alerts")
    parser.add_argument("--telegram-chat", default="", help="Telegram chat ID for alerts")
    args = parser.parse_args()

    engine = TradingEngine(
        symbol=args.symbol,
        timeframe_seconds=args.timeframe,
        paper_mode=(args.mode == "paper"),
        model_path=args.model,
        api_key=args.api_key,
        api_secret=args.api_secret,
        telegram_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat,
    )
    engine.start()


if __name__ == "__main__":
    main()
