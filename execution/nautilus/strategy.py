"""GARIC strategy implemented on top of NautilusTrader."""

from __future__ import annotations

from collections import deque
from decimal import Decimal

import numpy as np
import pandas as pd

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

from execution.nautilus.features import NautilusFeatureBuilder
from execution.nautilus.model import GaricModelAdapter
from execution.nautilus.state import NautilusStateWriter


class GaricNautilusStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    model_path: str
    state_path: str
    history_bars: PositiveInt = 160
    request_history_days: PositiveInt = 3
    starting_balance: float = 10_000.0
    mode: str = "backtest"
    close_positions_on_stop: bool = True
    reduce_only_on_stop: bool = True


class GaricNautilusStrategy(Strategy):
    def __init__(self, config: GaricNautilusStrategyConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self.model = GaricModelAdapter(config.model_path)
        self.features = NautilusFeatureBuilder(
            config.history_bars,
            include_forecast=self.model.feature_dim > 25,
        )
        self.state_writer = NautilusStateWriter(config.state_path)
        self._bars: deque[dict] = deque(maxlen=self.features.history_bars)
        self._last_bar_ts_event: int | None = None
        self._latest_price = 0.0
        self._last_signal = 0.0
        self._last_confidence = 0.0
        self._last_probabilities = {"short": 0.0, "flat": 0.0, "long": 0.0}
        self._flat_steps = 0
        self._pos_steps = 0
        self._last_turnover = 0.0
        self._last_realized_pnl = 0.0
        self._n_trades = 0
        self._n_wins = 0
        self._n_losses = 0
        self._action_counts = {"short": 0, "flat": 0, "long": 0}
        self._seen_event_ids: set[str] = set()
        self._peak_equity = float(config.starting_balance)
        self._max_drawdown = 0.0

    def snapshot(self) -> dict:
        return {
            "n_trades": self._n_trades,
            "n_wins": self._n_wins,
            "n_losses": self._n_losses,
            "win_rate": self._win_rate(),
            "action_counts": dict(self._action_counts),
            "last_signal": self._last_signal,
            "model_family": self.model.model_family,
            "model_path": str(self.model.model_path).replace("\\", "/"),
            "max_drawdown": float(self._max_drawdown),
        }

    def on_start(self) -> None:
        self.state_writer.update(
            status="STARTING",
            mode=self.config.mode,
            symbol=self.config.instrument_id.symbol.value,
            instrument_id=str(self.config.instrument_id),
            model_path=str(self.model.model_path).replace("\\", "/"),
            model_family=self.model.model_family,
            trade_size=str(self.config.trade_size),
            bar_type=str(self.config.bar_type),
            starting_balance=self.config.starting_balance,
            recent_price=0.0,
            event=f"Starting GARIC Nautilus strategy ({self.config.mode})",
        )
        self.request_instrument(self.config.instrument_id)
        self.subscribe_instrument(self.config.instrument_id)
        self.request_bars(
            self.config.bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=int(self.config.request_history_days)),
        )
        self.subscribe_bars(self.config.bar_type)

    def on_instrument(self, instrument) -> None:
        if instrument.id == self.config.instrument_id:
            self.instrument = instrument
            self.state_writer.update(event=f"Instrument ready: {instrument.id}")

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.config.bar_type:
            return
        if self._last_bar_ts_event == bar.ts_event:
            return
        self._last_bar_ts_event = int(bar.ts_event)

        row = {
            "open_time": pd.Timestamp(bar.ts_event, unit="ns", tz="UTC"),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        self._latest_price = row["close"]
        self._bars.append(row)

        if len(self._bars) < self.features.warmup_bars:
            self.state_writer.update(
                status="WARMUP",
                warmup_progress=len(self._bars),
                warmup_target=self.features.warmup_bars,
                recent_price=row["close"],
                history={
                    "ts": int(bar.ts_event),
                    "price": row["close"],
                    "equity": self.config.starting_balance,
                    "position": 0.0,
                    "upnl": 0.0,
                },
            )
            return

        frame = pd.DataFrame(list(self._bars))
        snapshot = self.features.build_latest(frame)
        current_position = self._current_position_state()
        if current_position == 0.0:
            self._flat_steps += 1
            self._pos_steps = 0
        else:
            self._pos_steps += 1
            self._flat_steps = 0

        upnl = self._portfolio_pnl("unrealized", snapshot.latest_price)
        total_pnl = self._portfolio_pnl("total", snapshot.latest_price)
        equity = float(self.config.starting_balance) + total_pnl
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (equity / self._peak_equity) - 1.0
            self._max_drawdown = min(self._max_drawdown, float(drawdown))

        prediction = self.model.predict(
            snapshot.feature_array,
            position_state=current_position,
            flat_steps=self._flat_steps,
            pos_steps=self._pos_steps,
            upnl=upnl,
            equity_ratio=(equity / max(float(self.config.starting_balance), 1e-9)) - 1.0,
            drawdown=abs(drawdown),
            rolling_volatility=self._rolling_volatility(frame),
            turnover_last_step=self._last_turnover,
        )
        self._record_action(prediction.direction)
        self._rebalance(prediction.direction, snapshot.latest_price)
        self._last_signal = prediction.direction
        self._last_confidence = prediction.confidence
        self._last_probabilities = dict(prediction.probabilities)
        post_position = self._current_position_state()
        realized_after = self._portfolio_pnl("realized", snapshot.latest_price)
        self._sync_trade_counters(
            previous_position=current_position,
            current_position=post_position,
            realized_pnl_after=realized_after,
        )
        self._publish_runtime_state(
            status="RUNNING",
            price=snapshot.latest_price,
            history={
                "ts": int(bar.ts_event),
                "price": snapshot.latest_price,
                "equity": equity,
                "position": post_position,
                "upnl": upnl,
            },
        )

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(
                instrument_id=self.config.instrument_id,
                reduce_only=self.config.reduce_only_on_stop,
            )
        self.unsubscribe_bars(self.config.bar_type)
        self.unsubscribe_instrument(self.config.instrument_id)
        self.state_writer.update(status="STOPPED", event="Strategy stopped")

    def on_event(self, event) -> None:
        event_name = type(event).__name__
        event_id = getattr(event, "id", None) or getattr(event, "event_id", None)
        event_key = (
            str(event_id)
            if event_id is not None
            else f"{event_name}:{getattr(event, 'ts_event', 0)}:{id(event)}"
        )
        if event_key in self._seen_event_ids:
            return
        self._seen_event_ids.add(event_key)

        price = self._safe_float(getattr(event, "last_px", 0.0)) or self._latest_price
        if event_name == "OrderFilled":
            side = "BUY" if getattr(event, "is_buy", False) else "SELL" if getattr(event, "is_sell", False) else "FILL"
            qty = self._safe_float(getattr(event, "last_qty", 0.0))
            self._publish_runtime_state(
                price=price,
                event=f"Order filled: {side} {qty:g} @ {price:,.2f}",
            )
            return

        if event_name not in {"PositionOpened", "PositionChanged", "PositionClosed"}:
            return

        realized_pnl = self._safe_float(getattr(event, "realized_pnl", 0.0))
        realized_return = self._safe_float(getattr(event, "realized_return", 0.0))
        side = getattr(getattr(event, "side", None), "value", str(getattr(event, "side", "")))
        qty = self._safe_float(getattr(event, "signed_qty", 0.0))

        if event_name == "PositionOpened":
            message = f"Position opened: {side} {qty:+.4f} @ {price:,.2f}"
        elif event_name == "PositionClosed":
            message = f"Position closed: PnL {realized_pnl:+.2f} ({realized_return:+.2%})"
        else:
            message = f"Position changed: realized PnL {realized_pnl:+.2f}"

        self._publish_runtime_state(
            price=price,
            event=message,
        )

    def _record_action(self, direction: float) -> None:
        if direction > 0:
            self._action_counts["long"] += 1
        elif direction < 0:
            self._action_counts["short"] += 1
        else:
            self._action_counts["flat"] += 1

    def _current_position_state(self) -> float:
        if self.portfolio.is_net_long(self.config.instrument_id):
            return 1.0
        if self.portfolio.is_net_short(self.config.instrument_id):
            return -1.0
        return 0.0

    def _portfolio_pnl(self, kind: str, price: float) -> float:
        if self.instrument is None:
            return 0.0
        price_obj = self.instrument.make_price(price)
        if kind == "unrealized":
            value = self.portfolio.unrealized_pnl(self.config.instrument_id, price=price_obj)
        elif kind == "realized":
            value = self.portfolio.realized_pnl(self.config.instrument_id)
        else:
            value = self.portfolio.total_pnl(self.config.instrument_id, price=price_obj)
        return self._safe_float(value)

    def _safe_float(self, value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except Exception:
            try:
                return float(value.as_decimal())
            except Exception:
                return 0.0

    def _rebalance(self, desired_direction: float, price: float) -> None:
        current_direction = self._current_position_state()
        self._last_turnover = abs(desired_direction - current_direction)
        if abs(desired_direction - current_direction) < 1e-9:
            self._last_turnover = 0.0
            return

        if current_direction != 0.0:
            self.close_all_positions(self.config.instrument_id)

        if desired_direction > 0:
            self._submit_market(OrderSide.BUY)
        elif desired_direction < 0:
            self._submit_market(OrderSide.SELL)
        else:
            self._publish_runtime_state(
                price=price,
                event="Flattening position",
            )

    def _rolling_volatility(self, frame: pd.DataFrame, window: int = 32) -> float:
        closes = frame["close"].astype(float).to_numpy()[-(max(int(window), 1) + 1):]
        if closes.size < 2:
            return 0.0
        returns = pd.Series(np.log(np.maximum(closes, 1e-9))).diff().dropna().to_numpy()
        if returns.size == 0:
            return 0.0
        return float(np.std(returns))

    def _submit_market(self, side: OrderSide) -> None:
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self._order_qty(),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(f"Submitted {side.value} market order", color=LogColor.CYAN)

    def _order_qty(self) -> Quantity:
        if self.instrument is not None:
            return self.instrument.make_qty(self.config.trade_size)
        return Quantity.from_str(str(self.config.trade_size))

    def _sync_trade_counters(
        self,
        *,
        previous_position: float,
        current_position: float,
        realized_pnl_after: float,
    ) -> None:
        realized_delta = float(realized_pnl_after - self._last_realized_pnl)

        if previous_position == 0.0 and current_position != 0.0:
            self._n_trades += 1
        elif previous_position != 0.0 and current_position == 0.0:
            if realized_delta > 0:
                self._n_wins += 1
            elif realized_delta < 0:
                self._n_losses += 1
        elif previous_position != 0.0 and current_position != 0.0 and previous_position != current_position:
            if realized_delta > 0:
                self._n_wins += 1
            elif realized_delta < 0:
                self._n_losses += 1
            self._n_trades += 1

        self._last_realized_pnl = float(realized_pnl_after)

    def _publish_runtime_state(
        self,
        *,
        price: float | None = None,
        status: str | None = None,
        event: str | None = None,
        history: dict | None = None,
    ) -> None:
        mark_price = float(price or self._latest_price or 0.0)
        current_position = self._current_position_state() if mark_price else 0.0
        upnl = self._portfolio_pnl("unrealized", mark_price) if mark_price else 0.0
        total_pnl = self._portfolio_pnl("total", mark_price) if mark_price else 0.0
        equity = float(self.config.starting_balance) + total_pnl
        self.state_writer.update(
            status=status or self.state_writer.state.get("status", "RUNNING"),
            recent_price=mark_price,
            position=current_position,
            action=self._direction_label(self._last_signal),
            action_value=self._last_signal,
            confidence=self._last_confidence,
            probabilities=dict(self._last_probabilities),
            total_pnl=total_pnl,
            unrealized_pnl=upnl,
            equity=equity,
            total_return=(equity / max(float(self.config.starting_balance), 1e-9)) - 1.0,
            n_trades=self._n_trades,
            n_wins=self._n_wins,
            n_losses=self._n_losses,
            win_rate=self._win_rate(),
            action_counts=dict(self._action_counts),
            history=history,
            event=event,
        )

    def _win_rate(self) -> float:
        total = self._n_wins + self._n_losses
        return float(self._n_wins / total) if total > 0 else 0.0

    def _direction_label(self, direction: float) -> str:
        if direction > 0:
            return "Long"
        if direction < 0:
            return "Short"
        return "Flat"
