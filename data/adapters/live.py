"""Live data adapter — แปลง WebSocket stream เป็น StandardFeatureVector.

*** กฎสำคัญที่สุด ***
- Accumulate tick/orderbook data ระหว่างแท่งเทียน
- ส่ง signal ให้โมเดลเฉพาะเมื่อแท่งเทียนปิด (candle close)
- ห้าม action ทุกครั้งที่ดึงข้อมูล ไม่งั้นระบบพัง
"""

import logging
import time
from collections import deque
from threading import Lock

import numpy as np

from data.adapters.base import BaseDataAdapter
from data.schema import OHLCV, StandardFeatureVector

logger = logging.getLogger(__name__)


class CandleAggregator:
    """Aggregate tick data เป็น candle แล้ว signal เมื่อ candle close.

    *** นี่คือ component ที่สำคัญที่สุดใน live pipeline ***
    ถ้าไม่มี aggregator นี้ โมเดลจะ action ทุก tick → overtrade → พัง
    """

    def __init__(self, timeframe_seconds: int = 900):  # default 15m
        self.timeframe_seconds = timeframe_seconds
        self._current_candle: OHLCV | None = None
        self._last_closed_candle: OHLCV | None = None
        self._candle_closed = False
        self._lock = Lock()

        # Accumulate features ระหว่างแท่งเทียน
        self._tick_volumes: list[float] = []
        self._tick_prices: list[float] = []
        self._bid_ask_spreads: list[float] = []

    @property
    def candle_closed(self) -> bool:
        with self._lock:
            return self._candle_closed

    def _get_candle_open_time(self, timestamp_ms: int) -> int:
        """คำนวณ open time ของแท่งเทียนที่ timestamp นี้อยู่."""
        interval_ms = self.timeframe_seconds * 1000
        return (timestamp_ms // interval_ms) * interval_ms

    def on_trade(self, price: float, volume: float, timestamp_ms: int):
        """รับ trade tick — accumulate แต่ไม่ action."""
        with self._lock:
            candle_open = self._get_candle_open_time(timestamp_ms)

            if self._current_candle is None:
                # เริ่มแท่งเทียนใหม่
                self._start_new_candle(price, volume, candle_open)
                return

            if candle_open > self._current_candle.timestamp:
                # แท่งเทียนเก่าปิดแล้ว → signal!
                self._last_closed_candle = OHLCV(
                    timestamp=self._current_candle.timestamp,
                    open=self._current_candle.open,
                    high=self._current_candle.high,
                    low=self._current_candle.low,
                    close=self._current_candle.close,
                    volume=self._current_candle.volume,
                )
                self._candle_closed = True
                self._start_new_candle(price, volume, candle_open)
                return

            # ยังอยู่ในแท่งเทียนเดิม → update
            self._current_candle.high = max(self._current_candle.high, price)
            self._current_candle.low = min(self._current_candle.low, price)
            self._current_candle.close = price
            self._current_candle.volume += volume

            self._tick_prices.append(price)
            self._tick_volumes.append(volume)

    def on_orderbook(self, best_bid: float, best_ask: float):
        """รับ orderbook update — accumulate spread แต่ไม่ action."""
        with self._lock:
            if best_bid > 0 and best_ask > 0:
                self._bid_ask_spreads.append(best_ask - best_bid)

    def _start_new_candle(self, price: float, volume: float, open_time: int):
        self._current_candle = OHLCV(
            timestamp=open_time,
            open=price, high=price, low=price, close=price,
            volume=volume,
        )
        self._tick_volumes = [volume]
        self._tick_prices = [price]
        self._bid_ask_spreads = []

    def consume_closed_candle(self) -> OHLCV | None:
        """ดึง candle ที่ปิดแล้ว แล้ว reset flag.

        *** เรียกเมื่อ candle_closed == True เท่านั้น ***
        """
        with self._lock:
            if not self._candle_closed:
                return None
            self._candle_closed = False
            candle = self._last_closed_candle
            self._last_closed_candle = None
            if candle is not None:
                return candle
            # Return copy ของ candle ก่อนหน้า (ที่ปิดแล้ว)
            # Note: _current_candle ตอนนี้คือแท่งใหม่แล้ว
            # ต้องเก็บ previous candle ไว้
            return self._current_candle  # TODO: เก็บ previous candle

    def get_accumulated_features(self) -> dict:
        """ดึง feature ที่ accumulate ระหว่างแท่งเทียน."""
        with self._lock:
            vwap = 0.0
            if self._tick_volumes and sum(self._tick_volumes) > 0:
                vwap = (
                    np.dot(self._tick_prices, self._tick_volumes)
                    / sum(self._tick_volumes)
                )

            avg_spread = 0.0
            if self._bid_ask_spreads:
                avg_spread = np.mean(self._bid_ask_spreads)

            return {
                "vwap": vwap,
                "avg_spread": avg_spread,
                "tick_count": len(self._tick_prices),
                "volume_sum": sum(self._tick_volumes),
            }


class LiveAdapter(BaseDataAdapter):
    """Live data adapter — ใช้ CandleAggregator เพื่อ action เฉพาะ candle close."""

    def __init__(
        self,
        timeframe_seconds: int = 900,
        lookback: int = 60,
    ):
        self.aggregator = CandleAggregator(timeframe_seconds)
        self.lookback = lookback
        self._candle_history: deque[OHLCV] = deque(maxlen=lookback + 10)
        self._return_periods = [1, 5, 15, 60, 240]

    def on_trade(self, price: float, volume: float, timestamp_ms: int):
        """รับ trade data จาก WebSocket. ไม่ action ที่นี่."""
        self.aggregator.on_trade(price, volume, timestamp_ms)

    def on_orderbook(self, best_bid: float, best_ask: float):
        """รับ orderbook update จาก WebSocket. ไม่ action ที่นี่."""
        self.aggregator.on_orderbook(best_bid, best_ask)

    def is_candle_closed(self) -> bool:
        """*** ตรวจว่าแท่งเทียนปิดแล้วหรือยัง — action เฉพาะเมื่อ True ***"""
        return self.aggregator.candle_closed

    def get_feature_vector(self, symbol: str, timestamp: int = 0) -> StandardFeatureVector:
        """สร้าง feature vector เฉพาะเมื่อ candle close.

        *** ต้องเรียก is_candle_closed() ก่อนเสมอ ***
        """
        candle = self.aggregator.consume_closed_candle()
        if candle is not None:
            self._candle_history.append(candle)

        if len(self._candle_history) < self.lookback:
            raise ValueError(
                f"Not enough candles: {len(self._candle_history)} < {self.lookback}. "
                "Wait for warmup."
            )

        # Build OHLCV window (same as historical)
        recent = list(self._candle_history)[-self.lookback:]
        ohlcv = np.array([
            [c.open, c.high, c.low, c.close, c.volume] for c in recent
        ])

        # Log returns
        closes = ohlcv[:, 3]
        all_returns = np.log(closes[1:] / closes[:-1])
        returns = np.array([
            all_returns[-min(p, len(all_returns)):].sum()
            for p in self._return_periods
        ])

        # Placeholder features (เติมใน Phase 2)
        return StandardFeatureVector(
            ohlcv=ohlcv,
            returns=returns,
            ta_indicators=np.zeros(15, dtype=np.float32),
            microstructure=np.zeros(5, dtype=np.float32),
            price_forecast=np.zeros(12, dtype=np.float32),
            forecast_uncertainty=0.0,
            funding_rate=0.0,
            open_interest_change=0.0,
            sentiment_score=0.0,
            onchain_metrics=np.zeros(5, dtype=np.float32),
            timestamp=candle.timestamp if candle else 0,
            symbol=symbol,
        )
