"""Binance Futures WebSocket client สำหรับ live data.

เชื่อมกับ CandleAggregator — ดึงข้อมูล real-time แต่ action เฉพาะ candle close.

Streams:
- kline: OHLCV real-time
- depth: order book L2
- trade: individual trades
- markPrice: mark price + funding rate
"""

import json
import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://fstream.binance.com/ws"


class BinanceFuturesWS:
    """Binance Futures WebSocket client."""

    def __init__(
        self,
        symbol: str = "btcusdt",
        on_trade: Callable | None = None,
        on_kline: Callable | None = None,
        on_depth: Callable | None = None,
        on_mark_price: Callable | None = None,
        kline_interval: str = "15m",
    ):
        self.symbol = symbol.lower()
        self.kline_interval = kline_interval
        self._on_trade = on_trade
        self._on_kline = on_kline
        self._on_depth = on_depth
        self._on_mark_price = on_mark_price
        self._running = False
        self._ws = None
        self._thread: threading.Thread | None = None

    def start(self):
        """Start WebSocket connection in background thread."""
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client not installed: pip install websocket-client")
            return

        streams = [
            f"{self.symbol}@trade",
            f"{self.symbol}@kline_{self.kline_interval}",
            f"{self.symbol}@depth5@100ms",
            f"{self.symbol}@markPrice@1s",
        ]
        url = f"{BINANCE_WS_BASE}/{'/'.join(streams)}"

        self._running = True
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket started: {self.symbol}")

    def _run_forever(self):
        while self._running:
            try:
                self._ws.run_forever(ping_interval=30)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            if self._running:
                logger.info("Reconnecting in 5s...")
                time.sleep(5)

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info("WebSocket stopped")

    def _on_open(self, ws):
        logger.info(f"WebSocket connected: {self.symbol}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        logger.info(f"WebSocket closed: {close_status} {close_msg}")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            event = data.get("e", "")

            if event == "trade" and self._on_trade:
                self._on_trade(
                    price=float(data["p"]),
                    volume=float(data["q"]),
                    timestamp_ms=data["T"],
                )
            elif event == "kline" and self._on_kline:
                k = data["k"]
                self._on_kline(
                    open_time=k["t"],
                    open=float(k["o"]),
                    high=float(k["h"]),
                    low=float(k["l"]),
                    close=float(k["c"]),
                    volume=float(k["v"]),
                    is_closed=k["x"],
                )
            elif event == "depthUpdate" and self._on_depth:
                bids = data.get("b", [])
                asks = data.get("a", [])
                if bids and asks:
                    self._on_depth(
                        best_bid=float(bids[0][0]),
                        best_ask=float(asks[0][0]),
                    )
            elif event == "markPriceUpdate" and self._on_mark_price:
                self._on_mark_price(
                    mark_price=float(data["p"]),
                    funding_rate=float(data.get("r", 0)),
                    next_funding_time=data.get("T", 0),
                )
        except Exception as e:
            logger.error(f"Message parse error: {e}")
