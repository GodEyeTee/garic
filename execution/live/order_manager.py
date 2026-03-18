"""Order manager — ส่ง order ผ่าน CCXT (multi-exchange).

รองรับ:
- Paper mode: จำลอง order ไม่ส่งจริง
- Live mode: ส่ง order ผ่าน Binance Futures API via CCXT
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str           # "buy" or "sell"
    amount: float       # quantity
    price: float        # execution price
    cost: float         # total cost
    fee: float
    status: str         # "filled", "partial", "rejected"
    is_paper: bool
    timestamp: int


class OrderManager:
    """จัดการ order ทั้ง paper และ live."""

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        secret: str = "",
        paper_mode: bool = True,
    ):
        self.paper_mode = paper_mode
        self._exchange = None
        self._paper_balance = 10000.0
        self._paper_position = 0.0
        self._order_history: list[OrderResult] = []

        if not paper_mode:
            self._init_exchange(exchange_id, api_key, secret)

    def _init_exchange(self, exchange_id: str, api_key: str, secret: str):
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_id)
            self._exchange = exchange_class({
                "apiKey": api_key,
                "secret": secret,
                "options": {"defaultType": "future"},
                "enableRateLimit": True,
            })
            self._exchange.load_markets()
            logger.info(f"Exchange connected: {exchange_id}")
        except ImportError:
            logger.error("ccxt not installed: pip install ccxt")
        except Exception as e:
            logger.error(f"Exchange init failed: {e}")

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        order_type: str = "market",
    ) -> OrderResult:
        """ส่ง order — paper หรือ live ตาม mode."""
        if self.paper_mode:
            return self._paper_order(symbol, side, amount, current_price)
        else:
            return self._live_order(symbol, side, amount, order_type)

    def _paper_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> OrderResult:
        """จำลอง order — ไม่ส่งจริง."""
        slippage = price * 0.0001  # 1 bps simulated slippage
        exec_price = price + slippage if side == "buy" else price - slippage
        fee = amount * exec_price * 0.0005  # taker fee
        cost = amount * exec_price

        if side == "buy":
            self._paper_position += amount
            self._paper_balance -= cost + fee
        else:
            self._paper_position -= amount
            self._paper_balance += cost - fee

        result = OrderResult(
            order_id=f"paper_{int(time.time() * 1000)}",
            symbol=symbol,
            side=side,
            amount=amount,
            price=exec_price,
            cost=cost,
            fee=fee,
            status="filled",
            is_paper=True,
            timestamp=int(time.time() * 1000),
        )
        self._order_history.append(result)
        logger.info(f"PAPER {side.upper()} {amount} {symbol} @ {exec_price:.2f} (fee={fee:.4f})")
        return result

    def _live_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str,
    ) -> OrderResult:
        """ส่ง order จริงผ่าน CCXT."""
        if self._exchange is None:
            return OrderResult(
                order_id="", symbol=symbol, side=side, amount=amount,
                price=0, cost=0, fee=0, status="rejected",
                is_paper=False, timestamp=int(time.time() * 1000),
            )

        try:
            order = self._exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
            )

            result = OrderResult(
                order_id=str(order["id"]),
                symbol=symbol,
                side=side,
                amount=float(order.get("filled", amount)),
                price=float(order.get("average", order.get("price", 0))),
                cost=float(order.get("cost", 0)),
                fee=float(order.get("fee", {}).get("cost", 0)),
                status=order.get("status", "unknown"),
                is_paper=False,
                timestamp=int(time.time() * 1000),
            )
            self._order_history.append(result)
            logger.info(f"LIVE {side.upper()} {amount} {symbol} @ {result.price:.2f}")
            return result

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(
                order_id="", symbol=symbol, side=side, amount=amount,
                price=0, cost=0, fee=0, status="rejected",
                is_paper=False, timestamp=int(time.time() * 1000),
            )

    def get_balance(self) -> float:
        if self.paper_mode:
            return self._paper_balance
        if self._exchange:
            try:
                balance = self._exchange.fetch_balance()
                return float(balance.get("total", {}).get("USDT", 0))
            except Exception as e:
                logger.error(f"Balance fetch failed: {e}")
        return 0.0

    def get_position(self, symbol: str = "") -> float:
        if self.paper_mode:
            return self._paper_position
        if self._exchange:
            try:
                positions = self._exchange.fetch_positions([symbol] if symbol else None)
                for pos in positions:
                    if pos["symbol"] == symbol:
                        return float(pos.get("contracts", 0))
            except Exception as e:
                logger.error(f"Position fetch failed: {e}")
        return 0.0
