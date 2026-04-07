"""Low-level Nautilus backtest runner for GARIC."""

from __future__ import annotations

import argparse
import json
import logging
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

try:
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.data import BarSpecification
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AccountType
    from nautilus_trader.model.enums import AggregationSource
    from nautilus_trader.model.enums import BarAggregation
    from nautilus_trader.model.enums import OmsType
    from nautilus_trader.model.enums import PriceType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    _NAUTILUS_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    BacktestEngine = None  # type: ignore[assignment]
    dt_to_unix_nanos = None  # type: ignore[assignment]
    USDT = None  # type: ignore[assignment]
    Bar = Any  # type: ignore[assignment]
    BarSpecification = None  # type: ignore[assignment]
    BarType = Any  # type: ignore[assignment]
    AccountType = None  # type: ignore[assignment]
    AggregationSource = None  # type: ignore[assignment]
    BarAggregation = None  # type: ignore[assignment]
    OmsType = None  # type: ignore[assignment]
    PriceType = None  # type: ignore[assignment]
    Venue = None  # type: ignore[assignment]
    Money = None  # type: ignore[assignment]
    TestInstrumentProvider = None  # type: ignore[assignment]
    _NAUTILUS_IMPORT_ERROR = exc

from execution.nautilus.config import load_nautilus_config
from execution.nautilus.state import NautilusStateWriter
from execution.nautilus.strategy import GaricNautilusStrategy
from execution.nautilus.strategy import GaricNautilusStrategyConfig


logger = logging.getLogger(__name__)


def _require_nautilus() -> None:
    if _NAUTILUS_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "nautilus_trader is required for Nautilus backtests"
        ) from _NAUTILUS_IMPORT_ERROR


def _is_running(obj: object | None) -> bool:
    if obj is None:
        return False
    try:
        return bool(getattr(obj, "is_running", False))
    except Exception:
        return False


def _is_disposed(obj: object | None) -> bool:
    if obj is None:
        return False
    try:
        return bool(getattr(obj, "is_disposed", False))
    except Exception:
        return False


def _safe_dispose_engine(engine) -> None:
    """Shut down a backtest engine safely on Windows and on failed runs.

    Nautilus `run(streaming=False)` normally finalizes via `end()` itself, but if an
    exception interrupts the run the engine can still be in a RUNNING state. Calling
    `dispose()` directly in that case causes `InvalidStateTrigger('RUNNING -> DISPOSE')`.
    """
    if engine is None:
        return

    trader = getattr(engine, "trader", None)
    kernel = getattr(engine, "kernel", None)
    kernel_parts = [
        getattr(kernel, "trader", None),
        getattr(kernel, "data_engine", None),
        getattr(kernel, "risk_engine", None),
        getattr(kernel, "exec_engine", None),
        getattr(kernel, "emulator", None),
    ]
    engine_running = _is_running(trader) or any(_is_running(part) for part in kernel_parts)

    if engine_running:
        try:
            engine.end()
        except Exception as exc:
            logger.warning("Nautilus engine.end() during cleanup failed: %s", exc)

    if _is_disposed(trader):
        return

    try:
        engine.dispose()
    except Exception as exc:
        message = str(exc)
        if "InvalidStateTrigger" in message:
            logger.warning("Ignoring Nautilus dispose state conflict during cleanup: %s", message)
            return
        raise


def _aggregate_to_15m(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    frame = df.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame = frame.set_index("open_time").sort_index()
    aggregated = frame.resample(f"{minutes}min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna().reset_index()
    aggregated["close_time"] = aggregated["open_time"] + pd.Timedelta(minutes=minutes) - pd.Timedelta(milliseconds=1)
    return aggregated


def _prepare_15m_frame(frame: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    prepared = frame.copy()
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in prepared.columns]
    if missing:
        raise ValueError(f"15m frame missing required columns: {missing}")

    if "open_time" not in prepared.columns:
        prepared["open_time"] = pd.date_range(
            "2020-01-01",
            periods=len(prepared),
            freq=f"{bar_minutes}min",
            tz="UTC",
        )
    prepared["open_time"] = pd.to_datetime(prepared["open_time"], utc=True)

    if "volume" not in prepared.columns:
        prepared["volume"] = 1.0
    prepared["volume"] = prepared["volume"].astype(float)
    if "close_time" not in prepared.columns:
        prepared["close_time"] = (
            prepared["open_time"]
            + pd.Timedelta(minutes=bar_minutes)
            - pd.Timedelta(milliseconds=1)
        )
    prepared["close_time"] = pd.to_datetime(prepared["close_time"], utc=True)
    return prepared[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _instrument_for_symbol(symbol: str):
    _require_nautilus()
    symbol = symbol.upper()
    if symbol == "BTCUSDT":
        return TestInstrumentProvider.btcusdt_perp_binance()
    if symbol == "ETHUSDT":
        return TestInstrumentProvider.ethusdt_perp_binance()
    raise ValueError(f"Backtest instrument template not implemented for symbol: {symbol}")


def _bars_from_frame(frame: pd.DataFrame, instrument, bar_minutes: int) -> tuple[BarType, list[Bar]]:
    _require_nautilus()
    bar_type = BarType(
        instrument.id,
        BarSpecification(bar_minutes, BarAggregation.MINUTE, PriceType.LAST),
        AggregationSource.EXTERNAL,
    )
    bars: list[Bar] = []
    for row in frame.itertuples(index=False):
        ts_event = dt_to_unix_nanos(row.close_time.to_pydatetime())
        bars.append(
            Bar(
                bar_type=bar_type,
                open=instrument.make_price(float(row.open)),
                high=instrument.make_price(float(row.high)),
                low=instrument.make_price(float(row.low)),
                close=instrument.make_price(float(row.close)),
                volume=instrument.make_qty(float(row.volume)),
                ts_event=ts_event,
                ts_init=ts_event,
            )
        )
    return bar_type, bars


def _action_mix_metrics(action_counts: dict[str, int]) -> dict[str, float]:
    short_count = int(action_counts.get("short", 0))
    flat_count = int(action_counts.get("flat", 0))
    long_count = int(action_counts.get("long", 0))
    total_actions = short_count + flat_count + long_count
    if total_actions <= 0:
        return {
            "flat_ratio": 1.0,
            "position_ratio": 0.0,
            "eval_action_entropy": 0.0,
            "eval_dominant_action": 1.0,
            "eval_dominant_action_ratio": 1.0,
            "eval_short_actions": 0.0,
            "eval_flat_actions": 0.0,
            "eval_long_actions": 0.0,
        }

    probs = np.array([short_count, flat_count, long_count], dtype=np.float64) / float(total_actions)
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(3.0))
    if abs(entropy) < 1e-12:
        entropy = 0.0
    dominant_idx = int(np.argmax(probs))
    dominant_ratio = float(np.max(probs))
    return {
        "flat_ratio": float(flat_count / total_actions),
        "position_ratio": float((short_count + long_count) / total_actions),
        "eval_action_entropy": entropy,
        "eval_dominant_action": float(dominant_idx),
        "eval_dominant_action_ratio": dominant_ratio,
        "eval_short_actions": float(short_count),
        "eval_flat_actions": float(flat_count),
        "eval_long_actions": float(long_count),
    }


def _server_cost_for_bars(
    *,
    monthly_server_cost_usd: float,
    periods_per_day: int,
    n_bars: int,
) -> float:
    if monthly_server_cost_usd <= 0 or periods_per_day <= 0 or n_bars <= 0:
        return 0.0
    return float(monthly_server_cost_usd) * (float(n_bars) / float(periods_per_day * 30))


def _summarize_result(
    *,
    result,
    snapshot: dict,
    cfg: dict,
    frame_15m: pd.DataFrame,
) -> dict:
    stats_pnls = getattr(result, "stats_pnls", {})
    usdt_stats = stats_pnls.get("USDT", {}) if isinstance(stats_pnls, dict) else {}
    total_positions = int(getattr(result, "total_positions", 0) or 0)
    engine_win_rate = float(usdt_stats.get("Win Rate", snapshot["win_rate"]) or 0.0)
    engine_wins = int(round(total_positions * engine_win_rate)) if total_positions > 0 else snapshot["n_wins"]
    engine_losses = max(total_positions - engine_wins, 0) if total_positions > 0 else snapshot["n_losses"]
    total_pnl_usdt = float(usdt_stats.get("PnL (total)", 0.0) or 0.0)
    initial_balance = float(cfg["initial_balance_usdt"])
    server_cost_paid = _server_cost_for_bars(
        monthly_server_cost_usd=float(cfg.get("monthly_server_cost_usd", 0.0)),
        periods_per_day=int(cfg.get("periods_per_day", 96)),
        n_bars=len(frame_15m),
    )
    gross_total_return = total_pnl_usdt / max(initial_balance, 1e-9)
    net_total_return = (total_pnl_usdt - server_cost_paid) / max(initial_balance, 1e-9)
    bh_return = 0.0
    if len(frame_15m) >= 2:
        first_price = float(frame_15m["close"].iloc[0])
        last_price = float(frame_15m["close"].iloc[-1])
        if first_price > 0:
            bh_return = last_price / first_price - 1.0
    action_mix = _action_mix_metrics(snapshot.get("action_counts", {}))
    summary = {
        "run_id": getattr(result, "run_id", ""),
        "total_orders": int(getattr(result, "total_orders", 0) or 0),
        "total_positions": total_positions,
        "stats_pnls": stats_pnls,
        "stats_returns": getattr(result, "stats_returns", {}),
        **snapshot,
        **action_mix,
        "n_trades": max(int(snapshot["n_trades"]), total_positions),
        "n_wins": engine_wins,
        "n_losses": engine_losses,
        "win_rate": engine_win_rate if total_positions > 0 else snapshot["win_rate"],
        "server_cost_paid": float(server_cost_paid),
        "gross_total_return": float(gross_total_return),
        "total_return": float(net_total_return),
        "bh_eval_return": float(bh_return),
        "outperformance_vs_bh": float(net_total_return - bh_return),
        "avg_trades_per_episode": float(max(int(snapshot["n_trades"]), total_positions)),
        "eval_episodes": 1,
        "status": "COMPLETE",
    }
    return summary


def run_backtest_frame(
    frame_15m: pd.DataFrame,
    *,
    symbol: str,
    model_path: str,
    venue: str = "BINANCE",
    bar_minutes: int = 15,
    history_bars: int = 160,
    request_history_days: int = 3,
    trade_size: str = "0.002",
    initial_balance_usdt: float = 10_000.0,
    leverage: float = 1.0,
    state_path: str = "checkpoints/nautilus_dashboard_state.json",
    mode: str = "backtest",
    close_positions_on_stop: bool = True,
    reduce_only_on_stop: bool = True,
    monthly_server_cost_usd: float = 100.0,
    periods_per_day: int = 96,
) -> dict:
    _require_nautilus()
    prepared = _prepare_15m_frame(frame_15m, bar_minutes)
    state = NautilusStateWriter(state_path)
    state.reset(
        status="STARTING",
        mode=mode,
        symbol=symbol,
        instrument_id=f"{symbol.upper()}-PERP.{venue.upper()}",
        model_path=model_path,
        event=f"Loading {len(prepared):,} prepared 15m bars for Nautilus {mode}",
    )

    instrument = _instrument_for_symbol(symbol)
    bar_type, bars = _bars_from_frame(prepared, instrument, bar_minutes)
    strategy = GaricNautilusStrategy(
        GaricNautilusStrategyConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            trade_size=Decimal(str(trade_size)),
            model_path=model_path,
            state_path=state_path,
            history_bars=int(history_bars),
            request_history_days=max(int(request_history_days), 1),
            starting_balance=float(initial_balance_usdt),
            mode=mode,
            close_positions_on_stop=bool(close_positions_on_stop),
            reduce_only_on_stop=bool(reduce_only_on_stop),
        )
    )

    engine = BacktestEngine()
    try:
        engine.add_venue(
            venue=Venue(venue),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(float(initial_balance_usdt), USDT)],
            default_leverage=Decimal(str(leverage)),
            bar_execution=True,
            trade_execution=True,
            use_reduce_only=True,
        )
        engine.add_instrument(instrument)
        engine.add_strategy(strategy)
        engine.add_data(bars)
        state.update(status="RUNNING", total_bars=len(bars), event=f"Starting Nautilus {mode} engine")
        engine.run()
        result = engine.get_result()
        summary = _summarize_result(
            result=result,
            snapshot=strategy.snapshot(),
            cfg={
                "initial_balance_usdt": float(initial_balance_usdt),
                "monthly_server_cost_usd": float(monthly_server_cost_usd),
                "periods_per_day": int(periods_per_day),
            },
            frame_15m=prepared,
        )
        total_pnl_usdt = float(summary["gross_total_return"] * float(initial_balance_usdt))
        ending_equity = float(initial_balance_usdt) + total_pnl_usdt - float(summary["server_cost_paid"])
        state.update(
            status="COMPLETE",
            model_family=summary["model_family"],
            n_trades=summary["n_trades"],
            n_wins=summary["n_wins"],
            n_losses=summary["n_losses"],
            win_rate=summary["win_rate"],
            action_counts=summary["action_counts"],
            total_pnl=total_pnl_usdt,
            unrealized_pnl=0.0,
            equity=ending_equity,
            total_return=summary["total_return"],
            gross_total_return=summary["gross_total_return"],
            server_cost_paid=summary["server_cost_paid"],
            bh_eval_return=summary["bh_eval_return"],
            outperformance_vs_bh=summary["outperformance_vs_bh"],
            flat_ratio=summary["flat_ratio"],
            position_ratio=summary["position_ratio"],
            eval_action_entropy=summary["eval_action_entropy"],
            eval_dominant_action_ratio=summary["eval_dominant_action_ratio"],
            max_drawdown=summary.get("max_drawdown", 0.0),
            backtest=summary,
            event=f"Nautilus {mode} complete",
        )
        logger.info("Nautilus %s summary: %s", mode, summary)
        return summary
    except Exception as exc:
        state.update(status="ERROR", error=str(exc), event=f"Nautilus {mode} failed: {exc}")
        raise
    finally:
        _safe_dispose_engine(engine)


def run_backtest(config_path: str | None = None, limit_bars: int | None = None) -> dict:
    config = load_nautilus_config(config_path)
    cfg = config["nautilus"]
    state = NautilusStateWriter(cfg["state_path"])
    state.reset(
        status="STARTING",
        mode="backtest",
        symbol=cfg["symbol"],
        instrument_id=cfg["instrument_id"],
        model_path=cfg["model_path"],
        event="Loading local parquet data for Nautilus backtest",
    )

    df_1m = pd.read_parquet(cfg["data_path"])
    df_15m = _aggregate_to_15m(df_1m, cfg["bar_minutes"])
    if limit_bars:
        df_15m = df_15m.tail(int(limit_bars)).reset_index(drop=True)

    return run_backtest_frame(
        df_15m,
        symbol=cfg["symbol"],
        model_path=cfg["model_path"],
        venue=cfg["venue"],
        bar_minutes=cfg["bar_minutes"],
        history_bars=cfg["history_bars"],
        request_history_days=cfg["request_history_days"],
        trade_size=cfg["trade_size"],
        initial_balance_usdt=cfg["initial_balance_usdt"],
        leverage=cfg["leverage"],
        state_path=cfg["state_path"],
        mode="backtest",
        close_positions_on_stop=cfg["close_positions_on_stop"],
        reduce_only_on_stop=cfg["reduce_only_on_stop"],
        monthly_server_cost_usd=cfg.get("monthly_server_cost_usd", 100.0),
        periods_per_day=cfg.get("periods_per_day", 96),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="GARIC Nautilus backtest runner")
    parser.add_argument("--config", default="configs/nautilus.yaml")
    parser.add_argument("--limit-bars", type=int, default=None)
    parser.add_argument("--frame-parquet", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--venue", default="BINANCE")
    parser.add_argument("--bar-minutes", type=int, default=15)
    parser.add_argument("--history-bars", type=int, default=160)
    parser.add_argument("--request-history-days", type=int, default=3)
    parser.add_argument("--trade-size", default="0.002")
    parser.add_argument("--initial-balance-usdt", type=float, default=10000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--state-path", default="checkpoints/nautilus_dashboard_state.json")
    parser.add_argument("--mode", default="backtest")
    parser.add_argument("--monthly-server-cost-usd", type=float, default=100.0)
    parser.add_argument("--periods-per-day", type=int, default=96)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("nautilus.log", mode="w", encoding="utf-8"),
        ],
    )
    if args.frame_parquet:
        frame = pd.read_parquet(args.frame_parquet)
        summary = run_backtest_frame(
            frame,
            symbol=args.symbol,
            model_path=str(args.model_path),
            venue=args.venue,
            bar_minutes=int(args.bar_minutes),
            history_bars=int(args.history_bars),
            request_history_days=int(args.request_history_days),
            trade_size=str(args.trade_size),
            initial_balance_usdt=float(args.initial_balance_usdt),
            leverage=float(args.leverage),
            state_path=str(args.state_path),
            mode=str(args.mode),
            monthly_server_cost_usd=float(args.monthly_server_cost_usd),
            periods_per_day=int(args.periods_per_day),
        )
        if args.summary_json:
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    else:
        run_backtest(args.config, limit_bars=args.limit_bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
