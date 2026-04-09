"""Nautilus live/paper runner for GARIC."""

from __future__ import annotations

import argparse
import logging
from decimal import Decimal

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.binance.config import BinanceExecClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance.factories import BinanceLiveExecClientFactory
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live.config import LiveExecEngineConfig
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId

from execution.nautilus.config import load_live_credentials
from execution.nautilus.config import load_nautilus_config
from execution.nautilus.state import NautilusStateWriter
from execution.nautilus.strategy import GaricNautilusStrategy
from execution.nautilus.strategy import GaricNautilusStrategyConfig


logger = logging.getLogger(__name__)


def run_live(mode: str, config_path: str | None = None) -> None:
    config = load_nautilus_config(config_path)
    cfg = config["nautilus"]
    state = NautilusStateWriter(cfg["state_path"])
    instrument_id = InstrumentId.from_str(cfg["instrument_id"])
    bar_type = BarType(
        instrument_id,
        BarSpecification(int(cfg["bar_minutes"]), BarAggregation.MINUTE, PriceType.LAST),
        AggregationSource.EXTERNAL,
    )

    provider = InstrumentProviderConfig(load_ids=(str(instrument_id),))
    data_cfg = BinanceDataClientConfig(
        instrument_provider=provider,
        account_type=BinanceAccountType.USDT_FUTURES,
        environment=BinanceEnvironment[cfg["environment"]],
    )

    data_clients = {"BINANCE": data_cfg}
    exec_clients = {}

    api_key, api_secret = load_live_credentials(config)
    if mode == "paper":
        exec_clients["BINANCE"] = SandboxExecutionClientConfig(
            venue=cfg["venue"],
            starting_balances=[f"{float(cfg['initial_balance_usdt']):.2f} USDT"],
            base_currency="USDT",
            oms_type="NETTING",
            account_type="MARGIN",
            default_leverage=Decimal(str(cfg["leverage"])),
            use_reduce_only=True,
            bar_execution=True,
            trade_execution=True,
        )
    else:
        if not api_key or not api_secret:
            raise RuntimeError(
                f"Live mode requires env vars {cfg['api_key_env']} and {cfg['api_secret_env']}"
            )
        data_clients["BINANCE"] = BinanceDataClientConfig(
            instrument_provider=provider,
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment[cfg["environment"]],
            api_key=api_key,
            api_secret=api_secret,
        )
        exec_clients["BINANCE"] = BinanceExecClientConfig(
            instrument_provider=provider,
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment[cfg["environment"]],
            api_key=api_key,
            api_secret=api_secret,
        )

    state.reset(
        status="STARTING",
        mode=mode,
        symbol=cfg["symbol"],
        instrument_id=str(instrument_id),
        model_path=cfg["model_path"],
        event=f"Starting Nautilus {mode} node",
    )

    exec_engine_cfg = (
        LiveExecEngineConfig(reconciliation=False, reconciliation_startup_delay_secs=0.0)
        if mode == "paper"
        else LiveExecEngineConfig()
    )
    node = TradingNode(
        TradingNodeConfig(
            trader_id=f"GARIC-{mode.upper()}-001",
            data_clients=data_clients,
            exec_clients=exec_clients,
            exec_engine=exec_engine_cfg,
        )
    )
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    if mode == "paper":
        node.add_exec_client_factory("BINANCE", SandboxLiveExecClientFactory)
    else:
        node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)

    strategy = GaricNautilusStrategy(
        GaricNautilusStrategyConfig(
            instrument_id=instrument_id,
            bar_type=bar_type,
            trade_size=Decimal(str(cfg["trade_size"])),
            model_path=cfg["model_path"],
            state_path=cfg["state_path"],
            history_bars=int(cfg["history_bars"]),
            request_history_days=max(int(cfg["request_history_days"]), 1),
            starting_balance=float(cfg["initial_balance_usdt"]),
            mode=mode,
            close_positions_on_stop=bool(cfg["close_positions_on_stop"]),
            reduce_only_on_stop=bool(cfg["reduce_only_on_stop"]),
            monthly_server_cost_usd=float(cfg.get("monthly_server_cost_usd", 0.0)),
            periods_per_day=max(int(cfg.get("periods_per_day", 96)), 1),
        )
    )
    node.trader.add_strategy(strategy)
    node.build()
    node.run()


def main() -> int:
    parser = argparse.ArgumentParser(description="GARIC Nautilus live/paper runner")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--config", default="configs/nautilus.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("nautilus.log", mode="w", encoding="utf-8"),
        ],
    )
    run_live(args.mode, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
