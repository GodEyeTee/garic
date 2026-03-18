"""Live trading dashboard — Streamlit-based.

แสดง: PnL, Sharpe, drawdown, position, agent confidence.
ใช้สำหรับ development. Production ใช้ Grafana.

Usage:
  streamlit run monitoring/live/dashboard.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from performance import format_drawdown_pct, summarize_equity_curve

logger = logging.getLogger(__name__)


class DashboardData:
    """Collect and serve data สำหรับ dashboard."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._records: list[dict] = []

    def add_record(
        self,
        timestamp: int,
        balance: float,
        position: float,
        pnl: float,
        action: float,
        confidence: float,
        symbol: str = "",
    ):
        self._records.append({
            "timestamp": timestamp,
            "balance": balance,
            "position": position,
            "pnl": pnl,
            "action": action,
            "confidence": confidence,
            "symbol": symbol,
        })
        if len(self._records) > self.max_history:
            self._records = self._records[-self.max_history:]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_metrics(self) -> dict:
        if len(self._records) < 2:
            return {}

        balances = [r["balance"] for r in self._records]
        equity = np.array(balances, dtype=np.float64)
        summary = summarize_equity_curve(equity)
        returns = np.diff(equity) / np.where(np.abs(equity[:-1]) > 1e-12, equity[:-1], 1.0)
        returns = returns[np.isfinite(returns)]

        return {
            "sharpe": float(summary["sharpe"]),
            "total_pnl": float(equity[-1] - equity[0]),
            "total_return_pct": float(summary["total_return"] * 100),
            "max_drawdown_pct": float(summary["max_drawdown"] * 100),
            "win_rate": float((returns > 0).mean() * 100) if len(returns) > 0 else 0.0,
            "n_records": len(self._records),
            "current_position": self._records[-1]["position"],
            "current_balance": self._records[-1]["balance"],
        }

    def save(self, path: str = "data/dashboard_history.parquet"):
        df = self.to_dataframe()
        if not df.empty:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)

    def load(self, path: str = "data/dashboard_history.parquet"):
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p)
            self._records = df.to_dict("records")


def run_streamlit_dashboard():
    """Launch Streamlit dashboard. เรียกด้วย: streamlit run monitoring/live/dashboard.py"""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        logger.error("streamlit/plotly not installed: pip install streamlit plotly")
        return

    st.set_page_config(page_title="GARIC Dashboard", layout="wide")
    st.title("GARIC — Live Trading Dashboard")

    # Load saved data
    data = DashboardData()
    data.load()
    df = data.to_dataframe()

    if df.empty:
        st.warning("No trading data yet. Start paper/live trading first.")
        return

    metrics = data.get_metrics()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Balance", f"${metrics.get('current_balance', 0):,.2f}")
    col2.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
    col3.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
    col4.metric("Max Drawdown", format_drawdown_pct(metrics.get("max_drawdown_pct", 0) / 100))

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["balance"], name="Equity"))
    fig.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Balance ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Position history
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["timestamp"], y=df["position"], name="Position"))
    fig2.update_layout(title="Position History", yaxis_title="Position Ratio")
    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    run_streamlit_dashboard()
