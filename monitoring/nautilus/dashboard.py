"""Realtime browser dashboard for GARIC Nautilus mode."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


STATE_PATH = Path("checkpoints/nautilus_dashboard_state.json")


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_pct(value: float) -> str:
    return f"{value:+.2%}"


def _fmt_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.2f}"


def _metric_card(label: str, value: str, detail: str = "", tone: str = "neutral") -> str:
    detail_html = f"<div class='detail'>{detail}</div>" if detail else ""
    return f"""
    <div class="metric tone-{tone}">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {detail_html}
    </div>
    """


def _bootstrap():
    import streamlit as st

    st.set_page_config(page_title="GARIC Nautilus", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #07111d 0%, #0b1726 100%); color: #e8f0ff; }
        .hero { display:flex; justify-content:space-between; gap:16px; padding:20px 24px; border:1px solid rgba(105,196,255,.22);
                border-radius:24px; background:rgba(7,17,29,.72); box-shadow:0 18px 60px rgba(0,0,0,.32); }
        .hero h1 { margin:0; font-size:2rem; }
        .hero .meta { color:#87a1bc; font-size:.95rem; margin-top:6px; }
        .pill { border:1px solid rgba(253, 201, 76, .35); border-radius:999px; padding:8px 14px; font-size:.86rem; }
        .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
        .metric { border:1px solid rgba(255,255,255,.08); border-radius:18px; background:rgba(255,255,255,.025); padding:16px 18px; }
        .metric .label { color:#7f96ad; font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; }
        .metric .value { color:#f3f7ff; font-size:1.7rem; font-weight:700; margin-top:8px; }
        .metric .detail { color:#8ca4bc; font-size:.92rem; margin-top:6px; }
        .tone-good { border-color: rgba(64, 216, 154, .30); }
        .tone-warn { border-color: rgba(247, 201, 75, .30); }
        .tone-bad { border-color: rgba(255, 106, 127, .30); }
        .glass { border:1px solid rgba(255,255,255,.08); border-radius:22px; background:rgba(255,255,255,.025); padding:14px 16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_dashboard():
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    _bootstrap()

    @st.fragment(run_every=1.0)
    def _render():
        state = _load_state()
        if not state:
            st.warning("No Nautilus state yet. Run `python run_nautilus_browser.py --mode backtest` or `--mode paper` first.")
            return

        updated_at = state.get("updated_at", 0.0)
        updated = datetime.fromtimestamp(updated_at).strftime("%H:%M:%S") if updated_at else "--:--:--"
        status = str(state.get("status", "IDLE")).upper()
        mode = str(state.get("mode", "backtest")).upper()
        symbol = state.get("symbol", "BTCUSDT")
        instrument_id = state.get("instrument_id", "")
        model_path = state.get("model_path", "")

        st.markdown(
            f"""
            <div class="hero">
              <div>
                <div style="color:#6ec5ff; letter-spacing:.12em; font-size:.82rem;">NAUTILUS EXECUTION</div>
                <h1>GARIC Nautilus Desk</h1>
                <div class="meta">{symbol} | {instrument_id} | Model: {model_path or "pending"} | Updated {updated}</div>
              </div>
              <div style="display:flex; gap:10px; align-items:flex-start;">
                <div class="pill">Mode {mode}</div>
                <div class="pill">Status {status}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        equity = float(state.get("equity", state.get("starting_balance", 0.0)))
        total_pnl = float(state.get("total_pnl", 0.0))
        total_return = float(state.get("total_return", 0.0))
        position = float(state.get("position", 0.0))
        confidence = float(state.get("confidence", 0.0))
        trades = int(state.get("n_trades", 0))
        win_rate = float(state.get("win_rate", 0.0))
        recent_price = float(state.get("recent_price", 0.0))
        upnl = float(state.get("unrealized_pnl", 0.0))

        tone_return = "good" if total_return > 0 else "bad" if total_return < 0 else "warn"
        tone_pnl = "good" if total_pnl > 0 else "bad" if total_pnl < 0 else "warn"
        tone_pos = "good" if position != 0 else "warn"

        st.markdown(
            "<div class='metrics'>"
            + _metric_card("Equity Estimate", _fmt_money(equity), detail=f"Last price {recent_price:,.2f}", tone=tone_pnl)
            + _metric_card("Net PnL", _fmt_money(total_pnl), detail=f"UPnL {_fmt_money(upnl)}", tone=tone_pnl)
            + _metric_card("Total Return", _fmt_pct(total_return), detail=f"Confidence {confidence:.1%}", tone=tone_return)
            + _metric_card("Position", f"{position:+.2f}", detail=f"Action {state.get('action', 'Flat')}", tone=tone_pos)
            + _metric_card("Trades", f"{trades}", detail=f"W/L {state.get('n_wins', 0)}/{state.get('n_losses', 0)}", tone="neutral")
            + _metric_card("Win Rate", f"{win_rate:.1%}", detail=f"Model {state.get('model_family', '-')}", tone="neutral")
            + _metric_card("Warmup", f"{int(state.get('warmup_progress', 0))}", detail=f"Target {int(state.get('warmup_target', 0))}", tone="neutral")
            + _metric_card("Backtest Orders", f"{int(state.get('backtest', {}).get('total_orders', 0))}", detail=f"Positions {int(state.get('backtest', {}).get('total_positions', 0))}", tone="neutral")
            + "</div>",
            unsafe_allow_html=True,
        )

        history = state.get("history", {})
        ts = history.get("ts", [])
        if ts:
            df = pd.DataFrame(
                {
                    "ts": pd.to_datetime(ts, unit="ns", utc=True),
                    "price": history.get("price", []),
                    "equity": history.get("equity", []),
                    "position": history.get("position", []),
                    "upnl": history.get("upnl", []),
                }
            )
            col1, col2 = st.columns(2)
            with col1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=df["ts"],
                        y=df["price"],
                        mode="lines",
                        name="Price",
                        line=dict(color="#4cc9f0", width=3),
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["ts"],
                        y=df["position"],
                        mode="lines",
                        name="Position",
                        line=dict(color="#ff9f1c", width=2),
                    ),
                    secondary_y=True,
                )
                fig.update_layout(
                    title="Price And Position",
                    height=320,
                    margin=dict(l=10, r=10, t=42, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8f0ff"),
                )
                fig.update_yaxes(title_text="Price", secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.06)")
                fig.update_yaxes(title_text="Position", secondary_y=True, showgrid=False)
                st.plotly_chart(fig, use_container_width=True, key="nautilus-price-position")

            with col2:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["ts"],
                        y=df["equity"],
                        mode="lines",
                        name="Equity",
                        line=dict(color="#52d7a5", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(82,215,165,0.12)",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["ts"],
                        y=df["upnl"],
                        mode="lines",
                        name="UPnL",
                        line=dict(color="#f87171", width=2),
                        yaxis="y2",
                    )
                )
                fig.update_layout(
                    title="Equity And UPnL",
                    height=320,
                    margin=dict(l=10, r=10, t=42, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8f0ff"),
                    yaxis2=dict(overlaying="y", side="right"),
                )
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
                st.plotly_chart(fig, use_container_width=True, key="nautilus-equity-upnl")

        col3, col4 = st.columns(2)
        with col3:
            counts = state.get("action_counts", {"short": 0, "flat": 0, "long": 0})
            pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Short", "Flat", "Long"],
                        values=[counts.get("short", 0), counts.get("flat", 0), counts.get("long", 0)],
                        hole=0.68,
                        marker=dict(colors=["#f97373", "#f7c948", "#40d89a"]),
                    )
                ]
            )
            pie.update_layout(
                title="Action Mix",
                height=320,
                margin=dict(l=10, r=10, t=42, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8f0ff"),
            )
            st.plotly_chart(pie, use_container_width=True, key="nautilus-action-mix")

        with col4:
            probs = state.get("probabilities", {"short": 0.0, "flat": 0.0, "long": 0.0})
            bar = go.Figure(
                data=[
                    go.Bar(
                        x=["Short", "Flat", "Long"],
                        y=[float(probs.get("short", 0.0)), float(probs.get("flat", 0.0)), float(probs.get("long", 0.0))],
                        marker_color=["#f97373", "#f7c948", "#40d89a"],
                    )
                ]
            )
            bar.update_layout(
                title="Current Model Probabilities",
                height=320,
                margin=dict(l=10, r=10, t=42, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8f0ff"),
                yaxis=dict(tickformat=".0%", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(bar, use_container_width=True, key="nautilus-probabilities")

        details_col, events_col = st.columns(2)
        with details_col:
            st.markdown("<div class='glass'><h4>Run Details</h4></div>", unsafe_allow_html=True)
            st.json(
                {
                    "mode": state.get("mode"),
                    "status": state.get("status"),
                    "symbol": state.get("symbol"),
                    "instrument_id": state.get("instrument_id"),
                    "model_family": state.get("model_family"),
                    "model_path": state.get("model_path"),
                    "backtest": state.get("backtest", {}),
                },
                expanded=False,
            )

        with events_col:
            st.markdown("<div class='glass'><h4>Recent Events</h4></div>", unsafe_allow_html=True)
            events = state.get("recent_events", [])
            if events:
                event_df = pd.DataFrame(events)
                event_df["ts"] = pd.to_datetime(event_df["ts"], unit="s")
                st.dataframe(event_df.sort_values("ts", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("No events yet.")

    _render()


if __name__ == "__main__":
    run_dashboard()

