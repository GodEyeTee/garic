"""Real-time browser dashboard for GARIC PPO training.

Usage:
  streamlit run monitoring/training/dashboard.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


STATE_PATH = Path("checkpoints/training_dashboard_state.json")


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}

    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_pct(value: float) -> str:
    return f"{value:+.2%}"


def _fmt_money(value: float, signed: bool = True) -> str:
    sign = "+" if signed and value > 0 else ""
    return f"{sign}${value:,.2f}"


def _fmt_drawdown(value: float) -> str:
    return f"-{abs(value):.2%}"


def _clean_markup(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text).strip()


def _metric_card(label: str, value: str, tone: str = "neutral", detail: str = "") -> str:
    detail_html = f"<div class='metric-detail'>{detail}</div>" if detail else ""
    return f"""
    <div class="metric-card tone-{tone}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {detail_html}
    </div>
    """


def _progress_block(step: int, total: int, fps: float, eta_sec: float, updated_at: float) -> str:
    total = max(int(total), 1)
    pct = min(max(step / total, 0.0), 1.0)
    updated = datetime.fromtimestamp(updated_at).strftime("%H:%M:%S") if updated_at else "--:--:--"
    return f"""
    <div class="glass-card">
      <div class="section-label">Training Progress</div>
      <div class="progress-meta">
        <span>{step:,} / {total:,} steps</span>
        <span>{pct:.1%}</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="width:{pct * 100:.2f}%"></div>
      </div>
      <div class="progress-grid">
        <div><span>FPS</span><strong>{fps:.0f}</strong></div>
        <div><span>ETA</span><strong>{eta_sec:.0f}s</strong></div>
        <div><span>Last Update</span><strong>{updated}</strong></div>
      </div>
    </div>
    """


def _normalize_series(values, default=0.0) -> list[float]:
    if not values:
        return [default]
    return [float(v) for v in values]


def _x_axis(series_len: int, step_history: list[int]) -> list[int]:
    if len(step_history) == series_len and series_len > 0:
        return [int(v) for v in step_history]
    return list(range(series_len))


def _line_area_figure(go, x_values, y_values, title: str, positive_color: str, negative_color: str, y_title: str):
    positive = [v if v >= 0 else None for v in y_values]
    negative = [v if v <= 0 else None for v in y_values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=positive,
            mode="lines",
            line=dict(color=positive_color, width=3),
            fill="tozeroy",
            fillcolor="rgba(64, 196, 160, 0.16)",
            name=title,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=negative,
            mode="lines",
            line=dict(color=negative_color, width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.14)",
            name=title,
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.18)")
    fig.update_layout(
        title=title,
        height=280,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(title="Step", showgrid=False, zeroline=False),
        yaxis=dict(title=y_title, showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        font=dict(color="#eaf2ff"),
    )
    return fig


def _training_quality_figure(go, loss_values, entropy_values):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(loss_values))),
            y=loss_values,
            mode="lines",
            line=dict(color="#ff5a7a", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 90, 122, 0.10)",
            name="Loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(entropy_values))),
            y=entropy_values,
            mode="lines",
            line=dict(color="#f7c94b", width=3),
            name="Entropy",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Training Quality",
        height=300,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Update", showgrid=False),
        yaxis=dict(title="Loss", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(title="Entropy", overlaying="y", side="right", showgrid=False),
        font=dict(color="#eaf2ff"),
    )
    return fig


def _trade_quality_figure(go, make_subplots, step_history, trade_history, win_rate_history):
    x_trade = _x_axis(len(trade_history), step_history[-len(trade_history):] if step_history else [])
    x_win = _x_axis(len(win_rate_history), step_history[-len(win_rate_history):] if step_history else [])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=x_trade,
            y=trade_history,
            mode="lines",
            line=dict(color="#45c0ff", width=3),
            fill="tozeroy",
            fillcolor="rgba(69, 192, 255, 0.10)",
            name="Trades",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_win,
            y=[v * 100.0 for v in win_rate_history],
            mode="lines",
            line=dict(color="#40d89a", width=3),
            name="Win Rate",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Trade Activity",
        height=300,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Step", showgrid=False),
        font=dict(color="#eaf2ff"),
    )
    fig.update_yaxes(title_text="Trades", secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Win Rate %", secondary_y=True, showgrid=False)
    return fig


def _distribution_figure(go, state: dict):
    eval_values = [
        float(state.get("eval_short_actions", 0.0)),
        float(state.get("eval_flat_actions", 0.0)),
        float(state.get("eval_long_actions", 0.0)),
    ]
    if sum(eval_values) > 0:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Short", "Flat", "Long"],
                    values=eval_values,
                    hole=0.62,
                    marker=dict(colors=["#ff6b6b", "#f3c84c", "#4de2ae"]),
                    textinfo="label+percent",
                )
            ]
        )
        fig.update_layout(
            title="Eval Action Mix",
            height=300,
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eaf2ff"),
            showlegend=False,
        )
        return fig

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Longs", "Shorts", "Wins", "Losses"],
                y=[
                    float(state.get("n_longs", 0.0)),
                    float(state.get("n_shorts", 0.0)),
                    float(state.get("n_wins", 0.0)),
                    float(state.get("n_losses", 0.0)),
                ],
                marker=dict(color=["#4de2ae", "#ff6b6b", "#40d89a", "#ff8b5f"]),
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Trade Outcome Mix",
        height=300,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaf2ff"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    return fig


def _return_stack_figure(go, state: dict):
    labels = ["Net", "Gross", "B&H", "Alpha"]
    values = [
        float(state.get("rl_return", 0.0)) * 100.0,
        float(state.get("gross_return", 0.0)) * 100.0,
        float(state.get("bh_return", 0.0)) * 100.0,
        float(state.get("alpha_vs_bh", 0.0)) * 100.0,
    ]
    colors = ["#45c0ff", "#4de2ae", "#a78bfa", "#ff8b5f"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors),
                text=[f"{v:+.2f}%" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.18)")
    fig.update_layout(
        title="Return Stack",
        height=300,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaf2ff"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Percent", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    return fig


def _summary_card(state: dict) -> str:
    collapsed = (
        float(state.get("selection_gate_passed", 1.0)) <= 0.0
        or float(state.get("eval_dominant_action_ratio", 0.0)) >= 0.95
    )
    inactive = float(state.get("flat_ratio", 0.0)) >= 0.95 or float(state.get("n_trades", 0.0)) <= 0
    badge_tone = "negative" if collapsed or inactive else "positive"
    status = "COLLAPSED" if collapsed else "INACTIVE" if inactive else "ACTIVE"
    return f"""
    <div class="glass-card">
      <div class="section-row">
        <div class="section-label">Performance Snapshot</div>
        <span class="pill pill-{badge_tone}">{status}</span>
      </div>
      <div class="mini-grid">
        <div><span>Sharpe</span><strong>{float(state.get('sharpe', 0.0)):.3f}</strong></div>
        <div><span>Sortino</span><strong>{float(state.get('sortino', 0.0)):.3f}</strong></div>
        <div><span>Max DD</span><strong>{_fmt_drawdown(float(state.get('max_dd', 0.0)))}</strong></div>
        <div><span>Flat Ratio</span><strong>{float(state.get('flat_ratio', 0.0)):.1%}</strong></div>
        <div><span>Position Ratio</span><strong>{float(state.get('position_ratio', 0.0)):.1%}</strong></div>
        <div><span>Action Entropy</span><strong>{float(state.get('eval_action_entropy', 0.0)):.3f}</strong></div>
        <div><span>Wrong-Side</span><strong>{float(state.get('wrong_side_moves', 0.0)):.3f}</strong></div>
        <div><span>Avg Reward / Ep</span><strong>{float(state.get('avg_reward_sum', 0.0)):+.2f}</strong></div>
      </div>
    </div>
    """


def _phase_card(state: dict) -> str:
    phases = state.get("phases_done", [])
    if not phases:
        items = "<li>Waiting for pipeline phases...</li>"
    else:
        items = "".join(f"<li>{_clean_markup(item)}</li>" for item in phases[-8:])

    model_path = state.get("model_path", "") or "training..."
    chart_path = state.get("chart_path", "") or "pending"
    return f"""
    <div class="glass-card">
      <div class="section-label">Pipeline Status</div>
      <ul class="phase-list">{items}</ul>
      <div class="footer-grid">
        <div><span>Model</span><strong>{model_path}</strong></div>
        <div><span>Chart</span><strong>{chart_path}</strong></div>
        <div><span>Log</span><strong>pipeline.log</strong></div>
        <div><span>Eval Episodes</span><strong>{float(state.get('eval_episodes', 0.0)):.0f}</strong></div>
        <div><span>Gate Passed</span><strong>{'YES' if float(state.get('selection_gate_passed', 1.0)) > 0 else 'NO'}</strong></div>
        <div><span>Best Score</span><strong>{float(state.get('selection_best_score', 0.0)):.3f}</strong></div>
      </div>
    </div>
    """


def run_dashboard():
    try:
        import plotly.graph_objects as go
        import streamlit as st
        from plotly.subplots import make_subplots
    except ImportError:
        print("streamlit/plotly not installed")
        return

    st.set_page_config(page_title="GARIC Training Dashboard", page_icon="G", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(62, 199, 255, 0.15), transparent 30%),
                radial-gradient(circle at 90% 10%, rgba(167, 139, 250, 0.15), transparent 30%),
                radial-gradient(circle at 50% 90%, rgba(77, 226, 174, 0.10), transparent 40%),
                linear-gradient(180deg, #040910 0%, #060d16 55%, #03060a 100%);
            color: #eef5ff;
        }
        .block-container {
            max-width: 1680px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            font-family: 'Inter', system-ui, sans-serif;
        }
        .hero-card,
        .glass-card,
        div[data-testid="stPlotlyChart"] {
            background: linear-gradient(145deg, rgba(16, 26, 38, 0.8) 0%, rgba(10, 16, 24, 0.9) 100%);
            border: 1px solid rgba(112, 230, 255, 0.2);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .hero-card:hover, .glass-card:hover {
            box-shadow: 0 12px 48px rgba(62, 199, 255, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        }
        div[data-testid="stPlotlyChart"] {
            padding: 12px 12px 0 12px;
        }
        .hero-card { padding: 28px 32px; }
        .glass-card { padding: 22px 26px; }
        .eyebrow {
            color: #3ec7ff;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
            text-shadow: 0 0 12px rgba(62, 199, 255, 0.4);
        }
        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.1;
            background: linear-gradient(90deg, #ffffff 0%, #a2cbeb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.6rem;
        }
        .hero-sub {
            color: #94b0c3;
            font-size: 1.1rem;
            line-height: 1.5;
            font-weight: 400;
        }
        .hero-sub strong {
            color: #eef5ff;
            font-weight: 600;
        }
        .hero-row,
        .progress-meta,
        .section-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }
        .badge-row {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 18px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            border: 1px solid rgba(255, 255, 255, 0.10);
            background: rgba(255, 255, 255, 0.05);
            color: #eef5ff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            text-transform: uppercase;
        }
        .pill-positive { border-color: rgba(77, 226, 174, 0.5); color: #4de2ae; background: rgba(77, 226, 174, 0.1); box-shadow: 0 0 16px rgba(77, 226, 174, 0.2); }
        .pill-negative { border-color: rgba(255, 107, 107, 0.5); color: #ff6b6b; background: rgba(255, 107, 107, 0.1); box-shadow: 0 0 16px rgba(255, 107, 107, 0.2); }
        .pill-warning { border-color: rgba(243, 200, 76, 0.5); color: #f3c84c; background: rgba(243, 200, 76, 0.1); box-shadow: 0 0 16px rgba(243, 200, 76, 0.2); }
        .metric-card {
            min-height: 130px;
            padding: 20px 22px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(16, 26, 38, 0.7) 0%, rgba(8, 14, 22, 0.9) 100%);
            border: 1px solid rgba(112, 230, 255, 0.15);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: rgba(255,255,255,0.1);
        }
        .tone-positive::before { background: linear-gradient(90deg, #4de2ae 0%, transparent 100%); }
        .tone-negative::before { background: linear-gradient(90deg, #ff6b6b 0%, transparent 100%); }
        .tone-warning::before { background: linear-gradient(90deg, #f3c84c 0%, transparent 100%); }
        
        .metric-label {
            color: #7f9db1;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .metric-value {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.02em;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }
        .metric-detail {
            color: #8eaec1;
            font-size: 0.95rem;
            margin-top: 0.8rem;
            font-weight: 500;
        }
        .section-label {
            color: #7f9db1;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .progress-track {
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            overflow: hidden;
            margin-top: 16px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
        }
        .progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #3ec7ff 0%, #4de2ae 100%);
            box-shadow: 0 0 20px rgba(77, 226, 174, 0.5);
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .progress-grid,
        .mini-grid,
        .footer-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-top: 18px;
        }
        .footer-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .progress-grid div,
        .mini-grid div,
        .footer-grid div {
            border-radius: 12px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            transition: background 0.2s ease;
        }
        .progress-grid div:hover,
        .mini-grid div:hover,
        .footer-grid div:hover {
            background: rgba(255,255,255,0.05);
        }
        .progress-grid span,
        .mini-grid span,
        .footer-grid span {
            display: block;
            color: #7f9db1;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .progress-grid strong,
        .mini-grid strong,
        .footer-grid strong {
            color: #f3f9ff;
            font-size: 1.05rem;
            font-weight: 700;
            word-break: break-word;
        }
        .phase-list {
            margin: 0;
            padding-left: 20px;
            color: #d7e4f0;
            line-height: 1.8;
            font-size: 0.95rem;
        }
        .phase-list li {
            margin-bottom: 4px;
        }
        .phase-list li::marker {
            color: #3ec7ff;
        }
        @media (max-width: 980px) {
            .hero-title { font-size: 2.2rem; }
            .progress-grid,
            .mini-grid,
            .footer-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    fragment_fn = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)

    def render_live_view():
        state = _load_state()
        updated_at = float(state.get("updated_at", 0.0))

        step = int(state.get("train_step", 0))
        total = max(int(state.get("train_total", 0)), 1)
        fps = float(state.get("train_fps", 0.0))
        eta_sec = max(total - step, 0) / max(fps, 1.0)
        current_pos = float(state.get("current_pos", 0.0))
        current_pnl = float(state.get("current_pnl", 0.0))
        current_action = str(state.get("current_action", "Flat"))

        action_tone = "warning"
        if current_action == "Long":
            action_tone = "positive"
        elif current_action == "Short":
            action_tone = "negative"

        st.markdown(
            f"""
            <div class="hero-card">
              <div class="hero-row">
                <div>
                  <div class="eyebrow">Real-Time Browser Dashboard</div>
                  <div class="hero-title">GARIC Trading Lab</div>
                  <div class="hero-sub">
                    {state.get('status_msg', 'Waiting for live training state...')}<br/>
                    Symbol <strong>{state.get('symbol', '--')}</strong> |
                    15m candles <strong>{int(state.get('data_15m', 0)):,}</strong> |
                    Features <strong>{int(state.get('features', 0)):,}</strong> |
                    GPU <strong>{state.get('gpu', '--')}</strong>
                  </div>
                </div>
                <div class="badge-row">
                  <span class="pill pill-{action_tone}">Action {current_action}</span>
                  <span class="pill pill-warning">Position {current_pos:+.2f}</span>
                  <span class="pill {'pill-positive' if current_pnl >= 0 else 'pill-negative'}">UPnL {_fmt_pct(current_pnl)}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        cols[0].markdown(
            _metric_card(
                "Net Return",
                _fmt_pct(float(state.get("rl_return", 0.0))),
                "positive" if float(state.get("rl_return", 0.0)) >= 0 else "negative",
                f"B&H eval {_fmt_pct(float(state.get('bh_return', 0.0)))}",
            ),
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            _metric_card(
                "Alpha vs B&H",
                _fmt_pct(float(state.get("alpha_vs_bh", 0.0))),
                "positive" if float(state.get("alpha_vs_bh", 0.0)) >= 0 else "negative",
                f"Gross {_fmt_pct(float(state.get('gross_return', 0.0)))}",
            ),
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            _metric_card(
                "Current Position",
                f"{current_pos:+.2f}",
                "positive" if current_pos > 0 else "negative" if current_pos < 0 else "warning",
                f"Action {current_action}",
            ),
            unsafe_allow_html=True,
        )
        cols[3].markdown(
            _metric_card(
                "Current UPnL",
                _fmt_pct(current_pnl),
                "positive" if current_pnl >= 0 else "negative",
                f"Server cost {_fmt_money(-float(state.get('total_server_cost_paid', 0.0)))}",
            ),
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        cols[0].markdown(
            _metric_card(
                "Total Trades",
                f"{float(state.get('n_trades', 0.0)):.0f}",
                "warning",
                f"Avg/ep {float(state.get('avg_trades_per_episode', 0.0)):.1f}",
            ),
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            _metric_card(
                "Win Rate",
                f"{float(state.get('win_rate', 0.0)):.1%}",
                "positive" if float(state.get("win_rate", 0.0)) >= 0.5 else "warning",
                f"W {float(state.get('n_wins', 0.0)):.0f} / L {float(state.get('n_losses', 0.0)):.0f}",
            ),
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            _metric_card(
                "Flat Ratio",
                f"{float(state.get('flat_ratio', 0.0)):.1%}",
                "negative" if float(state.get("flat_ratio", 0.0)) >= 0.8 else "warning",
                f"Position ratio {float(state.get('position_ratio', 0.0)):.1%}",
            ),
            unsafe_allow_html=True,
        )
        cols[3].markdown(
            _metric_card(
                "Speed",
                f"{fps:.0f} fps",
                "warning",
                f"Step {step:,}/{total:,}",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            _progress_block(step, total, fps, eta_sec, updated_at),
            unsafe_allow_html=True,
        )

        step_history = [int(v) for v in state.get("step_history", [])]
        position_history = _normalize_series(state.get("position_history", []), default=0.0)
        pnl_history = _normalize_series(state.get("pnl_history", []), default=0.0)
        trade_history = _normalize_series(state.get("trade_history", []), default=0.0)
        win_rate_history = _normalize_series(state.get("win_rate_history", []), default=0.0)
        loss_history = _normalize_series(state.get("loss_history", []), default=0.0)
        entropy_history = _normalize_series(state.get("entropy_history", []), default=0.0)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _line_area_figure(
                    go,
                    _x_axis(len(position_history), step_history[-len(position_history):] if step_history else []),
                    position_history,
                    "Active Position",
                    "#4de2ae",
                    "#ff6b6b",
                    "Position",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key="chart-active-position",
            )
        with right:
            st.plotly_chart(
                _line_area_figure(
                    go,
                    _x_axis(len(pnl_history), step_history[-len(pnl_history):] if step_history else []),
                    pnl_history,
                    "Unrealized PnL",
                    "#45c0ff",
                    "#ff8b5f",
                    "PnL",
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key="chart-unrealized-pnl",
            )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _training_quality_figure(go, loss_history, entropy_history),
                use_container_width=True,
                config={"displayModeBar": False},
                key="chart-training-quality",
            )
        with right:
            st.plotly_chart(
                _trade_quality_figure(go, make_subplots, step_history, trade_history, win_rate_history),
                use_container_width=True,
                config={"displayModeBar": False},
                key="chart-trade-activity",
            )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _distribution_figure(go, state),
                use_container_width=True,
                config={"displayModeBar": False},
                key="chart-distribution",
            )
        with right:
            st.plotly_chart(
                _return_stack_figure(go, state),
                use_container_width=True,
                config={"displayModeBar": False},
                key="chart-return-stack",
            )

        left, right = st.columns(2)
        with left:
            st.markdown(_summary_card(state), unsafe_allow_html=True)
        with right:
            st.markdown(_phase_card(state), unsafe_allow_html=True)

    if fragment_fn is None:
        st.warning("Installed Streamlit version does not support fragments. Live refresh is disabled.")
        render_live_view()
    else:
        fragment_fn(run_every=1.0)(render_live_view)()


if __name__ == "__main__":
    run_dashboard()
