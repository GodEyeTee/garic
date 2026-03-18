"""GARIC dashboard state manager with optional Rich terminal view.

Update in-place on single screen (no scrolling).
Uses rich library for colors, tables, progress bars, panels.

Usage:
    from monitoring.display import Dashboard
    dash = Dashboard()
    dash.update_system(gpu="RTX 2060", ...)
    dash.update_training(step=1000, total=100000, ...)
    dash.update_results(metrics, bh_return)
    dash.stop()
"""

import time
import json
import os
from pathlib import Path

import psutil
from rich.console import Console
from rich.columns import Columns
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from performance import format_drawdown_pct

try:
    import torch
except ImportError:
    torch = None

# Initialize psutil CPU percent
psutil.cpu_percent(interval=None)


console = Console()
SPARKLINE_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def make_bar(value, max_val, width=20, color_pos="green", color_neg="red"):
    """Create a text-based bar."""
    if max_val == 0:
        pct = 0
    else:
        pct = min(abs(value / max_val), 1.0)
    filled = int(pct * width)
    empty = width - filled
    color = color_pos if value >= 0 else color_neg
    return f"[{color}]{'#' * filled}[/][dim]{'.' * empty}[/]"


def make_sparkline(values, width=26, color="cyan"):
    """Render a compact sparkline without ANSI clear codes."""
    if not values:
        return Text("·" * width, style="dim")

    arr = list(values[-width:])
    if len(arr) < width:
        arr = ([arr[0]] * (width - len(arr))) + arr

    vmin = min(arr)
    vmax = max(arr)
    spread = vmax - vmin
    if spread <= 1e-12:
        glyphs = SPARKLINE_CHARS[0] * len(arr)
    else:
        glyphs = "".join(
            SPARKLINE_CHARS[min(int((v - vmin) / spread * (len(SPARKLINE_CHARS) - 1)), len(SPARKLINE_CHARS) - 1)]
            for v in arr
        )
    return Text(glyphs, style=color)


def stat_chip(label, value, color="cyan"):
    return Text.from_markup(f"[dim]{label}[/] [{color}]{value}[/]")


class Dashboard:
    """Live terminal dashboard -- updates in place."""

    def __init__(self):
        self._state_path = Path("checkpoints/training_dashboard_state.json")
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = os.environ.get("GARIC_DASHBOARD_MODE", "web").strip().lower()
        self._data = {
            "gpu": "", "gpu_vram": 0, "cuda": "",
            "symbol": "", "data_1m": 0, "data_15m": 0, "features": 0,
            "phase": "", "phase_status": "",
            "phases_done": [],
            "train_step": 0, "train_total": 0, "train_fps": 0,
            "train_elapsed": 0, "entropy": 0, "loss": 0,
            "loss_history": [], "entropy_history": [],
            "step_history": [], "position_history": [], "pnl_history": [],
            "trade_history": [], "win_rate_history": [],
            "epoch": 1, "total_epochs": 10,  # Added epoch tracking
            "bh_return": 0, "bh_full_return": 0,
            "rl_return": 0, "gross_return": 0, "server_cost_paid": 0,
            "total_server_cost_paid": 0, "avg_trades_per_episode": 0, "eval_episodes": 0,
            "flat_ratio": 0, "position_ratio": 0, "alpha_vs_bh": 0, "avg_reward_sum": 0,
            "eval_short_actions": 0, "eval_flat_actions": 0, "eval_long_actions": 0,
            "sharpe": 0, "sortino": 0, "max_dd": 0,
            "n_trades": 0, "n_longs": 0, "n_shorts": 0,
            "n_wins": 0, "n_losses": 0, "win_rate": 0,
            "current_pos": 0.0, "current_pnl": 0.0, "current_action": "Flat",
            "status_msg": "Initializing...",
            "model_path": "", "chart_path": "",
        }
        self._live = None
        if self._mode != "web":
            self._live = Live(
                self._render(),
                console=console,
                transient=False,
                auto_refresh=False,
                vertical_overflow="crop",
            )
            self._live.start()
        else:
            console.print(
                "[cyan]Web dashboard mode enabled.[/] "
                "[dim]Run:[/] [green]streamlit run monitoring/training/dashboard.py[/]"
            )
        self._persist_state()

    def update(self, **kwargs):
        if "loss" in kwargs and kwargs["loss"] != 0:
            if len(self._data["loss_history"]) == 0 or self._data["loss_history"][-1] != kwargs["loss"]:
                self._data["loss_history"].append(kwargs["loss"])
                if len(self._data["loss_history"]) > 40: self._data["loss_history"].pop(0)
                
        if "entropy" in kwargs and kwargs["entropy"] != 0:
            if len(self._data["entropy_history"]) == 0 or self._data["entropy_history"][-1] != kwargs["entropy"]:
                self._data["entropy_history"].append(kwargs["entropy"])
                if len(self._data["entropy_history"]) > 40: self._data["entropy_history"].pop(0)

        self._data.update(kwargs)
        self._append_realtime_histories(kwargs)
        self._persist_state()
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def add_phase(self, name, status="ok", time_sec=0):
        icon = "[green]OK[/]" if status == "ok" else "[red]FAIL[/]" if status == "fail" else "[yellow]...[/]"
        ts = f" ({time_sec:.1f}s)" if time_sec > 0 else ""
        self._data["phases_done"].append(f"  {icon}  {name}[dim]{ts}[/]")
        self._persist_state()
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def stop(self):
        self._persist_state()
        if self._live is not None:
            self._live.update(self._render(), refresh=True)
            self._live.stop()

    def _persist_state(self):
        """Persist dashboard state for the web dashboard."""
        payload = dict(self._data)
        payload["updated_at"] = time.time()
        self._state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _append_realtime_histories(self, updated_fields):
        """Keep short rolling histories for browser charts."""
        if "train_step" not in updated_fields:
            return

        step = int(self._data.get("train_step", 0))
        step_history = self._data["step_history"]
        if step_history and step_history[-1] == step:
            return

        step_history.append(step)
        self._data["position_history"].append(float(self._data.get("current_pos", 0.0)))
        self._data["pnl_history"].append(float(self._data.get("current_pnl", 0.0)))
        self._data["trade_history"].append(float(self._data.get("n_trades", 0.0)))
        self._data["win_rate_history"].append(float(self._data.get("win_rate", 0.0)))

        for key in [
            "step_history",
            "position_history",
            "pnl_history",
            "trade_history",
            "win_rate_history",
        ]:
            if len(self._data[key]) > 240:
                self._data[key] = self._data[key][-240:]

    def _render(self):
        d = self._data
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="system", size=9),  # Increased size for more bars and target line
            Layout(name="middle"),
            Layout(name="results", size=16),
            Layout(name="footer", size=3),
        )

        # === Header ===
        header_text = Text.from_markup(
            "[bold white]GARIC[/]  [dim]Reinforcement Learning Trading Lab[/]\n"
            f"[bold cyan]{d['status_msg']}[/]  [dim]|[/]  [white]BTCUSDT 15m[/]  [dim]|[/]  "
            f"[white]Steps[/] [green]{d['train_step']:,}[/]/[green]{max(d['train_total'], 1):,}[/]"
        )
        layout["header"].update(
            Panel(
                header_text,
                style="white",
                box=box.SQUARE,
                border_style="bright_cyan",
                title="[bold]Live Training Dashboard[/]",
            )
        )

        # === System ===
        cpu_pct = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        ram_pct = ram.percent
        
        gpu_pct = 0.0
        vram_pct = 0.0
        if torch and torch.cuda.is_available() and d["gpu_vram"] > 0:
            try:
                vram_used = torch.cuda.memory_allocated(0) / 1e9
                vram_pct = (vram_used / d["gpu_vram"]) * 100
            except:
                pass
            
            # Simple simulation or fallback for GPU utilization if pynvml not loaded
            gpu_pct = min(100.0, max(5.0, vram_pct * 1.5)) # proxy if genuine util isn't easy to fetch synchronously
        
        cpu_bar = make_bar(cpu_pct/100, 1.0, 20, "magenta", "magenta")
        ram_bar = make_bar(ram_pct/100, 1.0, 20, "magenta", "magenta")
        gpu_bar = make_bar(gpu_pct/100, 1.0, 20, "cyan", "cyan")
        vram_bar = make_bar(vram_pct/100, 1.0, 20, "cyan", "cyan")

        gpu_name = f"[bold green]{d['gpu']}[/]" if d["gpu"] else "[bold yellow]CPU[/]"
        sys_grid = Table.grid(expand=True)
        sys_grid.add_column(ratio=1)
        sys_grid.add_column(ratio=1)
        sys_grid.add_row(
            Text.from_markup(
                f"[bold]GPU[/] {gpu_bar} [cyan]{gpu_pct:04.1f}%[/]\n"
                f"[bold]VRAM[/] {vram_bar} [cyan]{vram_pct:04.1f}%[/]\n"
                f"[dim]Device:[/] {gpu_name}"
            ),
            Text.from_markup(
                f"[bold]CPU[/] {cpu_bar} [magenta]{cpu_pct:04.1f}%[/]\n"
                f"[bold]RAM[/] {ram_bar} [magenta]{ram_pct:04.1f}%[/]\n"
                f"[dim]CUDA:[/] [white]{d['cuda'] or '--'}[/]"
            ),
        )
        sys_footer = Text.from_markup(
            f"[bold]Symbol:[/] [cyan]{d['symbol']}[/]  [dim]|[/]  "
            f"[bold]15m Candles:[/] [white]{d['data_15m']:,}[/]  [dim]|[/]  "
            f"[bold]Eval B&H:[/] [green]{d['bh_return']:+.2%}[/]  [dim]|[/]  "
            f"[bold]Full B&H:[/] [green]{d['bh_full_return']:+.2%}[/]"
        )
        layout["system"].update(
            Panel(
                Group(sys_grid, Text(""), sys_footer),
                title="[bold]System Metrics[/]",
                box=box.ROUNDED,
                border_style="cyan",
            )
        )

        # === Middle: phases + training ===
        mid = Layout()
        mid.split_row(
            Layout(name="phases", ratio=1),
            Layout(name="training", ratio=1),
        )

        # Phases
        phases_text = "\n".join(d["phases_done"][-10:]) if d["phases_done"] else "  [dim]Waiting...[/]"
        mid["phases"].update(Panel(
            Text.from_markup(phases_text),
            title="[bold]Pipeline Phases[/]", box=box.ROUNDED,
        ))

        # Training progress
        step = d["train_step"]
        total = max(d["train_total"], 1)
        pct = step / total
        
        # Estimate epoch from step vs total
        # SB3 doesn't expose epoch seamlessly to callbacks without digging, so we estimate
        current_epoch = min(10, max(1, int((pct * 10) + 1)))
        if pct == 1.0: current_epoch = 10
        
        eta = (d["train_elapsed"] / max(step, 1)) * (total - step) if step > 0 else 0
        bar = make_bar(pct, 1.0, width=45, color_pos="green")

        latest_loss = d["loss_history"][-1] if d["loss_history"] else d.get("loss", 0)
        latest_entropy = d["entropy_history"][-1] if d["entropy_history"] else d.get("entropy", 0)
        trend_columns = Columns(
            [
                Panel(
                    Group(
                        stat_chip("latest", f"{latest_loss:.4f}", "red"),
                        Text(""),
                        make_sparkline(d["loss_history"], width=30, color="red"),
                    ),
                    title="[bold red]Loss[/]",
                    border_style="red",
                    box=box.ROUNDED,
                ),
                Panel(
                    Group(
                        stat_chip("latest", f"{latest_entropy:.4f}", "yellow"),
                        Text(""),
                        make_sparkline(d["entropy_history"], width=30, color="yellow"),
                    ),
                    title="[bold yellow]Entropy[/]",
                    border_style="yellow",
                    box=box.ROUNDED,
                ),
            ],
            equal=True,
            expand=True,
        )

        action_str = d.get('current_action', 'None')
        act_color = "red" if action_str == "Short" else "green" if action_str == "Long" else "yellow"
        
        train_text = Text.from_markup(
            f"  [bold]Progress:[/] {bar} [green]{pct:.0%}[/]\n"
            f"  [bold]Epochs:[/]   [cyan]{current_epoch} / 10[/]  |  [bold]Steps:[/] {step:,} / {total:,}\n"
            f"  [bold]Speed:[/]    {d['train_fps']:.0f} fps  |  [bold]ETA:[/] {eta:.0f}s\n"
            f"  [dim]{'-'*75}[/]\n"
            f"  [bold]Action:[/]    [{act_color}]{action_str}[/]  |  [bold]Pos:[/] [white]{d.get('current_pos', 0):.1f}[/]  |  [bold]UPnL:[/] [{ 'green' if d.get('current_pnl', 0) >= 0 else 'red' }]{d.get('current_pnl', 0):+.2%}\n"
        )
        mid["training"].update(
            Panel(
                Group(train_text, trend_columns),
                title="[bold]PPO Training Progress[/]",
                box=box.ROUNDED,
                border_style="green",
            )
        )

        layout["middle"].update(mid)

        # === Results ===
        res = Layout()
        res.split_row(
            Layout(name="perf", ratio=1),
            Layout(name="trades", ratio=1),
        )

        # Performance
        rl_ret = d["rl_return"]
        gross_ret = d["gross_return"]
        bh_ret = d["bh_return"]
        alpha = d["alpha_vs_bh"]
        rl_color = "green" if rl_ret > bh_ret else "red"
        gross_color = "green" if gross_ret > 0 else "red"
        sharpe_color = "green" if d["sharpe"] > 0 else "red"
        dd_mag = abs(d["max_dd"])
        dd_color = "red" if dd_mag > 0.3 else "yellow" if dd_mag > 0.1 else "green"
        inactive = d["n_trades"] <= 0 or d["flat_ratio"] >= 0.95
        status = "INACTIVE" if inactive else "ACTIVE"
        status_color = "red" if inactive else "green"
        alpha_color = "green" if alpha > 0 else "red"

        perf_text = Text.from_markup(
            f"  [bold]Status:[/] [{status_color}]{status}[/]  |  [bold]Alpha vs B&H:[/] [{alpha_color}]{alpha:+.2%}[/]  |  "
            f"[bold]Flat Ratio:[/] [yellow]{d['flat_ratio']:.1%}[/]\n"
            f"  [bold]{'Metric':<16} {'RL Agent':<14} {'B&H':<14}[/]\n"
            f"  [dim]{'-' * 42}[/]\n"
            f"  {'Avg Net Return':<16} [{rl_color}]{rl_ret:+.2%}[/]{'':>2} [green]{bh_ret:+.2%}[/]\n"
            f"  {'Gross Return':<16} [{gross_color}]{gross_ret:+.2%}[/]\n"
            f"  {'Avg Server Cost':<16} [red]-${d['server_cost_paid']:.2f}[/]\n"
            f"  {'Total Server Cost':<16} [red]-${d['total_server_cost_paid']:.2f}[/]\n"
            f"  {'Sharpe':<16} [{sharpe_color}]{d['sharpe']:.3f}[/]\n"
            f"  {'Sortino':<16} [{sharpe_color}]{d['sortino']:.3f}[/]\n"
            f"  {'Max DD':<16} [{dd_color}]{format_drawdown_pct(d['max_dd'])}[/]\n"
            f"  {'Position Ratio':<16} [cyan]{d['position_ratio']:.1%}[/]\n"
            f"  {'Avg Reward/Ep':<16} [magenta]{d['avg_reward_sum']:+.2f}[/]\n"
        )
        res["perf"].update(Panel(perf_text, title="[bold]Performance[/]", box=box.ROUNDED))

        # Trades
        nt = d["n_trades"]
        nl = d["n_longs"]
        ns = d["n_shorts"]
        nw = d["n_wins"]
        nlo = d["n_losses"]
        wr = d["win_rate"]
        eval_short = d["eval_short_actions"]
        eval_flat = d["eval_flat_actions"]
        eval_long = d["eval_long_actions"]

        total_ls = max(nl + ns, 1)
        total_wl = max(nw + nlo, 1)
        total_actions = max(eval_short + eval_flat + eval_long, 1)
        ls_bar = make_bar(nl / total_ls, 1.0, 20, "green", "green") if nl > 0 else "[dim]" + "." * 20 + "[/]"
        wl_bar = make_bar(nw / total_wl, 1.0, 20, "green", "green") if nw > 0 else "[dim]" + "." * 20 + "[/]"

        wr_color = "green" if wr > 0.5 else "yellow" if wr > 0.4 else "red"
        flat_lock = "YES" if d["flat_ratio"] >= 0.95 else "NO"

        trades_text = Text.from_markup(
            f"  [bold]Total Trades:[/] [cyan]{nt:.0f}[/]  |  [bold]Avg/Ep:[/] [cyan]{d['avg_trades_per_episode']:.1f}[/]\n"
            f"  [bold]Long/Short:[/]  {ls_bar}\n"
            f"    [green]L:{nl:.0f}[/] / [red]S:{ns:.0f}[/]\n"
            f"  [bold]Win/Loss:[/]    {wl_bar}\n"
            f"    [green]W:{nw:.0f}[/] / [red]L:{nlo:.0f}[/]\n"
            f"  [bold]Win Rate:[/]    [{wr_color}]{wr:.1%}[/]  |  [bold]Eval Eps:[/] [cyan]{d['eval_episodes']:.0f}[/]\n"
            f"  [bold]Eval Actions:[/] [red]S:{eval_short:.0f}[/] / [yellow]F:{eval_flat:.0f}[/] / [green]L:{eval_long:.0f}[/]\n"
            f"  [bold]Action Mix:[/]  [red]{eval_short/total_actions:.1%}[/] / [yellow]{eval_flat/total_actions:.1%}[/] / [green]{eval_long/total_actions:.1%}[/]\n"
            f"  [bold]Flat-Locked:[/] [{'red' if flat_lock == 'YES' else 'green'}]{flat_lock}[/]\n"
        )
        res["trades"].update(Panel(trades_text, title="[bold]Trade Stats[/]", box=box.ROUNDED))

        layout["results"].update(res)

        # === Footer ===
        model_str = f"[green]{d['model_path']}[/]" if d["model_path"] else "[dim]training...[/]"
        chart_str = f"[green]{d['chart_path']}[/]" if d["chart_path"] else "[dim]pending[/]"
        footer_text = Text.from_markup(
            f"  [bold]Model:[/] {model_str}  |  [bold]Chart:[/] {chart_str}  |  [bold]Log:[/] [green]pipeline.log[/]"
        )
        layout["footer"].update(Panel(footer_text, style="dim", box=box.ROUNDED))

        return layout


# === Simple non-live fallback (for logging) ===

def print_header():
    console.print()
    console.rule("[bold cyan]GARIC RL Trading System v2.0[/]", style="cyan")
    console.print("[cyan]GPU-Accelerated Reinforcement Intelligence for Crypto[/]", justify="center")
    console.rule(style="cyan")


def print_system_status(gpu_name="", gpu_vram=0, cuda="",
                        symbol="", data_rows=0, candles_15m=0, features=0):
    t = Table(show_header=False, box=None, padding=(0, 2))
    gpu_str = f"[green]{gpu_name}[/] ({gpu_vram:.1f}GB)" if gpu_name else "[yellow]CPU[/]"
    t.add_row("[bold]GPU[/]", gpu_str, "[bold]CUDA[/]", cuda or "--")
    t.add_row("[bold]Symbol[/]", f"[cyan]{symbol}[/]", "[bold]1m rows[/]", f"{data_rows:,}")
    t.add_row("[bold]15m candles[/]", f"{candles_15m:,}", "[bold]Features[/]", str(features))
    console.print(Panel(t, title="[bold]System[/]"))


def print_training_config(total_steps, n_steps, ent_coef, episode_len, lr):
    console.print(f"  [bold]Training:[/] [cyan]{total_steps:,}[/] steps | "
                  f"episode={episode_len} | ent_coef={ent_coef} | lr={lr}")
    console.print(f"  [bold]Actions:[/] [magenta]Discrete(11)[/] [-1.0 ... [yellow]0[/] ... +1.0]")


def print_baseline(bh_return, bh_sharpe=0):
    color = "green" if bh_return > 0 else "red"
    console.print(f"  [bold]>>> TARGET: Buy & Hold[/] = [{color}]{bh_return:+.2%}[/]")


def print_phase(name, status="ok", time_sec=0):
    icons = {"ok": "[green]OK[/]", "fail": "[red]FAIL[/]", "skip": "[yellow]SKIP[/]", "running": "[cyan]...[/]"}
    icon = icons.get(status, icons["running"])
    ts = f" ({time_sec:.1f}s)" if time_sec > 0 else ""
    console.print(f"  [{icon}]  [bold]{name}[/][dim]{ts}[/]")


def print_eval_results(metrics, bh_return):
    rl_ret = metrics.get("total_return", 0)
    gross_ret = metrics.get("gross_total_return", rl_ret)
    bh_eval_ret = metrics.get("bh_eval_return", bh_return)
    alpha = metrics.get("outperformance_vs_bh", rl_ret - bh_eval_ret)
    flat_ratio = metrics.get("flat_ratio", 0.0)
    beats = rl_ret > bh_eval_ret and metrics.get("n_trades", 0) > 5

    if beats:
        console.rule("[bold green]>>> RL AGENT BEATS BUY & HOLD <<<[/]", style="green")
    elif metrics.get("n_trades", 0) <= 1:
        console.rule("[bold red]>>> AGENT INACTIVE[/]", style="red")
    else:
        console.rule("[bold yellow]>>> TRAINING COMPLETE[/]", style="yellow")

    # Comparison table
    t = Table(title="[bold]RL Agent vs Buy & Hold[/]", box=box.ROUNDED)
    t.add_column("Metric", style="bold")
    t.add_column("RL Agent", justify="right")
    t.add_column("Buy & Hold", justify="right")

    rl_color = "green" if rl_ret > bh_eval_ret else "red"
    bh_color = "green" if bh_eval_ret > 0 else "red"
    t.add_row("Status", "[red]INACTIVE[/]" if metrics.get("n_trades", 0) <= 0 or flat_ratio >= 0.95 else "[green]ACTIVE[/]", "--")
    t.add_row("Alpha vs B&H", f"[{'green' if alpha > 0 else 'red'}]{alpha:+.2%}[/]", "--")
    t.add_row("Flat Ratio", f"[yellow]{flat_ratio:.1%}[/]", "--")
    t.add_row("Avg Net Return", f"[{rl_color}]{rl_ret:+.2%}[/]", f"[{bh_color}]{bh_eval_ret:+.2%}[/]")
    t.add_row("Gross Return", f"[{'green' if gross_ret > 0 else 'red'}]{gross_ret:+.2%}[/]", "--")
    t.add_row("Avg Server Cost", f"[red]-${metrics.get('server_cost_paid', 0):.2f}[/]", "--")
    t.add_row("Total Server Cost", f"[red]-${metrics.get('total_server_cost_paid', 0):.2f}[/]", "--")

    sh = metrics.get("sharpe", 0)
    t.add_row("Sharpe", f"[{'green' if sh > 0 else 'red'}]{sh:.3f}[/]", "--")
    t.add_row("Sortino", f"{metrics.get('sortino', 0):.3f}", "--")
    t.add_row("Max DD", f"[red]{format_drawdown_pct(metrics.get('max_drawdown', 0))}[/]", "--")

    nt = metrics.get("n_trades", 0)
    t.add_row("Trades", f"[cyan]{nt:.0f}[/]", "1")
    t.add_row("Avg Trades/Ep", f"[cyan]{metrics.get('avg_trades_per_episode', 0):.1f}[/]", "--")
    t.add_row("Eval Actions", f"S:{metrics.get('eval_short_actions', 0):.0f} / F:{metrics.get('eval_flat_actions', 0):.0f} / L:{metrics.get('eval_long_actions', 0):.0f}", "--")
    t.add_row("Long", f"[green]{metrics.get('n_longs', 0):.0f}[/]", "--")
    t.add_row("Short", f"[red]{metrics.get('n_shorts', 0):.0f}[/]", "--")
    t.add_row("Wins", f"[green]{metrics.get('n_wins', 0):.0f}[/]", "--")
    t.add_row("Losses", f"[red]{metrics.get('n_losses', 0):.0f}[/]", "--")

    wr = metrics.get("win_rate", 0)
    t.add_row("Win Rate", f"[{'green' if wr > 0.5 else 'yellow'}]{wr:.1%}[/]", "--")
    console.print(t)


def print_final_summary(total_time, model_path, chart_path):
    console.rule(style="cyan")
    console.print(f"  [bold]Time:[/]   [cyan]{total_time:.1f}s ({total_time/60:.1f} min)[/]")
    console.print(f"  [bold]Model:[/]  [green]{model_path}[/]")
    console.print(f"  [bold]Chart:[/]  [green]{chart_path}[/]")
    console.print(f"  [bold]Log:[/]    [green]pipeline.log[/]")
    console.rule(style="cyan")


if __name__ == "__main__":
    # Demo live dashboard
    dash = Dashboard()

    dash.update(
        gpu="NVIDIA GeForce RTX 2060", gpu_vram=6.4, cuda="12.8",
        symbol="BTCUSDT", data_1m=3_241_440, data_15m=216_096, features=24,
        bh_return=8.33, status_msg="Loading data...",
    )
    time.sleep(1)

    for name, t in [("Data Loading", 8.0), ("15m Aggregation", 2.3), ("Feature Engineering", 3.7)]:
        dash.add_phase(name, "ok", t)
        time.sleep(0.5)

    # Simulate training
    dash.update(status_msg="PPO Training...")
    for step in range(0, 200001, 20000):
        dash.update(
            train_step=step, train_total=200000,
            train_fps=280, train_elapsed=step / 280,
            entropy=max(2.0 - step / 100000, 0.5),
            loss=max(30 - step / 10000, 5),
            status_msg=f"Training... step {step:,}",
        )
        time.sleep(0.3)

    dash.add_phase("PPO Training (200K steps)", "ok", 350.0)
    dash.add_phase("Multi-Episode Eval (10 eps)", "ok", 25.0)

    # Results
    dash.update(
        rl_return=0.15, sharpe=1.2, sortino=1.8, max_dd=0.12,
        n_trades=850, n_longs=470, n_shorts=380,
        n_wins=420, n_losses=340, win_rate=0.553,
        model_path="checkpoints/rl_agent_final.zip",
        chart_path="checkpoints/training_results.png",
        status_msg="COMPLETE",
    )
    time.sleep(3)
    dash.stop()
