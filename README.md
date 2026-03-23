# GARIC

GARIC is a trading research project for Binance USDT-M futures. It combines:

- a fast internal training environment for PPO and supervised fallback models
- a browser-first training dashboard
- a NautilusTrader execution layer for event-driven backtest, paper, and live paths
- a cost model that separates `gross` trading performance from `net` performance after server cost

The project is designed for Windows and can run from the existing project `venv`.

## Current Status (2026-03-23)

### Reward v2 — Simplified for Profitability

The reward function was completely redesigned to fix PPO collapse (agent learning to always stay flat).

**Previous reward (v1)** had 12 conflicting penalty terms:
- missed_move_penalty (40x), wrong_side_penalty (20x), alpha benchmark, server cost multiplier (6x), flat penalty, same position penalty, closed trade bonus — all fighting each other
- Result: PPO collapsed to 100% flat, 0 trades, 0 reward

**New reward (v2)** uses only 2 clean components:
- `pnl × 200` — net PnL from trading (fees + funding included)
- `abs(market_move) × 30` — opportunity cost when flat (proportional to actual price movement)
- Server cost tracked in balance only, not in reward (sunk cost the agent can't control)
- No double-counting, no alpha benchmark comparison

**Incentive ordering**: correct position (+) > flat (−) > wrong position (−−)

### CryptoMamba Forecast — With Accuracy Validation

CryptoMamba (Mamba SSM) generates 5 forecast features for the RL model.

**Previous**: no accuracy validation — trained and used blindly, predictions could be pure noise

**Current**:
- Per-window normalization (matches inference behavior exactly)
- Train/validation split (80/20) with early stopping
- Quality gate: directional accuracy > 50% AND RMSE < naive baseline
- If quality gate fails → forecast features zeroed out automatically
- Model size: d_model=64, n_layers=4, context=128 bars (~50K params, fits RTX 2060)

### Data Coverage

- **Source**: Binance USDT-M futures, 1-minute candles
- **Range**: 2020-01-01 to 2026-02-28 (6.2 years, 3.2M rows)
- **Decision timeframe**: 15-minute candles (216K bars)
- **Split**: Train 72% (2020-01 → 2024-06) / Val 8% (2024-06 → 2024-12) / Test 20% (2024-12 → 2026-02)

### Model Selection Gates

Models must pass these gates to be selected:

- Net return > -10% (hard reject if losing too much)
- Dominant action ratio < 85% (must use diverse actions)
- Action entropy > 0.05 (must explore)
- Average trades per episode > 1.5

Profitability is heavily weighted in the scoring function.

### Feature Cache Validation

Pipeline now validates cached features against raw data size. Stale caches from test runs (e.g. 13K rows instead of 216K) are automatically invalidated and rebuilt.

### Test Suite

63 tests passing across all modules.

## Architecture

GARIC currently uses two execution layers on purpose:

1. Fast training environment
   - Used for PPO rollouts and fast model iteration
   - Lets training finish in practical time
   - Uses the same portfolio/server-cost logic as the internal backtest path

2. NautilusTrader validation and execution
   - Used for more realistic event-driven validation during model selection
   - Used for standalone Nautilus backtest
   - Used for Nautilus paper/live execution

Important:

- GARIC does **not** use Nautilus as the per-step PPO gym environment
- That would be far too slow for RL training
- Instead, training now uses Nautilus in the place where it matters most:
  - candidate validation
  - final model selection
  - held-out realistic test

## How Nautilus Is Used In This Project

### 1. During training

When you run the training pipeline, GARIC now:

- trains PPO in the fast internal env
- optionally trains supervised fallback candidates
- scores those candidates on the internal validation split
- runs Nautilus validation on candidate model files
- uses the Nautilus result to help select the final model
- runs a Nautilus held-out test on the selected model

This logic lives mainly in:

- `pipeline.py`
- `execution/nautilus/backtest_runner.py`
- `execution/nautilus/strategy.py`

The Nautilus-backed training validation is controlled by:

- `configs/train_rtx2060.yaml`
- `configs/default.yaml`
- `configs/test_rtx2060.yaml`

Look for this section in config:

```yaml
training:
  nautilus_validation:
    enabled: true
    use_for_model_selection: true
    evaluate_final_test: true
```

### 2. As a standalone execution engine

Nautilus is also exposed directly for:

- local event-driven backtest
- Binance paper mode
- Binance live mode

That path loads GARIC models directly from:

- `checkpoints/rl_agent_final.zip`
- `checkpoints/rl_agent_best.zip`
- `checkpoints/rl_agent_supervised.joblib`

## Installation

Create and activate the project virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you already have the project `venv`, reuse it.

## Recommended Windows Workflow

### Train with browser dashboard

```powershell
python run_training_browser.py --config configs\train_rtx2060.yaml
```

Or train directly (first run use `--no-cache` to rebuild features with full data):

```powershell
python pipeline.py --mode train --config configs\train_rtx2060.yaml --no-cache
```

### Run Nautilus backtest directly

```powershell
python run_nautilus_browser.py --mode backtest
```

### Run Nautilus paper mode

```powershell
python run_nautilus_browser.py --mode paper
```

### Run Nautilus live mode

```powershell
python run_nautilus_browser.py --mode live
```

Live mode requires `BINANCE_API_KEY` and `BINANCE_API_SECRET`.

## Main Project Commands

Run tests:

```powershell
pytest tests\
```

Run the Nautilus browser dashboard only:

```powershell
streamlit run monitoring\nautilus\dashboard.py
```

Run the training browser dashboard only:

```powershell
streamlit run monitoring\training\dashboard.py
```

## Current Model/Metric Rules

### Action space

The policy is discrete: `Short`, `Flat`, `Long`

### Feature stack

The current observation uses:

- `30` market features (TA 15 + Micro 5 + Returns 4 + CryptoMamba forecast 5 + Vol 1)
- `4` agent-state features (position, unrealized PnL, flat steps, position steps)

### Reward v2

```
reward = pnl × pnl_reward_scale           (default 200)
       − abs(market_move) × opp_cost_scale (default 30, only when flat)
```

Server cost is applied to balance tracking only (not reward). Episode-end penalty of 3.0 for zero-trade episodes forces exploration.

### Cost model

- `gross_total_return` = trading result before server cost
- `total_return` = net result after server cost
- if the model does nothing, equity still decays from server costs
- taker fee: 0.05% + slippage: 1 bps per trade

### Why Nautilus validation matters

The fast env is good for training speed, but it can still overvalue collapsed policies. Nautilus validation helps expose whether a model is just holding one direction, overtrading, or winning only because of a narrow synthetic path.

## Outputs

Important runtime files:

- `pipeline.log`
- `dashboard.log`
- `nautilus.log`
- `checkpoints/training_results.png`
- `checkpoints/training_dashboard_state.json`
- `checkpoints/rl_agent_final.zip`
- `checkpoints/rl_agent_best.zip`
- `checkpoints/rl_agent_supervised.joblib`
- `checkpoints/crypto_mamba_15m.pt`

## Repository Layout

```text
garic/
  configs/
  data/
  execution/
    backtest/
    live/
    nautilus/
  features/
  models/
    forecast/
    rl/
    moe/
  monitoring/
    training/
    live/
    nautilus/
  risk/
  tests/
  pipeline.py
  run_training_browser.py
  run_nautilus_browser.py
```

## Verified Status

Verified on Windows project `venv` (Python 3.12, PyTorch 2.10+cu128, RTX 2060):

- 63/63 pytest suite passing
- Reward v2 confirmed: correct position > flat > wrong position
- PPO no longer collapses to flat (entropy 0.938, diverse actions)
- CryptoMamba quality gate working (rejects noise, accepts real patterns)
- Feature cache validation prevents stale data

## License

This repository is source-available, not open source.

It is licensed under the custom `LICENSE` in this repo. Practical rules:

- non-commercial use only
- attribution to `GodEyeTee` is required
- modification and derivative works are not allowed
- commercial/profit use requires a separate paid license

## Disclaimer

This project is experimental and not financial advice. A profitable backtest, validation run, or paper run does not guarantee live profitability.
