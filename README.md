# GARIC

GARIC is a reinforcement-learning trading lab for Binance USDT-M futures. The current project focuses on a PPO-based discrete policy with realistic cost modeling, browser-first training monitoring, and evaluation that separates gross trading performance from net performance after infrastructure cost.

## Current Status

- Trading action space is discrete: `Short`, `Flat`, `Long`
- Feature stack uses `30` market features plus `4` agent-state features in the observation
- Training dashboard is browser-first via Streamlit
- Terminal output is kept for logs and errors
- Evaluation reports both:
  - `gross_total_return`: trading performance before server cost
  - `total_return`: net result after server cost

## What Was Fixed

The recent updates focus on making the system train and report honestly instead of looking better than it is.

### Cost model and metrics

- Monthly server cost is charged against portfolio equity every step
- Gross and net returns are tracked separately
- Max drawdown is displayed with the correct sign
- Sharpe and Sortino are clamped for no-trade / near-zero-variance cases
- Backtest and RL environment now use the same cost logic

### Reward shaping

- The agent is penalized for staying flat too long
- Missing meaningful market moves while flat is penalized
- Server cost is also reflected in reward shaping so inactivity is not treated as free
- Reward shaping is stronger than before to reduce flat-policy collapse

### Training stability

- Training uses periodic evaluation during learning
- The best checkpoint is selected during training instead of blindly trusting the final checkpoint
- PPO network size was increased to improve capacity
- Training config was extended to give the model more coverage

### Monitoring

- Training UI moved to the browser
- The dashboard now shows:
  - active position
  - unrealized PnL
  - training quality
  - trade activity
  - action mix
  - return stack
  - performance snapshot
  - pipeline status

## Repository Layout

```text
garic/
  configs/
  data/
  execution/
  features/
  models/
  monitoring/
  risk/
  tests/
  pipeline.py
  run_training_browser.py
  train_web.bat
```

## Installation

Create and activate a virtual environment first.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you use NVIDIA on Windows, install a CUDA-enabled PyTorch build that matches your driver.

## One-Command Training

This is the recommended way to train now:

```powershell
python run_training_browser.py
```

What it does:

- starts the Streamlit dashboard
- opens the browser automatically
- runs the training pipeline
- keeps logs in the terminal
- writes dashboard logs to `dashboard.log`
- writes pipeline logs to `pipeline.log`

You can also use:

```powershell
.\train_web.bat
```

## Main Commands

### Run tests

```powershell
pytest tests\test_performance_metrics.py tests\test_models.py tests\test_live.py
```

### Train with explicit config

```powershell
python pipeline.py --mode train --config configs\train_rtx2060.yaml
```

### Browser dashboard only

```powershell
streamlit run monitoring/training/dashboard.py
```

## Current Training Config Notes

The `configs/train_rtx2060.yaml` profile is tuned to reduce policy collapse:

- `total_timesteps: 600000`
- `batch_size: 250`
- `n_epochs: 15`
- `ent_coef: 0.03`
- periodic checkpoint evaluation every `50000` steps
- stronger inactivity penalties
- stronger missed-opportunity penalties
- stronger server-cost-aware reward shaping

## How Server Cost Works

Server cost is intentionally not optional in the reward/equity model.

- The portfolio pays the monthly server cost over time
- If the agent does nothing, equity still decays
- If the agent generates enough trading profit to beat the server cost, its net result improves
- This prevents the policy from learning that permanent inactivity is safe

## Logs and Outputs

Key artifacts after training:

- `checkpoints/rl_agent_final.zip`
- `checkpoints/rl_agent_best.zip`
- `checkpoints/training_results.png`
- `checkpoints/training_dashboard_state.json`
- `pipeline.log`
- `dashboard.log`

These files are runtime artifacts and are ignored from version control in this public repo.

## Live / Paper Trading Note

The live trading engine was updated so its observation format matches the RL environment more closely:

- it now passes `position`, `upnl`, `flat_steps`, and `pos_steps` style state
- it maps PPO discrete actions back to `-1 / 0 / 1`

This reduces the mismatch between training-time and live-time inference.

## Public Repo Policy

This public repository is intentionally kept clean.

Ignored from git:

- raw/clean/cache data
- checkpoints
- logs
- virtual environments
- local tool folders
- markdown files other than this `README.md`

## Disclaimer

This project is experimental and not financial advice. A profitable backtest or training run does not guarantee live profitability.
