# GARIC

GARIC is a Windows-first Binance USDT-M futures research repo. It trains PPO in a fast internal environment, validates candidates with NautilusTrader, and can run Nautilus backtest, paper, and live flows from the same workspace.

## Canonical Workflow

Use `garic.bat` or `python garic.py` as the only user-facing entrypoint.

| Goal | Recommended command | Notes |
| --- | --- | --- |
| Best current local training run | `garic.bat train --profile best --no-cache` | Starts the training dashboard and runs the full local recipe |
| Quick validation run | `garic.bat test` | Uses the lightweight smoke profile |
| Nautilus backtest | `garic.bat backtest` | Starts the Nautilus dashboard by default |
| Nautilus paper trading | `garic.bat paper` | Uses `configs/nautilus.yaml` unless overridden |
| Nautilus live trading | `garic.bat live` | Requires `BINANCE_API_KEY` and `BINANCE_API_SECRET` |

Equivalent Python form:

```powershell
python garic.py train --profile best --no-cache
```

Defaults:

- `train`, `backtest`, `paper`, and `live` start the Streamlit dashboard automatically.
- Add `--cli` to skip the dashboard and run the underlying module directly.
- Add `--config <yaml>` to override the profile mapping.
- Add `--no-browser` if you want the dashboard server without auto-opening a tab.

## Recommended Local Recipe

1. Validate the environment and code path first:

   ```powershell
   garic.bat test
   ```

2. Run the best current local training recipe:

   ```powershell
   garic.bat train --profile best --no-cache
   ```

3. Review the outputs in `checkpoints/`, `pipeline.log`, and `dashboard.log`.

`best` is the current recommended local profile in this repo. It maps to `configs/train_rtx2060.yaml` and currently means:

- BTCUSDT only
- 1,000,000 PPO timesteps
- balanced regime sampling
- strict checkpoint selection gates
- Nautilus validation enabled for model selection

## Installation

Create and activate a project virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Notes:

- `requirements.txt` is the full environment for training, dashboards, and Nautilus.
- `requirements-minimal.txt` is not enough for full training or execution.
- `garic.bat` automatically prefers `venv\Scripts\python.exe`, then `.venv\Scripts\python.exe`, then `python` from PATH.
- If GARIC says no compatible Python environment was found, recreate the project venv and reinstall `requirements.txt`.

If historical data is missing, the training pipeline can download Binance Futures OHLCV automatically into `data/raw/`.

## Config Model

GARIC always loads `configs/default.yaml` first. Any config you pass with `--config` is deep-merged on top of it.

Friendly profile mappings:

- `best` -> `configs/train_rtx2060.yaml`
- `smoke` -> `configs/test_rtx2060.yaml`
- `cloud` -> `configs/default.yaml`

Execution commands (`backtest`, `paper`, `live`) use `configs/nautilus.yaml` by default.

## What `train` Actually Does

When you run `garic.py train`, GARIC:

1. loads and cleans Binance 1-minute futures data
2. aggregates it to 15-minute decision bars
3. builds the compact market feature stack
4. optionally adds CryptoMamba forecast features if enabled and if they pass quality checks
5. trains PPO in the fast internal environment
6. evaluates checkpoints with internal selection gates
7. runs Nautilus validation on candidate models
8. selects the final model and writes artifacts to `checkpoints/`

The training path is intentionally split into two layers:

- Fast internal environment for practical PPO training speed
- NautilusTrader for more realistic candidate validation and final execution paths

## Current Defaults That Matter

- `pipeline.py` is the train/test engine only. Use `garic.py backtest|paper|live` for execution.
- Reward is v3: equity return after costs, minus drawdown-increase and turnover penalties.
- Base market feature stack is 25 dims: 5 returns + 15 TA + 5 microstructure.
- Agent state adds 8 dims, so the policy sees 33 dims when forecast features are disabled.
- CryptoMamba forecast features exist, but are disabled by default in the shipped profiles.
- Supervised fallback exists, but is disabled by default in the shipped profiles.
- Nautilus validation is enabled in the training profiles and is used during model selection.

## Main Outputs

Important runtime files:

- `pipeline.log`
- `dashboard.log`
- `nautilus.log`
- `checkpoints/training_results.png`
- `checkpoints/training_dashboard_state.json`
- `checkpoints/rl_agent_final.zip`
- `checkpoints/rl_agent_best.zip`
- `checkpoints/rl_agent_supervised.joblib`
- `checkpoints/crypto_mamba_15m.pt` when forecast features are enabled

## Key Files

- `garic.py`: unified launcher
- `garic.bat`: Windows launcher that prefers the project virtual environment
- `pipeline.py`: train/test pipeline engine
- `run_training_browser.py`: lower-level training dashboard launcher
- `run_nautilus_browser.py`: lower-level Nautilus dashboard launcher
- `configs/default.yaml`: base config
- `configs/train_rtx2060.yaml`: recommended local training override
- `configs/test_rtx2060.yaml`: lightweight smoke-test override
- `configs/nautilus.yaml`: default Nautilus execution config

## Legacy Entry Points

`run_training_browser.py`, `run_nautilus_browser.py`, and `pipeline.py` still exist because the unified launcher calls them internally. They are lower-level tools now.

For day-to-day use, stick to:

```powershell
garic.bat <command>
```

## License

This repository is source-available, not open source.

It is licensed under the custom `LICENSE` in this repo. Practical rules:

- non-commercial use only
- attribution to `GodEyeTee` is required
- modification and derivative works are not allowed
- commercial or profit use requires a separate paid license

## Disclaimer

This project is experimental and not financial advice. A profitable backtest, validation run, or paper run does not guarantee live profitability.
