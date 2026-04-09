# GARIC

GARIC is a Windows-first Binance USDT-M futures research workspace focused on one goal:

- train a compact directional model
- validate it against realistic execution
- keep a written record of what was tried and what actually worked

The current project direction is **supervised-first**.

- `supervised_logreg` is the primary model family
- PPO is still in the codebase for research, but it is **disabled by default** in the shipped training profiles
- `NautilusTrader` is the execution-aligned backtest / paper / live engine

## Current Reality

What is working now:

- compact 25-dim market feature stack
- 8-dim agent state stack
- supervised threshold search with side-specific long / short confidence
- validation-derived post-cost calibration and meta-label gating
- robust multi-window post-cost calibration for threshold selection and inference gating
- walk-forward validation inside the fast training pipeline
- Nautilus validation and final held-out execution-style testing
- browser dashboards for both training and Nautilus execution

What is not good enough yet:

- fast internal validation and execution-style validation still diverge on some runs
- the model is still too flat-heavy
- the strategy is not yet robust enough for unattended live deployment

If you are looking for the best verified reference run so far, read:

- [BASELINE_SUPERVISED_LOGREG.md](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/BASELINE_SUPERVISED_LOGREG.md)
- [EXPERIMENT_LOG.md](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/EXPERIMENT_LOG.md)

## Canonical Workflow

Use `garic.bat` or `python garic.py` as the normal entrypoint.

| Goal | Recommended command | Notes |
| --- | --- | --- |
| Full training run | `garic.bat train --profile best --no-cache` | Current main research path |
| Smoke / validation run | `garic.bat test` | Small run for fast iteration |
| Nautilus backtest | `garic.bat backtest` | Execution-style backtest |
| Nautilus paper trading | `garic.bat paper` | Browser dashboard + paper flow |
| Nautilus live trading | `garic.bat live` | Requires Binance credentials |

Equivalent Python form:

```powershell
python garic.py train --profile best --no-cache
```

## What `train` Does Today

The current `train` pipeline is:

1. Load and clean Binance 1-minute futures data
2. Aggregate to 15-minute decision bars
3. Build the compact feature stack
4. Optionally add forecast features if explicitly enabled and they pass quality gates
5. Train the selected primary family
6. Score candidates on fast internal validation
7. Stress candidates with contiguous walk-forward validation
8. Forward the strongest candidates into `NautilusTrader`
9. Select the final model using execution-aligned validation when enabled
10. Write charts, logs, metrics, and artifacts to `checkpoints/`

Current safety checks in the supervised path:

- calibrated confidence grids respect the conservative floors set in config
- post-cost edge is damped when calibration support comes from too few contiguous windows
- Windows held-out Nautilus tests use a longer timeout than validation windows

Important:

- `train` is no longer “PPO first with supervised rescue”
- it is now “supervised first, execution-aligned selection, PPO optional research path”

## How GARIC Uses Nautilus

`NautilusTrader` is not used as the PPO gym environment for every training step. That would be too slow for practical iteration.

Instead, GARIC uses Nautilus where realism matters most:

- candidate validation
- final held-out execution-style backtest
- paper trading
- live trading

Current `Nautilus` design in this repo:

- the pipeline builds an aligned 15-minute frame
- validation / test slices are passed to Nautilus with **prepended history warmup**
- scoring metrics are reconciled back to the true target slice, not the warmup bars
- browser dashboards read JSON state written by the running strategy / runner

Main Nautilus-related files:

- [execution/nautilus/backtest_runner.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/execution/nautilus/backtest_runner.py)
- [execution/nautilus/strategy.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/execution/nautilus/strategy.py)
- [execution/nautilus/model.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/execution/nautilus/model.py)
- [configs/nautilus.yaml](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/configs/nautilus.yaml)

## Current Best Verified Baseline

The best execution-aligned baseline currently kept on record is documented in:

- [BASELINE_SUPERVISED_LOGREG.md](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/BASELINE_SUPERVISED_LOGREG.md)

At the time of writing, the safer execution-aligned baseline is:

- model family: `supervised_logreg`
- net return: about `+0.63%`
- gross return: about `+1.33%`
- alpha vs B&H: about `+0.37%`
- max drawdown: about `2.05%`

There is also a stronger **fast-env** baseline that looks better on internal held-out evaluation, but it does **not** yet qualify as the safer production reference because Nautilus agreement is still weak.

## Experiment Record

This repo keeps explicit experiment history. Before changing thresholds, validation, or reward logic, read:

- [EXPERIMENT_LOG.md](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/EXPERIMENT_LOG.md)
- [BASELINE_SUPERVISED_LOGREG.md](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/BASELINE_SUPERVISED_LOGREG.md)

These files track:

- exact settings that were tried
- which run became the reference baseline
- which changes made things worse
- what should not be repeated

## Installation

Create and activate a project virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Notes:

- `requirements.txt` is the full environment for training, dashboards, and Nautilus
- `requirements-minimal.txt` is not enough for the full workflow
- `garic.bat` prefers `venv\Scripts\python.exe`, then `.venv\Scripts\python.exe`, then `python` from PATH

If historical data is missing, the training pipeline can download Binance Futures OHLCV into `data/raw/`.

## Profile Mapping

GARIC always loads [configs/default.yaml](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/configs/default.yaml) first, then deep-merges the profile override.

Profile shortcuts:

- `best` -> [configs/train_rtx2060.yaml](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/configs/train_rtx2060.yaml)
- `smoke` -> [configs/test_rtx2060.yaml](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/configs/test_rtx2060.yaml)
- `cloud` -> [configs/default.yaml](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/configs/default.yaml)

Execution commands use [configs/nautilus.yaml](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/configs/nautilus.yaml) by default.

## Shipped Defaults That Matter

Current shipped defaults are centered on:

- `primary_model: supervised_logreg`
- `training.rl.enabled: false`
- `training.nautilus_validation.enabled: true`
- `forecast_features.crypto_mamba.enabled: false`

Current compact market stack:

- 5 log-return features
- 15 TA features
- 5 microstructure features

Agent state stack:

- position
- unrealized PnL
- equity ratio
- current drawdown
- rolling volatility
- last-step turnover
- flat steps
- position steps

That means:

- base features: `25`
- agent state: `8`
- total observation dims without forecast: `33`

Important runtime note:

- the current default `supervised_logreg + Nautilus validation` path is mostly CPU-bound
- setting `training.device: "cuda"` does not make scikit-learn logistic regression or Nautilus backtests move to GPU
- GPU is only meaningfully used when PPO or CryptoMamba is enabled

## Main Outputs

Important runtime files:

- [pipeline.log](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/pipeline.log)
- [logs/](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/logs)
- [dashboard.log](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/dashboard.log)
- [nautilus.log](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/nautilus.log)
- [EXPERIMENT_LOG.md](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/EXPERIMENT_LOG.md)
- [BASELINE_SUPERVISED_LOGREG.md](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/BASELINE_SUPERVISED_LOGREG.md)
- [checkpoints/training_results.png](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/checkpoints/training_results.png)
- [checkpoints/rl_agent_supervised.joblib](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/checkpoints/rl_agent_supervised.joblib)
- [checkpoints/rl_agent_safe_flat.joblib](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/checkpoints/rl_agent_safe_flat.joblib)

## Key Files

- [garic.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/garic.py): unified launcher
- [garic.bat](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/garic.bat): Windows launcher
- [pipeline.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/pipeline.py): train / test pipeline
- [models/rl/supervised.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/models/rl/supervised.py): current primary model family
- [models/rl/environment.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/models/rl/environment.py): fast internal evaluation environment
- [execution/nautilus/strategy.py](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/execution/nautilus/strategy.py): execution strategy

## Current Research Priorities

The current high-value work is:

1. Make fast internal validation agree more closely with Nautilus contiguous execution
2. Improve post-cost calibration and meta-label gating so signals survive fees, slippage, and server cost
3. Reduce fast-env / Nautilus disagreement without falling back to `safe_flat`
3. Improve long / short threshold selection without overtrading
4. Reduce flat-heavy behavior without forcing trades
5. Keep server cost and execution cost in every honest net metric
6. Replace weak baselines only when a new full run beats them on the right reference path

## License

This repository is source-available, not open source.

It is licensed under the custom [LICENSE](/C:/Users/wanar/OneDrive/เดสก์ท็อป/NEXTROOM/garic/LICENSE) in this repo. Practical rules:

- non-commercial use only
- attribution to `GodEyeTee` is required
- modification and derivative works are not allowed
- commercial or profit use requires a separate paid license

## Disclaimer

This project is experimental and not financial advice. A profitable training run, validation run, backtest, or paper run does not guarantee live profitability.
