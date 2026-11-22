# CycleNet

CycleNet forecasts multivariate time series by learning recurring patterns and predicting the deviations from those patterns. This repository contains the implementation of *CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns* and research extensions that test when information from other channels improves a forecast. [CycleNet paper](https://arxiv.org/abs/2409.18479)

[Model](#how-cyclenet-works) · [Setup](#setup) · [Datasets](#datasets) · [Training](#train-a-baseline) · [Experiments](#experiment-workflows) · [Results](#read-the-results) · [Citation](#citation)

## How CycleNet works

Residual Cycle Forecasting learns a repeating template with one value per cycle position and channel. For hourly electricity data with a weekly cycle, the template has 168 positions. Given a history window, CycleNet:

1. Optionally normalizes each channel using the mean and variance of that input window.
2. Subtracts the learned template at the cycle positions covered by the history.
3. Forecasts the residuals with a linear layer or a two-layer MLP.
4. Adds the template at the future cycle positions and reverses the instance normalization when enabled.

The baseline shares the temporal forecasting weights across channels and processes each channel independently. The cycle length is configured in samples. Set it using the sampling frequency and a known period or use autocorrelation to identify a candidate period. The loaders compute cycle phase from row position, so keep observations in chronological order on a regular grid and resolve missing timestamps before training. [Baseline implementation](models/CycleNet.py) · [Data loaders](data_provider/data_loader.py)

![Residual Cycle Forecasting architecture](Figures/Figure2.png)

### Research extensions

The main experiments test whether one channel's deviations from its recurring pattern help forecast other channels. The three mixing variants apply a learnable gated residual update before the temporal forecast. Their gates start at zero.

| Experiment label | `--model` | Channel interaction |
| --- | --- | --- |
| CycleNet | `CycleNet` | Each channel uses its own history. |
| CycleNetMix | `CycleNetMixTune` in the fixed-lookback grids | An MLP mixes the channels at each lookback step. |
| PairMix | `CycleNetPairMix` | Each target channel receives a learned linear combination of selected source channels. |
| MixLowRank | `CycleNetMixLowRank` | A bottleneck of configurable rank limits the dimension of the channel interaction. |

The model registry also includes an earlier `CycleNetMix` implementation, `CycleNetMixLag`, trend and multiscale trend variants, and benchmark models including DLinear, PatchTST, and SparseTSF. Use the experiment's recorded model argument when reproducing a run. [Model registry](exp/exp_main.py) · [Model implementations](models/)

## Setup

Run the examples from the repository root. Use Python 3.10 or later for the full set of research utilities. The cluster scripts commonly use a virtual environment at `./venv`.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Install Optuna if you plan to use the `tune_*_optuna.py` scripts.

```bash
python -m pip install optuna
```

Single experiments can run on CPU when CUDA is unavailable. For GPU runs, use a PyTorch installation compatible with the cluster's CUDA environment. The Slurm grids request GPUs and may need site-specific partition, resource, and path settings.

## Datasets

Place the input files directly under `dataset/` or set `--root_path` for a standalone run. The channel counts and cycle lengths below match the multivariate experiments in this repository.

| Dataset | File | `--data` | Channels | Cycle in samples |
| --- | --- | --- | ---: | ---: |
| ETT hourly | `ETTh1.csv`, `ETTh2.csv` | `ETTh1`, `ETTh2` | 7 | 24 |
| ETT 15-minute | `ETTm1.csv`, `ETTm2.csv` | `ETTm1`, `ETTm2` | 7 | 96 |
| Electricity | `electricity.csv` | `custom` | 321 | 168 |
| Weather | `weather.csv` | `custom` | 21 | 144 |
| Solar | `solar_AL.txt` | `Solar` | 137 | 144 |
| PEMS08 | `PEMS08.npz` | `PEMS` | 170 | 288 |

The custom CSV loader expects a `date` column and a target column named `OT` by default. It places the target last in the channel order. Solar uses a headerless, comma-separated text file. The PEMS loader reads the first feature of the archive's `data` array for every sensor. Loaders create chronological train, validation, and test splits and fit their scalers on the training split. Keep the loader and channel order consistent between feature preparation and training. [Benchmark data archive](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view) · [PEMS source project](https://github.com/cure-lab/SCINet) · [Loader registry](data_provider/data_factory.py)

## Train a baseline

This example trains CycleNet/MLP on ETTh1 using 96 observed time steps to forecast the next 96. With `--features M`, the model forecasts all seven channels.

```bash
python -u run.py \
  --is_training 1 --model_id ETTh1_baseline_L96_H96 --model CycleNet \
  --root_path ./dataset/ --data_path ETTh1.csv --data ETTh1 --features M \
  --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --cycle 24 \
  --model_type mlp --d_model 512 --use_revin 1 \
  --batch_size 64 --learning_rate 0.001 --train_epochs 30 --patience 5 \
  --num_workers 4 --random_seed 2024 --itr 1
```

Use `--model_type linear` to train the linear baseline. The training loop saves the best validation checkpoint, reloads it, and evaluates the test split. Baseline recipes for multiple benchmarks are in [scripts/CycleNet/](scripts/CycleNet/), and the batch launcher is [old_slurm/run_main.sh](old_slurm/run_main.sh).

## Experiment workflows

### Build a PairMix dependency matrix

PairMix selects a fraction of the other channels for each target channel using `--mix_ratio`. The dependency builder loads a trained CycleNet checkpoint, reproduces its input normalization and cycle subtraction on training windows, and computes the absolute Pearson correlation between residual channels. It saves a square NumPy matrix with a zero diagonal. Channel indices must follow the forecast loader's channel order. [Dependency builder](create_dependency_scores.py) · [PairMix implementation](models/CycleNetPairMix.py)

After training the ETTh1 example above, generate the corresponding matrix with:

```bash
mkdir -p dependency_scores
python create_dependency_scores.py \
  --root_path ./dataset/ --data_path ETTh1.csv --data ETTh1 --features M \
  --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --cycle 24 \
  --model_type mlp --d_model 512 --use_revin 1 --batch_size 64 --num_workers 4 \
  --checkpoint ./checkpoints/ETTh1_baseline_L96_H96_CycleNet_ETTh1_ftM_sl96_pl96_cycle24_mlp_seed2024/checkpoint.pth \
  --output ./dependency_scores/ETTh1_example_dependency.npy
```

To train PairMix, rerun the training command with a distinct `--model_id`, set `--model CycleNetPairMix`, and append `--mix_ratio 0.4 --dependency_score_path ./dependency_scores/ETTh1_example_dependency.npy`. For benchmark matrices, use the recipes for [Electricity](scripts/experiments/electricity/electricity_dependency.sh), [ETT](scripts/experiments/ett/ett_dependency_matrices.sh), and [PEMS08](scripts/experiments/pems08/create_pems08_dependency.sh).

### Run the Electricity grid

The [Electricity grid](scripts/experiments/electricity/electricity_fixed_lookback_grid.sh) crosses four lookbacks and four horizons (96, 192, 336, and 720) with three mixing models. Each of its 48 Slurm array tasks covers one model, lookback, and horizon. The grid runs 1,408 configurations in total, varying learning rate and batch size as well as mixing depth, ratio, or rank as applicable. It uses seed 2024 and a weekly cycle.

```bash
export PROJECT_DIR="$PWD"
export PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
bash scripts/experiments/electricity/electricity_fixed_lookback_grid.sh --plan
sbatch scripts/experiments/electricity/electricity_fixed_lookback_grid.sh
bash scripts/experiments/electricity/electricity_fixed_lookback_grid.sh --summary
```

Before submission, ensure `dataset/electricity.csv` and `dependency_scores/Electricity_dependency.npy` are present and adapt the Slurm directives to your cluster. The script stores verified trials as JSON and skips those records on resubmission. `--summary` rebuilds the summary CSVs from the saved records.

### Run other benchmarks and tuning studies

| Workflow | Entry points |
| --- | --- |
| CycleNet MLP baseline sweeps | [Three-lookback baseline](scripts/experiments/baseline/cyclenet_mlp_three_lookbacks.sh) |
| ETTm1, Solar, and Weather mixing grids | [ETTm1](scripts/experiments/ett/srp_fixed_lookback_ettm1.sh), [Solar](scripts/experiments/solar/srp_fixed_lookback_solar.sh), [Weather](scripts/experiments/weather/srp_fixed_lookback_weather.sh) |
| ETTm2, ETTh1, and ETTh2 mixing grids | [ETT PairMix](scripts/experiments/ett/ett_pairmix_fixed_lookback_grid.sh), [ETTm2 mix and low rank](scripts/experiments/ett/srp_ettm2_mix_lowrank_grid.sh), [ETTh1 mix and low rank](scripts/experiments/ett/srp_etth1_mix_lowrank_grid.sh), [ETTh2 mix and low rank](scripts/experiments/ett/srp_etth2_mix_lowrank_grid.sh) |
| PEMS08 short-horizon experiments (12, 24, 48, and 96 steps) | [Baseline](scripts/experiments/pems08/srp_pems08_paper_baseline_12_24_48_96.sh), [Mixing grid](scripts/experiments/pems08/srp_pems08_mixing_12_24_48_96.sh), [Re-evaluate winners with RevIN disabled](scripts/experiments/pems08/pems08_winners_revin_off.sh) |
| Optuna studies | [PairMix](tune_pairmix_optuna.py), [Low rank](tune_lowrank_optuna.py), [Trend](tune_trend_optuna.py) |

Check each script's configuration before launching. Several research wrappers include cluster-specific paths and use `--summarize-only` for rebuilding tables, whereas the Electricity script uses `--summary`.

### Analyze channel dependence

The mutual-information study asks whether datasets with stronger channel dependence also show more frequent gains from mixing. It estimates quantile-binned normalized mutual information using training observations, compares raw and cycle-adjusted series, and uses circular shifts to estimate a bias control. It relates those statistics to benchmark win counts already recorded in the script.

```bash
python srp_mutual_information_test.py --data-dir ./dataset \
  --output-dir ./mi_srp_results --bins 8 --sample-size 8000 \
  --pair-cap 2500 --shift-controls 3 --seed 2024
```

The spread across datasets is exploratory. The analysis covers seven datasets and measures association at the dataset level. [Analysis implementation](srp_mutual_information_test.py) · [Slurm wrapper](scripts/experiments/analysis/run_srp_mutual_information.sh)

## Read the results

A standalone training run writes a checkpoint under `checkpoints/` by default, prints the best validation checkpoint and final test MSE and MAE, and appends the test metrics to `result.txt`. Grid scripts maintain their own result files. For Electricity, look under `fixed_lookback_grid/electricity/`.

| File or directory | Contents |
| --- | --- |
| `coverage.csv` | Completed and expected trial counts by model, lookback, and horizon. |
| `all_trials.csv` | Metrics and configuration for every verified trial. |
| `winning_configurations.csv` | Lowest-validation-MSE configuration within each completed model, lookback, and horizon group, with test results for that same configuration. |
| `provisional_winners.csv` | Best observed configurations with completion status, including partially searched groups. |
| `logs/`, `results/`, `checkpoints/` | Per-trial logs, verified JSON records, and trained weights. |

Use validation MSE to select configurations and report held-out test MSE and MAE for those configurations. When comparing against CycleNet, match the dataset split, lookback, horizon, normalization setting, and seed. Check grid coverage and search sizes when interpreting comparisons. The Electricity grid compares the three mixing variants at a single seed. Use a matched CycleNet baseline and additional seeds to assess the size and consistency of any improvement.

## Repository layout

| Path | Purpose |
| --- | --- |
| [run.py](run.py) | Training and evaluation CLI, configuration, and seeding. |
| [exp/](exp/) | Model registry, training loops, validation, checkpointing, and evaluation. |
| [models/](models/) and [layers/](layers/) | Forecasting models and building blocks. |
| [data_provider/](data_provider/) | File loaders, chronological splits, windowing, and cycle indices. |
| [utils/](utils/) | Metrics, time features, and training utilities. |
| [scripts/experiments/](scripts/experiments/) | Research launchers by dataset, plus baseline and analysis scripts. |
| [scripts/](scripts/) and [old_slurm/](old_slurm/) | Baseline recipes and historical launchers. |
| `fixed_lookback_grid/`, `dependency_scores/`, `mi_srp_results/` | Outputs from grid searches, dependency estimation, and mutual-information analysis. |

## Citation

```bibtex
@inproceedings{cyclenet,
  title={CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns},
  author={Lin, Shengsheng and Lin, Weiwei and Hu, Xinyi and Wu, Wentai and Mo, Ruichao and Zhong, Haocheng},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024}
}
```

Upstream project resources include the [poster](https://drive.google.com/file/d/1dBnmrjtTab4M5L9qfrdAp2Y_r53hBZ1/view), [slides](https://drive.google.com/file/d/1QcCCXRFtnFYPtaXmZiF4Zby5r-nl5tR5/view), and [TQNet successor project](https://github.com/ACAT-SCUT/TQNet).

The upstream repository credits SparseTSF, iTransformer, SegRNN, PatchTST, LTSF-Linear, Informer, Autoformer, FEDformer, Pyraformer, RevIN, and tsai for related code and datasets. [Apache License 2.0](LICENSE)
