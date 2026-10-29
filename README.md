# LOB Benchmark

This repository contains tools for testing short-horizon price prediction models on limit order book (LOB) data. It covers downloading and processing market data, preparing tabular and sequential inputs, tuning models, and evaluating them on held-out dates. It also supports efficient preprocessing and batch-wise data loading, allowing large datasets to be processed without loading them entirely into memory.

The current setup uses BTCUSDT snapshots from Binance Futures, obtained through [Tardis](https://tardis.dev/), but is designed to accommodate additional data providers. It supports regression of future mid-price returns and classification of price movements into down, neutral, and up. The models implemented so far are linear models, LightGBM, and a temporal convolutional network (TCN).

## How it works

```text
Tardis snapshots
    -> download and validate files
    -> compute features and targets with Polars
    -> save processed data as HDF5
    -> load tabular features or historical LOB windows
    -> tune models on training/validation data
    -> evaluate on a held-out test date
```

The code is split into a few modules:

| Path | Contents |
| --- | --- |
| `src/data/src_tardis.py` | Data download, preprocessing, target generation, and HDF5 export |
| `src/data/schema.py`, `src/data/features.py` | Column definitions and feature calculations |
| `src/data/datasets.py` | Tabular and sequence datasets, sampling, and normalization |
| `src/data/datasets_test.py` | Dataset correctness and throughput tests |
| `src/models/` | Model implementations and base interfaces |
| `src/trainers/` | Tabular and sequence training, evaluation, and DDP/FSDP support |
| `src/tuning/` | Model-specific hyperparameter searches and experiment orchestration |
| `scripts/setup_lob_snapshot_dataset.py` | Download and preprocess LOB snapshots into HDF5 datasets |
| `scripts/tune_model.py` | CLI entry point for tuning and evaluating registered models |

## Data

### Preprocessing

The data processor downloads Tardis book_snapshot_5 or book_snapshot_25 files, validates the gzip archives, and computes features and prediction targets using Polars. Preprocessing uses lazy execution and streaming to process large files without loading them entirely into memory, writing chunked HDF5 (`.h5`) files. The dataset interfaces support batch-wise reading, and sequence windows are generated incrementally, allowing deep learning models to train on datasets larger than available RAM. The HDF5 files contain transformed raw book features, engineered features, prediction targets, validity flags, timestamps, event indices, and other metadata.

The available features include bid/ask spread, depth and depth imbalance at several levels, top-of-book order flow imbalance (OFI) and rolling OFI, microprice deviation, lagged returns, volatility, interarrival time, and event rate.

There are two dataset classes:

- `TabularDataset` loads engineered features for models such as linear regression and LightGBM. It reads HDF5 data in batches and can sample a fixed number of observations. The tabular trainer supports in-memory fitting (`load`) and disk-backed arrays (`stream`).
- `SequenceDataset` produces windows of past LOB states, with optional engineered features. It reads HDF5 data in batches and yields sequence batches without keeping all overlapping windows in memory. Windows do not cross file boundaries.

### Prediction targets

The midpoint is the average of the best bid and best ask prices. For an event `t` and a horizon `H`, the return target is:

```text
mid(t) = (best_bid(t) + best_ask(t)) / 2
return_bps(t, H) = 10,000 * (mid(t + H) / mid(t) - 1)
```

In regression, the model predicts this return directly. In classification, a threshold in basis points determines the label:

| Label | Condition |
| --- | --- |
| Down (0) | Return is below `-threshold_bps` |
| Neutral (1) | Return is between `-threshold_bps` and `+threshold_bps`, inclusive |
| Up (2) | Return is above `+threshold_bps` |

The default horizons are 10, 50, and 100 **events**, not fixed amounts of time.

### Temporal split and leakage checks

By default, the tuning scripts sort compatible processed files by date and use the first four dates for training, the fifth for validation, and the sixth for testing. These split sizes can be changed with `--n-train-days`, `--n-val-days`, and `--n-test-days`. With the default download URLs, these are the first days of September 2020 through February 2021. They are six separate dates, not six consecutive trading days.

Inputs contain current and past observations; future mid-prices are used to build targets. Sequence windows are aligned to the event being predicted. Each sample also retains the observation index and timestamp, along with the event and timestamp at the end of its prediction horizon. Normalization statistics are calculated from the training files and reused for validation and testing.

Invalid book states and nonfinite inputs are filtered out. The dataset tests check target alignment, sequence windows, deterministic sampling, normalization, and whether changing later stored features affects earlier inputs. These checks cover specific sources of leakage; they do not rule out every possible issue in market-data preprocessing.

## Models and evaluation

| Model | Inputs | Tasks |
| --- | --- | --- |
| Ridge / logistic regression | Engineered features | Regression and classification |
| LightGBM | Engineered features | Regression and classification |
| TCN | Historical LOB windows, optionally with engineered features | Regression and classification |

Sequence models support single-device training and distributed training with PyTorch DistributedDataParallel (DDP) and Fully Sharded Data Parallel (FSDP). Tabular model training is single-process.

Each model has a tuning script with a parameter grid. Models are selected using validation loss and then evaluated once on the held-out test split. LightGBM and the TCN support early stopping. Runs save the tested configurations, selected hyperparameters, evaluation metrics, normalization information, and fitted model.

For regression, the reported metrics are MSE, MAE, and R². Classification uses log loss, accuracy, balanced accuracy, and macro-F1.

## Running the code

### Setup

```bash
git clone https://github.com/TuriAndrade/lob_benchmark.git
cd lob_benchmark

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `example.env` to `.env` and set the data and output paths:

```dotenv
DATA_DIR=/path/to/data
RESULTS_DIR=/path/to/results
```

### Process snapshots

From the repository root:

```bash
python -m scripts.setup_lob_snapshot_dataset
```

By default, this downloads six BTCUSDT depth-5 snapshot dates (the first day of each month from September 2020 through February 2021) and prepares targets for horizons 10, 50, and 100 events. You can supply other Tardis snapshot URLs as positional arguments and customize processing with `--horizons`, `--tick-size`, `--depth-levels`, `--return-lags`, `--ofi-windows`, `--volatility-window`, and `--event-rate-window`. Use `--overwrite` to regenerate existing outputs.

Processed files are written to `DATA_DIR/tardis_lob_snapshot/processed/binance-futures/BTCUSDT/` as `.h5` files. Each file stores raw and engineered features, targets, validity flags, metadata, and the preprocessing configuration in HDF5 attributes.

### Tune and evaluate

All tuning commands must include `--data-path`, pointing to the directory containing the processed `.h5` files (replace `/path/to/data` below with your configured `DATA_DIR`). By default, the scripts select four training dates, one validation date, and one test date; `--train-samples`, `--val-samples`, and `--test-samples` optionally limit the number of observations in each split.

Linear regression:

```bash
python -m scripts.tune_model \\
  --model linear \\
  --data-path /path/to/data/tardis_lob_snapshot/processed/binance-futures/BTCUSDT \\
  --task regression \\
  --horizon 50 \\
  --train-samples 20000 \\
  --val-samples 1000 \\
  --test-samples 1000
```

LightGBM classification:

```bash
python -m scripts.tune_model \\
  --model lightgbm \\
  --data-path /path/to/data/tardis_lob_snapshot/processed/binance-futures/BTCUSDT \\
  --task classification \\
  --horizon 50 \\
  --threshold 0.25 \\
  --train-samples 20000 \\
  --val-samples 1000 \\
  --test-samples 1000
```

Distributed TCN training with DDP on two GPUs:

```bash
torchrun --standalone --nproc-per-node=2 -m scripts.tune_model \\
  --model tcn \\
  --data-path /path/to/data/tardis_lob_snapshot/processed/binance-futures/BTCUSDT \\
  --task regression \\
  --horizon 50 \\
  --mode ddp \\
  --device auto \\
  --window-size 100 \\
  --batch-size 256 \\
  --train-samples 20000 \\
  --val-samples 1000 \\
  --test-samples 1000
```

For FSDP, use the same command with `--mode fsdp`. Sequence models also support `--mode single` (the default), as well as `--stride`, `--precision {fp32,bf16}`, `--include-engineered`, `--shuffle`/`--no-shuffle`, and `--drop-last`/`--no-drop-last`. If using `--normalize` for a sequence model, also set `--include-engineered`.

Tabular models offer `--data-mode {load,stream}` (default: `load`); `stream` stages data in disk-backed arrays during fitting. Common options include `--seed`, `--read-batch-size`, `--depth`, `--n-train-days`, `--n-val-days`, `--n-test-days`, `--normalize`/`--no-normalize`, `--save-dir`, and `--override`.

Runs save `trials.json`, `results.json`, and the selected fitted model (`model.joblib` for tabular models or `model.pt` for TCN). By default, outputs go under `RESULTS_DIR/<model>/`; `--save-dir` overrides this. Existing run directories cause an error unless `--override` is specified.

### Tests

The tests create their own small HDF5 fixtures, so preprocessing real market data is not required for the standard test suite.

```bash
python -m pytest -v -s src/data/datasets_test.py
python -m pytest -v -s src/trainers/tabular_trainer_test.py
python -m pytest -v -s src/trainers/sequence_trainer_test.py
```

The sequence trainer tests include two-process CPU DDP integration tests. CPU FSDP testing is opt-in:

```bash
LOB_TEST_FSDP_CPU=1 python -m pytest -v -s src/trainers/sequence_trainer_test.py
```

To additionally benchmark dataset throughput against a real processed HDF5 file:

```bash
LOB_BENCHMARK_PATH=/path/to/processed/file.h5 python -m pytest -v -s src/data/datasets_test.py
```


## Adding a model

Tabular models implement `TabularModel` and sequence models extend `SequenceModel`. To add a model, implement the corresponding interface, create a tuning entry point under `src/tuning/`, and register it in `src/tuning/__init__.py`. The existing dataset and evaluation code can then be reused.

## Roadmap

- [ ] **Models:** Implement DeepLOB, MLPLOB, TLOB, and other LOB models, with more extensive hyperparameter tuning and baseline comparisons.
- [ ] **Datasets:** Run experiments on more dates and additional instruments, and evaluate how results change across market conditions.
- [ ] **Efficiency:** Benchmark training time, inference latency, throughput, memory use, and model size.
- [ ] **Trading strategies:** Implement and evaluate strategies based on model predictions, accounting for transaction costs and execution constraints.

This is a research benchmark, not a live trading system. Mid-price prediction metrics do not account for whether trades can be executed profitably after spread, fees, slippage, and latency.
