# LOB Benchmark

An extensible research framework for short-horizon limit order book (LOB) prediction, with a focus on **large-scale data processing, memory-efficient HDF5 datasets, and distributed deep learning**.

The benchmark covers market-data preparation, causal feature engineering, regression and classification targets, hyperparameter tuning, and evaluation on chronologically held-out data. It currently supports historical **Bybit** order book data and implements linear models, LightGBM, and a temporal convolutional network (TCN). Its dataset and model interfaces are designed to accommodate other data providers and prediction architectures.

## Overview

```text
Bybit historical order book updates (snapshots + deltas)
    -> reconstruct order books and compute causal features
    -> write chunked HDF5 files and future-return targets
    -> stream tabular observations or historical LOB windows
    -> tune models on chronological training/validation splits
    -> evaluate on held-out test dates
```

Key capabilities:

- **Scalable preprocessing:** incrementally reconstruct order books from daily Bybit archives and write chunked HDF5 files without retaining full histories in RAM.
- **Efficient data loading:** select raw and engineered features, stream HDF5 batches, or generate sequence windows on demand without materializing all overlapping windows.
- **Distributed training:** train PyTorch sequence models on a single device or with DistributedDataParallel (DDP) and Fully Sharded Data Parallel (FSDP2), including rank-aware data partitioning and distributed evaluation.
- **Multiple forecasting tasks:** regress future mid-price returns or classify price movements using a configurable threshold in basis points.
- **Extensible experiments:** shared dataset, training, tuning, and evaluation interfaces for classical ML and neural sequence models.

## Data pipeline

### Bybit processing

`scripts/setup_bybit_dataset.py` downloads or accepts local daily Bybit order book archives. The processor reconstructs the book from snapshot and delta messages, tracks uninterrupted segments, and computes features in batches. A separate HDF5 file is created for each continuous segment so historical windows do not cross reconstruction resets.

The processed files are stored in `DATA_DIR/bybit/processed/`. Each `.h5` file contains:

| Dataset or attribute | Description |
| --- | --- |
| `raw_features` | Bid/ask prices and sizes at the selected book depth |
| `engineered_features` | Causal market microstructure features |
| `targets` | Future mid-price returns for the configured event horizons |
| `valid_book` | Validity flags for reconstructed order book states |
| `metadata/` | Timestamps, update identifiers, mid-prices, and target-end information |
| HDF5 attributes | Feature/target names, horizons, depth, symbol, date, and source information |

Engineered features include spread and microprice deviation, bid/ask notional depth, depth imbalance, order flow imbalance (OFI), lagged returns, rolling volatility, interarrival time, and order book update rates. The processor supports configurable book depth, aggregation levels, lookback windows, and prediction horizons.

The current data source is Bybit BTCUSDT linear contracts, but **other data providers can be integrated by exporting the same HDF5 schema**. The dataset and training code are independent of the original provider.

### Dataset interfaces

`src/data/datasets.py` provides two interfaces over the processed HDF5 files:

| Dataset | Use case | Loading behavior |
| --- | --- | --- |
| `TabularDataset` | Linear models and LightGBM | Streams selected feature batches or loads selected observations into arrays |
| `SequenceDataset` | TCN and other neural sequence models | Reads historical windows on demand and yields prebatched tensors; supports distributed ranks and data-loader workers |

The tabular trainer supports `--data-mode load` and `--data-mode stream`. The latter stages streamed observations in disk-backed arrays before fitting, reducing the need to materialize the input dataset as one in-memory array. Individual estimators may still allocate additional memory during fitting.

For sequence models, the dataset partitions batches across distributed processes, supports deterministic sampling, and preserves sequence boundaries. Normalization statistics are computed from training data and reused for validation and testing.

### Prediction tasks

The mid-price and return target at an event horizon `H` are defined as:

```text
mid(t) = (best_bid(t) + best_ask(t)) / 2
return_bps(t, H) = 10,000 * (mid(t + H) / mid(t) - 1)
```

- **Regression:** predict `return_bps(t, H)` directly.
- **Classification:** classify the same return as **down (0)** if it is below `-threshold`, **neutral (1)** if it falls within `[-threshold, +threshold]`, or **up (2)** if it exceeds `+threshold`.

Prediction horizons are measured in **order book events**, not elapsed seconds. The default processed targets use 10, 50, and 100 events; a specific target is selected at training time with `--target-name` (for example, `target_return_50_bps`).

### Chronological evaluation

The tuning scripts group compatible HDF5 parts by date and split dates chronologically. By default, they use **8 training days, 1 validation day, and 1 test day**, configurable through `--n-train-days`, `--n-val-days`, and `--n-test-days`. Hyperparameters are selected using validation loss, and the chosen model is then evaluated on the held-out test split.

Regression evaluation includes MSE, MAE, and R²; classification includes log loss, accuracy, balanced accuracy, and macro-F1. Experiments save trial configurations, evaluation metadata, and the selected model checkpoint.

## Models and training

| Model | Input | Tasks | Training |
| --- | --- | --- | --- |
| Linear (Ridge / logistic regression) | Engineered tabular features | Regression, classification | Single process |
| LightGBM | Engineered tabular features | Regression, classification | Single process |
| TCN | Historical LOB windows, optionally with engineered features | Regression, classification | Single device, DDP, FSDP2 |

The TCN is a causal, residual temporal convolutional network. Its tuning workflow first selects the input representation and window length, then searches architecture and optimization settings. The sequence trainer supports gradient accumulation, early stopping, gradient clipping, learning-rate scheduling, checkpointing, and optional BF16 precision on supported CUDA hardware.

To add a model, implement `TabularModel` or extend `SequenceModel` in `src/models/`, add a tuning entry point in `src/tuning/`, and register it in `src/tuning/__init__.py`. The existing HDF5 datasets, trainers, and evaluation utilities can then be reused.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/data/bybit/` | Bybit order book reconstruction, feature engineering, and HDF5 export |
| `src/data/datasets.py` | Source-independent tabular and sequence HDF5 datasets |
| `src/models/` | Model implementations and common interfaces |
| `src/trainers/` | Tabular/sequence training, metrics, optimization, and DDP/FSDP support |
| `src/tuning/` | Model-specific hyperparameter searches and chronological experiment workflow |
| `scripts/setup_bybit_dataset.py` | CLI for preparing Bybit datasets |
| `scripts/tune_model.py` | CLI for tuning and evaluating registered models |
| `src/**/*_test.py` | Dataset, trainer, and distributed integration tests |

## Running the code

### 1. Installation

From the repository root:

```bash
git clone https://github.com/TuriAndrade/lob_benchmark.git
cd lob_benchmark

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp example.env .env
```

Configure `.env`:

```dotenv
DATA_DIR=/path/to/data
RESULTS_DIR=/path/to/results
```

### 2. Prepare Bybit data

Process the previous ten days of Bybit BTCUSDT order book archives using the default settings:

```bash
python -m scripts.setup_bybit_dataset
```

Or customize the features and horizons:

```bash
python -m scripts.setup_bybit_dataset \
  --depth 10 \
  --depth-levels 1 5 10 \
  --return-lags 1 10 50 \
  --windows 10 50 100 \
  --horizons 10 50 100 \
  --batch-size 8192
```

The script also accepts Bybit archive URLs or local daily `.data.zip` files as positional arguments. Use `--overwrite` to regenerate processed files. The HDF5 files will be written to `/path/to/data/bybit/processed/` when `DATA_DIR=/path/to/data`.

### 3. Tune and evaluate models

All examples use the directory containing the processed `.h5` files. Sample limits are optional; omit them to use all eligible observations.

**Linear regression (disk-backed tabular data):**

```bash
python -m scripts.tune_model \
  --model linear \
  --data-path /path/to/data/bybit/processed \
  --task regression \
  --target-name target_return_50_bps \
  --data-mode stream \
  --n-train-samples 20000 \
  --n-val-samples 1000 \
  --n-test-samples 1000
```

**LightGBM classification (0.25 bps threshold):**

```bash
python -m scripts.tune_model \
  --model lightgbm \
  --data-path /path/to/data/bybit/processed \
  --task classification \
  --target-name target_return_50_bps \
  --threshold 0.25 \
  --data-mode stream \
  --n-train-samples 20000 \
  --n-val-samples 1000 \
  --n-test-samples 1000
```

**TCN with DDP on two GPUs:**

```bash
torchrun --standalone --nproc-per-node=2 -m scripts.tune_model \
  --model tcn \
  --data-path /path/to/data/bybit/processed \
  --task regression \
  --target-name target_return_50_bps \
  --training-mode ddp \
  --batch-size 256 \
  --precision fp32 \
  --n-train-samples 20000 \
  --n-val-samples 1000 \
  --n-test-samples 1000
```

**TCN with FSDP2 on two GPUs:**

```bash
torchrun --standalone --nproc-per-node=2 -m scripts.tune_model \
  --model tcn \
  --data-path /path/to/data/bybit/processed \
  --task regression \
  --target-name target_return_50_bps \
  --training-mode fsdp \
  --batch-size 256 \
  --precision fp32 \
  --n-train-samples 20000 \
  --n-val-samples 1000 \
  --n-test-samples 1000
```

Sequence training also supports `--training-mode single` (the default) and `--precision bf16` on CUDA GPUs with BF16 support. A run writes `trials.json`, `results.json`, and either `model.joblib` or `model.pt` under `RESULTS_DIR/<model>/<experiment>/`; `--save-dir` changes the destination and `--override` allows reusing an existing output directory.

### 4. Run tests

Run the dataset, tabular trainer, single-device, and distributed DDP tests:

```bash
python -m pytest -v -s
```

Enable the optional CPU FSDP integration test:

```bash
LOB_TEST_FSDP_CPU=1 python -m pytest -v -s
```

The tests use generated HDF5 fixtures. To additionally run the optional dataset-throughput test against a processed market-data file:

```bash
LOB_BENCHMARK_PATH=/path/to/data/bybit/processed/example.h5 \
  python -m pytest -v -s src/data/datasets_test.py
```

Replace `example.h5` with an actual processed file name.

## TODO

- [ ] **LOB architectures:** Implement and tune TLOB and other state-of-the-art methods for LOB prediction.
- [ ] **Trading policies:** Implement and evaluate trading policies driven by model predictions, accounting for fees, spread, slippage, latency, and execution constraints.
