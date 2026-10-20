# LOB Benchmark

This repository contains tools for testing short-horizon price prediction models on limit order book (LOB) data. It covers downloading and processing market data, preparing tabular and sequential inputs, tuning models, and evaluating them on held-out dates. It also supports efficient preprocessing and batch-wise data loading, allowing large datasets to be processed without loading them entirely into memory.

The current setup uses BTCUSDT snapshots from Binance Futures, obtained through [Tardis](https://tardis.dev/). It supports regression of future mid-price returns and classification of price movements into down, neutral, and up. The models implemented so far are linear models, LightGBM, and a temporal convolutional network (TCN).

The results reported below are from a small smoke test of the pipeline. They are not intended as a full comparison of the models.

## How it works

```text
Tardis snapshots
    -> download and validate files
    -> compute features and targets with Polars
    -> save processed data as Parquet
    -> load tabular features or historical LOB windows
    -> tune models on training/validation data
    -> evaluate on a held-out test date
```

The code is split into a few modules:

| Path | Contents |
| --- | --- |
| `src/data/src_tardis.py` | Data download, preprocessing, target generation, and Parquet export |
| `src/data/schema.py`, `src/data/features.py` | Column definitions and feature calculations |
| `src/data/datasets.py` | Tabular and sequence datasets, sampling, and normalization |
| `src/data/datasets_test.py` | Dataset and alignment tests |
| `src/models/` | Model implementations and base interfaces |
| `src/trainers/` | Training and evaluation code |
| `src/tuning/`, `src/tune.py` | Hyperparameter search and experiment scripts |

## Data

### Preprocessing

The data processor downloads Tardis book_snapshot_5 or book_snapshot_25 files, validates the gzip archives, and computes features and prediction targets using Polars. Preprocessing uses lazy execution and streaming to process large files without loading them entirely into memory, storing the results in compressed Parquet format. The dataset interfaces support batch-wise reading, and sequence windows are generated incrementally, allowing deep learning models to train on datasets larger than available RAM. The processed data includes raw book levels, timestamps, event indices, engineered features, and targets for the selected prediction horizons.

The available features include bid/ask spread, depth and depth imbalance at several levels, top-of-book order flow imbalance (OFI) and rolling OFI, microprice deviation, lagged returns, volatility, interarrival time, and event rate.

There are two dataset classes:

- `TabularDataset` loads engineered features for models such as linear regression and LightGBM. It can read Parquet data in batches and optionally sample a fixed number of observations. The tabular trainer currently loads the selected training arrays into memory for fitting.
- `SequenceDataset` produces windows of past LOB states, with optional engineered features. It reads Parquet data in batches and yields sequence batches without keeping all overlapping windows in memory. Windows do not cross file boundaries.

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

The tuning scripts sort the processed files by date and use the first four dates for training, the fifth for validation, and the sixth for testing. With the default download URLs, these are the first days of September 2020 through February 2021. They are six separate dates, not six consecutive trading days.

Inputs contain current and past observations; future mid-prices are used to build targets. Sequence windows are aligned to the event being predicted. Each sample also retains the observation index and timestamp, along with the event and timestamp at the end of its prediction horizon. Normalization statistics are calculated from the training files and reused for validation and testing.

Invalid book states and nonfinite inputs are filtered out. The dataset tests check target alignment, sequence windows, deterministic sampling, normalization, and whether changing later stored features affects earlier inputs. These checks cover specific sources of leakage; they do not rule out every possible issue in market-data preprocessing.

## Models and evaluation

| Model | Inputs | Tasks |
| --- | --- | --- |
| Ridge / logistic regression | Engineered features | Regression and classification |
| LightGBM | Engineered features | Regression and classification |
| TCN | Historical LOB windows, optionally with engineered features | Regression and classification |

Each model has a tuning script with a parameter grid. Models are selected using validation loss and then evaluated once on the held-out test split. LightGBM and the TCN support early stopping. Runs save the tested configurations, selected hyperparameters, evaluation metrics, normalization information, and fitted model.

For regression, the reported metrics are MSE, MAE, and R². Classification uses log loss, accuracy, balanced accuracy, and macro-F1.

## Initial experiments

I ran a small set of experiments with linear models and LightGBM to check that the complete workflow was working, from loading the data to saving evaluation results. Each run used **20,000 training samples, 1,000 validation samples, and 1,000 test samples**, sampled from the chronological partitions described above. The TCN is implemented but was not included in these runs.

The tables report **test-set** metrics for the configurations selected on validation data. Values are rounded to four decimal places.

### Regression

The target is future mid-price return in basis points. MAE is in basis points and MSE is in squared basis points.

| Horizon (events) | Model | MSE | MAE (bps) | R² |
| ---: | --- | ---: | ---: | ---: |
| 10 | Linear | 0.6061 | 0.4324 | 0.0995 |
| 10 | LightGBM | 0.6103 | 0.4385 | 0.0932 |
| 50 | Linear | 4.3877 | 1.4150 | 0.0381 |
| 50 | LightGBM | 4.3498 | 1.4086 | 0.0464 |
| 100 | Linear | 9.1636 | 2.1118 | 0.0191 |
| 100 | LightGBM | 9.0955 | 2.1059 | 0.0264 |

Both models obtained positive but low R² in this test. Linear regression was slightly better at 10 events, while LightGBM was slightly better at 50 and 100 events. The differences are small and were measured on one test date.

### Classification

Three-class classification was tested with movement thresholds of 0, 0.25, and 0.50 bps.

| Threshold (bps) | Horizon | Model | Log loss | Accuracy | Balanced accuracy | Macro-F1 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.00 | 10 | LightGBM | 0.8502 | 0.5911 | 0.5793 | 0.5797 |
| 0.00 | 10 | Linear | 0.8898 | 0.5534 | 0.5188 | 0.4985 |
| 0.00 | 50 | LightGBM | 0.8707 | 0.5706 | 0.4363 | 0.4307 |
| 0.00 | 50 | Linear | 0.9048 | 0.5117 | 0.4927 | 0.4714 |
| 0.00 | 100 | LightGBM | 0.8086 | 0.5619 | 0.3923 | 0.3830 |
| 0.00 | 100 | Linear | 0.8316 | 0.5246 | 0.3913 | 0.3994 |
| 0.25 | 10 | LightGBM | 0.8484 | 0.6183 | 0.4008 | 0.3830 |
| 0.25 | 10 | Linear | 0.8960 | 0.6064 | 0.3725 | 0.3317 |
| 0.25 | 50 | LightGBM | 1.0487 | 0.4614 | 0.4546 | 0.4484 |
| 0.25 | 50 | Linear | 1.0692 | 0.3915 | 0.4295 | 0.3915 |
| 0.25 | 100 | LightGBM | 0.9886 | 0.5019 | 0.3898 | 0.3651 |
| 0.25 | 100 | Linear | 1.0015 | 0.4563 | 0.4044 | 0.4042 |
| 0.50 | 10 | LightGBM | 0.7292 | 0.7107 | 0.3458 | 0.3037 |
| 0.50 | 10 | Linear | 0.7802 | 0.7083 | 0.3476 | 0.3095 |
| 0.50 | 50 | LightGBM | 1.1210 | 0.3150 | 0.3351 | 0.1631 |
| 0.50 | 50 | Linear | 1.1262 | 0.3622 | 0.3782 | 0.2867 |
| 0.50 | 100 | LightGBM | 1.0764 | 0.4262 | 0.4139 | 0.4116 |
| 0.50 | 100 | Linear | 1.0920 | 0.3683 | 0.3983 | 0.3677 |

Results vary considerably with horizon and threshold. In particular, accuracy can be high while macro-F1 is low when one class is much more common than the others. These runs are useful for checking the evaluation code, but more data and stronger baselines are needed before comparing models seriously.

## Running the code

### Setup

```bash
git clone https://github.com/TuriAndrade/lob_benchmark.git
cd lob_benchmark

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install tqdm
```

Copy `example.env` to `.env` and set the data and output paths:

```dotenv
DATA_DIR=/path/to/data
RESULTS_DIR=/path/to/results
```

### Process snapshots

```bash
python src/data/src_tardis.py
```

By default, this downloads six BTCUSDT depth-5 snapshot dates and prepares targets for horizons 10, 50, and 100. You can also pass different Tardis snapshot URLs and horizons. Processed Parquet files are saved under `DATA_DIR/tardis_book_snapshot/processed/`.

### Tune and evaluate

Linear regression:

```bash
python src/tune.py \
  --model linear_model \
  --task regression \
  --horizon 50 \
  --train-samples 20000 \
  --val-samples 1000 \
  --test-samples 1000
```

LightGBM classification:

```bash
python src/tune.py \
  --model lightgbm_model \
  --task classification \
  --horizon 50 \
  --threshold 0.25 \
  --train-samples 20000 \
  --val-samples 1000 \
  --test-samples 1000
```

The other registered model is `tcn_model`, which currently requires CUDA. The scripts save `trials.json`, `results.json`, and the fitted model (`model.joblib` or `model.pt`) under `RESULTS_DIR`. A run stops if its output directory already exists, so it must be moved or removed before running the same configuration again.

### Tests

```bash
pytest -q src/data/datasets_test.py
```

The tests use a sample from an existing processed Parquet file, so the data must be prepared first.

## Adding a model

Tabular models implement `TabularModel` and sequence models extend `SequenceModel`. To add a model, implement the corresponding interface, create a tuning entry point under `src/tuning/`, and register it in `src/tuning/__init__.py`. The existing dataset and evaluation code can then be reused.

## Roadmap

- [ ] **Models:** Implement DeepLOB, MLPLOB, TLOB, and other LOB models, with more extensive hyperparameter tuning and baseline comparisons.
- [ ] **Datasets:** Run experiments on more dates and additional instruments, and evaluate how results change across market conditions.
- [ ] **Efficiency:** Benchmark training time, inference latency, throughput, memory use, and model size.
- [ ] **Trading strategies:** Implement and evaluate strategies based on model predictions, accounting for transaction costs and execution constraints.

This is a research benchmark, not a live trading system. Mid-price prediction metrics do not account for whether trades can be executed profitably after spread, fees, slippage, and latency.
