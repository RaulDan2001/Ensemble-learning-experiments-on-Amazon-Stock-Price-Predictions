# Ensemble Learning for Amazon Stock Price Prediction

> Neural network ensemble (MLP, LSTM, GRU, BiLSTM) trained on 29 years of Amazon (AMZN) daily OHLCV data to predict next-day closing prices. Implements subset-based ensemble evaluation, mean and weighted aggregation, and short/long-term iterative forecasting.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Ensemble Strategies](#ensemble-strategies)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Graphs](#graphs)
- [Findings & Observations](#findings--observations)
- [Requirements](#requirements)

---

## Overview

This project explores **ensemble learning applied to financial time-series regression**. Five neural network architectures with different inductive biases (shallow MLP, deep MLP, LSTM, GRU, Bidirectional LSTM) are trained on Amazon stock data and combined using two aggregation strategies:

- **Mean ensemble** — uniform average of all model predictions
- **Weighted ensemble** — inverse-validation-MSE weighting (better models vote more)

The training data is also deliberately split into two chronological halves (Subset A = older data, Subset B = newer data) to measure how well ensembles generalise across time periods — a critical property for financial models.

A single **Baseline SmallMLP** (different random seed) trained on all data serves as the "what you'd get from one model" reference point.

---

## Key Results

### Experiment B — Full Training Set (Test Set Evaluation)

| Model | MSE (USD²) | MAE (USD) |
|---|---:|---:|
| SmallMLP | 85.30 | 7.24 |
| DeepMLP | 120.35 | 9.11 |
| LSTM | 137.79 | 8.60 |
| GRU | 88.97 | 7.15 |
| **BiLSTM** | **26.15** | **4.02** |
| Mean Ensemble | 58.85 | 6.00 |
| Weighted Ensemble | 62.14 | 5.92 |
| Baseline MLP | 118.43 | 8.75 |

**The ensemble reduces test MAE by 31% compared to the baseline single model.**

### Iterative Forecasting (from Jan 28, 2026)

| Horizon | Forecast |
|---|---|
| Short-term (5 days) | $227.05 → $222.89 |
| Long-term (30 days) | $227.05 → $219.60 |

---

## Project Structure

```
Proiect2/
├── README.md
├── requirements.txt
├── dataset/
│   └── Amazon_stock_data.csv        # 7,202 rows of daily OHLCV data (1997–2026)
├── src/
│   ├── data_preprocessing.py        # Data loading, feature engineering, DataLoaders
│   ├── models.py                    # 5 model class definitions + factory
│   ├── train.py                     # Training loop with early stopping
│   ├── ensemble.py                  # Aggregation helpers and evaluation utilities
│   └── main.py                      # Full experiment orchestration + graph generation
├── plots/                           # Auto-created; all 7 output PNGs saved here
│   ├── loss_curves.png
│   ├── test_predictions.png
│   ├── subsetA_metrics.png
│   ├── subsetB_metrics.png
│   ├── ensemble_vs_baseline.png
│   ├── short_term_forecast.png
│   └── long_term_forecast.png
├── logs/
│   └── main.log                     # Full console output from the last run
└── docs/
    ├── technical_documentation.md   # In-depth architecture and methodology
    └── project_plan.md              # Original requirements and implementation plan
```

---

## Dataset

**File:** `dataset/Amazon_stock_data.csv`

| Property | Value |
|---|---|
| Source | Amazon (AMZN) daily stock data |
| Date range | 1997-05-15 → 2026-01-28 |
| Rows | 7,202 trading days |
| Columns | Date, Close, High, Low, Open, Volume |
| Missing values | None |

### Feature Engineering

Three rolling statistics are derived from the raw `Close` column before any normalisation:

| Feature | Description |
|---|---|
| `MA_5` | 5-day rolling mean of Close |
| `MA_20` | 20-day rolling mean of Close |
| `Std_5` | 5-day rolling standard deviation of Close |

Combined with the 5 raw OHLCV columns this gives **8 features** per timestep.

### Splits (chronological — no shuffling)

| Split | Rows | Purpose |
|---|---|---|
| Train (full) | 5,731 sequences | Model training |
| Subset A | 2,865 sequences | Older half of training data |
| Subset B | 2,866 sequences | Newer half of training data |
| Validation | 720 sequences | Early stopping monitor |
| Test | 721 sequences | Final held-out evaluation |

---

## Models

All models receive sequences of shape `(batch, 30, 8)` — 30 timesteps × 8 features — and output a single scalar (next-day normalised Close price).

| # | Name | Type | Architecture | Seed |
|---|---|---|---|---|
| 1 | `SmallMLP` | MLP | Flatten → Linear(240→64) → ReLU → Dropout(0.2) → Linear(64→32) → ReLU → Linear(32→1) | 42 |
| 2 | `DeepMLP` | MLP | Flatten → Linear(240→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) | 123 |
| 3 | `LSTMModel` | LSTM | LSTM(input=8, hidden=64, layers=1) → FC(64→1), last timestep hidden state | 42 |
| 4 | `GRUModel` | GRU | GRU(input=8, hidden=64, layers=1) → FC(64→1), last timestep hidden state | 42 |
| 5 | `BiLSTMModel` | BiLSTM | BiLSTM(input=8, hidden=64, layers=1, bidirectional=True) → FC(128→1), concatenated final states | 42 |

**Baseline:** A sixth `SmallMLP` with `seed=999`, trained on the full dataset — represents a single-model ceiling.

---

## Ensemble Strategies

### Mean Ensemble

$$\hat{y}_{mean} = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_i$$

Every model contributes equally regardless of individual performance.

### Weighted Ensemble

$$w_i = \frac{1/\text{MSE}_{val,i}}{\sum_j 1/\text{MSE}_{val,j}}, \qquad \hat{y}_{weighted} = \sum_{i=1}^{N} w_i \hat{y}_i$$

Models with lower validation MSE receive proportionally higher weight. Weights are normalised to sum to 1.

**Weights from Experiment B:**

| Model | Weight |
|---|---|
| SmallMLP | 0.0674 |
| DeepMLP | 0.0576 |
| LSTM | 0.2663 |
| GRU | 0.3541 |
| BiLSTM | 0.2545 |

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (falls back to CPU automatically)

### Create environment

```bash
conda create -n ensemble-stock python=3.10
conda activate ensemble-stock
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
matplotlib
scikit-learn
scipy
torch
torchvision
```

---

## Usage

All scripts are run from the **project root** directory.

### Run the full experiment

```bash
python src/main.py
```

This will:
1. Load and preprocess the dataset
2. Run Experiment A (subset training × 2, 10 models total)
3. Run Experiment B (full training, 5 models + 1 baseline)
4. Generate iterative short-term and long-term forecasts
5. Save all 7 graphs to `plots/`

Expected runtime: **~10–20 minutes** with GPU, ~45–60 minutes CPU-only.

### Run individual module smoke tests

```bash
python src/data_preprocessing.py   # verify data loading and splits
python src/models.py               # verify all 5 model architectures
python src/train.py                # 3-epoch training smoke test
python src/ensemble.py             # 5-epoch evaluation smoke test
```

---

## Experiments

### Experiment A — Cross-Subset Generalization

**Goal:** Measure how well an ensemble trained on half the data generalises to the other (chronologically disjoint) half.

**Protocol:**
1. Train all 5 models on **Subset A** → evaluate on both Subset A and Subset B
2. Train all 5 models on **Subset B** → evaluate on both Subset A and Subset B
3. Compare per-model and ensemble metrics across both evaluation sets

**Key finding:** Models trained on Subset A (1997–~2009, low price era) collapse when evaluated on Subset B (~2009–2021, high price era). Recurrent models (LSTM, GRU, BiLSTM) generalise better because they learn relative temporal patterns rather than absolute price levels.

### Experiment B — Full Training & Baseline Comparison

**Goal:** Demonstrate ensemble benefit over a single baseline model trained on identical data.

**Protocol:**
1. Train all 5 models on the full training set
2. Evaluate each on the held-out test set
3. Compute mean and weighted ensemble predictions
4. Train a single Baseline SmallMLP (seed=999)
5. Compare all predictors

**Key finding:** The mean ensemble achieves a **50% reduction in MSE** compared to the Baseline MLP (58.85 vs 118.43 USD²).

---

## Graphs

All graphs are saved to `plots/` at 150 DPI.

| File | Description |
|---|---|
| `loss_curves.png` | 2×3 grid of train/val MSE loss curves for all 5 models + baseline |
| `test_predictions.png` | Actual vs all 5 individual models + both ensembles + baseline on the test set |
| `subsetA_metrics.png` | Bar chart: SubsetA-trained models, MSE evaluated on SubsetA vs SubsetB |
| `subsetB_metrics.png` | Bar chart: SubsetB-trained models, MSE evaluated on SubsetA vs SubsetB |
| `ensemble_vs_baseline.png` | Actual vs mean ensemble vs weighted ensemble vs baseline on the test set |
| `short_term_forecast.png` | 60 days of history + 5-day autoregressive ensemble forecast |
| `long_term_forecast.png` | 60 days of history + 30-day autoregressive ensemble forecast |

---

## Findings & Observations

| Finding | Explanation |
|---|---|
| **BiLSTM dominates individually** (MSE=26.15) | Bidirectional attention over both past and future context within the window captures non-causal price symmetries (support/resistance, reversals) |
| **Ensemble beats baseline by 50%** | Diverse architectures make uncorrelated errors; averaging suppresses individual model noise |
| **Mean ensemble outperforms weighted** in Experiment B | Weights are computed from normalised-space val MSE; GRU received highest weight (0.35) despite BiLSTM being the best in USD space |
| **Cross-subset generalisation fails for MLPs** | MLPs memorise absolute price levels; recurrent models learn scale-invariant temporal patterns |
| **Long-term forecast drifts to a mean** | Autoregressive models compound uncertainty — each predicted step becomes the next input, causing predictions to regress toward recent mean prices over 30 steps |

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
scipy>=1.11
torch>=2.0
torchvision>=0.15
```

---

## Academic Context

This project was developed as part of the **Neural Computing** (Calcul Neuronal) master's-level course. The primary learning objectives covered are:

- Ensemble construction and diversity
- Split-based subset evaluation methodology
- Mean vs. weighted aggregation analysis
- Short-term and long-term multi-step forecasting
- Comparison of ensemble vs. single-model baseline
