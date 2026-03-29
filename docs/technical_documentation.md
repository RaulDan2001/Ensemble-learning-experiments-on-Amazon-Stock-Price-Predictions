# Technical Documentation

## Ensemble Learning for Amazon Stock Price Prediction

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architectures](#3-model-architectures)
4. [Training Procedure](#4-training-procedure)
5. [Ensemble Methods](#5-ensemble-methods)
6. [Iterative Forecasting Algorithm](#6-iterative-forecasting-algorithm)
7. [Experiment Protocols](#7-experiment-protocols)
8. [Results & Interpretation](#8-results--interpretation)
9. [Module Reference](#9-module-reference)

---

## 1. System Overview

The system is a **supervised regression pipeline** that predicts the next trading day's Amazon closing price from a 30-day window of OHLCV data and derived rolling statistics.

```
Raw CSV
  │
  ▼
data_preprocessing.py
  ├── Feature engineering (rolling MA/Std)
  ├── MinMaxScaler (fit on train only)
  ├── Sliding window sequencing  (N, 30, 8)
  └── DataLoaders  {subsetA, subsetB, full_train, val, test}
  │
  ▼
models.py  ──  5 model classes + set_seed + factory
  │
  ▼
train.py  ──  Adam + MSELoss + early stopping
  │
  ▼
ensemble.py  ──  evaluate_model / mean_aggregate / weighted_aggregate
  │
  ▼
main.py  ──  Experiment A | Experiment B | Forecasting | Graphs
  │
  ▼
plots/  ──  7 PNG files
```

**Framework:** PyTorch 2.6  
**Device:** CUDA (automatic fallback to CPU)  
**Language:** Python 3.10

---

## 2. Data Pipeline

### 2.1 Raw Data

The dataset (`Amazon_stock_data.csv`) contains 7,202 rows of daily trading data for Amazon (AMZN) spanning 1997-05-15 to 2026-01-28. Columns: `Date`, `Close`, `High`, `Low`, `Open`, `Volume`. No missing values.

### 2.2 Feature Engineering

Three rolling features are computed from the `Close` column **before** any normalisation:

| Feature | Formula | Window |
|---|---|---|
| `MA_5` | `Close.rolling(5).mean()` | 5 days |
| `MA_20` | `Close.rolling(20).mean()` | 20 days |
| `Std_5` | `Close.rolling(5).std()` | 5 days |

The first 20 rows that contain `NaN` (due to the MA_20 lookback) are dropped, leaving **7,182 usable rows**.

Final feature matrix shape after engineering: `(7182, 8)`

**Feature order (preserved throughout):**

| Index | Feature |
|---|---|
| 0 | Close |
| 1 | Open |
| 2 | High |
| 3 | Low |
| 4 | Volume |
| 5 | MA_5 |
| 6 | MA_20 |
| 7 | Std_5 |

### 2.3 Normalisation

A `MinMaxScaler` is fitted **exclusively on training rows** (rows 0 to `train_end`) to prevent data leakage. This scaler is then applied to the entire dataset.

A separate `scaler_y` is fitted on the training `Close` column only. This scaler is used everywhere predictions are inverse-transformed back to USD.

**Why separate scalers?**
- `scaler_X` normalises the 8-feature input matrix
- `scaler_y` normalises only the scalar target (next-day Close) — allows inverse-transforming predictions independently of the input features

### 2.4 Sliding Window Sequencing

A sliding window of length `SEQ_LEN = 30` creates supervised samples:

```
Sequence i:
  X[i] = features_scaled[i : i+30]       shape: (30, 8)
  y[i] = target_scaled[i + 30]           next-day normalised Close
```

Total sequences: `N = 7182 - 30 = 7152` → after boundary adjustments: **7,172 usable sequences**.

### 2.5 Chronological Splits

All splits are strictly chronological. **No shuffling is applied to any split** to preserve the temporal structure of the time series.

```
Total sequences: 7,172
├── Train (full): 5,731  (80%)
│   ├── Subset A: 2,865  (first 50% of train — older data)
│   └── Subset B: 2,866  (second 50% of train — newer data)
├── Validation:     720  (10%)
└── Test:           721  (10%)
```

Boundary mapping from raw rows to sequence indices:

```
seq_train_end = train_end - SEQ_LEN
seq_val_end   = val_end   - SEQ_LEN
```

This ensures that the sequence label `y[i]` (at raw row `i+30`) does not straddle a split boundary.

### 2.6 DataLoaders

Seven DataLoaders are produced by `load_data()`:

| Key | Indices | Shuffle | Purpose |
|---|---|---|---|
| `subsetA` | Subset A | False | Training in Experiment A |
| `subsetA_eval` | Subset A | False | Evaluation after training |
| `subsetB` | Subset B | False | Training in Experiment A |
| `subsetB_eval` | Subset B | False | Evaluation after training |
| `full_train` | All train | False | Training in Experiment B |
| `val` | Validation | False | Early stopping in all experiments |
| `test` | Test | False | Final held-out evaluation |

Batch size: 32 for all loaders.

---

## 3. Model Architectures

All models accept input of shape `(batch, 30, 8)` and output shape `(batch, 1)`.

### 3.1 SmallMLP (seed=42)

```
Input: (batch, 30, 8)
  → Flatten()                     → (batch, 240)
  → Linear(240, 64)               → (batch, 64)
  → ReLU()
  → Dropout(p=0.2)
  → Linear(64, 32)                → (batch, 32)
  → ReLU()
  → Linear(32, 1)                 → (batch, 1)

Parameters: ~17,537
```

Dropout at 0.2 is applied during training only; disabled during evaluation via `model.eval()`.

### 3.2 DeepMLP (seed=123)

```
Input: (batch, 30, 8)
  → Flatten()                     → (batch, 240)
  → Linear(240, 128)              → (batch, 128)
  → ReLU()
  → Dropout(p=0.3)
  → Linear(128, 64)               → (batch, 64)
  → ReLU()
  → Linear(64, 32)                → (batch, 32)
  → ReLU()
  → Linear(32, 1)                 → (batch, 1)

Parameters: ~41,217
```

Higher dropout (0.3) compensates for the larger capacity.

### 3.3 LSTMModel (seed=42)

```
Input: (batch, 30, 8)
  → LSTM(input_size=8, hidden_size=64, num_layers=1, batch_first=True)
    Output: (batch, 30, 64)  |  h_n: (1, batch, 64)
  → out[:, -1, :]            → (batch, 64)   [last timestep hidden state]
  → Linear(64, 1)            → (batch, 1)

Parameters: ~19,009
```

The last timestep's hidden state `out[:, -1, :]` encodes the full sequential context of the 30-day window. This is equivalent to using `h_n[0]` directly.

### 3.4 GRUModel (seed=42)

```
Input: (batch, 30, 8)
  → GRU(input_size=8, hidden_size=64, num_layers=1, batch_first=True)
    Output: (batch, 30, 64)  |  h_n: (1, batch, 64)
  → out[:, -1, :]            → (batch, 64)
  → Linear(64, 1)            → (batch, 1)

Parameters: ~14,273
```

GRU has fewer parameters than LSTM (no separate cell state) but achieves comparable performance on shorter sequences. It converged fastest in experimentation.

### 3.5 BiLSTMModel (seed=42)

```
Input: (batch, 30, 8)
  → BiLSTM(input_size=8, hidden_size=64, num_layers=1,
           batch_first=True, bidirectional=True)
    h_n: (2, batch, 64)
    h_n[0] = forward direction final state  (has seen timesteps 0→29)
    h_n[1] = backward direction final state (has seen timesteps 29→0)
  → cat([h_n[0], h_n[1]], dim=1)  → (batch, 128)
  → Linear(128, 1)                → (batch, 1)

Parameters: ~38,017
```

The bidirectional architecture allows the model to capture both upward momentum (forward pass) and reversal patterns (backward pass) simultaneously. This explains its strong performance (best individual model with MSE=26.15 USD²).

### 3.6 Reproducibility — set_seed()

```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

`set_seed()` is called immediately before each model is instantiated via `build_all_models()`. This ensures weight initialisation is reproducible across runs. Using different seeds for SmallMLP (42) and DeepMLP (123) intentionally introduces diversity in the ensemble.

---

## 4. Training Procedure

### 4.1 Loss Function

**Mean Squared Error (MSE):**

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

Computed in **normalised space** (values in [0, 1]). This avoids the loss being dominated by large absolute price values in the later years.

### 4.2 Optimiser

**Adam** with default betas (β₁=0.9, β₂=0.999) and `lr=1e-3`.

Adam was chosen over SGD because:
- Adaptive per-parameter learning rates work across all 5 architectures without per-model tuning
- Robust to sparse gradients (relevant for volume features that are orders of magnitude larger than price features before normalisation)

### 4.3 Early Stopping

```
best_val_loss  ← ∞
patience       ← 10
no_improve     ← 0

for each epoch:
    train and compute val_loss
    if val_loss < best_val_loss:
        best_val_loss ← val_loss
        save model weights (deepcopy)
        no_improve ← 0
    else:
        no_improve += 1
    if no_improve >= patience:
        stop; restore best weights
```

`copy.deepcopy(model.state_dict())` is used to take a true snapshot of weights — a reference copy would be overwritten by subsequent epochs.

Early stopping epochs observed in Experiment B:

| Model | Stopped at epoch | Best val MSE |
|---|---|---|
| SmallMLP | 16 | 0.005874 |
| DeepMLP | 25 | 0.006878 |
| LSTM | 25 | 0.001487 |
| GRU | 31 | 0.001119 |
| BiLSTM | 37 | 0.001557 |
| Baseline MLP | 26 | 0.007271 |

GRU and BiLSTM converged slowest but to the lowest val loss, explaining their high weights in the weighted ensemble.

### 4.4 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `EPOCHS` | 100 | Upper ceiling; early stopping typically fires well before this |
| `LR` | 1e-3 | Standard Adam default; works across all architectures |
| `PATIENCE` | 10 | Conservative — prevents overfitting but allows genuine slow convergence |
| `BATCH_SIZE` | 32 | Balances GPU utilisation with gradient noise (regularisation effect) |
| `SEQ_LEN` | 30 | Approx. one trading month — sufficient temporal context without excessive memory cost |

---

## 5. Ensemble Methods

### 5.1 evaluate_model()

```python
def evaluate_model(model, loader, scaler_y, device):
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(device))
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y.numpy())
    preds   = scaler_y.inverse_transform(np.vstack(all_preds))
    actuals = scaler_y.inverse_transform(np.vstack(all_actuals))
    mse = np.mean((preds - actuals) ** 2)
    mae = np.mean(np.abs(preds - actuals))
    return preds.flatten(), actuals.flatten(), mse, mae
```

The `scaler_y.inverse_transform()` converts normalised predictions back to USD before computing metrics — ensuring MSE and MAE are interpretable in real price units.

### 5.2 Mean Ensemble

$$\hat{y}_{mean}[t] = \frac{1}{5} \sum_{i=1}^{5} \hat{y}_i[t]$$

Implementation: `np.stack(predictions_list, axis=0).mean(axis=0)`

Stacking produces shape `(5, N)` and taking the mean along axis 0 yields `(N,)`.

### 5.3 Weighted Ensemble

$$w_i = \frac{1 / \text{MSE}_{val,i}}{\sum_{j=1}^{5} 1 / \text{MSE}_{val,j}}$$

$$\hat{y}_{weighted}[t] = \sum_{i=1}^{5} w_i \cdot \hat{y}_i[t]$$

Weights are computed once from the minimum (best) validation MSE each model achieved during training, then held constant for all test predictions.

**Important note:** Validation MSE is measured in normalised space. This means a model with a small normalised val MSE (e.g., GRU=0.00112) gets a high weight even if another model (BiLSTM) is actually better in USD space on the test set. This mismatch explains why the weighted ensemble did not outperform the mean ensemble in Experiment B.

### 5.4 Ensemble Weights (Experiment B)

| Model | Val MSE (norm) | Weight |
|---|---|---|
| SmallMLP | 0.005874 | 0.0674 |
| DeepMLP | 0.006878 | 0.0576 |
| LSTM | 0.001487 | 0.2663 |
| GRU | 0.001119 | 0.3541 |
| BiLSTM | 0.001557 | 0.2545 |

---

## 6. Iterative Forecasting Algorithm

Because the models were trained to predict **one step ahead**, multi-step forecasting requires an **autoregressive loop**: the prediction at step `t+1` is fed back as input at step `t+2`.

### 6.1 Seed Window

The last 30 rows of the known feature matrix `features_scaled[-30:]` form the initial seed window of shape `(30, 8)`.

### 6.2 Rolling Buffer

A buffer of the last 20 known normalised Close values `features_scaled[-20:, 0]` is maintained to allow MA_5, MA_20, and Std_5 to be recomputed at each forecast step. After each predicted step, the predicted value is appended to the buffer.

### 6.3 Loop

```
for step in range(horizon):
    1. Compute weighted ensemble prediction on current window (1, 30, 8)
    2. Append prediction to close_buffer
    3. Construct new_row:
         new_row[0] = pred_norm          (Close)
         new_row[1:5] = window[-1][1:5]  (Open/High/Low/Vol — frozen at last known)
         new_row[5] = mean(buffer[-5:])  (MA_5)
         new_row[6] = mean(buffer[-20:]) (MA_20)
         new_row[7] = std(buffer[-5:])   (Std_5)
    4. Slide window: window = vstack([window[1:], new_row])
5. inverse_transform all forecast_norm → USD
```

**Limitation:** Open, High, Low, and Volume are frozen at their last known values for all forecast steps. This is a practical approximation that becomes increasingly inaccurate at longer horizons. The slow convergence of long-term forecasts toward a mean (~$220) reflects compounding uncertainty rather than genuine price discovery.

---

## 7. Experiment Protocols

### 7.1 Experiment A — Cross-Subset Generalisation

**Motivation:** Financial data is non-stationary — the price range from 1997–2009 ($0.07–$30) is completely different from 2009–2026 ($5–$230). Can an ensemble trained on one era make useful predictions about the other?

**Protocol:**

1. `run_subset_experiment(train_key='subsetA', eval_keys=['subsetA_eval', 'subsetB_eval'])`
2. `run_subset_experiment(train_key='subsetB', eval_keys=['subsetA_eval', 'subsetB_eval'])`

Each call trains all 5 models fresh (new `build_all_models()` call with fixed seeds), evaluates on both eval loaders, computes both ensemble types, and returns results for bar chart generation.

**Design choice — shared val set:** The validation set used for early stopping is always the global `val` loader (most recent 10% of data). This is intentional — it means early stopping uses a consistent and challenging out-of-distribution benchmark. The trade-off is that SubsetA models stop very early (epoch 14–21) because the global val set is from an entirely different price era.

### 7.2 Experiment B — Full Training and Baseline

**Protocol:**

1. Build and train all 5 models on `full_train`
2. Evaluate each on `test` → collect USD predictions and metrics
3. Compute `mean_aggregate` and `weighted_aggregate`
4. Build and train `Baseline SmallMLP` (seed=999) on `full_train`
5. Final comparison table including all predictors

The baseline uses a **different seed (999)** so its weight initialisation differs from the ensemble's SmallMLP (seed=42). This simulates the real-world scenario where a practitioner trains one model without knowing which seed would perform best.

---

## 8. Results & Interpretation

### 8.1 Experiment A Results Summary

**Trained on Subset A → evaluated on Subset B:**

| Model | MSE (USD²) |
|---|---:|
| DeepMLP | 110.97 |
| SmallMLP | 74.90 |
| GRU | 85.12 |
| BiLSTM | 144.94 |
| LSTM | 176.96 |
| Mean Ensemble | 112.73 |
| Weighted Ensemble | 92.74 |

All models fail severely because they learned the low-price regime of Subset A. The weighted ensemble outperforms mean here because the val set (high-price era) correctly penalises models that overfitted to old price levels.

**Trained on Subset B → evaluated on Subset B:**

| Model | MSE (USD²) |
|---|---:|
| BiLSTM | 2.20 |
| LSTM | 5.68 |
| GRU | 6.42 |
| DeepMLP | 117.46 |
| SmallMLP | 61.67 |
| Mean Ensemble | 21.99 |
| Weighted Ensemble | 7.01 |

Recurrent models dominate — the weighted ensemble (7.01) is 3× better than the mean ensemble (21.99) because MLPs received very low weights (their val MSE was high, as expected for fully connected networks on sequential data).

### 8.2 Experiment B Results Summary

| Model | Test MSE (USD²) | Test MAE (USD) |
|---|---:|---:|
| BiLSTM | **26.15** | **4.02** |
| GRU | 88.97 | 7.15 |
| SmallMLP | 85.30 | 7.24 |
| DeepMLP | 120.35 | 9.11 |
| LSTM | 137.79 | 8.60 |
| **Mean Ensemble** | **58.85** | **6.00** |
| Weighted Ensemble | 62.14 | 5.92 |
| Baseline MLP | 118.43 | 8.75 |

### 8.3 Key Observations

**1. BiLSTM is the best individual model by a large margin.**
MSE=26.15 vs GRU=88.97 (3.4× better). The bidirectional architecture processes the 30-day sequence in both directions simultaneously, extracting both momentum and mean-reversion signals. This property is particularly useful for Amazon stock, which alternates between strong uptrends and sharp corrections.

**2. The ensemble beats the baseline by 50% (MSE: 58.85 vs 118.43).**
This is the core ensemble result. Even though no individual component of the ensemble is trained with a special procedure, averaging their predictions reduces error variance substantially.

**3. The mean ensemble outperforms the weighted ensemble in Experiment B.**
This is a known risk of inverse-MSE weighting: GRU achieved the lowest normalised val MSE (0.00112) and received the highest weight (0.3541), but on the test set GRU (88.97) is second worst. BiLSTM, which was actually the best test performer, received only 0.2545 weight. Weighting is only beneficial when the validation distribution matches the test distribution.

**4. LSTM is the worst full-train model despite decent val MSE.**
LSTM's gating mechanism (forget/input/output gates) gave it a low val MSE but it overfitted to the most recent data pattern in the full training set and failed to generalise to the test split.

**5. Long-term forecasts converge toward a stable mean.**
After ~10 predicted steps, the autoregressive ensemble forecast stabilises around $220–$221. This is expected — compounding prediction errors with frozen non-Close features causes the model to extrapolate a weighted average of recent patterns rather than discovering genuine future price structure.

---

## 9. Module Reference

### `data_preprocessing.py`

| Symbol | Type | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | `int` | Sliding window length (30) |
| `FEATURE_COLS` | `list[str]` | Ordered list of 8 feature column names |
| `N_FEATURES` | `int` | 8 |
| `StockDataset` | `class` | PyTorch Dataset wrapping `(X, y)` numpy arrays |
| `load_data(csv_path, seq_len, batch_size)` | `function` | Returns `(loaders, scaler_X, scaler_y, features_scaled)` |

### `models.py`

| Symbol | Type | Description |
|---|---|---|
| `set_seed(seed)` | `function` | Sets all RNG seeds for reproducibility |
| `SmallMLP` | `nn.Module` | 2-hidden-layer MLP |
| `DeepMLP` | `nn.Module` | 3-hidden-layer MLP |
| `LSTMModel` | `nn.Module` | 1-layer LSTM + FC |
| `GRUModel` | `nn.Module` | 1-layer GRU + FC |
| `BiLSTMModel` | `nn.Module` | 1-layer bidirectional LSTM + FC |
| `MODEL_CONFIGS` | `list` | `[(name, class, seed), ...]` for all 5 models |
| `build_all_models()` | `function` | Returns `[(name, model), ...]` with fixed seeds |

### `train.py`

| Symbol | Type | Description |
|---|---|---|
| `train_model(model, train_loader, val_loader, epochs, lr, patience, device)` | `function` | Returns `(model, train_losses, val_losses)` |

### `ensemble.py`

| Symbol | Type | Description |
|---|---|---|
| `evaluate_model(model, loader, scaler_y, device)` | `function` | Returns `(preds_usd, actuals_usd, mse, mae)` |
| `mean_aggregate(predictions_list)` | `function` | Returns `np.ndarray (N,)` — uniform mean |
| `weighted_aggregate(predictions_list, val_mses)` | `function` | Returns `(np.ndarray (N,), weights (n_models,))` |
| `compute_metrics(preds, actuals)` | `function` | Returns `(mse, mae)` |
| `print_metrics_table(results, title)` | `function` | Pretty-prints a `[(name, mse, mae)]` table |

### `main.py`

| Symbol | Type | Description |
|---|---|---|
| `EPOCHS`, `LR`, `PATIENCE`, `BATCH` | `int/float` | Global hyperparameters |
| `SHORT_HORIZON`, `LONG_HORIZON` | `int` | Forecast lengths (5, 30) |
| `CONTEXT_DAYS` | `int` | Historical days shown in forecast graphs (60) |
| `run_subset_experiment(train_key, eval_keys, label)` | `function` | Full Experiment A for one subset |
| `iterative_forecast(models_list, val_mses, features_scaled, scaler_y, horizon, device)` | `function` | Autoregressive multi-step forecast |
| `plot_subset_metrics(...)` | `function` | Saves bar chart to `plots/` |
| `plot_forecast(...)` | `function` | Saves forecast line chart to `plots/` |
