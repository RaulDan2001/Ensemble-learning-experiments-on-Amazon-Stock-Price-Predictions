# Project Plan

## Ensemble Learning — Amazon Stock Price Prediction

**Course:** Neural Computing (Calcul Neuronal)  
**Level:** Master's  
**Date:** March 2026

---

## 1. Original Requirements

The project requirements specified the following mandatory components:

### Ensembles — Required Deliverables

| Requirement | Status |
|---|---|
| Split the training dataset into 2 subsets of similar size | Implemented — Subset A (2,865 seq) and Subset B (2,866 seq), chronological split |
| Test the ensemble individually on both subsets | Implemented — each subset-trained ensemble is evaluated on both SubsetA_eval and SubsetB_eval |
| Initialize at least 5 models with different seeds or different types | Implemented — SmallMLP(seed=42), DeepMLP(seed=123), LSTM(seed=42), GRU(seed=42), BiLSTM(seed=42) |
| Train at least 5 models on the same dataset | Implemented — all 5 trained on full_train in Experiment B |
| (2*) Aggregate the results — test each model and aggregate | Implemented — mean aggregate + inverse-MSE weighted aggregate, compared in metrics tables |
| Compare with a model trained on all the data | Implemented — Baseline SmallMLP (seed=999) trained on full_train |
| Short-term / long-term predictions with graphs | Implemented — 5-day and 30-day autoregressive ensemble forecasts with plots |

---

## 2. Technical Decisions

### 2.1 Task Type Decision

**Decision:** Regression (predict next-day Close price in USD)

**Alternatives considered:**
- Classification (predict direction: up/down) — rejected because regression is more informative and directly useful
- Multi-step supervised regression (train on H-step targets) — rejected in favour of autoregressive forecasting to keep models simpler

### 2.2 Framework Decision

**Decision:** PyTorch only

**Alternative considered:** mix of scikit-learn (classical models) and PyTorch. Rejected because all 5 required model types (MLP, LSTM, GRU, BiLSTM, Deep MLP) are naturally expressible in PyTorch and maintaining a unified training loop is cleaner.

### 2.3 Input Representation

**Decision:** Sliding window of 30 timesteps × 8 features

**Rationale:**
- 30 days ≈ one calendar month, a natural financial cycle
- 8 features: 5 raw OHLCV + 3 derived rolling statistics (MA_5, MA_20, Std_5)
- Recurrent models receive shape `(batch, 30, 8)` directly
- MLP models use `nn.Flatten()` to convert to `(batch, 240)` internally

### 2.4 Normalisation Strategy

**Decision:** MinMaxScaler fitted only on training rows; separate scaler for X and y

**Rationale:**
- Fitting on training rows only prevents the scaler from "seeing" future data — critical for avoiding data leakage in financial forecasting
- Separate `scaler_y` allows predictions to be inverse-transformed independently of the input features when computing USD metrics

### 2.5 No Shuffling

**Decision:** All DataLoaders use `shuffle=False`

**Rationale:** Time-series data must preserve temporal order. Shuffling would mix past and future data within a batch, introducing look-ahead bias and producing unrealistically optimistic training results.

### 2.6 Subset Splitting Strategy

**Decision:** Chronological 50/50 split of the training set

- Subset A = first 50% of training sequences (older, lower-price era)
- Subset B = second 50% of training sequences (newer, higher-price era)

**Rationale:** A chronological split tests whether the ensemble generalises across time periods — a more realistic scenario than a random split.

### 2.7 Ensemble Weighting

**Decision:** Inverse-val-MSE weighting computed from the minimum (best) val loss each model achieved

$$w_i = \frac{1 / \min(\text{val\_losses}_i)}{\sum_j 1 / \min(\text{val\_losses}_j)}$$

**Rationale:** The best checkpoint is restored by early stopping, so it is the most appropriate measure of each model's actual quality on the validation distribution.

**Known limitation:** Weights are computed in normalised space. If val and test distributions differ significantly (as they do in financial data with rapidly growing prices), the weighting may not reflect real-world USD performance.

### 2.8 Forecasting Approach

**Decision:** Autoregressive (recursive) iterative forecasting

Each model was trained to predict one step ahead. To produce H-step forecasts:
1. Feed the last 30 rows as seed
2. Predict step `t+1`
3. Update the window with the predicted Close value (and recomputed rolling stats)
4. Repeat H times

**Alternative:** Direct multi-output regression (train models to output all H steps at once). Rejected because it would require retraining all models with different output heads, and autoregressive forecasting better demonstrates the ensemble aggregation concept.

---

## 3. Implementation Plan

### Phase 1 — Data Preprocessing (`src/data_preprocessing.py`)

- [x] Load CSV, parse dates, sort chronologically
- [x] Engineer rolling features (MA_5, MA_20, Std_5)
- [x] Drop NaN rows (first 20 rows)
- [x] Fit MinMaxScaler on training rows only
- [x] Create sliding window sequences (30 × 8)
- [x] Map raw row boundaries to sequence index boundaries
- [x] Split Subset A / Subset B (chronological 50/50 of train)
- [x] Build 7 DataLoaders with shuffle=False
- [x] Smoke test: verify batch shapes `(32, 30, 8)` and `(32, 1)`

### Phase 2 — Model Definitions (`src/models.py`)

- [x] `set_seed(seed)` utility function
- [x] SmallMLP — 2 hidden layers, Dropout 0.2
- [x] DeepMLP — 3 hidden layers, Dropout 0.3
- [x] LSTMModel — 1-layer LSTM, last timestep
- [x] GRUModel — 1-layer GRU, last timestep
- [x] BiLSTMModel — 1-layer BiLSTM, concatenated final states
- [x] `MODEL_CONFIGS` list and `build_all_models()` factory
- [x] Smoke test: all 5 models output `(4, 1)` from dummy input `(4, 30, 8)`

### Phase 3 — Training Loop (`src/train.py`)

- [x] Adam optimizer with configurable LR
- [x] MSELoss (normalised space)
- [x] Per-epoch training pass (accumulate batch losses correctly)
- [x] Per-epoch validation pass (no gradient, deterministic)
- [x] Early stopping with `deepcopy` checkpoint
- [x] Best weight restoration on exit
- [x] Return `(model, train_losses, val_losses)`
- [x] Smoke test: 3 epochs on SmallMLP — loss decreasing

### Phase 4 — Ensemble Utilities (`src/ensemble.py`)

- [x] `evaluate_model()` — inference loop + inverse-transform + USD metrics
- [x] `mean_aggregate()` — uniform average across all model predictions
- [x] `weighted_aggregate()` — inverse-MSE weighted average
- [x] `compute_metrics()` — standalone MSE + MAE in USD
- [x] `print_metrics_table()` — formatted console table
- [x] Smoke test: 2-model ensemble weights sum to 1.0

### Phase 5 — Main Orchestration (`src/main.py`)

- [x] `run_subset_experiment()` — reusable function for Experiment A
- [x] Experiment A: SubsetA training → eval on both subsets
- [x] Experiment A: SubsetB training → eval on both subsets
- [x] Experiment B: all 5 models on full_train
- [x] Experiment B: Baseline MLP (seed=999)
- [x] Full comparison table: individual + mean + weighted + baseline
- [x] `iterative_forecast()` — autoregressive H-step forecasting
- [x] Short-term (5-day) and long-term (30-day) forecasts
- [x] Graph 1: Loss curves (2×3 grid including baseline)
- [x] Graph 2: All models + both ensembles + baseline on test set
- [x] Graph 3: SubsetA bar chart
- [x] Graph 4: SubsetB bar chart
- [x] Graph 5: Mean ensemble vs Weighted ensemble vs Baseline
- [x] Graph 6: Short-term forecast
- [x] Graph 7: Long-term forecast

---

## 4. Actual Results vs Expectations

| Item | Expected | Actual |
|---|---|---|
| Total sequences | ~7,140 | 7,172 |
| Train / Val / Test | ~5,720 / ~715 / ~715 | 5,731 / 720 / 721 |
| Best individual model | LSTM or GRU (expected due to sequence learning) | BiLSTM (MSE=26.15) |
| Ensemble benefit over baseline | ~20–30% improvement | **50% improvement** (MSE: 58.85 vs 118.43) |
| Cross-subset generalisation | Moderate degradation | Severe degradation for MLP; moderate for recurrent |
| Weighted > Mean ensemble | Expected (by design) | **Mean beat weighted in Experiment B**; weighted better in Experiment A Subset B scenario |
| Long-term forecast stability | Gradual convergence | Stabilised at ~$220–$221 after ~10 steps |

---

## 5. Identified Limitations and Potential Improvements

### High Priority

| Improvement | Effort | Expected Impact |
|---|---|---|
| Compute weighting in USD space (not normalised) | Low | Correct weighting would give BiLSTM a much higher weight |
| Use % returns as prediction target instead of absolute price | Medium | Would solve cross-era generalisation problem (scale-invariant) |
| Stacking meta-learner (linear regression on val predictions) | Medium | Learns optimal model combination from data rather than heuristic |

### Medium Priority

| Improvement | Effort | Expected Impact |
|---|---|---|
| Learning rate scheduler (ReduceLROnPlateau) | Low | Models converged at 16–37 epochs; scheduler could extract more |
| Increase early stopping patience to 15–20 for full-train | Low | More complete convergence for recurrent models |
| Per-subset local val set (last 10% of each subset) | Medium | More meaningful early stopping for Experiment A |
| Gradient clipping (`clip_grad_norm_`) | Low | Prevents early epoch instability spikes |

### Lower Priority

| Improvement | Effort | Expected Impact |
|---|---|---|
| Additional features: RSI, MACD, Bollinger Bands | Medium | More signal for recurrent models |
| Multi-layer LSTM/GRU (2 layers) | Low | May improve representational capacity |
| Attention mechanism over LSTM hidden states | High | Better long-range dependency modelling |
| Walk-forward validation instead of fixed split | High | More realistic out-of-sample evaluation |

---

## 6. File Outputs

| File | Description |
|---|---|
| `plots/loss_curves.png` | Training convergence of all 6 models |
| `plots/test_predictions.png` | All predictors overlaid on test actuals |
| `plots/subsetA_metrics.png` | Cross-subset bar chart (trained on Subset A) |
| `plots/subsetB_metrics.png` | Cross-subset bar chart (trained on Subset B) |
| `plots/ensemble_vs_baseline.png` | Primary ensemble vs baseline comparison |
| `plots/short_term_forecast.png` | 5-day weighted ensemble forecast |
| `plots/long_term_forecast.png` | 30-day weighted ensemble forecast |
| `logs/main.log` | Full console output from the last experiment run |
