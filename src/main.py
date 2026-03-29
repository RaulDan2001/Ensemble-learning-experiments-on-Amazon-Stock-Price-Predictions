"""
main.py — Full ensemble learning experiment orchestration.

Experiment A : Train all 5 models on Subset A -> evaluate on A and B.
               Repeat with Subset B.
Experiment B : Train all 5 models on full training set.
               Compare mean ensemble, weighted ensemble, and a single baseline.
Forecasting  : Short-term (5-day) and long-term (30-day) iterative predictions.
Graphs       : 7 PNG files saved to figs/.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # file-based backend — no display window needed
import matplotlib.pyplot as plt
import torch

# Ensure src/ is on sys.path so sibling modules can be imported
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data_preprocessing import load_data
from models import build_all_models, SmallMLP, set_seed
from train import train_model
from ensemble import (
    evaluate_model,
    mean_aggregate,
    weighted_aggregate,
    compute_metrics,
    print_metrics_table,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = SRC.parent
DATA_PATH    = PROJECT_ROOT / 'dataset' / 'Amazon_stock_data.csv'
PLOTS_DIR    = PROJECT_ROOT / 'figs'
PLOTS_DIR.mkdir(exist_ok=True)   # create plots/ folder if it doesn't exist

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS        = 100
LR            = 1e-3
PATIENCE      = 20
BATCH         = 32
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHORT_HORIZON = 5    # number of future days for the short-term forecast
LONG_HORIZON  = 30   # number of future days for the long-term forecast
CONTEXT_DAYS  = 60   # days of actual history shown before the forecast in graphs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  Loading data...")
print(f"{'═'*60}")

loaders, scaler_X, scaler_y, features_scaled = load_data(DATA_PATH, batch_size=BATCH)
print(f"  Running on: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPERIMENT A: SUBSET TRAINING
#
# Purpose: test whether an ensemble trained on half the data can generalise
# to the other (chronologically later or earlier) half.
#
# Steps:
#   1. Train all 5 models on loaders[train_key] (either subsetA or subsetB).
#   2. Evaluate each model individually on BOTH subsets.
#   3. Aggregate with mean and weighted ensemble on BOTH subsets.
#   4. Print a metrics table and return results for bar-chart plotting.
# ═══════════════════════════════════════════════════════════════════════════════

def run_subset_experiment(train_key, eval_keys, label):
    """
    Train all 5 models on loaders[train_key]; evaluate on every eval_key.

    Parameters
    ----------
    train_key : str        — loaders key used for training (e.g. 'subsetA')
    eval_keys : list[str]  — loaders keys to evaluate on after training
    label     : str        — human-readable name for console output

    Returns
    -------
    per_model_results : dict  eval_key → list[(name, mse, mae)]
    ensemble_results  : dict  eval_key → (mean_mse, mean_mae, w_mse, w_mae)
    """
    print(f"\n{'─'*60}")
    print(f"  EXPERIMENT A — Training on {label}")
    print(f"{'─'*60}")

    trained_models = []   # (name, model) after training
    val_mses_norm  = []   # best normalised val MSE per model (for weighting)

    # ── Train all 5 models sequentially ──────────────────────────────────────
    for name, model in build_all_models():
        print(f"\n  [{label}] Training {name}...")
        model, _, val_losses = train_model(
            model,
            train_loader=loaders[train_key],
            val_loader=loaders['val'],
            epochs=EPOCHS,
            lr=LR,
            patience=PATIENCE,
            device=DEVICE,
        )
        trained_models.append((name, model))
        val_mses_norm.append(min(val_losses))   # best (lowest) val loss this model achieved

    # ── Evaluate every trained model on every requested split ─────────────────
    per_model_results = {k: [] for k in eval_keys}   # (name, mse, mae) per split
    preds_by_eval     = {k: [] for k in eval_keys}   # prediction arrays for aggregation
    actuals_by_eval   = {}                            # ground-truth arrays (same for all models)

    for name, model in trained_models:
        for key in eval_keys:
            preds, actuals, mse, mae = evaluate_model(model, loaders[key], scaler_y, DEVICE)
            per_model_results[key].append((name, mse, mae))
            preds_by_eval[key].append(preds)
            actuals_by_eval[key] = actuals   # identical across models; safe to overwrite

    # ── Compute ensemble metrics for each eval split ──────────────────────────
    ensemble_results = {}
    for key in eval_keys:
        mean_p             = mean_aggregate(preds_by_eval[key])
        weighted_p, _      = weighted_aggregate(preds_by_eval[key], val_mses_norm)
        m_mse, m_mae       = compute_metrics(mean_p,     actuals_by_eval[key])
        w_mse, w_mae       = compute_metrics(weighted_p, actuals_by_eval[key])
        ensemble_results[key] = (m_mse, m_mae, w_mse, w_mae)

    # ── Print metrics tables ──────────────────────────────────────────────────
    for key in eval_keys:
        m_mse, m_mae, w_mse, w_mae = ensemble_results[key]
        print_metrics_table(
            per_model_results[key] + [
                ('Mean Ensemble', m_mse, m_mae),
                ('Weighted Ens.', w_mse, w_mae),
            ],
            title=f"Trained on {label} | Evaluated on {key}"
        )

    return per_model_results, ensemble_results


# Train on Subset A → evaluate on A and B
resultsA, ensA = run_subset_experiment(
    train_key='subsetA',
    eval_keys=['subsetA_eval', 'subsetB_eval'],
    label='Subset A',
)

# Train on Subset B → evaluate on A and B
resultsB, ensB = run_subset_experiment(
    train_key='subsetB',
    eval_keys=['subsetA_eval', 'subsetB_eval'],
    label='Subset B',
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPERIMENT B: FULL TRAINING + BASELINE COMPARISON
#
# Steps:
#   1. Train all 5 models on the full training set.
#   2. Evaluate each model on the test set.
#   3. Aggregate with mean ensemble and weighted ensemble.
#   4. Train a single Baseline SmallMLP (seed=999) on the same data.
#   5. Compare: each individual model, mean ensemble, weighted ensemble, baseline.
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  EXPERIMENT B — Full training set (all 5 models + baseline)")
print(f"{'═'*60}")

full_trained  = []   # (name, model, train_losses, val_losses)
val_mses_full = []   # best normalised val MSE per model (used for weighted agg.)

for name, model in build_all_models():
    print(f"\n  [FullTrain] Training {name}...")
    model, tr_losses, vl_losses = train_model(
        model,
        train_loader=loaders['full_train'],
        val_loader=loaders['val'],
        epochs=EPOCHS,
        lr=LR,
        patience=PATIENCE,
        device=DEVICE,
    )
    full_trained.append((name, model, tr_losses, vl_losses))
    val_mses_full.append(min(vl_losses))

# ── Evaluate each individually on the test set ────────────────────────────────
test_preds_list  = []   # one (N,) USD array per model
test_ind_results = []   # (name, mse, mae) for the metrics table
test_actuals     = None

for name, model, _, _ in full_trained:
    preds, actuals, mse, mae = evaluate_model(model, loaders['test'], scaler_y, DEVICE)
    test_preds_list.append(preds)
    test_ind_results.append((name, mse, mae))
    test_actuals = actuals   # same ground truth for every model

# ── Ensemble aggregation on the test set ─────────────────────────────────────
mean_test_preds              = mean_aggregate(test_preds_list)
weighted_test_preds, weights = weighted_aggregate(test_preds_list, val_mses_full)
mean_test_mse, mean_test_mae = compute_metrics(mean_test_preds,     test_actuals)
w_test_mse,    w_test_mae    = compute_metrics(weighted_test_preds, test_actuals)

# ── Baseline: a single SmallMLP with a different seed ────────────────────────
# This represents what you would get from training just ONE model on all the data
# — the comparison shows the benefit of ensembling.
print("\n  Training Baseline SmallMLP (seed=999)...")
set_seed(999)
baseline_model = SmallMLP()
baseline_model, bl_tr_losses, bl_vl_losses = train_model(
    baseline_model,
    train_loader=loaders['full_train'],
    val_loader=loaders['val'],
    epochs=EPOCHS,
    lr=LR,
    patience=PATIENCE,
    device=DEVICE,
)
bl_preds, _, bl_mse, bl_mae = evaluate_model(baseline_model, loaders['test'], scaler_y, DEVICE)

# ── Full comparison table ─────────────────────────────────────────────────────
print_metrics_table(
    test_ind_results + [
        ('Mean Ensemble', mean_test_mse, mean_test_mae),
        ('Weighted Ens.', w_test_mse,    w_test_mae),
        ('Baseline MLP',  bl_mse,        bl_mae),
    ],
    title="Experiment B — Test Set: Individual Models vs Ensembles vs Baseline"
)
print(f"\n  Weighted ensemble weights (sum=1): {np.round(weights, 4)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ITERATIVE MULTI-STEP FORECASTING
#
# The models were trained to predict ONE step ahead.
# To predict H steps ahead we use an autoregressive loop:
#   - Feed the last 30 rows (seed window) → predict step t+1.
#   - Append that prediction into the window; drop the oldest row.
#   - Repeat H times.
#
# For the non-Close features (Open, High, Low, Volume) we reuse the last
# known normalised values (practical approximation).
# Rolling statistics (MA_5, MA_20, Std_5) are recomputed from a running
# buffer of predicted normalised-Close values.
# ═══════════════════════════════════════════════════════════════════════════════

def iterative_forecast(models_list, val_mses, features_scaled, scaler_y, horizon, device):
    """
    Autoregressively predict `horizon` future Close prices (in USD) using
    the weighted ensemble.

    Parameters
    ----------
    models_list     : list[(name, model)]  — trained models in eval mode
    val_mses        : list[float]          — best normalised val MSE per model
    features_scaled : np.ndarray (N, 8)   — full scaled feature matrix
    scaler_y        : MinMaxScaler         — to inverse-transform predictions
    horizon         : int                  — number of steps to forecast
    device          : torch.device

    Returns
    -------
    forecast_usd : np.ndarray (horizon,) — predicted prices in USD
    """
    SEQ_LEN = 30

    # Seed window: the last SEQ_LEN rows of known history
    window = features_scaled[-SEQ_LEN:].copy()   # (30, 8)

    # Running buffer of normalised Close values (needed for rolling MA/Std).
    # Initialised with the last 20 known values so MA_20 can be computed
    # from the very first predicted step.
    close_buffer = list(features_scaled[-20:, 0])   # column 0 is Close (normalised)

    # Pre-compute inverse-MSE weights once — they are the same for every step
    inv_mses = np.array([1.0 / m for m in val_mses])
    weights  = inv_mses / inv_mses.sum()   # normalised so they sum to 1

    forecast_norm = []   # collect normalised predictions

    for _ in range(horizon):
        # ── Weighted ensemble prediction for the next step ────────────────────
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        # Shape: (1, 30, 8) — a single sequence
        step_preds = []
        for _, model in models_list:
            model.eval()
            with torch.no_grad():
                step_preds.append(model(x).item())   # scalar in normalised space

        pred_norm = float(np.dot(weights, step_preds))   # weighted mean
        forecast_norm.append(pred_norm)
        close_buffer.append(pred_norm)   # extend buffer for rolling stat update

        # ── Build the next input row ──────────────────────────────────────────
        new_row    = window[-1].copy()      # copy last row (preserves Open/High/Low/Vol)
        new_row[0] = pred_norm              # update Close (feature index 0)
        buf        = np.array(close_buffer)
        new_row[5] = buf[-5:].mean()        # MA_5  (feature index 5)
        new_row[6] = buf[-20:].mean() if len(buf) >= 20 else buf.mean()  # MA_20 (index 6)
        new_row[7] = buf[-5:].std()         # Std_5 (feature index 7)

        # Slide the window forward: drop oldest row, append the new row
        window = np.vstack([window[1:], new_row])   # still (30, 8)

    # Convert normalised Close predictions back to USD
    return scaler_y.inverse_transform(
        np.array(forecast_norm).reshape(-1, 1)
    ).flatten()


print(f"\n{'═'*60}")
print("  SECTION 4 — Forecasting")
print(f"{'═'*60}")

models_list    = [(n, m) for n, m, _, _ in full_trained]
short_forecast = iterative_forecast(
    models_list, val_mses_full, features_scaled, scaler_y, SHORT_HORIZON, DEVICE
)
long_forecast = iterative_forecast(
    models_list, val_mses_full, features_scaled, scaler_y, LONG_HORIZON, DEVICE
)

print(f"\n  Short-term ({SHORT_HORIZON}-day) forecast (USD): {np.round(short_forecast, 2)}")
print(f"  Long-term  ({LONG_HORIZON}-day)  forecast (USD): {np.round(long_forecast, 2)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GRAPHS
# 7 PNG files saved to plots/
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  SECTION 5 — Generating graphs...")
print(f"{'═'*60}")


# ── Graph 1: Loss curves — one subplot per model + baseline ───────────────────
# Lets you see convergence speed and where early stopping fired for each model.
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
fig.suptitle('Training & Validation Loss — Full-Training Experiment',
             fontsize=14, fontweight='bold')

for i, (name, _, tr_losses, vl_losses) in enumerate(full_trained):
    axes[i].plot(tr_losses, label='Train MSE', color='steelblue')
    axes[i].plot(vl_losses, label='Val MSE',   color='tomato', linestyle='--')
    axes[i].set_title(name)
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('MSE (normalised)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# 6th subplot: baseline model
axes[5].plot(bl_tr_losses, label='Train MSE', color='steelblue')
axes[5].plot(bl_vl_losses, label='Val MSE',   color='tomato', linestyle='--')
axes[5].set_title('Baseline MLP (seed=999)')
axes[5].set_xlabel('Epoch')
axes[5].set_ylabel('MSE (normalised)')
axes[5].legend()
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loss_curves.png', dpi=150)
plt.close()
print("  Saved: loss_curves.png")


# ── Graph 2: All models + both ensembles + baseline on the test set ────────────
# Gives a visual overview of every predictor on the same time axis.
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(test_actuals, label='Actual', color='black', linewidth=1.5, zorder=5)

model_colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
for (name, *_), preds, color in zip(full_trained, test_preds_list, model_colors):
    ax.plot(preds, label=name, alpha=0.55, linewidth=0.8, color=color)

ax.plot(mean_test_preds,     label='Mean Ensemble',     color='cyan',    linewidth=2,   linestyle='--')
ax.plot(weighted_test_preds, label='Weighted Ensemble', color='magenta', linewidth=2,   linestyle='-.')
ax.plot(bl_preds,            label='Baseline MLP',      color='gold',    linewidth=1.8, linestyle=':')

ax.set_title('Test-Set Predictions: All Models vs Both Ensembles vs Baseline',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Time step (test set)')
ax.set_ylabel('Close Price (USD)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'test_predictions.png', dpi=150)
plt.close()
print("  Saved: test_predictions.png")


# ── Graphs 3 & 4: Subset experiment bar charts ────────────────────────────────
# For each model, two bars: MSE when evaluated on SubsetA vs SubsetB.
# Horizontal lines show where the ensembles land for each eval split.
def plot_subset_metrics(per_model_results, ensemble_results, train_label, filename):
    eval_keys  = list(per_model_results.keys())
    names      = [r[0] for r in per_model_results[eval_keys[0]]]
    x          = np.arange(len(names))
    width      = 0.35
    bar_colors = ['steelblue', 'darkorange']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, key in enumerate(eval_keys):
        mses = [r[1] for r in per_model_results[key]]
        bars = ax.bar(x + i * width, mses, width,
                      label=f'Eval on {key}', color=bar_colors[i], alpha=0.8)
        ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=7)

    # Horizontal lines: mean and weighted ensemble MSE per eval split
    h_colors = [('steelblue', 'navy'), ('darkorange', 'saddlebrown')]
    for i, key in enumerate(eval_keys):
        m_mse, _, w_mse, _ = ensemble_results[key]
        ax.axhline(m_mse, color=h_colors[i][0], linestyle='--', linewidth=1.5,
                   label=f'Mean Ens ({key}): {m_mse:.1f} USD²')
        ax.axhline(w_mse, color=h_colors[i][1], linestyle=':',  linewidth=1.5,
                   label=f'Wtd Ens  ({key}): {w_mse:.1f} USD²')

    ax.set_title(f'Subset Experiment — Trained on {train_label} | Per-Model MSE (USD²)',
                 fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('MSE (USD²)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(names)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


plot_subset_metrics(resultsA, ensA, 'Subset A', 'subsetA_metrics.png')
print("  Saved: subsetA_metrics.png")

plot_subset_metrics(resultsB, ensB, 'Subset B', 'subsetB_metrics.png')
print("  Saved: subsetB_metrics.png")


# ── Graph 5: Mean Ensemble vs Weighted Ensemble vs Baseline ───────────────────
# The key comparison: does weighting by inverse-val-MSE improve over simple mean?
# Does the ensemble beat a single well-trained model?
# MSE and MAE are shown directly in the legend for quick reading.
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(test_actuals,
        label='Actual',
        color='black', linewidth=1.5)
ax.plot(mean_test_preds,
        label=f'Mean Ensemble    (MSE={mean_test_mse:.2f}  MAE={mean_test_mae:.2f})',
        color='steelblue',  linewidth=2, linestyle='--')
ax.plot(weighted_test_preds,
        label=f'Weighted Ensemble (MSE={w_test_mse:.2f}  MAE={w_test_mae:.2f})',
        color='darkorange', linewidth=2, linestyle='-.')
ax.plot(bl_preds,
        label=f'Baseline MLP     (MSE={bl_mse:.2f}  MAE={bl_mae:.2f})',
        color='tomato',     linewidth=2, linestyle=':')

ax.set_title('Mean Ensemble vs Weighted Ensemble vs Baseline — Test Set',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Time step (test set)')
ax.set_ylabel('Close Price (USD)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ensemble_vs_baseline.png', dpi=150)
plt.close()
print("  Saved: ensemble_vs_baseline.png")


# ── Graphs 6 & 7: Short-term and long-term forecasts ─────────────────────────
# Shows CONTEXT_DAYS of real history followed by the multi-step forecast.
# The vertical dashed line marks where actuals end and predictions begin.
def plot_forecast(forecast_usd, horizon, context_days,
                  features_scaled, scaler_y, filename, title):
    # Recover actual Close prices for the context window (inverse-transform column 0)
    recent_norm = features_scaled[-context_days:, 0].reshape(-1, 1)
    recent_usd  = scaler_y.inverse_transform(recent_norm).flatten()

    x_history  = np.arange(context_days)
    x_forecast = np.arange(context_days, context_days + horizon)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x_history,  recent_usd,
            label='Historical (actual)', color='black', linewidth=1.5)
    ax.plot(x_forecast, forecast_usd,
            label=f'{horizon}-Day Weighted Ensemble Forecast',
            color='darkorange', linewidth=2, marker='o', markersize=5)
    ax.axvline(x=context_days - 0.5, color='gray', linestyle='--',
               linewidth=1.2, label='Forecast start')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Days (relative to forecast origin)')
    ax.set_ylabel('Close Price (USD)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


plot_forecast(
    short_forecast, SHORT_HORIZON, CONTEXT_DAYS,
    features_scaled, scaler_y,
    'short_term_forecast.png',
    f'Short-Term Forecast — Next {SHORT_HORIZON} Days (Weighted Ensemble)',
)
print("  Saved: short_term_forecast.png")

plot_forecast(
    long_forecast, LONG_HORIZON, CONTEXT_DAYS,
    features_scaled, scaler_y,
    'long_term_forecast.png',
    f'Long-Term Forecast — Next {LONG_HORIZON} Days (Weighted Ensemble)',
)
print("  Saved: long_term_forecast.png")


# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  All experiments and graphs complete.")
print(f"  Plots saved to: {PLOTS_DIR}")
print(f"{'═'*60}\n")