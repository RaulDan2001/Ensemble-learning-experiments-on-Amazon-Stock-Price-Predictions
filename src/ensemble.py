import numpy as np
import torch


def evaluate_model(model, loader, scaler_y, device=None):
    """
    Run a trained model over a DataLoader and return predictions + actuals
    both in the original (inverse-scaled) price space (USD).

    Parameters
    ----------
    model    : nn.Module      — a trained model in eval mode
    loader   : DataLoader     — any of the 7 loaders from load_data()
    scaler_y : MinMaxScaler   — fitted on training Close prices; used to
                                invert the normalisation back to USD
    device   : torch.device   — inferred automatically if None

    Returns
    -------
    preds   : np.ndarray shape (N,) — predicted Close prices in USD
    actuals : np.ndarray shape (N,) — ground-truth Close prices in USD
    mse     : float  — mean squared error in USD²
    mae     : float  — mean absolute error in USD
    """

    # ── Device selection ─────────────────────────────────────────────────────
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()   # ensure Dropout is disabled during inference

    all_preds   = []
    all_actuals = []

    with torch.no_grad():   # no gradient computation needed for evaluation
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)                         # (batch, 1) normalised
            all_preds.append(out.cpu().numpy())          # move back to CPU
            all_actuals.append(y_batch.numpy())          # already on CPU

    # Stack all batches into single arrays -> (N, 1)
    all_preds   = np.vstack(all_preds)
    all_actuals = np.vstack(all_actuals)

    # Inverse-transform from [0,1] back to USD prices
    preds   = scaler_y.inverse_transform(all_preds).flatten()    # (N,)
    actuals = scaler_y.inverse_transform(all_actuals).flatten()  # (N,)

    # ── Metrics in original price space ──────────────────────────────────────
    mse = float(np.mean((preds - actuals) ** 2))
    mae = float(np.mean(np.abs(preds - actuals)))

    return preds, actuals, mse, mae


def mean_aggregate(predictions_list):
    """
    Simple ensemble: average the predictions of all models element-wise.

    Parameters
    ----------
    predictions_list : list[np.ndarray]  — one (N,) array per model, in USD

    Returns
    -------
    np.ndarray shape (N,) — ensemble mean prediction in USD
    """
    # Stack to (n_models, N) then average across the model axis
    stacked = np.stack(predictions_list, axis=0)   # (n_models, N)
    return stacked.mean(axis=0)                    # (N,)


def weighted_aggregate(predictions_list, val_mses):
    """
    Weighted ensemble: models with lower validation MSE get higher weight.
    Weight formula: w_i = (1 / val_mse_i) / sum(1 / val_mse_j)

    Parameters
    ----------
    predictions_list : list[np.ndarray]  — one (N,) array per model, in USD
    val_mses         : list[float]       — validation MSE (normalised space)
                                           for each model; same order as preds

    Returns
    -------
    weighted : np.ndarray shape (N,) — weighted ensemble prediction in USD
    weights  : np.ndarray shape (n_models,) — normalised weights (sum to 1)
    """
    # Compute inverse-MSE weights and normalise so they sum to 1
    inv_mses = np.array([1.0 / mse for mse in val_mses])
    weights  = inv_mses / inv_mses.sum()                   # (n_models,)

    # Stack predictions and apply weights along the model axis
    stacked  = np.stack(predictions_list, axis=0)          # (n_models, N)
    weighted = (stacked * weights[:, None]).sum(axis=0)    # (N,)

    return weighted, weights


def compute_metrics(preds, actuals):
    """
    Compute MSE and MAE given arrays already in USD (not normalised).

    Parameters
    ----------
    preds   : np.ndarray (N,)
    actuals : np.ndarray (N,)

    Returns
    -------
    mse : float
    mae : float
    """
    mse = float(np.mean((preds - actuals) ** 2))
    mae = float(np.mean(np.abs(preds - actuals)))
    return mse, mae


def print_metrics_table(results, title="Results"):
    """
    Pretty-print a table of model metrics.

    Parameters
    ----------
    results : list of (name, mse, mae) tuples
    title   : str — printed as section header
    """
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    print(f"  {'Model':<14}  {'MSE':>12}  {'MAE':>10}")
    print(f"  {'─'*14}  {'─'*12}  {'─'*10}")
    for name, mse, mae in results:
        print(f"  {name:<14}  {mse:>12.4f}  {mae:>10.4f}")
    print(f"{'─'*50}")


if __name__ == '__main__':
    # ── Smoke test ────────────────────────────────────────────────────────────
    # Trains SmallMLP + DeepMLP for 5 epochs each, then exercises all ensemble
    # helpers to verify shapes and inverse-scaling work before main.py
    from pathlib import Path
    from data_preprocessing import load_data
    from models import build_all_models
    from train import train_model

    data_path = Path(__file__).resolve().parents[1] / 'dataset' / 'Amazon_stock_data.csv'
    loaders, _, scaler_y, _ = load_data(data_path)

    # Train two models for 5 epochs each to test aggregation with >1 model
    all_preds_list = []
    all_actuals    = None
    val_mses_norm  = []

    for name, model in build_all_models()[:2]:    # SmallMLP + DeepMLP only
        print(f"\nTraining {name} (5 epochs)...")
        model, _, val_losses = train_model(
            model, loaders['full_train'], loaders['val'], epochs=5, patience=10
        )
        preds, actuals, mse, mae = evaluate_model(model, loaders['test'], scaler_y)
        all_preds_list.append(preds)
        all_actuals = actuals                     # same ground truth for all models
        val_mses_norm.append(val_losses[-1])      # last epoch val MSE (normalised space)
        print(f"  {name}: test MSE={mse:.4f} USD²  MAE={mae:.4f} USD")

    # ── Test mean aggregation ─────────────────────────────────────────────────
    # Every model contributes equally regardless of individual performance
    mean_preds = mean_aggregate(all_preds_list)
    mean_mse, mean_mae = compute_metrics(mean_preds, all_actuals)

    # ── Test weighted aggregation ─────────────────────────────────────────────
    # Better-performing models (lower val MSE) receive a proportionally higher weight
    weighted_preds, weights = weighted_aggregate(all_preds_list, val_mses_norm)
    w_mse, w_mae = compute_metrics(weighted_preds, all_actuals)

    print(f"\nWeights (must sum to 1): {np.round(weights, 4)}  sum={weights.sum():.4f}")

    print_metrics_table([
        ('Mean Ensemble', mean_mse, mean_mae),
        ('Weighted Ens.', w_mse,    w_mae),
    ], title="Ensemble Smoke Test")

    print("\nensemble.py smoke test passed.")