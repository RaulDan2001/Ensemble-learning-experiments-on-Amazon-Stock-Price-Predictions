import copy
import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    patience=10,
    device=None,
):
    """
    Train a single model with early stopping.

    Parameters
    ----------
    model        : nn.Module  — any of the 5 model classes
    train_loader : DataLoader — batches used to update weights
    val_loader   : DataLoader — batches used to monitor generalisation
    epochs       : int        — maximum number of full passes over train_loader
    lr           : float      — Adam learning rate
    patience     : int        — stop if val loss doesn't improve for this many epochs
    device       : torch.device or None — inferred automatically if None

    Returns
    -------
    model        : nn.Module  — weights restored to the best val-loss checkpoint
    train_losses : list[float]— mean MSE loss per epoch on train set
    val_losses   : list[float]— mean MSE loss per epoch on val set
    """

    # ── Device selection ─────────────────────────────────────────────────────
    # Use GPU if available; fall back to CPU transparently.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # ── Loss and optimiser ───────────────────────────────────────────────────
    # MSELoss is standard for regression; Adam adapts the learning rate per
    # parameter, which works well across all 5 architectures without tuning.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Early-stopping state ─────────────────────────────────────────────────
    best_val_loss  = float('inf')   # lowest val loss seen so far
    best_weights   = None           # deep copy of model weights at best epoch
    epochs_no_improve = 0           # counter — reset to 0 whenever val improves
    train_losses   = []
    val_losses     = []

    for epoch in range(1, epochs + 1):

        # ── Training pass ────────────────────────────────────────────────────
        model.train()   # activates Dropout layers
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()           # clear gradients from previous step
            preds = model(X_batch)          # forward pass
            loss  = criterion(preds, y_batch)
            loss.backward()                 # backpropagation
            optimizer.step()                # update weights

            running_loss += loss.item() * len(X_batch)  # accumulate batch loss

        # Average loss over all training samples in this epoch
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ── Validation pass ──────────────────────────────────────────────────
        model.eval()    # disables Dropout (deterministic predictions)
        val_loss = 0.0

        with torch.no_grad():   # no gradient tracking needed for evaluation
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds    = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # ── Early-stopping check ─────────────────────────────────────────────
        if epoch_val_loss < best_val_loss:
            best_val_loss     = epoch_val_loss
            best_weights      = copy.deepcopy(model.state_dict())  # save snapshot
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print a progress line every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"train_MSE: {epoch_train_loss:.6f}  "
                f"val_MSE: {epoch_val_loss:.6f}  "
                f"(no_improve: {epochs_no_improve}/{patience})"
            )

        # Stop early if validation hasn't improved for `patience` epochs
        if epochs_no_improve >= patience:
            print(f"  ↳ Early stop at epoch {epoch}  best_val_MSE: {best_val_loss:.6f}")
            break

    # ── Restore best weights ─────────────────────────────────────────────────
    # Revert the model to the checkpoint that had the lowest val loss,
    # discarding any overfitting that happened in subsequent epochs.
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()   # always return in eval mode (Dropout off)
    return model, train_losses, val_losses


if __name__ == '__main__':
    # ── Quick smoke test ─────────────────────────────────────────────────────
    # Runs one model for a few epochs to confirm the training loop works
    # end-to-end before the full experiment in main.py.
    from pathlib import Path
    from data_preprocessing import load_data
    from models import build_all_models

    data_path = Path(__file__).resolve().parents[1] / 'dataset' / 'Amazon_stock_data.csv'
    loaders, _, _, _ = load_data(data_path)

    # Test with just the first model (SmallMLP) and 3 epochs
    name, model = build_all_models()[0]
    print(f"\nSmoke-testing training loop with: {name}")
    model, tr_losses, val_losses = train_model(
        model,
        train_loader=loaders['full_train'],
        val_loader=loaders['val'],
        epochs=3,
        patience=10,
    )
    print(f"\nTrain losses : {[round(l, 6) for l in tr_losses]}")
    print(f"Val   losses : {[round(l, 6) for l in val_losses]}")
    print("Smoke test passed.")