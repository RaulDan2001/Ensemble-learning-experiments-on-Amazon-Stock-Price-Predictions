import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path

SEQUENCE_LENGTH = 30
FEATURE_COLS = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'Std_5']
N_FEATURES = len(FEATURE_COLS)  # 8


class StockDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, seq_len, n_features)  y: (N, 1)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(csv_path, seq_len=SEQUENCE_LENGTH, batch_size=32):
    # ── 1. Load & sort chronologically ──────────────────────────────────────
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    df['MA_5']  = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Std_5'] = df['Close'].rolling(5).std()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 3. Raw split boundaries (on row indices before sequencing) ───────────
    n         = len(df)
    train_end = int(n * 0.80)   # row index where val starts
    val_end   = int(n * 0.90)   # row index where test starts

    # ── 4. Fit scalers on TRAINING ROWS ONLY — prevent data leakage ─────────
    raw_X = df[FEATURE_COLS].values          # (n, 8)
    raw_y = df[['Close']].values             # (n, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(raw_X[:train_end])
    scaler_y.fit(raw_y[:train_end])

    features_scaled = scaler_X.transform(raw_X)          # (n, 8)
    target_scaled   = scaler_y.transform(raw_y).flatten()  # (n,)

    # ── 5. Sliding window sequences ──────────────────────────────────────────
    # Sequence i: X[i] = features[i : i+seq_len]    (shape: seq_len, 8)
    #             y[i] = close[i + seq_len]          (next-day close, normalized)
    X_all, y_all = [], []
    for i in range(n - seq_len):
        X_all.append(features_scaled[i : i + seq_len])
        y_all.append(target_scaled[i + seq_len])

    X_all = np.array(X_all, dtype=np.float32)   # (N, 30, 8)
    y_all = np.array(y_all, dtype=np.float32)   # (N,)

    # ── 6. Map raw boundaries to sequence index space ────────────────────────
    # Sequence i has its label (y) at raw row i+seq_len.
    # Last training label sits at raw row train_end-1 → seq index = train_end - seq_len - 1
    seq_train_end = train_end - seq_len   # exclusive
    seq_val_end   = val_end   - seq_len   # exclusive

    idx_train = list(range(0,             seq_train_end))
    idx_val   = list(range(seq_train_end, seq_val_end))
    idx_test  = list(range(seq_val_end,   len(X_all)))

    # ── 7. Split training into two equal subsets (chronological) ────────────
    half     = len(idx_train) // 2
    idx_subA = idx_train[:half]   # older half
    idx_subB = idx_train[half:]   # newer half

    # ── 8. Build Dataset and DataLoaders ────────────────────────────────────
    dataset = StockDataset(X_all, y_all)

    def make_loader(indices, shuffle=False):
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    loaders = {
        'subsetA':      make_loader(idx_subA,  shuffle=False),
        'subsetA_eval': make_loader(idx_subA,  shuffle=False),
        'subsetB':      make_loader(idx_subB,  shuffle=False),
        'subsetB_eval': make_loader(idx_subB,  shuffle=False),
        'full_train':   make_loader(idx_train, shuffle=False),
        'val':          make_loader(idx_val,   shuffle=False),
        'test':         make_loader(idx_test,  shuffle=False),
    }

    print(f"Total raw rows  : {n}")
    print(f"Total sequences : {len(X_all):>6}  shape: {X_all.shape}")
    print(f"Train           : {len(idx_train):>6}  -> SubsetA: {len(idx_subA)}, SubsetB: {len(idx_subB)}")
    print(f"Val             : {len(idx_val):>6}")
    print(f"Test            : {len(idx_test):>6}")

    return loaders, scaler_X, scaler_y, features_scaled


if __name__ == '__main__':
    # Read dataset from project-relative path to avoid hardcoded machine-specific paths.
    data_path = Path(__file__).resolve().parents[1] / 'dataset' / 'Amazon_stock_data.csv'

    loaders, scaler_X, scaler_y, features_scaled = load_data(csv_path=data_path)
    # Quick shape smoke-test
    X_batch, y_batch = next(iter(loaders['full_train']))
    print(f"\nBatch shapes -> X: {X_batch.shape}, y: {y_batch.shape}")
    # Expected: X: (32, 30, 8)   y: (32, 1)