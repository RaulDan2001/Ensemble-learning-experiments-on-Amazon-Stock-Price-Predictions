import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model 1: Small MLP ───────────────────────────────────────────────────────
class SmallMLP(nn.Module):
    """2 hidden layers. Flattens (batch, 30, 8) -> 240 internally."""
    def __init__(self, seq_len=30, n_features=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):   # x: (batch, 30, 8)
        return self.net(x)


# ── Model 2: Deep MLP ────────────────────────────────────────────────────────
class DeepMLP(nn.Module):
    """3 hidden layers, wider. Flattens (batch, 30, 8) -> 240 internally."""
    def __init__(self, seq_len=30, n_features=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):   # x: (batch, 30, 8)
        return self.net(x)


# ── Model 3: LSTM ────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, n_features=8, hidden=64, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):            # x: (batch, 30, 8)
        out, _ = self.lstm(x)        # out: (batch, 30, 64)
        return self.fc(out[:, -1, :])  # last timestep → (batch, 1)


# ── Model 4: GRU ─────────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, n_features=8, hidden=64, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, n_layers, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x):            # x: (batch, 30, 8)
        out, _ = self.gru(x)         # out: (batch, 30, 64)
        return self.fc(out[:, -1, :])


# ── Model 5: Bidirectional LSTM ──────────────────────────────────────────────
class BiLSTMModel(nn.Module):
    def __init__(self, n_features=8, hidden=64, n_layers=1):
        super().__init__()
        self.bilstm = nn.LSTM(n_features, hidden, n_layers,
                              batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)   # *2 for both directions

    def forward(self, x):                          # x: (batch, 30, 8)
        _, (h_n, _) = self.bilstm(x)              # h_n: (2, batch, 64)
        # h_n[0] = forward final state, h_n[1] = backward final state
        h = torch.cat([h_n[0], h_n[1]], dim=1)    # (batch, 128)
        return self.fc(h)


# ── Factory ───────────────────────────────────────────────────────────────────
# (name, class, seed)  — seed controls weight initialization
MODEL_CONFIGS = [
    ('SmallMLP',  SmallMLP,    42),
    ('DeepMLP',   DeepMLP,    123),
    ('LSTM',      LSTMModel,   42),
    ('GRU',       GRUModel,    42),
    ('BiLSTM',    BiLSTMModel, 42),
]


def build_all_models():
    """Instantiate all 5 models with reproducible weight initialization."""
    models = []
    for name, cls, seed in MODEL_CONFIGS:
        set_seed(seed)
        models.append((name, cls()))
    return models


if __name__ == '__main__':
    dummy = torch.randn(4, 30, 8)   # batch=4, seq=30, features=8
    for name, model in build_all_models():
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<12}  output: {out.shape}  params: {params:,}")