"""
LSTM model for AQI prediction using PyTorch.

Reshapes flat tabular features into sequences using a sliding window,
then feeds them through a multi-layer LSTM followed by a linear head.

The model expects *already-scaled* inputs -- the training pipeline is
responsible for fitting the scaler and transforming data before calling
``train()``.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -- Defaults ----------------------------------------------------------------

SEQUENCE_LENGTH = 48  # hours of history per sample
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
MAX_EPOCHS = 150
PATIENCE = 15  # early-stopping patience (epochs without val-loss improvement)


# -- Dataset -----------------------------------------------------------------


class AQISequenceDataset(Dataset):
    """Sliding-window dataset that turns (N, F) arrays into (seq_len, F) windows."""

    def __init__(self, X, y, seq_len=SEQUENCE_LENGTH):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.seq_len]  # (seq_len, features)
        y_target = self.y[idx + self.seq_len]  # scalar
        return x_window, y_target


# -- Network -----------------------------------------------------------------


class AQILstm(nn.Module):
    """Multi-layer LSTM with a linear regression head."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take the last time-step
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)  # (batch,)


# -- Training ----------------------------------------------------------------


def train(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    seq_len=SEQUENCE_LENGTH,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
):
    """
    Train an LSTM model on pre-scaled feature arrays.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features (N, F) and target (N,).
    X_val, y_val : np.ndarray, optional
        Validation arrays for early stopping.
    seq_len : int
        Number of past time-steps in each input window.
    hidden_size, num_layers, dropout : int/float
        LSTM architecture parameters.
    batch_size, lr, max_epochs, patience : int/float
        Training loop parameters.

    Returns
    -------
    AQILstm
        Trained model (on CPU, in eval mode).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  LSTM device: {device}")

    # Build data loaders -- shuffle=True for training to break temporal ordering bias
    train_ds = AQISequenceDataset(X_train, y_train, seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_dl = None
    if X_val is not None and y_val is not None:
        val_ds = AQISequenceDataset(X_val, y_val, seq_len)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    input_size = X_train.shape[1]
    model = AQILstm(input_size, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=10.0)  # More robust to outlier AQI spikes than MSE
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_ds)

        # -- Validate --
        if val_dl is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item() * len(yb)
            val_loss /= len(val_ds)

            # Step the LR scheduler
            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                    break
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model.cpu().eval()
    return model


# -- Prediction helper -------------------------------------------------------


def predict(model, X, seq_len=SEQUENCE_LENGTH):
    """
    Generate predictions from the trained LSTM.

    Parameters
    ----------
    model : AQILstm
        Trained model (CPU, eval mode).
    X : np.ndarray
        Scaled feature array (N, F).
    seq_len : int
        Must match the value used during training.

    Returns
    -------
    np.ndarray
        Predictions of shape (N - seq_len,).
    """
    model.eval()
    ds = AQISequenceDataset(X, np.zeros(len(X)), seq_len)
    dl = DataLoader(ds, batch_size=128, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(model(xb).numpy())

    return np.concatenate(preds)
