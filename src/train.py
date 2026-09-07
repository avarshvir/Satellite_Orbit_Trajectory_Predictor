"""
train.py
============================================================
Trains the OrbitLSTM on the sliding-window sequences and saves
the best model as a full checkpoint (.pth).

Flow:
    1. prepare_data()   -> scaled train/test windows (+ saved scaler)
    2. DataLoaders      -> efficient mini-batches
    3. train loop       -> for each epoch: forward -> loss -> backprop -> step
    4. validate         -> measure test loss each epoch (watch it drop)
    5. save checkpoint  -> weights + config + scaler-path, best epoch only

We save a CHECKPOINT DICT (not just weights) so you can reproduce results
and resume training -- the professional approach we agreed on.

Run:
    python train.py
============================================================
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
from model import build_model
from preprocess import prepare_data


# ------------------------------------------------------------
# Build PyTorch DataLoaders from the numpy window arrays
# ------------------------------------------------------------
def make_loaders(data: Dict) -> tuple[DataLoader, DataLoader]:
    """Wrap numpy arrays into batched, tensor-based DataLoaders."""
    X_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"])
    X_test = torch.from_numpy(data["X_test"])
    y_test = torch.from_numpy(data["y_test"])

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    # Shuffle training batches (the SAMPLES are independent windows, so
    # shuffling batches is fine -- it does NOT leak time information).
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False)
    return train_loader, test_loader


# ------------------------------------------------------------
# Evaluate average loss over a loader (no gradient updates)
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> float:
    """Compute mean loss over an entire loader (used for validation)."""
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


# ------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------
def train() -> None:
    device = config.DEVICE
    print(f"[train] Device: {device}")

    # 1) Data
    data = prepare_data()
    train_loader, test_loader = make_loaders(data)

    # 2) Model, loss, optimizer
    model = build_model(device=device)
    criterion = nn.MSELoss()                       # regression -> mean sq error
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE)

    best_val = float("inf")
    history = {"train": [], "val": []}
    start = time.time()

    # 3) Epoch loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running, n = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()          # clear old gradients
            preds = model(xb)              # forward pass
            loss = criterion(preds, yb)    # how wrong are we?
            loss.backward()                # backprop the error
            optimizer.step()               # nudge weights to improve

            running += loss.item() * len(xb)
            n += len(xb)

        train_loss = running / max(n, 1)
        val_loss = evaluate_loss(model, test_loader, criterion, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # 4) Save the BEST model (lowest validation loss so far)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(model, optimizer, epoch, best_val, data)
            marker = "  <- saved (best)"

        print(f"Epoch {epoch:3d}/{config.EPOCHS} | "
              f"train {train_loss:.6f} | val {val_loss:.6f}{marker}")

    elapsed = time.time() - start
    print(f"\n[train] Done in {elapsed:.1f}s. Best val loss: {best_val:.6f}")
    print(f"[train] Best model saved -> {config.MODEL_PATH}")

    # Save the loss history so evaluate.py can plot the learning curve
    hist_path = config.MODELS_DIR / "history.npz"
    np.savez(hist_path, train=np.array(history["train"]),
             val=np.array(history["val"]))
    print(f"[train] Loss history saved -> {hist_path}")


# ------------------------------------------------------------
# Save a full checkpoint (weights + optimizer + config + scaler path)
# ------------------------------------------------------------
def _save_checkpoint(model, optimizer, epoch, val_loss, data) -> None:
    """Write a reproducible checkpoint dict to config.MODEL_PATH."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "feature_cols": data["feature_cols"],
        "scaler_path": str(config.MODELS_DIR / "scaler.joblib"),
        "hyperparams": {
            "input_size": config.INPUT_SIZE,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "output_size": config.OUTPUT_SIZE,
            "dropout": config.DROPOUT,
            "window_size": config.WINDOW_SIZE,
            "horizon": config.HORIZON,
        },
    }
    torch.save(checkpoint, config.MODEL_PATH)


# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Training OrbitLSTM")
    print("=" * 60)
    train()
    print("\nDone.")
