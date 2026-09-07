"""
model.py
============================================================
The LSTM model -- the "brain" of the orbit predictor.

Idea:
    Feed the LSTM a sequence of the last WINDOW_SIZE states
    (each state = 6 numbers: x, y, z, vx, vy, vz).
    It reads them in order, keeps a running "memory" of the motion,
    and outputs the predicted NEXT state (6 numbers).

Architecture (small + CPU-friendly on purpose):
    input (batch, 30, 6)
        -> LSTM (2 stacked layers, 64 hidden units, dropout 0.2)
        -> take the LAST timestep's hidden output
        -> Linear(64 -> 6)
    output (batch, 6)

Public API
----------
    OrbitLSTM        -> the nn.Module class
    build_model()    -> create one straight from config.py

Run directly for a shape sanity-check (no training):
    python model.py
============================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


# ------------------------------------------------------------
# THE MODEL
# ------------------------------------------------------------
class OrbitLSTM(nn.Module):
    """
    A compact LSTM regressor for satellite state prediction.

    Parameters
    ----------
    input_size  : number of features per timestep (6: x,y,z,vx,vy,vz)
    hidden_size : LSTM hidden units (memory width)
    num_layers  : how many LSTM layers are stacked
    output_size : numbers to predict (6: the next state)
    dropout     : regularization between stacked layers
    """

    def __init__(
        self,
        input_size: int = config.INPUT_SIZE,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        output_size: int = config.OUTPUT_SIZE,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True -> input/output shaped as (batch, seq_len, features)
        # dropout only applies BETWEEN layers, so it needs num_layers > 1.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Maps the LSTM's final hidden state to the 6 output features
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        returns : (batch, output_size)
        """
        # out: (batch, seq_len, hidden_size) -- hidden state at every timestep
        out, _ = self.lstm(x)

        # We only care about the LAST timestep's output: it has "seen" the
        # whole input sequence and summarizes it. Shape -> (batch, hidden_size)
        last_step = out[:, -1, :]

        # Project to the predicted next state -> (batch, output_size)
        return self.fc(last_step)


# ------------------------------------------------------------
# FACTORY: build a model straight from config
# ------------------------------------------------------------
def build_model(device: str | None = None) -> OrbitLSTM:
    """Create an OrbitLSTM using the hyperparameters in config.py."""
    model = OrbitLSTM()
    model.to(device or config.DEVICE)
    return model


# ------------------------------------------------------------
# Run directly: quick forward-pass shape check (no training)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("OrbitLSTM -- architecture sanity check")
    print("=" * 60)

    model = build_model(device="cpu")
    print(model)

    # Count trainable parameters (nice to quote in your report)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    # Fake a batch of 8 sequences to confirm the shapes flow through
    dummy = torch.randn(8, config.WINDOW_SIZE, config.INPUT_SIZE)
    out = model(dummy)

    print(f"\nInput  shape: {tuple(dummy.shape)}  (batch, window, features)")
    print(f"Output shape: {tuple(out.shape)}  (batch, predicted-features)")
    assert out.shape == (8, config.OUTPUT_SIZE), "Output shape mismatch!"
    print("\nShapes look correct. Model is ready to train.")
    print("\nDone.")
