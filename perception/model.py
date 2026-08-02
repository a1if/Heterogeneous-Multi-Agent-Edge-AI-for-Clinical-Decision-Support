"""
CNN-LSTM architecture, exact spec from KB Section 3.2.

Input: (batch, 1, 360) — single-lead ECG, 360 samples, z-score normalized.
Output: (logits [batch, 5], context_vector [batch, 32])

The 32-dim context_vector is the LSTM context layer's output BEFORE the
classification head — this is the exact extraction point specified in the
research gap brief, Section 4 ("the output of the Perception Agent's second
LSTM layer... configured with hidden=32, return_sequences=False"), and is
what Arm B's adapter will consume (Day 3+). Both classification and feature
extraction reuse this single forward pass — no duplicate compute.
"""
import torch
import torch.nn as nn

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN block 1: Conv1D(32) -> BatchNorm -> ReLU -> MaxPool(stride=2) -> (32, 180)
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # CNN block 2: Conv1D(64) -> BatchNorm -> ReLU -> MaxPool(stride=2) -> (64, 90)
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # CNN block 3: Conv1D(128) -> BatchNorm -> ReLU -> MaxPool(stride=2) -> (128, 45)
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # Bidirectional LSTM: hidden=64, forward+backward -> (45, 128)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # LSTM context layer: hidden=32, return_sequences=False -> (32,) context vector
        self.context_lstm = nn.LSTM(input_size=128, hidden_size=32, batch_first=True, bidirectional=False)
        # Classification head: Dense(64) + Dropout(0.3) -> Dense(5) + Softmax
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        # x: (batch, 1, 360)
        x = self.block1(x)   # (batch, 32, 180)
        x = self.block2(x)   # (batch, 64, 90)
        x = self.block3(x)   # (batch, 128, 45)
        x = x.permute(0, 2, 1)  # (batch, 45, 128) — timesteps x features for LSTM

        bilstm_out, _ = self.bilstm(x)          # (batch, 45, 128)
        _, (h_n, _) = self.context_lstm(bilstm_out)  # h_n: (1, batch, 32)
        context_vector = h_n.squeeze(0)          # (batch, 32) — the adapter's extraction point

        logits = self.head(context_vector)       # (batch, 5)
        return logits, context_vector
