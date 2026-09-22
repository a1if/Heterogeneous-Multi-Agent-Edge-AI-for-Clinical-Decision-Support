"""CNN-LSTM + RR branch (Phase 1 encoder sweep, plan §2.6.2).

Identical convolutional and BiLSTM stack to perception.model.CNNLSTM. The only
change: a small RR embedding (perception.rr_features, 5 standardised interval
features -> 16) is appended to every timestep entering the context LSTM. The
context vector is still that LSTM's 32-d final hidden state, the same extraction
point the adapter consumes, so every downstream component (adapter, probes) is
unchanged. Timing information reaches the context vector rather than bypassing
it, which is what the attribution protocol needs to see.

Input:  x (batch, 1, 360) z-scored window; rr (batch, 5) standardised RR features
Output: (logits [batch, 5], context_vector [batch, 32])
"""
import torch
import torch.nn as nn

from perception.model import CNNLSTM
from perception.rr_features import N_RR_FEATURES

RR_EMBED_DIM = 16


class CNNLSTMRR(CNNLSTM):
    def __init__(self):
        super().__init__()
        self.rr_embed = nn.Sequential(nn.Linear(N_RR_FEATURES, RR_EMBED_DIM), nn.ReLU())
        self.context_lstm = nn.LSTM(input_size=128 + RR_EMBED_DIM, hidden_size=32,
                                    batch_first=True, bidirectional=False)

    def forward(self, x, rr):
        x = self.block3(self.block2(self.block1(x)))  # (batch, 128, 45)
        bilstm_out, _ = self.bilstm(x.permute(0, 2, 1))  # (batch, 45, 128)
        rr_emb = self.rr_embed(rr).unsqueeze(1).expand(-1, bilstm_out.size(1), -1)
        _, (h_n, _) = self.context_lstm(torch.cat([bilstm_out, rr_emb], dim=2))
        context_vector = h_n.squeeze(0)  # (batch, 32)
        return self.head(context_vector), context_vector
