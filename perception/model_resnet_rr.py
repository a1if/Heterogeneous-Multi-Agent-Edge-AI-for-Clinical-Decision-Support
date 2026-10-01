"""ResNet1D + RR branch: the second non-transformer sender (Deviation 22).

Purely convolutional (no recurrence), unlike perception.model_rr.CNNLSTMRR. The RR
features enter through the same small embedding; the 32-d context vector is a tanh
bottleneck after global average pooling, the same extraction point and size the
adapter consumes, so every downstream component is unchanged.

Input:  x (batch, 1, 360) z-scored window; rr (batch, 5) standardised RR features
Output: (logits [batch, 5], context_vector [batch, 32])
"""
import torch
import torch.nn as nn

from perception.rr_features import N_RR_FEATURES

RR_EMBED_DIM = 16
CONTEXT_DIM = 32


class _Block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, 5, stride=stride, padding=2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 5, padding=2, bias=False), nn.BatchNorm1d(c_out))
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ResNet1DRR(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.stages = nn.Sequential(_Block(32, 32, 1), _Block(32, 32, 1),
                                    _Block(32, 64, 2), _Block(64, 64, 1),
                                    _Block(64, 128, 2), _Block(128, 128, 1))
        self.rr_embed = nn.Sequential(nn.Linear(N_RR_FEATURES, RR_EMBED_DIM), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(128 + RR_EMBED_DIM, CONTEXT_DIM), nn.Tanh())
        self.head = nn.Linear(CONTEXT_DIM, n_classes)

    def forward(self, x, rr):
        h = self.stages(self.stem(x)).mean(dim=2)  # global average pooling -> (batch, 128)
        context_vector = self.context(torch.cat([h, self.rr_embed(rr)], dim=1))
        return self.head(context_vector), context_vector
