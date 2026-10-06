"""
Auxiliary reconstruction decoder for the Perception Agent.

Tests whether the auditability ceiling found in day7_auditability_probe.py
(context-32d attribution probe recall 69.6%, see
aux_training_precommitted_analysis_plan.md) is a property of the
classification-only training objective rather than of the 32-dim bottleneck
itself. Attaches alongside the existing CNNLSTM classification head in
perception/model.py -- the encoder and classification head are unchanged;
this module only adds a second head trained jointly via combined_loss().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionDecoder(nn.Module):
    """
    Mirrors CNNLSTM's conv encoder in reverse: three transpose-conv blocks
    taking the 32-dim context vector back up to the 360-sample window.
    128 channels x 45 timesteps as the seed shape so three stride-2 upsamples
    (45 -> 90 -> 180 -> 360) land exactly on WINDOW_LEN, matching the
    encoder's own 360 -> 180 -> 90 -> 45 downsampling in model.py.
    """

    def __init__(self, context_dim: int = 32, target_length: int = 360):
        super().__init__()
        self.target_length = target_length

        self.fc = nn.Linear(context_dim, 128 * 45)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, context_vector: torch.Tensor) -> torch.Tensor:
        x = self.fc(context_vector)
        x = x.view(-1, 128, 45)
        x = self.deconv(x)
        x = x.squeeze(1)

        # Guard against off-by-a-few-samples from the stride/padding math above.
        if x.shape[-1] != self.target_length:
            x = F.interpolate(
                x.unsqueeze(1), size=self.target_length, mode="linear", align_corners=False
            ).squeeze(1)
        return x  # (batch, 360), same shape/scale as data_prep.py's normalized window


def combined_loss(
    classification_logits: torch.Tensor,
    classification_target: torch.Tensor,
    reconstructed_window: torch.Tensor,
    original_window: torch.Tensor,
    lam: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """L = L_classification + lam * L_reconstruction.

    Returns components separately (not just the sum) so loss_recon can be
    logged and checked for convergence independently of the downstream probe
    result, per the pre-committed plan's single-seed sanity check.
    """
    l_cls = F.cross_entropy(classification_logits, classification_target)
    l_recon = F.mse_loss(reconstructed_window, original_window)
    l_total = l_cls + lam * l_recon
    return l_total, {"loss_total": l_total.item(), "loss_cls": l_cls.item(), "loss_recon": l_recon.item()}
