"""Deviation 22: the ResNet1D-RR sender and its loading through PerceptionAgent (CPU)."""
import torch

from perception.model_resnet_rr import ResNet1DRR
from perception.perception_agent import PerceptionAgent


def test_shapes_and_context_range():
    m = ResNet1DRR().eval()
    logits, ctx = m(torch.randn(4, 1, 360), torch.randn(4, 5))
    assert logits.shape == (4, 5) and ctx.shape == (4, 32)
    assert ctx.abs().max() <= 1.0  # tanh bottleneck


def test_agent_loads_resnet_checkpoint(tmp_path):
    torch.manual_seed(0)
    m = ResNet1DRR()
    stats = {"mean": [0.0] * 5, "std": [1.0] * 5}  # lists, as fit_standardizer saves them
    path = tmp_path / "resnet1d_rr_seed0.pt"
    torch.save({"state_dict": m.state_dict(), "rr_standardizer": stats, "arch": "resnet1d_rr", "seed": 0}, path)
    agent = PerceptionAgent(checkpoint_path=str(path), device="cpu")
    assert isinstance(agent.model, ResNet1DRR) and agent.uses_rr
