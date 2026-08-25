import torch

from reasoning.virtual_adapter import VirtualTokenAdapter


def test_adapter_vectors_can_be_flattened_for_a_later_linear_probe():
    adapter = VirtualTokenAdapter(embedding_dim=8, num_tokens=4)
    vectors = adapter(torch.randn(3, 32)).flatten(start_dim=1)
    assert vectors.shape == (3, 32)
