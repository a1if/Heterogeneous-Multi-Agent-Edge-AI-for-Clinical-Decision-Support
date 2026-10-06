"""Read-only companion to day3_norm_check.py: re-derives the same three norm
arrays (real_vocab, virtual_init, virtual_trained) and saves the RAW per-sample
values to .npy, for a genuine histogram/violin figure in the dissertation.
Does not touch day3_norm_check.py or its saved results_ledger.json-backing
summary; this is purely additive, for plotting only.

Run:
    python export_norm_check_raw.py
"""
import numpy as np
import torch

from perception.perception_agent import PerceptionAgent
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model
from reasoning.virtual_adapter import VirtualTokenAdapter
from day7_auditability_probe import select_events
from project_config import ADAPTER_CHECKPOINT, DS2_PATH, PERCEPTION_CHECKPOINT

INIT_ADAPTER_SEED = 42
OUT_PATH = "results/norm_check_raw_arrays.npz"


def main():
    print("Loading Perception Agent + DS2, collecting held-out context vectors...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, record_ids = data["features"], data["labels"], data["record_ids"]
    context_vectors = []
    for idx in select_events(y, record_ids):
        agent.predict(X[idx])
        context_vectors.append(agent.get_last_context_vector().copy())
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ctx = torch.from_numpy(np.stack(context_vectors)).float()

    print("Loading Gemma 4 E4B + trained adapter...")
    model, _ = load_model()
    trained_adapter = load_trained_adapter(ADAPTER_CHECKPOINT, model)
    device = next(trained_adapter.parameters()).device
    ctx = ctx.to(device)

    real_vocab_norms = model.get_input_embeddings().weight.detach().float().norm(dim=-1).cpu().numpy()

    torch.manual_seed(INIT_ADAPTER_SEED)
    init_adapter = VirtualTokenAdapter.for_model(model, num_tokens=trained_adapter.num_tokens).to(device)
    with torch.no_grad():
        init_norms = init_adapter(ctx).norm(dim=-1).flatten().cpu().numpy()
        trained_norms = trained_adapter(ctx).norm(dim=-1).flatten().cpu().numpy()

    print(f"real_vocab: n={len(real_vocab_norms)} mean={real_vocab_norms.mean():.4f}")
    print(f"virtual_init: n={len(init_norms)} mean={init_norms.mean():.4f}")
    print(f"virtual_trained: n={len(trained_norms)} mean={trained_norms.mean():.4f}")

    np.savez(OUT_PATH,
             real_vocab=real_vocab_norms,
             virtual_init=init_norms,
             virtual_trained=trained_norms)
    print(f"Saved raw arrays to {OUT_PATH}")


if __name__ == "__main__":
    main()
