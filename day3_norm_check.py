"""Day 3: adapter internal-properties check -- three-way embedding-norm comparison.

Compares three L2-norm distributions (sprint plan, Day 3 "Adapter internal-
properties check"):
  (1) real_vocab      -- Gemma's own input token embedding table
  (2) virtual_init     -- an UNTRAINED adapter (fresh nn.init.normal_ scheme,
                           std=0.02/input_dim**0.5) on real held-out context vectors
  (3) virtual_trained  -- the TRAINED adapter on the same context vectors
(2) vs (3) isolates whether training drifted virtual embeddings off-manifold,
now backed by a paired Wilcoxon signed-rank test (n=320, matched positions) --
the properly-powered instrument, unlike any comparison against the full real
vocabulary (~250K rows), which is reported as a magnitude ratio only.
(1) vs (3) shows where they land relative to real tokens. Speaks directly to
Vision Wormhole (2026)'s claim that direct continuous-vector injection
"typically destabilizes generation."

Do NOT run before Day 3 -- alongside E1, per the sprint plan.

Run:
    python day3_norm_check.py
"""
import json

import numpy as np
import torch
from scipy import stats as scipy_stats

from perception.perception_agent import PerceptionAgent
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model
from reasoning.virtual_adapter import VirtualTokenAdapter
from day7_auditability_probe import select_events  # same 80-event selection as Day 6/7
from project_config import ADAPTER_CHECKPOINT, DS2_PATH, PERCEPTION_CHECKPOINT

RESULTS_PATH = "results/day3_norm_check_results.json"
INIT_ADAPTER_SEED = 42  # previously unset -- untrained baseline was non-reproducible run to run


def summarize(label: str, norms: np.ndarray) -> dict:
    print(f"{label}: n={len(norms)} mean={norms.mean():.4f} std={norms.std():.4f} "
          f"min={norms.min():.4f} max={norms.max():.4f}")
    return {"mean": float(norms.mean()), "std": float(norms.std()), "n": int(len(norms))}


def main():
    print("Loading Perception Agent + DS2, collecting held-out context vectors...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, record_ids = data["features"], data["labels"], data["record_ids"]
    context_vectors = []
    for idx in select_events(y, record_ids):  # same 80 events as Day 6/7
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

    torch.manual_seed(INIT_ADAPTER_SEED)  # untrained baseline is now reproducible run to run
    init_adapter = VirtualTokenAdapter.for_model(model, num_tokens=trained_adapter.num_tokens).to(device)
    with torch.no_grad():
        init_norms = init_adapter(ctx).norm(dim=-1).flatten().cpu().numpy()
        trained_norms = trained_adapter(ctx).norm(dim=-1).flatten().cpu().numpy()

    summary = {
        "real_vocab": summarize("real_vocab", real_vocab_norms),
        "virtual_init": summarize("virtual_init", init_norms),
        "virtual_trained": summarize("virtual_trained", trained_norms),
    }

    # Paired Wilcoxon: init vs trained, same 320 matched positions (80 events x
    # 4 virtual tokens). This is the properly-powered test -- both sides equal
    # N, unlike any comparison against the ~250K-row real vocabulary, which
    # stays a magnitude ratio only (see chapter prose) rather than a p-value,
    # since Mann-Whitney against a sample that large is mechanically
    # significant almost regardless of practical effect size.
    print("\nRunning paired Wilcoxon (init vs trained)...")
    try:
        stat, p = scipy_stats.wilcoxon(init_norms, trained_norms)
        stat, p = float(stat), float(p)
        interpretation = "drift is statistically supported" if p < 0.05 else "drift not statistically supported at this n"
        print(f"  statistic={stat:.4f}  p_value={p:.4e}  -> {interpretation}")
    except ValueError as e:
        # degenerate case: all paired differences are zero
        stat, p, interpretation = None, None, f"degenerate (all differences zero): {e}"
        print(f"  {interpretation}")

    summary["comparison"] = {
        "init_vs_trained_wilcoxon": {
            "statistic": stat,
            "p_value": p,
            "interpretation": interpretation,
            "n": len(init_norms),
        }
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()