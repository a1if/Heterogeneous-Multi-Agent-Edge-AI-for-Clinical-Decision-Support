"""Run s_class_expanded_check.py and day3_norm_check.py in one process so Gemma
loads once (reasoning/model_loader.py caches it at module level) instead of twice.

Run (from repo root):
    python run_gpu_checks_combined.py
"""
import s_class_expanded_check
import day3_norm_check

print("=" * 70)
print("S-CLASS EXPANDED CHECK (N=63)")
print("=" * 70)
s_class_expanded_check.main()

print("\n" + "=" * 70)
print("DAY 3 NORM CHECK (seeded, with Wilcoxon)")
print("=" * 70)
day3_norm_check.main()

print("\nBoth checks complete.")
