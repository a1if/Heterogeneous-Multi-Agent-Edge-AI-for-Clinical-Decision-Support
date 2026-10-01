"""Exploratory (cited in the Phase 1 report's limitations): why the RR encoder does not detect F beats. CPU/GPU, about 1 min.

Per split: F-beat count and the records they come from; median RR features of F vs N vs V;
correlation of the mean F beat with the mean N and V beats; what seeds 0, 6 and 7 predict
for DS2's F beats.

Run (from repo root):
    python p1_f_class_analysis.py
"""
import collections

import numpy as np
import torch

from p1_io import save_json_atomic
from perception.model_rr import CNNLSTMRR
from perception.rr_features import FEATURE_NAMES, standardize
from train_perception_agent import VAL_RECORDS
from train_perception_agent_rr import load_with_rr, loader, predict

C = ["N", "S", "V", "F"]


def main():
    X1, y1, rr1, r1 = load_with_rr("ds1_train")
    X2, y2, rr2, r2 = load_with_rr("ds2_test")
    tr = ~np.isin(r1, list(VAL_RECORDS))
    out = {"design": __doc__, "splits": {}, "rr_feature_medians": {}, "mean_beat_correlation": {}, "seed_predictions": {}}
    for name, y, r, m in (("ds1_train", y1, r1, tr), ("ds1_val", y1, r1, ~tr), ("ds2_test", y2, r2, np.ones_like(y2, bool))):
        f = collections.Counter(int(x) for x in r[m][y[m] == 3])
        out["splits"][name] = {"f_beats": sum(f.values()), "beats": int(m.sum()), "by_record": dict(f.most_common())}
    for name, rr, y in (("ds1_train", rr1[tr], y1[tr]), ("ds2_test", rr2, y2)):
        out["rr_feature_medians"][name] = {C[c]: dict(zip(FEATURE_NAMES, np.median(rr[y == c], 0).round(3).tolist())) for c in (0, 2, 3)}
    for name, X, y in (("ds1_train", X1[tr], y1[tr]), ("ds2_test", X2, y2)):
        f, n, v = (X[y == c].mean(0) for c in (3, 0, 2))
        out["mean_beat_correlation"][name] = {"F_vs_N": float(np.corrcoef(f, n)[0, 1]), "F_vs_V": float(np.corrcoef(f, v)[0, 1])}
    out["mean_beat_correlation"]["F_ds1_vs_ds2"] = float(np.corrcoef(X1[tr][y1[tr] == 3].mean(0), X2[y2 == 3].mean(0))[0, 1])
    out["mean_beat_correlation"]["N_ds1_vs_ds2"] = float(np.corrcoef(X1[tr][y1[tr] == 0].mean(0), X2[y2 == 0].mean(0))[0, 1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in (0, 6, 7):
        ck = torch.load(f"perception/checkpoints/cnn_lstm_rr_seed{seed}.pt", map_location=dev)
        model = CNNLSTMRR().to(dev)
        model.load_state_dict(ck["state_dict"])
        p, _ = predict(model, loader(X2, standardize(rr2, ck["rr_standardizer"]), y2, batch_size=1024), dev, True)
        out["seed_predictions"][str(seed)] = {"f_predicted_as": dict(collections.Counter(C[i] for i in p[y2 == 3])),
                                              "predicted_f_total": int((p == 3).sum()),
                                              "predicted_f_correct": int(((p == 3) & (y2 == 3)).sum())}
    save_json_atomic("results/p1_f_class_analysis.json", out)
    for k in ("splits", "mean_beat_correlation", "seed_predictions"):
        print(k, out[k])


if __name__ == "__main__":
    main()
