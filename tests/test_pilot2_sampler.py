import numpy as np

import p1_pilot2_stratified as p2


def test_stratified_windows_quota_record_and_tier():
    rng = np.random.default_rng(1)
    record_ids = np.repeat(np.arange(30), 400)  # 30 records x 400 beats
    tiers = rng.choice(p2.TIERS, size=len(record_ids), p=[0.9, 0.07, 0.03]).tolist()
    cells = p2.stratified_windows(tiers, record_ids, np.random.default_rng(0))
    rank = np.array([p2.RANK[t] for t in tiers])
    for (n, tier), starts in cells.items():
        assert len(starts) == p2.PER_CELL
        recs = [int(record_ids[s]) for s in starts]
        assert max(recs.count(r) for r in set(recs)) <= p2.MAX_PER_RECORD
        for s in starts:
            assert record_ids[s] == record_ids[s + n - 1]                    # inside one record
            assert p2.TIERS[rank[s:s + n].max()] == tier                     # window tier = its cell
