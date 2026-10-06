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


def test_class_round_robin_spreads_top_classes():
    rng = np.random.default_rng(2)
    record_ids = np.repeat(np.arange(40), 300)
    tiers = rng.choice(p2.TIERS, size=len(record_ids), p=[0.85, 0.1, 0.05]).tolist()
    classes = rng.choice(["S", "V", "F"], size=len(record_ids), p=[0.8, 0.15, 0.05]).tolist()
    cells = p2.stratified_windows(tiers, record_ids, np.random.default_rng(0), ns=(5,), per_cell=12,
                                  classes=classes)
    rank = np.array([p2.RANK[t] for t in tiers])
    tops = [classes[s + int(np.argmax(rank[s:s + 5]))] for s in cells[(5, "priority")]]
    # without round-robin the dominant class (S, 80%) would fill the cell; with it, all three appear
    assert set(tops) == {"S", "V", "F"}
