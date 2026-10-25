import numpy as np
import pandas as pd
from slmtriage import core, noise


def _df(n=600, seed=0, eps=0.15):
    """Synthetic world: true labels; gold wrong with prob eps; three systems that mostly see the truth."""
    rng = np.random.default_rng(seed)
    true = rng.choice(core.CLASSES, n)
    gold = true.copy(); wrong = rng.random(n) < eps
    for i in np.flatnonzero(wrong):
        gold[i] = rng.choice([c for c in core.CLASSES if c != true[i]])
    df = pd.DataFrame({"id": [f"T-{i:05d}" for i in range(n)], "text": ["x"] * n, "gold": gold, "n_words": 5})
    for s, acc in [("f1", 0.9), ("f2", 0.9), ("f3", 0.92)]:
        pred = np.where(rng.random(n) < acc, true, rng.choice(core.CLASSES, n))
        df[s] = pred
        df[f"{s}_ok"] = core.strict_correct(core.to_sets(df.gold), core.to_sets(pred))
    df["stratum"] = noise.compute_strata(df)
    return df, wrong


def test_strata_definitions():
    df = pd.DataFrame({"gold": ["bug report", "bug report", "bug report|other", "other"],
                       "f1": ["bug report", "other", "other", "bug report"], "f2": ["bug report", "other", "other", "other"],
                       "f3": ["bug report", "other", "other", "bug report,feature request"]})
    for s in ["f1", "f2", "f3"]:
        df[f"{s}_ok"] = core.strict_correct(core.to_sets(df.gold), core.to_sets(df[s]))
    assert list(noise.compute_strata(df)) == [1, 2, 1, 3]  # row 3: 'other' is in the multi-label gold -> all agree with gold


def test_overlap_detects_shared_errors_and_null_is_calibrated():
    df, wrong = _df()
    res = noise.error_overlap(df, n_perm=2000, seed=1)
    assert len(res) == 3 and (res.p_perm < 0.01).all()           # shared label errors induce overlap
    rng = np.random.default_rng(3)                                 # independent errors -> no detection
    for s in ["f1", "f2"]:
        df[f"{s}_ok"] = rng.random(len(df)) < 0.8
    r2 = noise.error_overlap(df, systems=("f1", "f2"), n_perm=2000, seed=1)
    assert r2.p_perm.iloc[0] > 0.01 and abs(r2.J_obs.iloc[0] - r2.J_null_mean.iloc[0]) < 0.05


def test_sampling_and_queues(tmp_path):
    df, _ = _df()
    s = noise.draw_sample(df, "DSX", n_per_stratum=50, seed=7)
    assert set(s.stratum) <= {1, 2, 3} and s.groupby("stratum").size().max() <= 50 and s.item_key.is_unique
    assert (s.N_h.values >= s.n_h.values).all()
    info = noise.make_queues([s], tmp_path, core.DEFINITIONS)
    qa, qb = pd.read_csv(tmp_path / "queue_A.csv"), pd.read_csv(tmp_path / "queue_B.csv")
    assert list(qa.columns) == ["queue_pos", "item_key", "text", "label", "multi_intent", "note"]
    assert set(qa.item_key) == set(qb.item_key) == set(s.item_key) and list(qa.item_key) != list(qb.item_key)
    assert "gold" not in qa.columns and "stratum" not in qa.columns and info["n_items"] == len(s)
    ext = noise.external_subset(s, n=30)
    assert 25 <= len(ext) <= 35 and ext.item_key.is_unique


def test_stratified_rate_recovers_truth():
    df, wrong = _df(n=3000, eps=0.12)
    N_h = df.stratum.value_counts().to_dict()
    s = noise.draw_sample(df, "DSX", n_per_stratum=200, seed=11)
    n_h = s.stratum.value_counts().to_dict()
    truth = pd.Series(wrong, index=df.id)
    m_h = {h: int(truth.loc[s.id[s.stratum == h]].sum()) for h in n_h}
    r = noise.stratified_rate(N_h, n_h, m_h)
    assert r["ci"][0] <= wrong.mean() <= r["ci"][1] + 0.01, (r, wrong.mean())
    assert all(0 <= v["ci"][0] <= v["rate"] <= v["ci"][1] <= 1 for v in r["per_stratum"].values())
    assert noise.clopper_pearson(0, 100)[0] == 0.0 and noise.clopper_pearson(100, 100)[1] == 1.0


def test_ceiling_and_risk_ratio_and_adjudication():
    df, wrong = _df(n=1500, eps=0.1)
    s = noise.draw_sample(df, "DSX", n_per_stratum=100, seed=5)
    s["adjudicated_label"] = s.gold                                   # perfect agreement with gold -> ceiling 100
    assert abs(noise.weighted_ceiling(s, B=100)["ceiling"] - 100) < 1e-9
    err = np.array([True] * 60 + [False] * 240); flagged = np.array([True] * 42 + [False] * 18 + [True] * 33 + [False] * 207)
    rr = noise.risk_ratio(err, flagged)
    assert 2.6 < rr["RR"] < 3.0 and rr["ci"][0] < rr["RR"] < rr["ci"][1] and rr["fisher_p"] < 1e-6
    assert noise.assign_status("bug report", "bug report", "bug report", 0, 0, "") == "gold correct"
    assert noise.assign_status("bug report", "other", "other", 0, 0, "") == "gold wrong"
    assert noise.assign_status("bug report", "other", "bug report", 0, 0, "") == "contested"
    assert noise.assign_status("bug report", "other", "bug report", 0, 0, "other") == "gold wrong"
    assert noise.assign_status("bug report", "bug report", "bug report", 1, 1, "") == "contested"
    assert noise.assign_status("bug report|other", "other", "other", 0, 0, "") == "gold correct"
    qa = pd.DataFrame({"item_key": s.item_key, "label": s.gold.str.split("|").str[0], "multi_intent": 0})
    qb = qa.copy(); qb.loc[0, "label"] = "other" if qb.loc[0, "label"] != "other" else "bug report"
    adj = noise.adjudicate(s, qa, qb)
    assert adj.status.iloc[0] == "contested" and (adj.status.iloc[1:] == "gold correct").all()
