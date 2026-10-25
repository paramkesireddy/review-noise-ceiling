"""Synthetic-world tests for slmtriage.analysis: planted gold errors must be recovered; artifact swings must be arithmetic."""
import json

import numpy as np
import pandas as pd

from slmtriage import analysis, core, noise


def _world(tmp_path, n=900, eps=0.12, seed=0):
    rng = np.random.default_rng(seed)
    proc, jl, tb, ann = (tmp_path / d for d in ["proc", "jl", "tb", "ann"])
    for d in (proc, jl, tb, ann):
        d.mkdir()
    true = rng.choice(core.CLASSES, n)
    gold = true.copy(); wrong = rng.random(n) < eps
    for i in np.flatnonzero(wrong):
        gold[i] = rng.choice([c for c in core.CLASSES if c != true[i]])
    ids = [f"DSX-{i:05d}" for i in range(n)]
    ev = pd.DataFrame(dict(id=ids, text=[f"review number {i}" for i in range(n)], gold=gold, n_words=3))
    ev.to_csv(proc / "DSX_eval.csv", index=False)

    def noisy(acc):
        return np.where(rng.random(n) < acc, true, rng.choice(core.CLASSES, n))

    f1, f2, f3 = noisy(0.9), noisy(0.9), noisy(0.92)
    pd.DataFrame(dict(id=ids, ref_label_gpt4=f3)).to_csv(proc / "DSX_reference_predictions.csv", index=False)
    with open(jl / "distilbert__DSX__seed1.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps(dict(id=ids[i], label_argmax=f1[i], pred_ovr_mask=f1[i])) + "\n")
    with open(jl / "DSX__qwen3-4b__P2__G__bf16.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps(dict(id=ids[i], label=f2[i], valid=True, multi=False, label_ref=f2[i], ms_per_review=1.0)) + "\n")
    return proc, jl, tb, ann, true, wrong, ids


def test_noise_ceiling_recovers_planted_errors(tmp_path):
    proc, jl, tb, ann, true, wrong, ids = _world(tmp_path)
    df = noise.load_system_labels("DSX", proc, jl)
    s = noise.draw_sample(df, "DSX", n_per_stratum=100, seed=7)
    noise.make_queues([s], ann, core.DEFINITIONS)
    id2true = dict(zip(ids, true)); key2id = dict(zip(s.item_key, s.id))
    rng = np.random.default_rng(1)
    for name in "AB":
        q = pd.read_csv(ann / f"queue_{name}.csv")
        q["label"] = [id2true[key2id[k]] for k in q.item_key]
        if name == "B":  # 5% disagreements -> contested items, as in real life
            flip = rng.random(len(q)) < 0.05
            q.loc[flip, "label"] = [rng.choice([c for c in core.CLASSES if c != l]) for l in q.label[flip]]
        q["multi_intent"] = 0
        q.to_csv(ann / f"annotator_{name}.csv", index=False)
    out = analysis.noise_ceiling("DSX", proc, jl, ann, tb, B=200)
    assert out is not None and out["n_sample"] == len(s) and set(out["status_counts"]) <= {"gold correct", "gold wrong", "contested"}
    lo, hi = out["epsilon_confirmed"], out["epsilon_if_contested_wrong"]
    assert lo["ci"][0] - 0.01 <= wrong.mean() <= hi["ci"][1] + 0.01, (lo["estimate"], hi["estimate"], wrong.mean())
    assert hi["estimate"] >= lo["estimate"] and 0 < out["ceiling"]["ceiling"] < 100
    assert abs(out["ceiling_multilabel_only"]["ceiling"] - 100) < 1e-9          # no multi-label gold in this world
    for f, v in out["systems"].items():                                          # systems see the truth -> errors sit on wrong gold
        assert v["RR"] > 1.5 and v["F1_vs_adjudicated_weighted"] > v["F1_vs_gold_weighted"], (f, v)
    assert out["agreement"]["kappa"] > 0.8
    assert (tb / "noise_ceiling_DSX.json").exists() and len(pd.read_csv(tb / "noise_ceiling_DSX.csv")) > 10
    assert analysis.noise_ceiling("DSX", proc, jl, tmp_path / "nowhere", tb) is None


def test_artifact_decomposition_arithmetic(tmp_path):
    tb, jl = tmp_path / "tb", tmp_path / "jl"; tb.mkdir(); jl.mkdir()
    rows = [dict(run="distilbert__DSX__seed1", model="distilbert", prompt="-", mode="argmax|ovr", macroF1_strict=79.8, macroF1_refparser=78.6, gpt4o_same_rows=79.55)]
    vals = {("P1", "G"): (79.08, 80.0), ("P1", "S"): (74.22, 74.22), ("P2", "G"): (79.41, 79.9), ("P2", "S"): (79.64, 79.64),
            ("P3", "G"): (76.07, 77.5), ("P3", "S"): (77.41, 77.41)}
    for (p, m), (st, rf) in vals.items():
        rows.append(dict(run=f"DSX__qwen3-4b__{p}__{m}__bf16", model="qwen3-4b", prompt=p, mode=m, macroF1_strict=st, macroF1_refparser=rf, gpt4o_same_rows=79.55))
    pd.DataFrame(rows).to_csv(tb / "summary_DSX.csv", index=False)
    for seed, (a, mk, nm) in enumerate([(79.7, 78.6, 75.9), (79.9, 78.0, 76.5), (79.8, 79.2, 75.3)], 1):
        (jl / f"distilbert__DSX__seed{seed}.summary.json").write_text(json.dumps(dict(seed=seed, macroF1_argmax_single_label=a, macro_ovr=dict(mask=dict(F1=mk), nomask=dict(F1=nm)))))
    tab = analysis.artifact_decomposition(tb, jl, datasets=("DSX",), models=("qwen3-4b",))
    a2 = tab[tab.artifact.str.startswith("A2")].set_index("artifact").points
    assert a2.filter(like="mode G").iloc[0] == round(79.41 - 76.07, 2) and a2.filter(like="mode S").iloc[0] == round(79.64 - 74.22, 2)
    assert tab[tab.artifact.str.startswith("A1")].points.iloc[0] == round(77.5 - 76.07, 2)  # P3 has the largest parser swing
    assert tab[tab.artifact.str.startswith("A3")].points.iloc[0] == round(79.08 - 74.22, 2)
    assert tab[tab.artifact.str.startswith("A4")].points.iloc[0] == round(np.mean([79.7 - 75.9, 79.9 - 76.5, 79.8 - 75.3]), 2)
    gap = tab[tab.artifact.str.contains("pre-registered config P2-G")].points.iloc[0]
    assert gap == round(79.55 - 79.41, 2)
    assert (tb / "artifact_decomposition.csv").exists()


def test_contamination_and_transfer_tables(tmp_path):
    proc, jl, tb, ann, true, wrong, ids = _world(tmp_path)
    for ds, flagged in [("DSX", 1)]:
        (jl / f"contam__{ds}__qwen3-4b__base.summary.json").write_text(json.dumps(dict(
            run_id="x", model="qwen3-4b", variant="base", hf="Qwen/Qwen3-4B-Base", dataset=ds, n_total=3000, n_eligible=1500, eligible_share=0.5,
            flagged=flagged, flagged_rate=0.07, ci95=[0.0, 0.37], exceeds_threshold=False, mean_matches=0.55)))
    (jl / "contam__control__qwen3-4b__base.summary.json").write_text(json.dumps(dict(flagged=10, n_eligible=10, mean_matches=10.0)))
    ct = analysis.contamination_table(jl, tb)
    assert len(ct) == 1 and ct.control_flagged.iloc[0] == 10 and not ct.exceeds_threshold.iloc[0] and (tb / "contamination.csv").exists()
    # transfer: reuse the in-domain file as a "transfer" run DSX->DSX and a fake in-domain summary
    rows = [json.loads(l) for l in open(jl / "distilbert__DSX__seed1.jsonl")]
    with open(jl / "distilbert_transfer__DSX_to_DSX__seed1.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (jl / "distilbert__DSX__seed1.summary.json").write_text(json.dumps(dict(seed=1, macroF1_argmax_single_label=90.0, macro_ovr=dict(mask=dict(F1=89.0), nomask=dict(F1=88.0)))))
    tt = analysis.transfer_summary(proc, jl, tb, pairs=(("DSX", "DSX"),), seeds=[1], B=50)
    assert len(tt) == 1 and tt.indomain_argmax.iloc[0] == 90.0 and abs(tt.drop_argmax.iloc[0] - (90.0 - tt.transfer_argmax.iloc[0])) < 1e-6
    assert analysis.transfer_summary(proc, jl, tb, pairs=(("DSX", "DSZ"),), seeds=[1]) is None
