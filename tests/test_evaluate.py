import json
import numpy as np
import pandas as pd
from slmtriage import core, evaluate


def _frozen(tmp_path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    gold = rng.choice(core.CLASSES, n)
    ids = [f"DSX-{i:05d}" for i in range(n)]
    ev = pd.DataFrame({"id": ids, "text": ["t"] * n, "gold": gold})
    ev.to_csv(tmp_path / "DSX_eval.csv", index=False)
    g4 = np.where(rng.random(n) < 0.8, gold, rng.choice(core.CLASSES, n))  # GPT-4o agrees with gold 80% of the time
    pd.DataFrame({"id": ids, "ref_label_gpt4": g4}).to_csv(tmp_path / "DSX_reference_predictions.csv", index=False)
    return ids, gold, g4


def test_local_run_scoring(tmp_path):
    ids, gold, g4 = _frozen(tmp_path)
    run = tmp_path / "DSX__toy__P2__G__bf16.jsonl"
    with open(run, "w") as f:
        for i, lab in zip(ids, g4):  # a model that copies GPT-4o exactly
            f.write(json.dumps(dict(id=i, raw=lab, label=lab, valid=True, multi=False, label_ref=lab, ms_per_review=1.0)) + "\n")
    r = evaluate.evaluate_run(run, "DSX", tmp_path, B=100)
    assert r["ratio_vs_gpt4o"] == 1.0 and r["kappa_vs_gpt4o"] == 1.0 and r["better_worse"] == "0/0" and r["mcnemar_p"] == 1.0
    assert r["macroF1_strict"] == r["macroF1_refparser"] == r["gpt4o_same_rows"] and r["invalid_pct"] == 0.0


def test_distilbert_run_scoring_and_summary(tmp_path):
    ids, gold, g4 = _frozen(tmp_path)
    run = tmp_path / "distilbert__DSX__seed1.jsonl"
    with open(run, "w") as f:
        for i, lab in zip(ids, gold):  # a perfect baseline
            f.write(json.dumps(dict(id=i, label_argmax=lab, pred_ovr_mask=lab)) + "\n")
    r = evaluate.evaluate_run(run, "DSX", tmp_path, B=100)
    assert r["macroF1_strict"] == 100.0 and r["ratio_vs_gpt4o"] > 1.0 and r["mode"] == "argmax|ovr"
    run2 = tmp_path / "DSX__toy__P1__S__bf16.jsonl"
    with open(run2, "w") as f:
        for i, lab in zip(ids, gold):
            f.write(json.dumps(dict(id=i, scores={c: 0.0 for c in core.CLASSES}, label=lab, valid=True, label_mass=0.9, ms_per_review=2.0)) + "\n")
    tab = evaluate.summarize("DSX", tmp_path, tmp_path, tmp_path / "tables", B=50)
    assert len(tab) == 2 and set(tab["mode"]) == {"S", "argmax|ovr"} and (tmp_path / "tables" / "summary_DSX.csv").exists()
    assert tab.loc[tab["mode"] == "S", "label_mass_med"].iloc[0] == 0.9
