"""End-to-end on a synthetic repository: make_all -> paper/make_tables.py -> LaTeX tables + numbers.tex, before and after annotation."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from slmtriage import analysis, core, data, noise

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_fake_repo(root: Path, n: int = 600, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    proc, jl, tb = root / "data/processed", root / "results/jsonl", root / "results/tables"
    for d in (proc, jl, tb, root / "paper", root / "data/annotation"):
        d.mkdir(parents=True, exist_ok=True)
    (proc / "prompts.json").write_text("{}")
    manifest = {"prompts_sha256": data.sha256(proc / "prompts.json"), "datasets": {}, "amendments": []}
    trues, wrongs = {}, {}
    for ds in ["DS3", "DS4"]:
        true = rng.choice(core.CLASSES, n); gold = true.copy(); wrong = rng.random(n) < 0.1
        for i in np.flatnonzero(wrong):
            gold[i] = rng.choice([c for c in core.CLASSES if c != true[i]])
        trues[ds], wrongs[ds] = true, wrong
        ids = [f"{ds}-{i:05d}" for i in range(n)]
        ev = pd.DataFrame(dict(id=ids, text_sha1="x", text=[f"review {i} of {ds}" for i in range(n)], text_pre="", gold=gold, n_words=5))
        for c, col in [("bug report", "gold_bug"), ("feature request", "gold_feature"), ("other", "gold_other")]:
            ev[col] = (ev.gold == c).astype(int)
        ev.to_csv(proc / f"{ds}_eval.csv", index=False)

        def noisy(acc, t=true):
            return np.where(rng.random(n) < acc, t, rng.choice(core.CLASSES, n))

        g4 = noisy(0.92)
        pd.DataFrame(dict(id=ids, ref_label_gpt4=g4, strict_label_gpt4=g4)).to_csv(proc / f"{ds}_reference_predictions.csv", index=False)
        manifest["datasets"][ds] = dict(eval_sha256=data.sha256(proc / f"{ds}_eval.csv"), n=n, multilabel_gold=0,
                                        reference_predictions_sha256=data.sha256(proc / f"{ds}_reference_predictions.csv"))
        for model, acc in [("qwen3-4b", 0.9), ("qwen3-1.7b", 0.8)]:
            for P in ["P1", "P2", "P3"]:
                for mode in ["G", "S"]:
                    lab = noisy(acc - (0.03 if P == "P3" else 0.0))
                    _jsonl(jl / f"{ds}__{model}__{P}__{mode}__bf16.jsonl",
                           [dict(id=ids[i], label=lab[i], valid=True, multi=False, label_ref=lab[i], ms_per_review=1.0, label_mass=0.9) for i in range(n)])
        for s in [1, 2, 3]:
            lab = noisy(0.9)
            _jsonl(jl / f"distilbert__{ds}__seed{s}.jsonl", [dict(id=ids[i], label_argmax=lab[i], pred_ovr_mask=lab[i]) for i in range(n)])
            f1 = core.ovr_f1(core.to_sets(gold), core.to_sets(lab))[0]
            (jl / f"distilbert__{ds}__seed{s}.summary.json").write_text(json.dumps(dict(
                seed=s, macroF1_argmax_single_label=f1, macro_ovr=dict(mask=dict(F1=f1 - 1.0), nomask=dict(F1=f1 - 3.0)))))
    for a, b in [("DS3", "DS4"), ("DS4", "DS3")]:
        ids = [f"{b}-{i:05d}" for i in range(n)]
        for s in [1, 2, 3]:
            lab = np.where(rng.random(n) < 0.8, trues[b], rng.choice(core.CLASSES, n))
            _jsonl(jl / f"distilbert_transfer__{a}_to_{b}__seed{s}.jsonl", [dict(id=ids[i], label_argmax=lab[i], pred_ovr_mask=lab[i]) for i in range(n)])
    for model in ["qwen3-4b", "qwen3-1.7b"]:
        for variant in ["base", "instruct"]:
            (jl / f"contam__control__{model}__{variant}.summary.json").write_text(json.dumps(dict(flagged=9, n_eligible=10, mean_matches=9.0)))
            for ds in ["DS3", "DS4"]:
                (jl / f"contam__{ds}__{model}__{variant}.summary.json").write_text(json.dumps(dict(
                    run_id="x", model=model, variant=variant, hf="h", dataset=ds, n_total=n, n_eligible=300, eligible_share=0.5, flagged=0,
                    flagged_rate=0.0, ci95=[0.0, 1.22], exceeds_threshold=False, mean_matches=0.5)))
    (proc / "manifest.json").write_text(json.dumps(manifest))
    return dict(trues=trues, wrongs=wrongs)


def _run_make_tables(root: Path) -> str:
    shutil.copy(ROOT / "paper/make_tables.py", root / "paper/make_tables.py")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, str(root / "paper/make_tables.py")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return (root / "paper/numbers.tex").read_text()


def test_make_tables_before_and_after_annotation(tmp_path):
    world = build_fake_repo(tmp_path)
    man = analysis.make_all(tmp_path, B=50, verbose=False)
    assert man["sections"]["perf"] and man["sections"]["transfer"] and man["sections"]["contamination"] and not man["sections"]["noise_DS3"]
    numbers = _run_make_tables(tmp_path)
    tabs = {p.name: p.read_text() for p in (tmp_path / "paper/tables").glob("*.tex")}
    assert {"tab_main.tex", "tab_artifacts.tex", "tab_overlap.tex", "tab_transfer.tex", "tab_noise.tex"} <= set(tabs)
    assert "Annotation in progress" in tabs["tab_noise.tex"] and r"\newcommand{\epsDSthree}{\textbf{[pending]}}" in numbers
    for m in [r"\nDSthree", r"\gapQfourDSthree", r"\gapDbDSfour", r"\dropDSthreeToDSfour", r"\contamUpperQfourBaseDSthree", r"\artMetricDSthree", r"\overlapHolmP"]:
        assert m + "}" in numbers, m
    assert tabs["tab_main.tex"].count(r"\\") >= 7 and "Qwen3-4B" in tabs["tab_main.tex"] and "DistilBERT" in tabs["tab_main.tex"]
    assert tabs["tab_transfer.tex"].count("rightarrow") == 3  # header + two directions
    # --- annotation arrives: both queues labelled with the planted truth (B disagrees on 5%)
    ann, proc, jl = tmp_path / "data/annotation", tmp_path / "data/processed", tmp_path / "results/jsonl"
    samples = [noise.draw_sample(noise.load_system_labels(ds, proc, jl), ds, n_per_stratum=60, seed=7) for ds in ["DS3", "DS4"]]
    noise.make_queues(samples, ann, core.DEFINITIONS)
    key2true = {}
    for s in samples:
        t = world["trues"][s.dataset.iloc[0]]
        key2true.update({k: t[int(i.split("-")[1])] for k, i in zip(s.item_key, s.id)})
    rng = np.random.default_rng(3)
    for name in "AB":
        q = pd.read_csv(ann / f"queue_{name}.csv"); q["label"] = [key2true[k] for k in q.item_key]; q["multi_intent"] = 0
        if name == "B":
            flip = rng.random(len(q)) < 0.05
            q.loc[flip, "label"] = [rng.choice([c for c in core.CLASSES if c != l]) for l in q.label[flip]]
        q.to_csv(ann / f"annotator_{name}.csv", index=False)
    man = analysis.make_all(tmp_path, skip_perf=True, B=50, verbose=False)
    assert man["sections"]["noise_DS3"] and man["sections"]["noise_DS4"]
    numbers = _run_make_tables(tmp_path)
    tn = (tmp_path / "paper/tables/tab_noise.tex").read_text()
    assert "RR of" in tn and "Annotation in progress" not in tn
    eps = float(numbers.split(r"\newcommand{\epsDSthree}{")[1].split("}")[0])
    assert 4.0 < eps < 16.0, eps                                     # planted 10%, confirmed-only estimate on a 180-item sample
    assert r"\newcommand{\hThreeVerdictDSthree}{" in numbers and r"\newcommand{\rrFtwoDSfour}{" in numbers
