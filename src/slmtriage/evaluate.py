"""slmtriage.evaluate — score run files against the frozen gold and the reference GPT-4o predictions on the same rows.

A run file is either a local-model JSONL (fields: id, label, valid, multi?, label_ref?, ms_per_review, label_mass?) or a
DistilBERT JSONL (fields: id, label_argmax, pred_ovr_mask). Both are scored with the reference metric (one-vs-rest
macro-F1 on set-valued gold) plus the strict single-label view, with paired bootstrap CIs and McNemar vs GPT-4o.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import core

COLS = ["model", "prompt", "mode", "n", "macroF1_strict", "ci_lo", "ci_hi", "F1_bug", "F1_feat", "F1_other", "macroF1_refparser",
        "gpt4o_same_rows", "ratio_vs_gpt4o", "ratio_ci_lo", "ratio_ci_hi", "kappa_vs_gpt4o", "mcnemar_p", "better_worse",
        "invalid_pct", "multi_pct", "label_mass_med", "ms_per_review"]


def load_gold_and_ref(proc_dir: Path, dataset: str):
    ev = pd.read_csv(Path(proc_dir) / f"{dataset}_eval.csv").set_index("id")
    ref = pd.read_csv(Path(proc_dir) / f"{dataset}_reference_predictions.csv").set_index("id")
    return ev, ref


def evaluate_labels(ids, labels, dataset, proc_dir, ref_col="ref_label_gpt4", B=2000, ref_parser_labels=None) -> dict:
    ev, ref = load_gold_and_ref(proc_dir, dataset)
    gold = core.to_sets(ev.loc[ids, "gold"]); g4 = core.to_sets(ref.loc[ids, ref_col])
    pred = core.to_sets(labels)
    f, per = core.ovr_f1(gold, pred); f_g4 = core.ovr_f1(gold, g4)[0]
    ci, rci = core.paired_bootstrap(gold, pred, g4, B=B)
    p, b, c = core.mcnemar_exact(core.strict_correct(gold, pred), core.strict_correct(gold, g4))
    out = dict(n=len(ids), macroF1_strict=round(f, 2), ci_lo=round(ci[0], 2), ci_hi=round(ci[1], 2),
               F1_bug=round(per[0], 2), F1_feat=round(per[1], 2), F1_other=round(per[2], 2),
               gpt4o_same_rows=round(f_g4, 2), ratio_vs_gpt4o=round(f / f_g4, 4) if f_g4 else None,
               ratio_ci_lo=round(rci[0], 4), ratio_ci_hi=round(rci[1], 4), kappa_vs_gpt4o=round(core.kappa_single(pred, g4), 4),
               mcnemar_p=float(f"{p:.3g}"), better_worse=f"{b}/{c}", acc_strict=round(100 * core.strict_correct(gold, pred).mean(), 2))
    if ref_parser_labels is not None:
        out["macroF1_refparser"] = round(core.ovr_f1(gold, core.to_sets(ref_parser_labels))[0], 2)
    return out


def evaluate_run(path: Path, dataset: str, proc_dir: Path, B: int = 2000) -> dict:
    path = Path(path); rows = [json.loads(l) for l in open(path)]; rid = path.stem
    ids = [r["id"] for r in rows]
    if rid.startswith("distilbert"):  # baseline file: single-label argmax is the comparable view; OVR set reported too
        res = evaluate_labels(ids, [r["label_argmax"] for r in rows], dataset, proc_dir, B=B,
                              ref_parser_labels=[r.get("pred_ovr_mask", r["label_argmax"]) for r in rows])
        parts = rid.split("__")
        res.update(run=rid, model=parts[0], prompt="-", mode="argmax|ovr", invalid_pct=0.0, multi_pct=0.0, label_mass_med=None, ms_per_review=None)
        return res
    res = evaluate_labels(ids, [r["label"] for r in rows], dataset, proc_dir, B=B,
                          ref_parser_labels=[r.get("label_ref", r["label"]) for r in rows])
    ds, model, prompt, mode = rid.split("__")[:4]
    res.update(run=rid, model=model, prompt=prompt, mode=mode,
               invalid_pct=round(100 * np.mean([not r["valid"] for r in rows]), 2),
               multi_pct=round(100 * np.mean([bool(r.get("multi", False)) for r in rows]), 2),
               label_mass_med=round(float(np.median([r["label_mass"] for r in rows])), 3) if mode == "S" else None,
               ms_per_review=round(float(np.mean([r["ms_per_review"] for r in rows])), 1))
    return res


def summarize(dataset: str, jsonl_dir: Path, proc_dir: Path, tables_dir: Path, pattern: str = "*", B: int = 2000) -> pd.DataFrame:
    paths = sorted(p for p in Path(jsonl_dir).glob(f"{dataset}__{pattern}.jsonl") if ".partial" not in p.name)
    paths += sorted(Path(jsonl_dir).glob(f"distilbert__{dataset}__seed*.jsonl"))  # baseline rows join the same table
    tab = pd.DataFrame([evaluate_run(p, dataset, proc_dir, B=B) for p in paths])
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    tab.to_csv(Path(tables_dir) / f"summary_{dataset}.csv", index=False)
    return tab
