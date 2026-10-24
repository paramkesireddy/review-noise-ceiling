"""slmtriage.distilbert_cv — the reference fine-tuned baseline, re-implemented faithfully in PyTorch.

Recipe (Abedini & Heydarnoori 2025 notebook, 'Fine-Tined Models' cell): distilbert-base-uncased, one binary classifier per
class (one-vs-rest), StratifiedKFold(5, shuffle, random_state=1234) per class, balanced class weights, Adam lr 5e-5,
batch 16, 3 epochs, max_length 50, fp32. Their notebook passes input_ids WITHOUT attention masks at prediction time;
we evaluate every fold model both ways (mask = correct, nomask = faithful) and report both.

Outputs per (dataset, seed): results/jsonl/distilbert__{ds}__seed{s}.jsonl with one line per item (out-of-fold
probabilities and predictions) plus a .summary.json; a summary dict is returned.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from . import core

CFG_DEFAULT = dict(model="distilbert-base-uncased", max_length=50, batch=16, lr=5e-5, epochs=3, folds=5,
                   kfold_seed=core.SEEDS["kfold"], eval_batch=256, dtype="fp32")
GOLD_COLS = {"bug report": "gold_bug", "feature request": "gold_feature", "other": "gold_other"}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _encode(tok, texts, max_length):
    enc = tok(list(texts), padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _train_one(model_name, ids, mask, y, train_idx, seed, cfg, device):
    """Train one binary classifier on train_idx; returns the model (eval mode)."""
    import torch
    from transformers import AutoModelForSequenceClassification

    core.set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    counts = np.bincount(y[train_idx], minlength=2)
    cw = torch.tensor(len(train_idx) / (2 * np.maximum(counts, 1)), dtype=torch.float, device=device)  # sklearn 'balanced'
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(train_idx), generator=g).numpy()
        for s in range(0, len(train_idx), cfg["batch"]):
            b = train_idx[perm[s:s + cfg["batch"]]]
            out = model(input_ids=ids[b].to(device), attention_mask=mask[b].to(device))
            loss = lossf(out.logits.float(), torch.tensor(y[b], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def _predict(model, ids, mask, idx, use_mask, eval_batch, device):
    import torch

    probs = []
    with torch.no_grad():
        for s in range(0, len(idx), eval_batch):
            b = idx[s:s + eval_batch]
            out = model(input_ids=ids[b].to(device), attention_mask=(mask[b].to(device) if use_mask else None))
            probs.append(torch.softmax(out.logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def _summarize(df: pd.DataFrame, prob_cols: dict, pred_cols: dict) -> dict:
    """Per-class P/R/F1 for the thresholded one-vs-rest predictions, plus macro; plus single-label argmax macro-F1."""
    rows = []
    for variant, cols in pred_cols.items():
        for c in core.CLASSES:
            y = df[GOLD_COLS[c]].values; p = df[cols[c]].values
            P, R, F, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
            rows.append({"variant": variant, "class": c, "P": 100 * P, "R": 100 * R, "F1": 100 * F, "n_pos": int(y.sum())})
    per = pd.DataFrame(rows)
    macro = {v: {m: float(per[per.variant == v][m].mean()) for m in ["P", "R", "F1"]} for v in pred_cols}
    gold = core.to_sets(df["gold"])
    argmax = core.to_sets(df["label_argmax"])
    f_arg, per_arg = core.ovr_f1(gold, argmax)
    return {"per_class": rows, "macro_ovr": macro, "macroF1_argmax_single_label": f_arg,
            "per_class_F1_argmax": dict(zip(core.CLASSES, per_arg))}


def run_cv(dataset: str, proc_dir: Path, out_dir: Path, seed: int, repo: Path | None = None,
           cfg: dict | None = None, device: str = "cuda", verbose: bool = True) -> dict:
    """5-fold out-of-fold predictions for every class; pooled; both inference variants. Idempotent per (dataset, seed)."""
    import torch
    from transformers import AutoTokenizer

    cfg = {**CFG_DEFAULT, **(cfg or {})}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"distilbert__{dataset}__seed{seed}"
    path = out_dir / f"{run_id}.jsonl"
    df = pd.read_csv(Path(proc_dir) / f"{dataset}_eval.csv")
    if path.exists() and sum(1 for _ in open(path)) == len(df):
        if verbose:
            print("cached:", run_id)
        return json.loads((out_dir / f"{run_id}.summary.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    ids, mask = _encode(tok, df.text.astype(str), cfg["max_length"])
    n = len(df); t0 = time.time()
    prob = {c: np.zeros(n) for c in core.CLASSES}
    pred_mask = {c: np.zeros(n, dtype=int) for c in core.CLASSES}
    pred_nomask = {c: np.zeros(n, dtype=int) for c in core.CLASSES}
    fold_of = {c: np.full(n, -1, dtype=int) for c in core.CLASSES}
    for c in core.CLASSES:
        y = df[GOLD_COLS[c]].values.astype(int)
        skf = StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["kfold_seed"])
        for k, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
            model = _train_one(cfg["model"], ids, mask, y, tr, seed * 100 + k, cfg, device)
            p = _predict(model, ids, mask, te, True, cfg["eval_batch"], device)
            prob[c][te] = p; pred_mask[c][te] = (p > 0.5).astype(int)
            pred_nomask[c][te] = (_predict(model, ids, mask, te, False, cfg["eval_batch"], device) > 0.5).astype(int)
            fold_of[c][te] = k
            if verbose:
                fm = precision_recall_fscore_support(y[te], pred_mask[c][te], average="binary", zero_division=0)[2]
                print(f"  {run_id} {c:16s} fold {k}: F1(mask)={100 * fm:5.1f}  ({time.time() - t0:4.0f}s)")
            del model; torch.cuda.empty_cache()
    P = np.vstack([prob[c] for c in core.CLASSES]).T
    df_out = df[["id", "gold", "gold_bug", "gold_feature", "gold_other"]].copy()
    for c in core.CLASSES:
        key = c.split()[0]
        df_out[f"prob_{key}"] = prob[c]; df_out[f"pred_mask_{key}"] = pred_mask[c]
        df_out[f"pred_nomask_{key}"] = pred_nomask[c]; df_out[f"fold_{key}"] = fold_of[c]
    df_out["label_argmax"] = [core.CLASSES[i] for i in P.argmax(1)]
    df_out["pred_ovr_mask"] = ["|".join(c for c in core.CLASSES if pred_mask[c][i]) for i in range(n)]
    df_out["pred_ovr_nomask"] = ["|".join(c for c in core.CLASSES if pred_nomask[c][i]) for i in range(n)]
    with open(path, "w") as f:
        for r in df_out.to_dict("records"):
            f.write(json.dumps(r) + "\n")
    summary = _summarize(df_out, None,
                         {"mask": {c: f"pred_mask_{c.split()[0]}" for c in core.CLASSES},
                          "nomask": {c: f"pred_nomask_{c.split()[0]}" for c in core.CLASSES}})
    summary.update({"run_id": run_id, "dataset": dataset, "seed": seed, "n": n, "config": cfg, "wall_s": round(time.time() - t0, 1),
                    "git_commit": _git_commit(repo) if repo else "unknown", "finished_utc": core_now(),
                    "gpu": _gpu_name(), "torch": torch.__version__})
    (out_dir / f"{run_id}.summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        m = summary["macro_ovr"]
        print(f"{run_id}: macro-F1 mask={m['mask']['F1']:.1f} nomask={m['nomask']['F1']:.1f} | argmax single-label={summary['macroF1_argmax_single_label']:.1f} | {summary['wall_s']}s")
    return summary


def run_transfer(train_ds: str, test_ds: str, proc_dir: Path, out_dir: Path, seed: int, repo: Path | None = None,
                 cfg: dict | None = None, device: str = "cuda", verbose: bool = True) -> dict:
    """Train on ALL of train_ds (per class), predict ALL of test_ds. For RQ5 (cross-dataset shift)."""
    import torch
    from transformers import AutoTokenizer

    cfg = {**CFG_DEFAULT, **(cfg or {})}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"distilbert_transfer__{train_ds}_to_{test_ds}__seed{seed}"
    path = out_dir / f"{run_id}.jsonl"
    dtr = pd.read_csv(Path(proc_dir) / f"{train_ds}_eval.csv"); dte = pd.read_csv(Path(proc_dir) / f"{test_ds}_eval.csv")
    if path.exists() and sum(1 for _ in open(path)) == len(dte):
        if verbose:
            print("cached:", run_id)
        return json.loads((out_dir / f"{run_id}.summary.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    ids_tr, mask_tr = _encode(tok, dtr.text.astype(str), cfg["max_length"])
    ids_te, mask_te = _encode(tok, dte.text.astype(str), cfg["max_length"])
    n = len(dte); t0 = time.time(); prob = {}; pred = {}
    for c in core.CLASSES:
        y = dtr[GOLD_COLS[c]].values.astype(int)
        model = _train_one(cfg["model"], ids_tr, mask_tr, y, np.arange(len(dtr)), seed * 100 + 7, cfg, device)
        prob[c] = _predict(model, ids_te, mask_te, np.arange(n), True, cfg["eval_batch"], device)
        pred[c] = (prob[c] > 0.5).astype(int)
        del model; torch.cuda.empty_cache()
    P = np.vstack([prob[c] for c in core.CLASSES]).T
    df_out = dte[["id", "gold", "gold_bug", "gold_feature", "gold_other"]].copy()
    for c in core.CLASSES:
        key = c.split()[0]; df_out[f"prob_{key}"] = prob[c]; df_out[f"pred_mask_{key}"] = pred[c]
    df_out["label_argmax"] = [core.CLASSES[i] for i in P.argmax(1)]
    df_out["pred_ovr_mask"] = ["|".join(c for c in core.CLASSES if pred[c][i]) for i in range(n)]
    with open(path, "w") as f:
        for r in df_out.to_dict("records"):
            f.write(json.dumps(r) + "\n")
    summary = _summarize(df_out, None, {"mask": {c: f"pred_mask_{c.split()[0]}" for c in core.CLASSES}})
    summary.update({"run_id": run_id, "train": train_ds, "test": test_ds, "seed": seed, "n": n, "config": cfg,
                    "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(repo) if repo else "unknown",
                    "finished_utc": core_now(), "gpu": _gpu_name(), "torch": torch.__version__})
    (out_dir / f"{run_id}.summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"{run_id}: macro-F1 mask={summary['macro_ovr']['mask']['F1']:.1f} | argmax={summary['macroF1_argmax_single_label']:.1f} | {summary['wall_s']}s")
    return summary


def core_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"
