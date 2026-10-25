"""slmtriage.noise — convergence (RQ2), strata and annotation sampling (RQ3), stratified estimators, adjudication (RQ3/RQ4).

Systems (pre-registered, configs/prereg.md): f1 = DistilBERT seed-1 out-of-fold argmax, f2 = Qwen3-4B P2 mode G (strict parser),
f3 = GPT-4o reference-parser label. "Agrees with gold" = single predicted label contained in the (possibly multi-label) gold set.
Strata: h1 all three agree with gold; h2 all three output the same single label and it is not in gold; h3 otherwise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact

from . import core

SYSTEMS = {"f1": "distilbert seed1 argmax", "f2": "qwen3-4b P2 G strict", "f3": "gpt-4o ref-parser"}


# ----------------------------------------------------------------------------- systems and strata
def load_system_labels(dataset: str, proc_dir: Path, jsonl_dir: Path, distilbert_seed: int = 1,
                       qwen_run: str = "qwen3-4b__P2__G__bf16") -> pd.DataFrame:
    proc_dir, jsonl_dir = Path(proc_dir), Path(jsonl_dir)
    ev = pd.read_csv(proc_dir / f"{dataset}_eval.csv")[["id", "text", "gold", "n_words"]]
    ref = pd.read_csv(proc_dir / f"{dataset}_reference_predictions.csv")[["id", "ref_label_gpt4"]]
    db = pd.DataFrame([json.loads(l) for l in open(jsonl_dir / f"distilbert__{dataset}__seed{distilbert_seed}.jsonl")])[["id", "label_argmax"]]
    qw = pd.DataFrame([json.loads(l) for l in open(jsonl_dir / f"{dataset}__{qwen_run}.jsonl")])[["id", "label"]]
    df = ev.merge(db, on="id", how="left").merge(qw, on="id", how="left").merge(ref, on="id", how="left")
    df = df.rename(columns={"label_argmax": "f1", "label": "f2", "ref_label_gpt4": "f3"})
    assert df[["f1", "f2", "f3"]].notna().all().all(), "missing predictions for some ids"
    gold = core.to_sets(df.gold)
    for s in ["f1", "f2", "f3"]:
        pred = core.to_sets(df[s])
        df[f"{s}_ok"] = core.strict_correct(gold, pred)
    df["stratum"] = compute_strata(df)
    return df


def compute_strata(df: pd.DataFrame) -> np.ndarray:
    gold = core.to_sets(df.gold)
    f = [core.to_sets(df[s]) for s in ["f1", "f2", "f3"]]
    h = np.full(len(df), 3, dtype=int)
    for i in range(len(df)):
        preds = [f[k][i] for k in range(3)]
        if all(df[f"{s}_ok"].iloc[i] for s in ["f1", "f2", "f3"]):
            h[i] = 1
        elif all(len(p) == 1 for p in preds) and preds[0] == preds[1] == preds[2] and not (preds[0] <= gold[i]):
            h[i] = 2
    return h


# ----------------------------------------------------------------------------- RQ2: error overlap
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def error_overlap(df: pd.DataFrame, systems=("f1", "f2", "f3"), n_perm: int = 10000, seed: int = 1) -> pd.DataFrame:
    """Observed Jaccard of error sets vs a permutation null (shuffle one system's error indicator over items)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            a, b = ~df[f"{systems[i]}_ok"].values, ~df[f"{systems[j]}_ok"].values
            j_obs = jaccard(a, b)
            null = np.empty(n_perm)
            for t in range(n_perm):
                null[t] = jaccard(a, rng.permutation(b))
            p = (1 + (null >= j_obs).sum()) / (1 + n_perm)
            rows.append(dict(pair=f"{systems[i]}-{systems[j]}", err_a=int(a.sum()), err_b=int(b.sum()), inter_obs=int((a & b).sum()),
                             inter_expected=float(len(a) * a.mean() * b.mean()), J_obs=round(j_obs, 4),
                             J_null_mean=round(float(null.mean()), 4), J_null_sd=round(float(null.std()), 4), p_perm=float(p)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- RQ3: sampling
def item_key(dataset: str, item_id: str, salt: str = "rnc-2026") -> str:
    return hashlib.sha1(f"{salt}|{dataset}|{item_id}".encode()).hexdigest()[:10]


def draw_sample(df: pd.DataFrame, dataset: str, n_per_stratum: int = 100, seed: int = core.SEEDS["annotation_sample"]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for h in [1, 2, 3]:
        pool = df.index[df.stratum == h].to_numpy()
        take = pool if len(pool) <= n_per_stratum else rng.choice(pool, n_per_stratum, replace=False)
        parts.append(df.loc[np.sort(take), ["id", "text", "gold", "stratum"]])
    s = pd.concat(parts).reset_index(drop=True)
    s.insert(0, "dataset", dataset)
    s["item_key"] = [item_key(dataset, i) for i in s.id]
    s["N_h"] = s.stratum.map(df.stratum.value_counts()); s["n_h"] = s.stratum.map(s.stratum.value_counts())
    return s


def make_queues(samples: list[pd.DataFrame], out_dir: Path, definitions: dict, seeds=(20261101, 20261102)) -> dict:
    """sample.csv (the key; never opened during annotation), queue_A/B.csv (text only, different orders), guidelines.md."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    allk = pd.concat(samples).reset_index(drop=True)
    allk.to_csv(out_dir / "sample.csv", index=False)
    for name, sd in zip("AB", seeds):
        q = allk[["item_key", "text"]].sample(frac=1.0, random_state=sd).reset_index(drop=True)
        q.insert(0, "queue_pos", range(1, len(q) + 1))
        q["label"] = ""; q["multi_intent"] = ""; q["note"] = ""
        q.to_csv(out_dir / f"queue_{name}.csv", index=False)
    lines = ["# Annotation guidelines", "", "Label each review with exactly ONE of: `bug report`, `feature request`, `other`.",
             "Decide from the review text alone. Do not consult the original dataset, model outputs, or the other annotator.", "",
             "Class definitions (verbatim from the evaluation prompt):", ""]
    lines += [f"- **{k}**: {v}" for k, v in definitions.items()]
    lines += ["", "Set `multi_intent` = 1 only when the reviewer expresses two intents in separate clauses (e.g. a crash AND a request).",
              "Put the dominant intent in `label` anyway. Use `note` for anything a third reader should know.",
              "Work in queue order; do not re-sort. Take breaks every 100 items. Do not discuss items until both queues are complete."]
    (out_dir / "guidelines.md").write_text("\n".join(lines) + "\n")
    return {"n_items": len(allk), "per_dataset_stratum": allk.groupby(["dataset", "stratum"]).size().to_dict()}


def external_subset(sample_all: pd.DataFrame, n: int = 100, seed: int = 20261028) -> pd.DataFrame:
    """Stratified (dataset x stratum, proportional) subset for the external annotator."""
    rng = np.random.default_rng(seed)
    g = sample_all.groupby(["dataset", "stratum"])
    alloc = (g.size() / len(sample_all) * n).round().astype(int)
    parts = [grp.loc[rng.choice(grp.index, min(int(alloc[k]), len(grp)), replace=False)] for k, grp in g]
    return pd.concat(parts).sort_values("item_key").reset_index(drop=True)


# ----------------------------------------------------------------------------- RQ3/RQ4: estimators
def clopper_pearson(m: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if m == 0 else beta.ppf(alpha / 2, m, n - m + 1)
    hi = 1.0 if m == n else beta.ppf(1 - alpha / 2, m + 1, n - m)
    return float(lo), float(hi)


def stratified_rate(N_h: dict, n_h: dict, m_h: dict) -> dict:
    """Stratified estimate of a population proportion with finite-population correction (Section 5 of the proposal)."""
    N = sum(N_h.values()); est = 0.0; var = 0.0; per = {}
    for h in N_h:
        W = N_h[h] / N; n = n_h[h]; m = m_h[h]; p = m / n if n else 0.0
        est += W * p
        var += (W ** 2) * (1 - n / N_h[h]) * (p * (1 - p) / (n - 1)) if n > 1 else 0.0
        per[h] = dict(N=N_h[h], n=n, m=m, rate=p, ci=clopper_pearson(m, n))
    se = float(np.sqrt(var))
    return dict(estimate=est, se=se, ci=(est - 1.96 * se, est + 1.96 * se), per_stratum=per)


def weighted_ceiling(sample: pd.DataFrame, adj_col: str = "adjudicated_label", B: int = 2000, seed: int = 1) -> dict:
    """Macro-F1 that a perfect classifier (= adjudicated labels) receives against the original gold, stratum-weighted."""
    gold = core.to_sets(sample.gold); adj = core.to_sets(sample[adj_col])
    w = (sample.N_h / sample.n_h).to_numpy(dtype=float)
    point = core.ovr_f1(gold, adj, weights=w)[0]
    rng = np.random.default_rng(seed); boots = []
    groups = {h: np.flatnonzero(sample.stratum.values == h) for h in sorted(sample.stratum.unique())}
    for _ in range(B):  # stratified bootstrap: resample within each stratum
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups.values()])
        boots.append(core.ovr_f1(gold, adj, idx=idx, weights=w[idx])[0])
    return dict(ceiling=point, ci=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))


def risk_ratio(err: np.ndarray, flagged: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """RR = P(flagged | error) / P(flagged); log-normal CI from the (weighted) 2x2; Fisher exact on raw counts."""
    err, flagged = np.asarray(err, bool), np.asarray(flagged, bool)
    w = np.ones(len(err)) if weights is None else np.asarray(weights, float)
    a = (w * (err & flagged)).sum(); b = (w * (err & ~flagged)).sum(); c = (w * (~err & flagged)).sum(); d = (w * (~err & ~flagged)).sum()
    p_flag_err = a / (a + b) if a + b else np.nan; p_flag = (a + c) / (a + b + c + d)
    rr = p_flag_err / p_flag if p_flag else np.nan
    se = np.sqrt(1 / max(a, 1e-9) - 1 / max(a + b, 1e-9) + 1 / max(a + c, 1e-9) - 1 / (a + b + c + d))
    ra, rb, rc, rd = int((err & flagged).sum()), int((err & ~flagged).sum()), int((~err & flagged).sum()), int((~err & ~flagged).sum())
    return dict(RR=float(rr), ci=(float(np.exp(np.log(rr) - 1.96 * se)), float(np.exp(np.log(rr) + 1.96 * se))) if rr and rr > 0 else (np.nan, np.nan),
                fisher_p=float(fisher_exact([[ra, rb], [rc, rd]])[1]), counts=dict(a=ra, b=rb, c=rc, d=rd))


# ----------------------------------------------------------------------------- adjudication
def assign_status(gold: str, label_a: str, label_b: str, multi_a, multi_b, reconciled: str | None) -> str:
    """gold correct / gold wrong / contested, mechanically from the labels (see prereg.md)."""
    gset = core.to_sets([gold])[0]
    flagged_multi = bool(int(multi_a or 0)) and bool(int(multi_b or 0))
    final = reconciled if reconciled not in (None, "", "nan") else (label_a if label_a == label_b else None)
    if flagged_multi or final is None:
        return "contested"
    return "gold correct" if final in gset else "gold wrong"


def adjudicate(sample: pd.DataFrame, qa: pd.DataFrame, qb: pd.DataFrame, reconciled: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join annotator queues (by item_key) to the sample; attach statuses; never edits the inputs."""
    a = qa.set_index("item_key")[["label", "multi_intent"]].rename(columns={"label": "label_A", "multi_intent": "multi_A"})
    b = qb.set_index("item_key")[["label", "multi_intent"]].rename(columns={"label": "label_B", "multi_intent": "multi_B"})
    out = sample.set_index("item_key").join(a).join(b)
    rec = reconciled.set_index("item_key")["reconciled_label"] if reconciled is not None else pd.Series(dtype=object)
    out["reconciled_label"] = rec.reindex(out.index)
    out["adjudicated_label"] = [r if isinstance(r, str) and r else (la if la == lb else "")
                                for r, la, lb in zip(out.reconciled_label.fillna(""), out.label_A, out.label_B)]
    out["status"] = [assign_status(g, la, lb, ma, mb, r) for g, la, lb, ma, mb, r in
                     zip(out.gold, out.label_A, out.label_B, out.multi_A, out.multi_B, out.reconciled_label.fillna(""))]
    return out.reset_index()


# ----------------------------------------------------------------------------- returned queues: validation and agreement
def validate_queue(returned_csv: Path, queue_csv: Path, allow_blank: int = 0) -> dict:
    """Check a returned annotation file against the issued queue: same keys, labels in the class set, no silent edits."""
    q = pd.read_csv(queue_csv); r = pd.read_csv(returned_csv)
    problems = []
    if set(q.item_key) != set(r.item_key):
        problems.append(f"item_key mismatch: missing {len(set(q.item_key) - set(r.item_key))}, extra {len(set(r.item_key) - set(q.item_key))}")
    if "label" not in r.columns:
        problems.append("no label column")
    else:
        lab = r.label.fillna("").astype(str).str.strip().str.lower()
        bad = sorted(set(lab[~lab.isin(core.CLASSES + [""])]))
        blanks = int((lab == "").sum())
        if bad:
            problems.append(f"labels outside the class set: {bad}")
        if blanks > allow_blank:
            problems.append(f"{blanks} blank labels")
    if "multi_intent" in r.columns:
        mi = r.multi_intent.fillna("").astype(str).str.strip()
        badmi = sorted(set(mi[~mi.isin(["", "1", "1.0", "0", "0.0"])]))
        if badmi:
            problems.append(f"multi_intent values other than 1/blank: {badmi}")
    merged = q[["item_key", "text"]].merge(r[["item_key", "text"]], on="item_key", suffixes=("_q", "_r"), how="inner")
    edited = int((merged.text_q.astype(str) != merged.text_r.astype(str)).sum())
    if edited:
        problems.append(f"{edited} review texts differ from the issued queue (edited or re-encoded)")
    dist = r.label.fillna("").astype(str).str.strip().str.lower().value_counts().to_dict() if "label" in r.columns else {}
    return {"ok": not problems, "problems": problems, "n": len(r), "label_distribution": dist}


def agreement(a: pd.DataFrame, b: pd.DataFrame, key: str = "item_key") -> dict:
    """Cohen's kappa and percent agreement between two returned queues; lists disagreements (key, A, B)."""
    from sklearn.metrics import cohen_kappa_score

    m = a[[key, "label"]].merge(b[[key, "label"]], on=key, suffixes=("_A", "_B"))
    la = m.label_A.astype(str).str.strip().str.lower(); lb = m.label_B.astype(str).str.strip().str.lower()
    dis = m[la.values != lb.values]
    return {"n": len(m), "percent_agreement": round(100 * float((la.values == lb.values).mean()), 1),
            "kappa": round(float(cohen_kappa_score(la, lb)), 3), "n_disagree": len(dis),
            "disagreements": dis.rename(columns={"label_A": "A", "label_B": "B"}).to_dict("records")}
