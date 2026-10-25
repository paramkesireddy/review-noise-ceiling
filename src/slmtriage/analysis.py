"""slmtriage.analysis — regenerate every derived table from the frozen data and the run files. No GPU, no network.

    python -m slmtriage.analysis [--skip-perf]      # from the repo root; rebuilds results/tables/*

Sections (each skipped, with a message, when its inputs are not there yet)
  1 performance tables      summary_{DS}.csv                 evaluate.summarize (paired bootstrap, McNemar vs GPT-4o)
  2 convergence (RQ2, H2)   strata_sizes.csv, error_overlap.csv   permutation p with Holm over the 3 pairs per dataset
  3 artifacts (RQ1)         artifact_decomposition.csv       A1 parser, A2 prompt, A3 decision rule, A4 metric view, vs model gaps
  4 shift (RQ5, H5)         transfer_summary.csv             cross-dataset DistilBERT vs in-domain OOF and same-row GPT-4o
  5 contamination           contamination.csv                from the stored probe summaries
  6 noise ceiling (RQ3/4)   noise_ceiling_{DS}.json/.csv     H3 and H4 verdicts; only once data/annotation/annotator_{A,B}.csv exist
Decision rules are the pre-registered ones (configs/prereg.md, tag prereg-v1); verdict strings quote them.
A MANIFEST.json with the sha256 of every table, the git commit and the timestamp closes the run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import core, data, evaluate, noise

PRIMARY = ("P2", "G")  # pre-registered primary LLM configuration (configs/prereg.md)
# pre-registered thresholds (configs/prereg.md, "Hypotheses and decision rules")
H2_ALPHA = 0.01                 # permutation p, Holm over the 3 pairs per dataset
H3_EPS_SUPPORT, H3_EPS_REFUTE, H3_UPPER_REFUTE = 0.08, 0.04, 0.08
H4_RR = 3.0                     # errors >= 3x enriched on wrong-or-contested items; CI excluding 1; Holm over 3 systems
H5_DROP = 10.0                  # cross-dataset drop >= 10 points
NEGLIGIBLE = 2.0                # differences under 2 F1 points are pre-declared practically negligible


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj.tolist()


# ----------------------------------------------------------------------------- 1 + 2
def perf_tables(proc: Path, jl: Path, tb: Path, datasets=("DS3", "DS4"), B: int = 2000) -> dict:
    return {ds: evaluate.summarize(ds, jl, proc, tb, B=B) for ds in datasets}


def convergence(proc: Path, jl: Path, tb: Path, datasets=("DS3", "DS4"), n_perm: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    sizes, overlaps = [], []
    for ds in datasets:
        df = noise.load_system_labels(ds, proc, jl)
        vc = df.stratum.value_counts()
        sizes.append(dict(dataset=ds, n=len(df), h1=int(vc.get(1, 0)), h2=int(vc.get(2, 0)), h3=int(vc.get(3, 0)),
                          **{f"err_{s}": int((~df[f"{s}_ok"]).sum()) for s in ["f1", "f2", "f3"]}))
        o = noise.error_overlap(df, n_perm=n_perm); o.insert(0, "dataset", ds)
        o["p_holm"] = holm(o.p_perm.tolist()); o["H2"] = np.where(o.p_holm < H2_ALPHA, "supported", "not supported")
        overlaps.append(o)
    S, O = pd.DataFrame(sizes), pd.concat(overlaps, ignore_index=True)
    S.to_csv(tb / "strata_sizes.csv", index=False); O.to_csv(tb / "error_overlap.csv", index=False)
    return S, O


# ----------------------------------------------------------------------------- 3 artifact decomposition
def _distilbert_views(jl: Path, ds: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(jl).glob(f"distilbert__{ds}__seed*.summary.json")):
        s = json.loads(p.read_text())
        rows.append(dict(seed=s.get("seed"), argmax=s["macroF1_argmax_single_label"], mask=s["macro_ovr"]["mask"]["F1"],
                         nomask=s["macro_ovr"]["nomask"]["F1"]))
    return pd.DataFrame(rows)


def artifact_decomposition(tb: Path, jl: Path, datasets=("DS3", "DS4"), models=("qwen3-4b", "qwen3-1.7b"),
                           primary=PRIMARY) -> pd.DataFrame:
    """Size of each measurement choice (max swing in macro-F1 points, same model, same data) next to the model gaps."""
    rows = []

    def add(ds, model, artifact, points, detail):
        rows.append(dict(dataset=ds, model=model, artifact=artifact, points=round(float(points), 2), detail=detail))

    for ds in datasets:
        t = pd.read_csv(Path(tb) / f"summary_{ds}.csv")
        g4 = float(t.gpt4o_same_rows.iloc[0])
        V = _distilbert_views(jl, ds)
        if len(V):
            views = V[["argmax", "mask", "nomask"]]
            add(ds, "distilbert", "A4 metric view: argmax vs OVR-mask vs OVR-nomask (mean over seeds of max-min)",
                (views.max(axis=1) - views.min(axis=1)).mean(), "seed means: " + ", ".join(f"{c} {views[c].mean():.2f}" for c in views))
            add(ds, "distilbert", "training seed: sd of argmax macro-F1 over seeds", views.argmax.std(ddof=1) if len(V) > 1 else 0.0,
                "seeds " + ", ".join(f"{v:.2f}" for v in views.argmax))
        for m in models:
            tm = t[t.model == m]
            if tm.empty:
                continue
            G = tm[tm["mode"] == "G"].set_index("prompt").sort_index(); S = tm[tm["mode"] == "S"].set_index("prompt").sort_index()
            if len(G):
                d = G.macroF1_refparser - G.macroF1_strict
                add(ds, m, "A1 parser: reference substring vs strict first-mention (mode G, max |diff| over prompts)", d.abs().max(),
                    "; ".join(f"{p}: {v:+.2f}" for p, v in d.items()))
            for mode, X in [("G", G), ("S", S)]:
                if len(X) >= 2:
                    add(ds, m, f"A2 prompt: P1-P3 range (mode {mode})", X.macroF1_strict.max() - X.macroF1_strict.min(),
                        "; ".join(f"{p}: {v:.2f}" for p, v in X.macroF1_strict.items()))
            common = G.index.intersection(S.index)
            if len(common):
                d = G.loc[common].macroF1_strict - S.loc[common].macroF1_strict
                add(ds, m, "A3 decision rule: greedy generation vs label scoring (max |diff| over prompts)", d.abs().max(),
                    "; ".join(f"{p}: G-S {v:+.2f}" for p, v in d.items()))
            X = G if primary[1] == "G" else S
            if primary[0] in X.index:
                prim = float(X.loc[primary[0]].macroF1_strict)
                add(ds, m, f"model gap: GPT-4o minus {m} at the pre-registered config {primary[0]}-{primary[1]}", g4 - prim,
                    f"GPT-4o {g4:.2f}; {m} {prim:.2f}")
                if len(V):
                    add(ds, m, f"model gap: {m} ({primary[0]}-{primary[1]}) minus DistilBERT argmax (mean over seeds)",
                        prim - V.argmax.mean(), f"{m} {prim:.2f}; DistilBERT {V.argmax.mean():.2f}")
            best = tm.loc[tm.macroF1_strict.idxmax()]
            add(ds, m, f"model gap: GPT-4o minus {m} at its best post-hoc config (optimistic)", g4 - float(best.macroF1_strict),
                f"best = {best.prompt}-{best['mode']} {best.macroF1_strict:.2f}")
        add(ds, "gpt-4o", "model gap: GPT-4o minus DistilBERT argmax (mean over seeds)", g4 - V.argmax.mean() if len(V) else np.nan,
            f"GPT-4o {g4:.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(Path(tb) / "artifact_decomposition.csv", index=False)
    return out


# ----------------------------------------------------------------------------- 4 cross-dataset transfer
def transfer_summary(proc: Path, jl: Path, tb: Path, pairs=(("DS3", "DS4"), ("DS4", "DS3")), seeds=core.SEEDS["train"],
                     B: int = 2000) -> pd.DataFrame | None:
    rows = []
    for a, b in pairs:
        for seed in seeds:
            p = Path(jl) / f"distilbert_transfer__{a}_to_{b}__seed{seed}.jsonl"
            q = Path(jl) / f"distilbert__{b}__seed{seed}.summary.json"
            if not (p.exists() and q.exists()):
                continue
            r = evaluate.evaluate_run(p, b, proc, B=B); indom = json.loads(q.read_text())
            rows.append(dict(train=a, test=b, seed=seed, transfer_argmax=r["macroF1_strict"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             transfer_ovr_mask=r["macroF1_refparser"], F1_bug=r["F1_bug"], F1_feat=r["F1_feat"], F1_other=r["F1_other"],
                             indomain_argmax=round(indom["macroF1_argmax_single_label"], 2),
                             indomain_ovr_mask=round(indom["macro_ovr"]["mask"]["F1"], 2), gpt4o_same_rows=r["gpt4o_same_rows"],
                             kappa_vs_gpt4o=r["kappa_vs_gpt4o"], mcnemar_p=r["mcnemar_p"], better_worse=r["better_worse"]))
    if not rows:
        return None
    tab = pd.DataFrame(rows); tab["drop_argmax"] = (tab.indomain_argmax - tab.transfer_argmax).round(2)
    tab.to_csv(Path(tb) / "transfer_summary.csv", index=False)
    agg = tab.groupby(["train", "test"], as_index=False).agg(
        n_seeds=("seed", "size"), transfer_mean=("transfer_argmax", "mean"), transfer_sd=("transfer_argmax", "std"),
        indomain_mean=("indomain_argmax", "mean"), drop_mean=("drop_argmax", "mean"), drop_min=("drop_argmax", "min"),
        drop_max=("drop_argmax", "max"), gpt4o_same_rows=("gpt4o_same_rows", "first")).round(2)
    agg["H5"] = np.where(agg.drop_mean >= H5_DROP, f"met (drop >= {H5_DROP:g})", f"not met (drop < {H5_DROP:g})")
    agg["transfer_below_gpt4o_by"] = (agg.gpt4o_same_rows - agg.transfer_mean).round(2)
    agg.to_csv(Path(tb) / "transfer_agg.csv", index=False)
    return tab


# ----------------------------------------------------------------------------- 5 contamination (from stored summaries)
def contamination_table(jl: Path, tb: Path) -> pd.DataFrame | None:
    recs = []
    for p in sorted(Path(jl).glob("contam__DS*__*__*.summary.json")):
        s = json.loads(p.read_text())
        cp = Path(jl) / f"contam__control__{s['model']}__{s['variant']}.summary.json"
        c = json.loads(cp.read_text()) if cp.exists() else {}
        recs.append(dict(model=s["model"], variant=s["variant"], hf=s["hf"], dataset=s["dataset"], n_total=s["n_total"],
                         n_eligible=s["n_eligible"], eligible_share=s["eligible_share"], flagged=s["flagged"], flagged_rate=s["flagged_rate"],
                         ci_lo=s["ci95"][0], ci_hi=s["ci95"][1], exceeds_threshold=s["exceeds_threshold"], mean_matches=s["mean_matches"],
                         control_flagged=c.get("flagged"), control_n=c.get("n_eligible"), control_mean_matches=c.get("mean_matches")))
    if not recs:
        return None
    tab = pd.DataFrame(recs).sort_values(["variant", "model", "dataset"], ascending=[True, False, True]).reset_index(drop=True)
    tab.to_csv(Path(tb) / "contamination.csv", index=False)
    return tab


# ----------------------------------------------------------------------------- 6 noise ceiling
def _normalise_queue(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q["label"] = q.label.astype(str).str.strip().str.lower()
    q["multi_intent"] = pd.to_numeric(q.get("multi_intent", 0), errors="coerce").fillna(0).astype(int)
    return q


def multilabel_only_ceiling(sample: pd.DataFrame, B: int = 2000) -> dict:
    """Ceiling of a single-label oracle that always picks a gold label: the part of the ceiling owed to multi-label gold alone."""
    s = sample.copy()
    s["adjudicated_label"] = [sorted(g)[0] if g else "" for g in core.to_sets(s.gold)]
    return noise.weighted_ceiling(s, B=B)


def noise_ceiling(dataset: str, proc: Path, jl: Path, ann: Path, tb: Path, B: int = 2000) -> dict | None:
    """RQ3/RQ4 estimates for one dataset from the two blind queues (+ optional reconciled.csv). Returns None until the files exist."""
    ann = Path(ann)
    need = [ann / "sample.csv", ann / "annotator_A.csv", ann / "annotator_B.csv"]
    if not all(p.exists() for p in need):
        return None
    sample = pd.read_csv(ann / "sample.csv"); s = sample[sample.dataset == dataset].reset_index(drop=True)
    qa, qb = (_normalise_queue(pd.read_csv(ann / f"annotator_{x}.csv")) for x in "AB")
    rec = pd.read_csv(ann / "reconciled.csv") if (ann / "reconciled.csv").exists() else None
    adj = noise.adjudicate(s, qa, qb, rec)
    sysdf = noise.load_system_labels(dataset, proc, jl)
    adj = adj.merge(sysdf[["id", "f1", "f2", "f3", "f1_ok", "f2_ok", "f3_ok"]], on="id", how="left")
    assert adj[["f1", "f2", "f3", "label_A", "label_B"]].notna().all().all(), "unmatched items between sample, queues and run files"
    N_h = sysdf.stratum.value_counts().to_dict(); n_h = adj.stratum.value_counts().to_dict()
    for h in n_h:
        assert int(adj.N_h[adj.stratum == h].iloc[0]) == N_h[h], f"stratum {h} size changed since the sample was drawn"
    m_h = {h: int(((adj.stratum == h) & (adj.status == "gold wrong")).sum()) for h in n_h}
    c_h = {h: int(((adj.stratum == h) & (adj.status == "contested")).sum()) for h in n_h}
    eps_lo = noise.stratified_rate(N_h, n_h, m_h)                                   # confirmed gold errors only
    eps_hi = noise.stratified_rate(N_h, n_h, {h: m_h[h] + c_h[h] for h in n_h})      # if every contested item were a gold error
    # ceilings: contested items keep the original gold (they are not counted as errors); sensitivity: contested excluded
    keep = adj.copy(); keep.loc[keep.status == "contested", "adjudicated_label"] = keep.gold[keep.status == "contested"]
    excl = adj[adj.status != "contested"].reset_index(drop=True)
    ceiling = noise.weighted_ceiling(keep, B=B)
    ceiling_excl = noise.weighted_ceiling(excl, B=B) if len(excl) else None
    ml_only = multilabel_only_ceiling(adj, B=B)
    # per system (RQ4 / H4): enrichment of its errors on wrong-or-contested items (pre-registered flag set) and on wrong-only items;
    # F1 vs gold and vs adjudicated labels inside the sample (stratum-weighted); population F1 on the whole dataset for H3
    w = (adj.N_h / adj.n_h).to_numpy(float)
    flag_primary = adj.status.isin(["gold wrong", "contested"]).to_numpy(bool); flag_wrong = (adj.status == "gold wrong").to_numpy(bool)
    gold = core.to_sets(adj.gold); corrected = core.to_sets(keep.adjudicated_label); gold_pop = core.to_sets(sysdf.gold)
    systems = {}
    for f, name in noise.SYSTEMS.items():
        err = ~adj[f"{f}_ok"].to_numpy(bool); pred = core.to_sets(adj[f])
        rr, rr2 = noise.risk_ratio(err, flag_primary, w), noise.risk_ratio(err, flag_wrong, w)
        we = float((w * err).sum())
        systems[f] = dict(system=name, RR=rr["RR"], RR_ci=list(rr["ci"]), fisher_p=rr["fisher_p"], counts=rr["counts"],
                          RR_wrong_only=rr2["RR"], RR_wrong_only_ci=list(rr2["ci"]), fisher_p_wrong_only=rr2["fisher_p"],
                          share_of_errors_on_wrong_or_contested=float((w * (err & flag_primary)).sum() / we) if we else float("nan"),
                          share_of_errors_on_wrong_gold=float((w * (err & flag_wrong)).sum() / we) if we else float("nan"),
                          F1_vs_gold_weighted=core.ovr_f1(gold, pred, weights=w)[0],
                          F1_vs_adjudicated_weighted=core.ovr_f1(corrected, pred, weights=w)[0],
                          F1_population=core.ovr_f1(gold_pop, core.to_sets(sysdf[f]))[0])
    for f, p_adj in zip(systems, holm([systems[f]["fisher_p"] for f in systems])):
        v = systems[f]; v["fisher_p_holm"] = p_adj
        sig = p_adj < 0.05 and v["RR_ci"][0] > 1
        v["H4"] = ("supported" if sig and v["RR"] >= H4_RR else "enriched, below threshold" if sig else "not supported")
        v["H4_rule"] = f"RR >= {H4_RR:g} on wrong-or-contested items, 95% CI excluding 1, Holm over 3 systems"
    # H3 (pre-registered for DS3; computed for every dataset, DS4 is descriptive)
    eps_pt, eps_ci = eps_lo["estimate"], eps_lo["ci"]
    in_ci = {f: bool(ceiling["ci"][0] <= v["F1_population"] <= ceiling["ci"][1]) for f, v in systems.items()}
    conv_f1 = float(np.mean([v["F1_population"] for v in systems.values()]))   # "convergence F1": the common level of the 3 systems
    conv_in = bool(ceiling["ci"][0] <= conv_f1 <= ceiling["ci"][1])
    if eps_pt >= H3_EPS_SUPPORT and conv_in:
        h3 = "supported"
    elif eps_pt < H3_EPS_REFUTE and eps_ci[1] < H3_UPPER_REFUTE:
        h3 = "refuted"
    else:
        h3 = "indeterminate"
    H3 = dict(verdict=h3, pre_registered_for="DS3", epsilon_point=eps_pt, epsilon_ci=list(eps_ci), ceiling_ci=list(ceiling["ci"]),
              convergence_f1=conv_f1, convergence_f1_within_ceiling_ci=conv_in, systems_within_ceiling_ci=in_ci,
              rule=f"epsilon >= {H3_EPS_SUPPORT:g} and convergence F1 within the ceiling CI -> supported; epsilon < {H3_EPS_REFUTE:g} "
                   f"with upper CI < {H3_UPPER_REFUTE:g} -> refuted; otherwise indeterminate")
    # agreement before reconciliation, after reconciliation (reconciled label replaces both), and vs the external annotator
    qa_s, qb_s = qa[qa.item_key.isin(s.item_key)], qb[qb.item_key.isin(s.item_key)]
    agreement = {"before_reconciliation": _agree(qa_s, qb_s)}
    if rec is not None:
        r = rec.set_index("item_key")["reconciled_label"].dropna().astype(str).str.strip().str.lower()
        qa_r, qb_r = qa_s.copy(), qb_s.copy()
        for q in (qa_r, qb_r):
            hit = q.item_key.isin(r.index); q.loc[hit, "label"] = q.item_key[hit].map(r).values
        agreement["after_reconciliation"] = _agree(qa_r, qb_r)
    ext_p = ann / "annotator_external.csv"
    if ext_p.exists():
        ext = _normalise_queue(pd.read_csv(ext_p)); ext = ext[ext.item_key.isin(s.item_key)]
        if len(ext):
            adjl = keep[["item_key", "adjudicated_label"]].rename(columns={"adjudicated_label": "label"})
            agreement["external"] = dict(n=int(len(ext)), vs_A=_agree(ext, qa_s), vs_B=_agree(ext, qb_s), vs_adjudicated=_agree(ext, adjl))
    out = dict(dataset=dataset, n_sample=int(len(adj)), status_counts=adj.status.value_counts().to_dict(),
               strata=dict(N_h={int(k): int(v) for k, v in N_h.items()}, n_h={int(k): int(v) for k, v in n_h.items()},
                           gold_wrong={int(k): v for k, v in m_h.items()}, contested={int(k): v for k, v in c_h.items()}),
               epsilon_confirmed=eps_lo, epsilon_if_contested_wrong=eps_hi, ceiling=ceiling, ceiling_contested_excluded=ceiling_excl,
               ceiling_multilabel_only=ml_only, systems=systems, H3=H3, agreement=agreement)
    tb = Path(tb); tb.mkdir(parents=True, exist_ok=True)
    (tb / f"noise_ceiling_{dataset}.json").write_text(json.dumps(out, indent=2, default=_jsonable))
    flat = [dict(dataset=dataset, quantity="epsilon_confirmed", value=eps_lo["estimate"], ci_lo=eps_lo["ci"][0], ci_hi=eps_lo["ci"][1], verdict=f"H3 {h3}"),
            dict(dataset=dataset, quantity="epsilon_if_contested_wrong", value=eps_hi["estimate"], ci_lo=eps_hi["ci"][0], ci_hi=eps_hi["ci"][1], verdict=""),
            dict(dataset=dataset, quantity="ceiling_F1", value=ceiling["ceiling"], ci_lo=ceiling["ci"][0], ci_hi=ceiling["ci"][1], verdict=""),
            dict(dataset=dataset, quantity="ceiling_F1_multilabel_only", value=ml_only["ceiling"], ci_lo=ml_only["ci"][0], ci_hi=ml_only["ci"][1], verdict="")]
    for h, v in eps_lo["per_stratum"].items():
        flat.append(dict(dataset=dataset, quantity=f"epsilon_h{h} (m/n={v['m']}/{v['n']})", value=v["rate"], ci_lo=v["ci"][0], ci_hi=v["ci"][1], verdict=""))
    for f, v in systems.items():
        flat.append(dict(dataset=dataset, quantity=f"RR_{f} wrong-or-contested ({v['system']})", value=v["RR"], ci_lo=v["RR_ci"][0], ci_hi=v["RR_ci"][1],
                         verdict=f"H4 {v['H4']} (Holm p={v['fisher_p_holm']:.3g})"))
        flat.append(dict(dataset=dataset, quantity=f"RR_{f} wrong-only", value=v["RR_wrong_only"], ci_lo=v["RR_wrong_only_ci"][0], ci_hi=v["RR_wrong_only_ci"][1], verdict=""))
        flat.append(dict(dataset=dataset, quantity=f"F1_{f}_population", value=v["F1_population"], ci_lo=np.nan, ci_hi=np.nan,
                         verdict="within ceiling CI" if in_ci[f] else "outside ceiling CI"))
        flat.append(dict(dataset=dataset, quantity=f"F1_{f}_vs_gold_weighted", value=v["F1_vs_gold_weighted"], ci_lo=np.nan, ci_hi=np.nan, verdict=""))
        flat.append(dict(dataset=dataset, quantity=f"F1_{f}_vs_adjudicated_weighted", value=v["F1_vs_adjudicated_weighted"], ci_lo=np.nan, ci_hi=np.nan, verdict=""))
    for k, v in agreement.items():
        if "kappa" in v:
            flat.append(dict(dataset=dataset, quantity=f"kappa_{k}", value=v["kappa"], ci_lo=np.nan, ci_hi=np.nan, verdict=f"{v['percent_agreement']}% agreement, n={v['n']}"))
    pd.DataFrame(flat).round(4).to_csv(tb / f"noise_ceiling_{dataset}.csv", index=False)
    return out


def _agree(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    r = noise.agreement(a, b)
    return {k: v for k, v in r.items() if k != "disagreements"}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, tuple)):
        return list(o)
    raise TypeError(type(o))


# ----------------------------------------------------------------------------- driver
def make_all(repo: Path, skip_perf: bool = False, B: int = 2000, verbose: bool = True) -> dict:
    repo = Path(repo)
    proc, jl, tb, ann = repo / "data/processed", repo / "results/jsonl", repo / "results/tables", repo / "data/annotation"
    tb.mkdir(parents=True, exist_ok=True)
    assert data.integrity_check(proc), "processed data does not match manifest.json"
    done = {}

    def log(msg):
        if verbose:
            print(msg)

    if skip_perf:
        log("1 performance tables: skipped (--skip-perf)")
    else:
        perf_tables(proc, jl, tb, B=B); done["perf"] = True; log("1 performance tables: summary_DS3.csv, summary_DS4.csv")
    convergence(proc, jl, tb); done["convergence"] = True; log("2 convergence: strata_sizes.csv, error_overlap.csv")
    artifact_decomposition(tb, jl); done["artifacts"] = True; log("3 artifact decomposition: artifact_decomposition.csv")
    t = transfer_summary(proc, jl, tb, B=B); done["transfer"] = t is not None
    log("4 transfer: transfer_summary.csv" if t is not None else "4 transfer: skipped (no transfer runs)")
    c = contamination_table(jl, tb); done["contamination"] = c is not None
    log("5 contamination: contamination.csv" if c is not None else "5 contamination: skipped (no probe summaries)")
    for ds in ("DS3", "DS4"):
        r = noise_ceiling(ds, proc, jl, ann, tb, B=B); done[f"noise_{ds}"] = r is not None
        log(f"6 noise ceiling {ds}: noise_ceiling_{ds}.json/.csv" if r is not None else f"6 noise ceiling {ds}: skipped (annotator files missing)")
    tables = sorted(p for p in tb.glob("*.csv")) + sorted(tb.glob("*.json"))
    import datetime

    manifest = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), git_commit=_git(repo), sections=done,
                    tables={p.name: core.sha256_file(p) for p in tables if p.name != "MANIFEST.json"})
    (tb / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _git(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--skip-perf", action="store_true", help="do not recompute summary_{DS}.csv (the slow bootstrap part)")
    ap.add_argument("-B", type=int, default=2000)
    a = ap.parse_args()
    m = make_all(Path(a.repo), skip_perf=a.skip_perf, B=a.B)
    print(json.dumps(m["sections"]))
