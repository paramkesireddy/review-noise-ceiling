"""paper/make_tables.py — write paper/tables/*.tex and paper/numbers.tex from results/tables (run after `python -m slmtriage.analysis`).

Every number in the manuscript is either \\input from paper/tables/ or a macro from paper/numbers.tex; nothing is typed by hand.
Sections whose inputs do not exist yet (the noise ceiling before annotation) get a placeholder so the PDF always compiles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from slmtriage import analysis, core  # noqa: E402

TB, JL, PROC, OUT = REPO / "results/tables", REPO / "results/jsonl", REPO / "data/processed", REPO / "paper/tables"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["DS3", "DS4"]
WORD = {"DS3": "DSthree", "DS4": "DSfour", "f1": "Fone", "f2": "Ftwo", "f3": "Fthree"}
macros: dict[str, str] = {}


def macro(name: str, value) -> None:
    assert name.isalpha(), name  # LaTeX macro names: letters only
    macros[name] = str(value)


def f(x, d=1) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(lo, hi, d=1) -> str:
    return f"[{f(lo, d)}, {f(hi, d)}]"


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body.strip() + "\n")
    print("wrote", OUT / name)


# ----------------------------------------------------------------------------- Table 1: main results
rows = {}
for ds in DATASETS:
    t = pd.read_csv(TB / f"summary_{ds}.csv")
    ev = pd.read_csv(PROC / f"{ds}_eval.csv"); ref = pd.read_csv(PROC / f"{ds}_reference_predictions.csv")
    gold, g4 = core.to_sets(ev.gold), core.to_sets(ref.ref_label_gpt4)
    f_g4 = core.ovr_f1(gold, g4)[0]; ci_g4, _ = core.paired_bootstrap(gold, g4, g4, B=2000)
    V = analysis._distilbert_views(JL, ds)
    db = t[t.model == "distilbert"]
    man = json.loads((PROC / "manifest.json").read_text())
    macro(f"n{WORD[ds]}", f"{len(ev):,}")
    macro(f"multilabel{WORD[ds]}", man["datasets"][ds].get("multilabel_gold", "--"))
    macro(f"gptFone{WORD[ds]}", f(f_g4))

    def row(key, label, r=None, mean=None, sd=None, kappa=None):
        rows.setdefault(key, {"label": label})
        if r is not None:
            rows[key][ds] = dict(f1=f(r.macroF1_strict), ci=ci(r.ci_lo, r.ci_hi), kappa=f(r.kappa_vs_gpt4o, 2))
        else:
            rows[key][ds] = dict(f1=f"{f(mean)} $\\pm$ {f(sd)}", ci="", kappa=f(kappa, 2) if kappa is not None else "--")

    rows.setdefault("gpt4o", {"label": "GPT-4o (reference outputs, reference parser)"})
    rows["gpt4o"][ds] = dict(f1=f(f_g4), ci=ci(*ci_g4), kappa="--")
    for key, mk, label, m in [("q4", "qFour", "Qwen3-4B, P2, greedy, strict parser", "qwen3-4b"),
                              ("q17", "qSeventeen", "Qwen3-1.7B, P2, greedy, strict parser", "qwen3-1.7b")]:
        r = t[(t.model == m) & (t.prompt == "P2") & (t["mode"] == "G")]
        if len(r):
            row(key, label, r=r.iloc[0])
            macro(f"{mk}F{WORD[ds]}", f(r.iloc[0].macroF1_strict)); macro(f"{mk}Kappa{WORD[ds]}", f(r.iloc[0].kappa_vs_gpt4o, 2))
            macro(f"{mk}Ratio{WORD[ds]}", f(r.iloc[0].ratio_vs_gpt4o, 3))
            macro(f"{mk}RatioCi{WORD[ds]}", ci(r.iloc[0].ratio_ci_lo, r.iloc[0].ratio_ci_hi, 3))
            macro(f"{mk}McNemar{WORD[ds]}", f"{r.iloc[0].mcnemar_p:.2g}")
    if len(V):
        row("dbArg", "DistilBERT, single-label argmax (3 seeds)", mean=V.argmax.mean(), sd=V.argmax.std(ddof=1), kappa=db.kappa_vs_gpt4o.mean() if len(db) else None)
        row("dbMask", "DistilBERT, one-vs-rest with attention mask (reference recipe)", mean=V["mask"].mean(), sd=V["mask"].std(ddof=1))
        row("dbNomask", "DistilBERT, one-vs-rest without attention mask", mean=V.nomask.mean(), sd=V.nomask.std(ddof=1))
        macro(f"dbArg{WORD[ds]}", f(V.argmax.mean())); macro(f"dbArgSd{WORD[ds]}", f(V.argmax.std(ddof=1)))
        macro(f"dbMask{WORD[ds]}", f(V["mask"].mean())); macro(f"dbNomask{WORD[ds]}", f(V.nomask.mean()))
        macro(f"dbKappa{WORD[ds]}", f(db.kappa_vs_gpt4o.mean(), 2) if len(db) else "--")
        macro(f"gapDb{WORD[ds]}", f"{f_g4 - V.argmax.mean():+.1f}")
        macro(f"gapDbMask{WORD[ds]}", f"{f_g4 - V['mask'].mean():+.1f}")
    r4 = t[(t.model == "qwen3-4b") & (t.prompt == "P2") & (t["mode"] == "G")]
    if len(r4):
        macro(f"gapQfour{WORD[ds]}", f"{f_g4 - r4.iloc[0].macroF1_strict:+.2f}")
    r17 = t[(t.model == "qwen3-1.7b") & (t.prompt == "P2") & (t["mode"] == "G")]
    if len(r17):
        macro(f"gapQseventeen{WORD[ds]}", f"{f_g4 - r17.iloc[0].macroF1_strict:+.1f}")

order = ["gpt4o", "q4", "q17", "dbArg", "dbMask", "dbNomask"]
lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{DS3 ($n$=\nDSthree)} & \multicolumn{2}{c}{DS4 ($n$=\nDSfour)} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"System & macro-F1 [95\% CI] & $\kappa$ & macro-F1 [95\% CI] & $\kappa$ \\", r"\midrule"]
for k in order:
    if k not in rows:
        continue
    r = rows[k]; cells = []
    for ds in DATASETS:
        c = r.get(ds, {"f1": "--", "ci": "", "kappa": "--"})
        cells += [f"{c['f1']} {c['ci']}".strip(), c["kappa"]]
    lines.append(f"{r['label']} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_main.tex", "\n".join(lines))

# ----------------------------------------------------------------------------- Table 2: artifact decomposition
A = pd.read_csv(TB / "artifact_decomposition.csv")


def pick(ds, model, prefix):
    r = A[(A.dataset == ds) & (A.model == model) & A.artifact.str.startswith(prefix)]
    return float(r.points.iloc[0]) if len(r) else np.nan


spec = [("A1 parser (reference substring vs.\\ strict first-mention), Qwen3-4B", "qwen3-4b", "A1"),
        ("A2 prompt wording (P1--P3), Qwen3-4B, greedy", "qwen3-4b", "A2 prompt: P1-P3 range (mode G)"),
        ("A2 prompt wording (P1--P3), Qwen3-4B, label scoring", "qwen3-4b", "A2 prompt: P1-P3 range (mode S)"),
        ("A3 decision rule (greedy vs.\\ label scoring), Qwen3-4B", "qwen3-4b", "A3"),
        ("A4 metric view (argmax vs.\\ OVR mask vs.\\ OVR no-mask), DistilBERT", "distilbert", "A4"),
        ("Training seed (s.d.\\ over 3 seeds), DistilBERT", "distilbert", "training seed"),
        ("GPT-4o $-$ Qwen3-4B (pre-registered P2, greedy)", "qwen3-4b", "model gap: GPT-4o minus qwen3-4b at the pre-registered"),
        ("GPT-4o $-$ DistilBERT argmax", "gpt-4o", "model gap: GPT-4o minus DistilBERT"),
        ("GPT-4o $-$ Qwen3-1.7B (P2, greedy)", "qwen3-1.7b", "model gap: GPT-4o minus qwen3-1.7b at the pre-registered")]
lines = [r"\begin{tabular}{@{}p{5.4cm}rr@{}}", r"\toprule", r"Measurement choice (same model, same data) & DS3 & DS4 \\", r"\midrule"]
for label, model, prefix in spec:
    if label.startswith("GPT-4o $-$ Qwen3-4B"):
        lines.append(r"\midrule")
    vals = [pick(ds, model, prefix) for ds in DATASETS]
    lines.append(f"{label} & {f(vals[0])} & {f(vals[1])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_artifacts.tex", "\n".join(lines))
for ds in DATASETS:
    macro(f"artMetric{WORD[ds]}", f(pick(ds, "distilbert", "A4"))); macro(f"artParser{WORD[ds]}", f(pick(ds, "qwen3-4b", "A1")))
    macro(f"artPromptG{WORD[ds]}", f(pick(ds, "qwen3-4b", "A2 prompt: P1-P3 range (mode G)")))
    macro(f"artPromptS{WORD[ds]}", f(pick(ds, "qwen3-4b", "A2 prompt: P1-P3 range (mode S)")))
    macro(f"artRule{WORD[ds]}", f(pick(ds, "qwen3-4b", "A3"))); macro(f"seedSd{WORD[ds]}", f(pick(ds, "distilbert", "training seed")))
allart = [pick(ds, m, p) for ds in DATASETS for (m, p) in [("qwen3-4b", "A1"), ("qwen3-4b", "A2 prompt: P1-P3 range (mode G)"),
                                                           ("qwen3-4b", "A2 prompt: P1-P3 range (mode S)"), ("qwen3-4b", "A3"), ("distilbert", "A4")]]
macro("artMin", f(np.nanmin(allart))); macro("artMax", f(np.nanmax(allart)))

# ----------------------------------------------------------------------------- Table 3: error overlap
O = pd.read_csv(TB / "error_overlap.csv")
S = pd.read_csv(TB / "strata_sizes.csv").set_index("dataset")
names = {"f1-f2": "DB--Q4B", "f1-f3": "DB--GPT-4o", "f2-f3": "Q4B--GPT-4o"}
lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Data & Pair & $J_{\mathrm{obs}}$ & $J_{\mathrm{null}}$ & shared errors & Holm $p$ \\", r"\midrule"]
for ds in DATASETS:
    o = O[O.dataset == ds]
    for _, r in o.iterrows():
        lines.append(f"{ds} & {names.get(r.pair, r.pair)} & {r.J_obs:.2f} & {r.J_null_mean:.2f} & {int(r.inter_obs)} / {r.inter_expected:.0f} & {r.p_holm:.4f} \\\\")
    macro(f"strataHone{WORD[ds]}", int(S.loc[ds, "h1"])); macro(f"strataHtwo{WORD[ds]}", int(S.loc[ds, "h2"])); macro(f"strataHthree{WORD[ds]}", int(S.loc[ds, "h3"]))
    macro(f"jMin{WORD[ds]}", f"{o.J_obs.min():.2f}"); macro(f"jMax{WORD[ds]}", f"{o.J_obs.max():.2f}"); macro(f"jNull{WORD[ds]}", f"{o.J_null_mean.mean():.2f}")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_overlap.tex", "\n".join(lines))
macro("overlapHolmP", f"{O.p_holm.max():.4f}")

# ----------------------------------------------------------------------------- Table 4: cross-dataset transfer
if (TB / "transfer_agg.csv").exists():
    G = pd.read_csv(TB / "transfer_agg.csv"); T = pd.read_csv(TB / "transfer_summary.csv")
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Train $\rightarrow$ test & transfer & drop & GPT-4o & bug & feat. & other \\", r"\midrule"]
    for _, g in G.iterrows():
        pc = T[(T.train == g.train) & (T.test == g.test)][["F1_bug", "F1_feat", "F1_other"]].mean()
        lines.append(f"{g.train} $\\rightarrow$ {g.test} & {g.transfer_mean:.1f} $\\pm$ {g.transfer_sd:.1f} & {g.drop_mean:.1f} & {g.gpt4o_same_rows:.1f} & {pc.F1_bug:.1f} & {pc.F1_feat:.1f} & {pc.F1_other:.1f} \\\\")
        key = f"{WORD[g.train]}To{WORD[g.test]}"
        macro(f"drop{key}", f(g.drop_mean)); macro(f"transfer{key}", f(g.transfer_mean)); macro(f"transferSd{key}", f(g.transfer_sd))
        macro(f"belowGpt{key}", f(g.transfer_below_gpt4o_by)); macro(f"dropMin{key}", f(g.drop_min)); macro(f"dropMax{key}", f(g.drop_max))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_transfer.tex", "\n".join(lines))

# ----------------------------------------------------------------------------- contamination macros
if (TB / "contamination.csv").exists():
    C = pd.read_csv(TB / "contamination.csv")
    for _, r in C.iterrows():
        key = f"{'Qfour' if r.model == 'qwen3-4b' else 'Qseventeen'}{r.variant.capitalize()}{WORD[r.dataset]}"
        macro(f"contamFlagged{key}", int(r.flagged)); macro(f"contamElig{key}", f"{int(r.n_eligible):,}")
        macro(f"contamUpper{key}", f"{r.ci_hi:.2f}"); macro(f"contamRate{key}", f"{r.flagged_rate:.2f}")
        macro(f"contamControl{'Qfour' if r.model == 'qwen3-4b' else 'Qseventeen'}{r.variant.capitalize()}", f"{int(r.control_flagged)}/{int(r.control_n)}")
        macro(f"eligShare{WORD[r.dataset]}", f"{100 * r.eligible_share:.0f}")

# ----------------------------------------------------------------------------- Table 5: noise ceiling (placeholder until annotation)
noise_lines = [r"\begin{tabular}{@{}p{3.6cm}ll@{}}", r"\toprule", r"Quantity & DS3 & DS4 \\", r"\midrule"]
have = {ds: (TB / f"noise_ceiling_{ds}.json").exists() for ds in DATASETS}
if any(have.values()):
    N = {ds: json.loads((TB / f"noise_ceiling_{ds}.json").read_text()) for ds in DATASETS if have[ds]}

    def cell(ds, fn):
        return fn(N[ds]) if ds in N else "--"

    noise_lines.append("Agreement before reconciliation ($\\kappa$) & " + " & ".join(cell(ds, lambda n: f(n['agreement']['before_reconciliation']['kappa'], 2)) for ds in DATASETS) + r" \\")
    noise_lines.append("Gold-label error rate $\\hat\\varepsilon$ [95\\% CI] & " + " & ".join(cell(ds, lambda n: f"{100 * n['epsilon_confirmed']['estimate']:.1f}\\% {ci(100 * n['epsilon_confirmed']['ci'][0], 100 * n['epsilon_confirmed']['ci'][1])}") for ds in DATASETS) + r" \\")
    noise_lines.append("Noise ceiling, macro-F1 [95\\% CI] & " + " & ".join(cell(ds, lambda n: f"{f(n['ceiling']['ceiling'])} {ci(*n['ceiling']['ci'])}") for ds in DATASETS) + r" \\")
    noise_lines.append("Convergence F1 (mean of 3 systems) & " + " & ".join(cell(ds, lambda n: f(n['H3']['convergence_f1'])) for ds in DATASETS) + r" \\")
    for fk, lab in [("f1", "DistilBERT"), ("f2", "Qwen3-4B"), ("f3", "GPT-4o")]:
        noise_lines.append(f"RR of {lab} errors on wrong-or-contested items [95\\% CI] & " + " & ".join(cell(ds, lambda n, fk=fk: f"{f(n['systems'][fk]['RR'], 2)} {ci(n['systems'][fk]['RR_ci'][0], n['systems'][fk]['RR_ci'][1], 2)}") for ds in DATASETS) + r" \\")
    for ds, n in N.items():
        macro(f"eps{WORD[ds]}", f"{100 * n['epsilon_confirmed']['estimate']:.1f}"); macro(f"epsLo{WORD[ds]}", f"{100 * n['epsilon_confirmed']['ci'][0]:.1f}")
        macro(f"epsHi{WORD[ds]}", f"{100 * n['epsilon_confirmed']['ci'][1]:.1f}"); macro(f"ceiling{WORD[ds]}", f(n["ceiling"]["ceiling"]))
        macro(f"ceilingLo{WORD[ds]}", f(n["ceiling"]["ci"][0])); macro(f"ceilingHi{WORD[ds]}", f(n["ceiling"]["ci"][1]))
        macro(f"hThreeVerdict{WORD[ds]}", n["H3"]["verdict"]); macro(f"kappaBefore{WORD[ds]}", f(n["agreement"]["before_reconciliation"]["kappa"], 2))
        for fk in ["f1", "f2", "f3"]:
            macro(f"rr{WORD[fk]}{WORD[ds]}", f(n["systems"][fk]["RR"], 2)); macro(f"hFourVerdict{WORD[fk]}{WORD[ds]}", n["systems"][fk]["H4"])
else:
    noise_lines.append(r"\multicolumn{3}{l}{\textit{Annotation in progress; this table is generated from results/tables/noise\_ceiling\_*.json.}} \\")
noise_lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_noise.tex", "\n".join(noise_lines))
for ds in DATASETS:
    for name in [f"eps{WORD[ds]}", f"epsLo{WORD[ds]}", f"epsHi{WORD[ds]}", f"ceiling{WORD[ds]}", f"ceilingLo{WORD[ds]}", f"ceilingHi{WORD[ds]}",
                 f"hThreeVerdict{WORD[ds]}", f"kappaBefore{WORD[ds]}"] + [f"rr{WORD[fk]}{WORD[ds]}" for fk in ["f1", "f2", "f3"]] + \
                [f"hFourVerdict{WORD[fk]}{WORD[ds]}" for fk in ["f1", "f2", "f3"]]:
        macros.setdefault(name, r"\textbf{[pending]}")

# ----------------------------------------------------------------------------- numbers.tex
man = json.loads((TB / "MANIFEST.json").read_text()) if (TB / "MANIFEST.json").exists() else {}
macro("tablesCommit", man.get("git_commit", "unknown"))
body = "% generated by paper/make_tables.py — do not edit\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
(REPO / "paper/numbers.tex").write_text(body + "\n")
print(f"wrote paper/numbers.tex ({len(macros)} macros)")
