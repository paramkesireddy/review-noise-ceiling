"""slmtriage.core — constants, seeds, prompts, parsers and the evaluation metric.

Everything here is pure Python/NumPy (no torch) so it can be unit-tested anywhere.
The metric reproduces Abedini & Heydarnoori (2025), Table 2, to 0.1 (verified in the pilot).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- constants
CLASSES = ["bug report", "feature request", "other"]
CLASS_RE = {c: re.compile(r"\b" + re.escape(c) + r"\b") for c in CLASSES}
DATASETS = {"DS1": "guzman", "DS2": "maalej", "DS3": "jha", "DS4": "scalabrino"}
REF_MODELS = ["gpt3", "gpt4", "t5", "llama3"]

# Reference-study results recomputed from its replication package (pilot, Cell 6); coarse scheme, macro-F1.
REF_MACRO_F1_THREE = {
    "gpt4": {"DS1": 83.6, "DS2": 54.3, "DS3": 79.6, "DS4": 87.6},
    "gpt3": {"DS1": 84.6, "DS2": 55.5, "DS3": 78.7, "DS4": 86.6},
    "llama3": {"DS1": 81.3, "DS2": 52.7, "DS3": 78.8, "DS4": 85.8},
    "t5": {"DS1": 82.8, "DS2": 54.7, "DS3": 76.0, "DS4": 79.3},
}
# Paper Table 2, original scheme, GPT-4o macro-F1 (the metric gate target).
PAPER_TABLE2_GPT4O_ORIG = {"DS1": 47.2, "DS2": 58.3, "DS3": 82.0, "DS4": 75.8}
PAPER_TABLE2_DISTILBERT_ORIG = {"DS1": 40.3, "DS2": 51.8, "DS3": 63.8, "DS4": 59.2}

SEEDS = {"kfold": 1234, "train": [1, 2, 3], "bootstrap": 1, "annotation_sample": 20261027}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional here
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- prompts
# Class definitions copied from the reference notebook's `three_labels` (verified byte-equal at freeze time).
DEFINITIONS = {
    "bug report": (
        "report a problem, such as faulty behavior of the application. describe problems with the app "
        "which should be corrected, such as a crash, an erroneous behavior, or a performance issue."
    ),
    "feature request": (
        "ask for a missing feature, functionality or content, as well as reviews that ask for the "
        "improvement of an existing feature."
    ),
    "other": (
        "users express general appreciation with the application. It focuses on general judgment. They are "
        "simple text reflections of the numeric star rating. Ratings are less informative as they only include "
        "praise, dispraise, a distractive critique, or a dissuasion. Reviews where users describe workarounds, "
        "use cases and scenarios involving the app"
    ),
}


def reference_create_prompt(tlabels: list[list[str]]) -> str:
    """Verbatim port of the reference notebook's create_prompt."""
    prompt = "I will provide you with a textual user review of an Android app, aiming to categorize the reviews into specific intention classes:"
    for tlabel in tlabels:
        prompt += "\n" + tlabel[0] + ": " + tlabel[1]
    return prompt


def reference_create_query(tlabels: list[list[str]]) -> str:
    """Verbatim port of the reference notebook's create_query."""
    query = "Does the author's intention in this app review relate to a"
    for tlabel in tlabels:
        query += " " + tlabel[0] + " or"
    query = query[:-3] + "?"
    return query


def build_prompts(definitions: dict[str, str] = DEFINITIONS) -> dict:
    """P1 byte-identical to the reference study (incl. its stray quote); P2/P3 share the definitions."""
    tlabels = [[k, v] for k, v in definitions.items()]
    system_p1 = reference_create_prompt(tlabels)
    query = reference_create_query(tlabels)
    # reference loop: get_output(query + "\nreview: " + text + "'\nintention: ", prompt) with user content f"'{...}'"
    user_p1 = "'" + query + "\nreview: {text}'\nintention: '"
    def_block = "\n".join(f"{k}: {v}" for k, v in definitions.items())
    return {
        "P1": {
            "name": "reference_verbatim",
            "system": system_p1,
            "user": user_p1,
            "note": "byte-identical to Abedini & Heydarnoori 2025 notebook incl. stray quote; temperature 0",
        },
        "P2": {
            "name": "clean_same_definitions",
            "system": "You classify user reviews of a mobile app into exactly one intention class. Classes:\n" + def_block,
            "user": "Review: {text}\n\nWhich class does the author's intention belong to: bug report, feature request, or other?\nAnswer with only the class name.\nIntention:",
        },
        "P3": {
            "name": "reworded_same_definitions",
            "system": "Task: read an app review and decide what the user is trying to communicate. Possible answers and what they mean:\n" + def_block,
            "user": 'Here is the review:\n"""{text}"""\n\nRespond with one of: bug report / feature request / other. Output the label and nothing else.\nLabel:',
        },
        "definitions": dict(definitions),
        "classes": list(definitions.keys()),
        "verbalizers": {c: c for c in definitions.keys()},
        "reference_parser": "get_class: lowercase, collapse whitespace, substring-match each class name; none matched -> other",
    }


def prompts_json_text(prompts: dict) -> str:
    """Canonical serialization (what gets hashed into the manifest)."""
    return json.dumps(prompts, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------- parsers
def parse_strict(raw: str) -> tuple[str | None, bool, bool]:
    """label = first class mentioned (word-boundary, case-insensitive); valid iff any; multi iff >1 distinct."""
    s = re.sub(r"\s+", " ", str(raw).lower())
    hits = sorted(((m.start(), c) for c in CLASSES for m in [CLASS_RE[c].search(s)] if m), key=lambda x: x[0])
    return (hits[0][1] if hits else None), bool(hits), len({c for _, c in hits}) > 1


def parse_ref(raw: str) -> str:
    """Reference get_class, verbatim behaviour: substring match; none -> 'other'; may return several (comma-joined)."""
    s = re.sub(r"\s+", " ", str(raw).lower())
    hits = [c for c in CLASSES if c in s]
    return ",".join(hits) if hits else "other"


# ----------------------------------------------------------------------------- metric
def to_sets(labels) -> list[frozenset]:
    """'a,b' / 'a|b' / None / NaN / '' -> frozenset of valid class names."""
    out = []
    for l in labels:
        if l is None or (isinstance(l, float) and np.isnan(l)) or str(l) == "":
            out.append(frozenset())
            continue
        out.append(frozenset(p.strip() for p in str(l).replace("|", ",").split(",") if p.strip() in CLASSES))
    return out


def ovr_f1(gold_sets: list[frozenset], pred_sets: list[frozenset], idx=None, weights=None) -> tuple[float, list[float]]:
    """Per-class one-vs-rest F1 on set-valued gold/pred, macro-averaged (the reference metric).

    `idx` selects rows (bootstrap); `weights` (same length as idx/rows) weights confusion counts (stratified estimates).
    """
    n = len(gold_sets)
    idx = np.arange(n) if idx is None else np.asarray(idx)
    w = np.ones(len(idx)) if weights is None else np.asarray(weights, dtype=float)
    f1s = []
    for c in CLASSES:
        yt = np.fromiter((c in gold_sets[i] for i in idx), dtype=bool, count=len(idx))
        yp = np.fromiter((c in pred_sets[i] for i in idx), dtype=bool, count=len(idx))
        tp = float((w * (yt & yp)).sum()); fp = float((w * (~yt & yp)).sum()); fn = float((w * (yt & ~yp)).sum())
        f1s.append(0.0 if tp == 0 else 100.0 * 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)), f1s


def strict_correct(gold_sets, pred_sets) -> np.ndarray:
    """Exactly one predicted label and it is in the gold set."""
    return np.array([len(p) == 1 and next(iter(p)) in g for g, p in zip(gold_sets, pred_sets)])


def paired_bootstrap(gold_sets, pred_sets, ref_sets, B: int = 2000, seed: int = SEEDS["bootstrap"]):
    """95% percentile CIs for macro-F1(pred) and for the ratio F1(pred)/F1(ref), resampling items."""
    rng = np.random.default_rng(seed)
    n = len(gold_sets); f1s, ratios = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        f = ovr_f1(gold_sets, pred_sets, idx)[0]; r = ovr_f1(gold_sets, ref_sets, idx)[0]
        f1s.append(f); ratios.append(f / r if r else np.nan)
    return tuple(np.percentile(f1s, [2.5, 97.5])), tuple(np.nanpercentile(ratios, [2.5, 97.5]))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[float, int, int]:
    """Exact binomial McNemar on paired correctness; returns (p, b=a_right_b_wrong, c=a_wrong_b_right)."""
    from scipy.stats import binomtest

    b = int((correct_a & ~correct_b).sum()); c = int((~correct_a & correct_b).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return float(p), b, c


def kappa_single(pred_a_sets, pred_b_sets) -> float:
    """Cohen's kappa on items where both predictions are single-label."""
    from sklearn.metrics import cohen_kappa_score

    both = [i for i in range(len(pred_a_sets)) if len(pred_a_sets[i]) == 1 and len(pred_b_sets[i]) == 1]
    if not both:
        return float("nan")
    return float(cohen_kappa_score([next(iter(pred_a_sets[i])) for i in both], [next(iter(pred_b_sets[i])) for i in both]))


def score_predictions(gold_labels, pred_labels, ref_labels=None, B: int = 2000) -> dict:
    """One-call evaluation of a label column against gold (and optionally against a reference system)."""
    gold, pred = to_sets(gold_labels), to_sets(pred_labels)
    f, per = ovr_f1(gold, pred)
    out = {"n": len(gold), "macroF1": round(f, 2), "F1_bug": round(per[0], 2), "F1_feat": round(per[1], 2),
           "F1_other": round(per[2], 2), "acc_strict": round(100 * strict_correct(gold, pred).mean(), 2)}
    if ref_labels is not None:
        ref = to_sets(ref_labels)
        f_ref = ovr_f1(gold, ref)[0]
        ci, rci = paired_bootstrap(gold, pred, ref, B=B)
        p, b, c = mcnemar_exact(strict_correct(gold, pred), strict_correct(gold, ref))
        out.update({"ref_macroF1": round(f_ref, 2), "ci_lo": round(ci[0], 2), "ci_hi": round(ci[1], 2),
                    "ratio": round(f / f_ref, 4) if f_ref else None, "ratio_ci_lo": round(rci[0], 4), "ratio_ci_hi": round(rci[1], 4),
                    "mcnemar_p": p, "better_worse": f"{b}/{c}", "kappa_vs_ref": round(kappa_single(pred, ref), 4)})
    return out
