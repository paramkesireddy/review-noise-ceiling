"""slmtriage.contamination — verbatim-completion probe for pre-training contamination (pre-registered check).

For every review with at least PREFIX+CONT tokens: feed the model the first PREFIX tokens of the raw review text (no chat
template, no instruction), greedily generate CONT tokens, count position-wise matches with the true continuation. An item is
flagged when matches >= FLAG_AT. The statistic is the flagged rate on eligible items (eligible share reported); the
pre-registered concern threshold is THRESHOLD_PCT.
Primary probe: the Qwen3 *base* checkpoints (before instruction tuning). Secondary: the instruct checkpoints we evaluated.
Positive control: public-domain passages that any web-scale corpus contains; the probe must flag most of them, else it has no power.
Outputs: results/jsonl/contam__{dataset|control}__{model}__{variant}.jsonl (+ .summary.json) and results/tables/contamination.csv.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .local_hf import REGISTRY, _git_commit, core_now

PREFIX, CONT, FLAG_AT = 12, 10, 8
THRESHOLD_PCT = 2.0
VARIANTS = {
    "base": {"qwen3-4b": "Qwen/Qwen3-4B-Base", "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base"},
    "instruct": {k: v["hf"] for k, v in REGISTRY.items()},
}
# Positive controls: public-domain texts (all first published before 1929) that any web-scale pre-training corpus contains.
CONTROL_PASSAGES = {
    "declaration_1776": "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.",
    "gettysburg_1863": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",
    "constitution_1787": "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity,",
    "two_cities_1859": "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light,",
    "pride_1813": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood,",
    "genesis_kjv_1611": "In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.",
    "alice_1865": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it,",
    "moby_dick_1851": "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
    "lorem_ipsum": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "lincoln_second_1865": "With malice toward none, with charity for all, with firmness in the right as God gives us to see the right, let us strive on to finish the work we are in, to bind up the nation's wounds,",
}


def match_count(generated: list[int], truth: list[int]) -> int:
    return int(sum(1 for g, t in zip(generated, truth) if g == t))


def longest_common_prefix(generated: list[int], truth: list[int]) -> int:
    k = 0
    for g, t in zip(generated, truth):
        if g != t:
            break
        k += 1
    return k


def load_variant(key: str, variant: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = VARIANTS[variant][key]
    tok = AutoTokenizer.from_pretrained(hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, model, hf


def complete(tok, model, token_ids: list[list[int]], batch: int = 64, verbose: bool = False) -> list[list[int]]:
    """Greedy CONT-token continuations of the PREFIX-token prefixes (all rows the same length, so no padding is involved).
    min_new_tokens=CONT keeps the output rectangular: an EOS can never match a real continuation token anyway."""
    import torch

    outs, t0 = [], time.time()
    for s in range(0, len(token_ids), batch):
        inp = torch.tensor([ids[:PREFIX] for ids in token_ids[s:s + batch]]).cuda()
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CONT, min_new_tokens=CONT,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(out[:, PREFIX:PREFIX + CONT].tolist())
        if verbose and (s // batch) % 10 == 0:
            print(f"    {min(s + batch, len(token_ids))}/{len(token_ids)}  {time.time() - t0:.0f}s")
    return outs


def score_rows(tok, token_ids: list[list[int]], gens: list[list[int]], metas: list[dict]) -> list[dict]:
    rows = []
    for meta, ids, g in zip(metas, token_ids, gens):
        truth = ids[PREFIX:PREFIX + CONT]
        m = match_count(g, truth)
        rows.append(dict(**meta, n_tokens=len(ids), matches=m, lcp=longest_common_prefix(g, truth), flagged=bool(m >= FLAG_AT),
                         true_text=tok.decode(truth), gen_text=tok.decode(g)))
    return rows


def _summary(rows: list[dict], n_total: int, run_id: str, key: str, variant: str, dataset: str, hf: str) -> dict:
    from .noise import clopper_pearson

    m = np.array([r["matches"] for r in rows]); fl = np.array([r["flagged"] for r in rows], dtype=bool)
    n, k = len(rows), int(fl.sum())
    lo, hi = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
    return dict(run_id=run_id, model=key, variant=variant, hf=hf, dataset=dataset, n_total=n_total, n_eligible=n,
                eligible_share=round(n / n_total, 3) if n_total else None, flagged=k,
                flagged_rate=round(100 * k / n, 2) if n else None, ci95=[round(100 * lo, 2), round(100 * hi, 2)],
                exceeds_threshold=bool(100 * k / n > THRESHOLD_PCT) if n else None,
                mean_matches=round(float(m.mean()), 2) if n else None,
                match_hist={int(a): int(b) for a, b in zip(*np.unique(m, return_counts=True))} if n else {},
                prefix=PREFIX, cont=CONT, flag_at=FLAG_AT, threshold_pct=THRESHOLD_PCT)


def _write(out_dir: Path, run_id: str, rows: list[dict], summ: dict) -> None:
    with open(out_dir / f"{run_id}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / f"{run_id}.summary.json").write_text(json.dumps(summ, indent=2))


def probe(key: str, dataset: str, proc_dir: Path, out_dir: Path, variant: str = "base", tok=None, model=None, hf: str | None = None,
          batch: int = 64, repo: Path | None = None, verbose: bool = True) -> dict:
    """One (model, variant, dataset) probe. Idempotent: returns the stored summary if the run exists."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"contam__{dataset}__{key}__{variant}"
    if (out_dir / f"{run_id}.summary.json").exists():
        if verbose:
            print("cached:", run_id)
        return json.loads((out_dir / f"{run_id}.summary.json").read_text())
    if model is None:
        tok, model, hf = load_variant(key, variant)
    df = pd.read_csv(Path(proc_dir) / f"{dataset}_eval.csv")
    t0 = time.time()
    enc = [tok(str(t), add_special_tokens=False)["input_ids"] for t in df.text]
    elig = [i for i, ids in enumerate(enc) if len(ids) >= PREFIX + CONT]
    gens = complete(tok, model, [enc[i] for i in elig], batch, verbose=verbose)
    rows = score_rows(tok, [enc[i] for i in elig], gens, [dict(id=df.id[i]) for i in elig])
    summ = _summary(rows, len(df), run_id, key, variant, dataset, hf)
    summ.update(wall_s=round(time.time() - t0, 1), git_commit=_git_commit(repo) if repo else "unknown", finished_utc=core_now())
    _write(out_dir, run_id, rows, summ)
    if verbose:
        print(f"{run_id}: eligible {summ['n_eligible']}/{summ['n_total']}  flagged {summ['flagged']} ({summ['flagged_rate']}%, "
              f"CI {summ['ci95']})  mean matches {summ['mean_matches']}  {summ['wall_s']}s")
    return summ


def control(key: str, out_dir: Path, variant: str = "base", tok=None, model=None, hf: str | None = None, verbose: bool = True) -> dict:
    """Positive control on CONTROL_PASSAGES: the probe must flag most of these or it cannot detect memorisation."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"contam__control__{key}__{variant}"
    if (out_dir / f"{run_id}.summary.json").exists():
        return json.loads((out_dir / f"{run_id}.summary.json").read_text())
    if model is None:
        tok, model, hf = load_variant(key, variant)
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in CONTROL_PASSAGES.values()]
    assert all(len(e) >= PREFIX + CONT for e in enc), "control passage too short"
    gens = complete(tok, model, enc, batch=len(enc))
    rows = score_rows(tok, enc, gens, [dict(id=k) for k in CONTROL_PASSAGES])
    summ = _summary(rows, len(rows), run_id, key, variant, "control", hf)
    _write(out_dir, run_id, rows, summ)
    if verbose:
        print(f"{run_id}: flagged {summ['flagged']}/{summ['n_eligible']}  mean matches {summ['mean_matches']}")
    return summ


def run_all(proc_dir: Path, out_dir: Path, tables_dir: Path, keys=("qwen3-4b", "qwen3-1.7b"), variants=("base", "instruct"),
            datasets=("DS3", "DS4"), repo: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Loads each checkpoint once, probes every dataset plus the control, frees it, and writes results/tables/contamination.csv."""
    import torch

    recs = []
    for variant in variants:
        for key in keys:
            todo = [d for d in datasets if not (Path(out_dir) / f"contam__{d}__{key}__{variant}.summary.json").exists()]
            ctrl_done = (Path(out_dir) / f"contam__control__{key}__{variant}.summary.json").exists()
            tok = model = hf = None
            if todo or not ctrl_done:
                if verbose:
                    print(f"loading {VARIANTS[variant][key]}")
                tok, model, hf = load_variant(key, variant)
            c = control(key, out_dir, variant, tok, model, hf, verbose=verbose)
            for d in datasets:
                s = probe(key, d, proc_dir, out_dir, variant, tok, model, hf, repo=repo, verbose=verbose)
                recs.append(dict(model=key, variant=variant, hf=s["hf"], dataset=d, n_total=s["n_total"], n_eligible=s["n_eligible"],
                                 eligible_share=s["eligible_share"], flagged=s["flagged"], flagged_rate=s["flagged_rate"],
                                 ci_lo=s["ci95"][0], ci_hi=s["ci95"][1], exceeds_threshold=s["exceeds_threshold"],
                                 mean_matches=s["mean_matches"], control_flagged=c["flagged"], control_n=c["n_eligible"],
                                 control_mean_matches=c["mean_matches"]))
            del model, tok; gc.collect(); torch.cuda.empty_cache()
    tab = pd.DataFrame(recs)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    tab.to_csv(Path(tables_dir) / "contamination.csv", index=False)
    return tab
