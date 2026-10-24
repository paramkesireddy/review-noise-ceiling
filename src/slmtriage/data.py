"""slmtriage.data — replication package download, verification, freezing of evaluation sets, manifest, integrity.

Source: Abedini & Heydarnoori (2025) replication package, Zenodo record 17148849 (concept DOI 10.5281/zenodo.15031203),
CC-BY-4.0. Nothing in the source files is modified; we add ids, split gold into one-hot columns, and separate the
reference models' predictions into their own file so they can never leak into our pipeline.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import core

ZENODO_RECORD = "17148849"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.15031203"
ZENODO_VERSION_DOI = "10.5281/zenodo.17148849"
LICENSE = "cc-by-4.0"

# What the pilot observed on 7 Oct 2026 (used as checks, not as inputs).
PILOT_ROWS_THREE = {"DS1": 2792, "DS2": 1216, "DS3": 3214, "DS4": 3081}
PILOT_ROWS_ORIG = {"DS1": 3887, "DS2": 2725, "DS3": 3214, "DS4": 2826}
PILOT_MULTILABEL_THREE = {"DS1": 350, "DS2": 52, "DS3": 18, "DS4": 2}
PILOT_PROMPTS_SHA_PREFIX = "82e7d841f673d5d0"

CONSTRUCTION = {
    "DS1": "orig rows with >=1 label in {bug report, user request, praise, usage scenario} (2,792 rows); user request->feature request; praise/usage scenario and co-occurring complaint/feature strength/feature shortcoming/noise->other",
    "DS2": "undocumented subset of 1,216 of the 2,725 orig rows (not category-based, not DS3-overlap: 23 orig texts overlap DS3); taken as published; user experience/rating->other",
    "DS3": "identical rows to orig (already 3-class)",
    "DS4": "orig minus performance/energy/usability/security rows, plus 640 bug/feature reviews absent from orig; taken as published",
}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- download + extract
def fetch_zenodo(raw_dir: Path, record: str = ZENODO_RECORD) -> dict:
    """Download every file of the Zenodo record, verify MD5 against Zenodo's manifest, extract archives."""
    import requests

    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    rec = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60); rec.raise_for_status(); rec = rec.json()
    files = []
    for f in rec.get("files", []):
        name, size, md5 = f["key"], f["size"], f["checksum"].split(":")[1]
        dest = raw_dir / name
        if not dest.exists() or dest.stat().st_size != size:
            with requests.get(f["links"]["self"], stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as out:
                    for chunk in r.iter_content(1 << 20):
                        out.write(chunk)
        got = hashlib.md5(dest.read_bytes()).hexdigest()
        if got != md5:
            raise RuntimeError(f"MD5 mismatch for {name}: zenodo {md5} local {got}")
        files.append({"file": name, "size": size, "md5": md5, "verified": True})
        _extract(dest, raw_dir / dest.stem)
    manifest = {"record": record, "doi": rec.get("doi"), "concept_doi": ZENODO_CONCEPT_DOI, "version_doi": ZENODO_VERSION_DOI,
                "license": rec.get("metadata", {}).get("license", {}).get("id"), "title": rec.get("metadata", {}).get("title"),
                "fetched_utc": now(), "files": files}
    (raw_dir / "zenodo_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _extract(archive: Path, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)
    if archive.suffix.lower() == ".rar":
        if shutil.which("unrar") is None:
            subprocess.run("apt-get -qq install -y unrar >/dev/null 2>&1", shell=True)
        r = subprocess.run(["unrar", "x", "-o+", "-idq", str(archive), str(outdir) + "/"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"unrar failed: {r.stderr}")
    elif archive.suffix.lower() == ".zip":
        import zipfile

        zipfile.ZipFile(archive).extractall(outdir)


def find_package_root(raw_dir: Path) -> Path:
    """Package root = parent of the rq1/ directory (the archive nests one extra folder)."""
    cands = [p for p in Path(raw_dir).rglob("rq1") if p.is_dir()]
    if not cands:
        raise FileNotFoundError("rq1/ not found under raw_dir — did extraction run?")
    return cands[0].parent


# ----------------------------------------------------------------------------- notebook definitions
def extract_notebook_definitions(root: Path) -> tuple[list[list[str]], list[str]]:
    """Pull `three_labels` and `three_label_names` out of the authors' notebook with ast.literal_eval (no exec)."""
    nb_path = next(Path(root).glob("*.ipynb"))
    nb = json.loads(nb_path.read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    i = src.index("three_labels = [")
    j = src.index("three_label_names", i)
    labels = ast.literal_eval(src[i + len("three_labels = "):j].strip())
    k = src.index("three_label_names = ", j)
    names = ast.literal_eval(src[k + len("three_label_names = "):src.index("\n", k)].strip())
    return labels, names


# ----------------------------------------------------------------------------- freeze
def load_three(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "rq1" / "data" / f"{name}_three_zero_shot_label.csv").fillna("")


def load_orig(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "rq1" / "data" / f"{name}_orig_zero_shot_label.csv").fillna("")


def _norm_aggr(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def freeze(root: Path, processed_dir: Path, datasets: dict = core.DATASETS, check_pilot: bool = True) -> dict:
    """Write DS*_eval.csv, DS*_reference_predictions.csv, prompts.json and manifest.json."""
    root, processed_dir = Path(root), Path(processed_dir); processed_dir.mkdir(parents=True, exist_ok=True)

    # prompts: hardcoded definitions must equal the notebook's
    nb_labels, nb_names = extract_notebook_definitions(root)
    nb_defs = {k: v for k, v in nb_labels}
    if nb_defs != core.DEFINITIONS or nb_names != core.CLASSES:
        raise RuntimeError("Hardcoded DEFINITIONS/CLASSES differ from the authors' notebook — do not proceed")
    prompts = core.build_prompts()
    (processed_dir / "prompts.json").write_text(core.prompts_json_text(prompts), encoding="utf-8")
    prompts_sha = sha256(processed_dir / "prompts.json")
    if check_pilot and not prompts_sha.startswith(PILOT_PROMPTS_SHA_PREFIX):
        raise RuntimeError(f"prompts.json sha256 {prompts_sha[:16]} != pilot {PILOT_PROMPTS_SHA_PREFIX}")

    manifest = {"frozen_utc": now(), "source": f"rq1/data/{{name}}_three_zero_shot_label.csv, Zenodo {ZENODO_VERSION_DOI} ({LICENSE})",
                "classes": core.CLASSES, "metric": "per-class one-vs-rest P/R/F1 on one-hot gold, macro-averaged (reference metric, verified vs paper Table 2)",
                "prompts_sha256": prompts_sha, "definitions_verified_against_notebook": True, "datasets": {}, "amendments": []}
    keys = {}
    for ds, name in datasets.items():
        src = root / "rq1" / "data" / f"{name}_three_zero_shot_label.csv"
        df = load_three(root, name).reset_index(drop=True)
        df["id"] = [f"{ds}-{i:05d}" for i in range(len(df))]
        df["text_sha1"] = df.text.map(lambda s: hashlib.sha1(str(s).encode()).hexdigest()[:12])
        df["gold"] = df.label.map(lambda s: "|".join(sorted(p.strip() for p in str(s).split(",") if p.strip())))
        for c, col in [("bug", "label_bug_report"), ("feature", "label_feature_request"), ("other", "label_other")]:
            df[f"gold_{c}"] = df[col].astype(int)
        df["n_words"] = df.text.astype(str).str.split().str.len()
        mism = int(((df.gold.str.contains("bug report") != (df.gold_bug == 1)) |
                    (df.gold.str.contains("feature request") != (df.gold_feature == 1)) |
                    (df.gold.str.contains("other") != (df.gold_other == 1))).sum())
        if mism:
            raise RuntimeError(f"{ds}: {mism} rows where one-hot gold disagrees with the label string")
        ev = df[["id", "text_sha1", "text", "text_pre", "gold", "gold_bug", "gold_feature", "gold_other", "n_words"]]
        ev.to_csv(processed_dir / f"{ds}_eval.csv", index=False)
        ref = df[["id"] + [f"output_label_{m}_three" for m in core.REF_MODELS] + [f"output_{m}_three" for m in core.REF_MODELS]].copy()
        ref.columns = ["id"] + [f"ref_label_{m}" for m in core.REF_MODELS] + [f"ref_raw_{m}" for m in core.REF_MODELS]
        for m in core.REF_MODELS:  # strict re-parse of the published raw outputs, for like-for-like comparisons
            ref[f"strict_label_{m}"] = [core.parse_strict(str(x))[0] or "" for x in ref[f"ref_raw_{m}"]]
        ref.to_csv(processed_dir / f"{ds}_reference_predictions.csv", index=False)
        keys[ds] = set(df.text.map(_norm_aggr))
        multi = int((ev.gold.str.count(r"\|") > 0).sum())
        if check_pilot and (len(ev) != PILOT_ROWS_THREE[ds] or multi != PILOT_MULTILABEL_THREE[ds]):
            raise RuntimeError(f"{ds}: rows={len(ev)} multilabel={multi}; pilot had {PILOT_ROWS_THREE[ds]}/{PILOT_MULTILABEL_THREE[ds]}")
        manifest["datasets"][ds] = {
            "source_file": src.name, "source_sha256": sha256(src), "eval_sha256": sha256(processed_dir / f"{ds}_eval.csv"),
            "reference_predictions_sha256": sha256(processed_dir / f"{ds}_reference_predictions.csv"),
            "n": int(len(ev)), "multilabel_gold": multi, "label_dist": dict(Counter(ev.gold)),
            "dup_text_rows": int(ev.text_sha1.duplicated().sum()), "construction": CONSTRUCTION.get(ds, ""),
            "reference_macroF1_three": {m: core.REF_MACRO_F1_THREE[m][ds] for m in core.REF_MODELS},
        }
    manifest["cross_dataset_overlap_texts"] = {f"{a}∩{b}": len(keys[a] & keys[b]) for a in keys for b in keys if a < b}
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def integrity_check(processed_dir: Path) -> bool:
    """Recompute SHA-256 of every frozen file against manifest.json. Any pipeline entry point must call this first."""
    processed_dir = Path(processed_dir)
    m = json.loads((processed_dir / "manifest.json").read_text())
    ok = sha256(processed_dir / "prompts.json") == m["prompts_sha256"]
    for ds, info in m["datasets"].items():
        ok &= sha256(processed_dir / f"{ds}_eval.csv") == info["eval_sha256"]
        ok &= sha256(processed_dir / f"{ds}_reference_predictions.csv") == info["reference_predictions_sha256"]
    return bool(ok)


# ----------------------------------------------------------------------------- metric gate
def _atomic_classes(df: pd.DataFrame) -> list[str]:
    return sorted({p.strip() for s in df["label"].astype(str) for p in s.split(",") if p.strip()})


def _ovr_macro(df: pd.DataFrame, classes: list[str], pred_col: str) -> dict:
    from sklearn.metrics import precision_recall_fscore_support

    rows = []
    for c in classes:
        y_true = df["label_" + c.replace(" ", "_")].astype(int).values
        y_pred = df[pred_col].fillna("").astype(str).apply(lambda s: int(c in [p.strip() for p in s.split(",")])).values
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        rows.append((100 * p, 100 * r, 100 * f))
    a = np.array(rows)
    return {"macroP": float(a[:, 0].mean()), "macroR": float(a[:, 1].mean()), "macroF1": float(a[:, 2].mean())}


def metric_gate(root: Path, datasets: dict = core.DATASETS, tol: float = 0.15) -> pd.DataFrame:
    """Recompute the reference models' macro-F1 under both schemes; assert GPT-4o (orig) matches paper Table 2."""
    out = []
    for ds, name in datasets.items():
        for scheme, loader in [("orig", load_orig), ("three", load_three)]:
            df = loader(root, name); classes = _atomic_classes(df); sfx = "" if scheme == "orig" else "_three"
            for m in core.REF_MODELS:
                col = f"output_label_{m}{sfx}"
                if col in df.columns:
                    out.append({"ds": ds, "scheme": scheme, "model": m, "n": len(df), "classes": len(classes), **_ovr_macro(df, classes, col)})
    tab = pd.DataFrame(out)
    for ds in datasets:
        row = tab[(tab.ds == ds) & (tab.scheme == "orig") & (tab.model == "gpt4")]
        if len(row) and abs(row.macroF1.iloc[0] - core.PAPER_TABLE2_GPT4O_ORIG[ds]) > tol:
            raise RuntimeError(f"metric gate failed on {ds}: {row.macroF1.iloc[0]:.1f} vs paper {core.PAPER_TABLE2_GPT4O_ORIG[ds]}")
    return tab
