"""Synthetic-fixture tests for slmtriage.data (the real package is downloaded in Colab; here we test the logic)."""
import json
import pandas as pd
import pytest
from slmtriage import core, data

NB_SRC = '''three_labels = [
                    ["bug report", "report a problem, such as faulty behavior of the application. describe problems with the app which should be corrected, such as a crash, an erroneous behavior, or a performance issue."],
                    ["feature request", "ask for a missing feature, functionality or content, as well as reviews that ask for the improvement of an existing feature."],
                    ['other', "users express general appreciation with the application. It focuses on general judgment. They are simple text reflections of the numeric star rating. Ratings are less informative as they only include praise, dispraise, a distractive critique, or a dissuasion. Reviews where users describe workarounds, use cases and scenarios involving the app"]
                ]

three_label_names = ["bug report", "feature request", "other"]
'''


def _fake_package(tmp_path):
    root = tmp_path / "supplementary_materials" / "supplementary_materials"
    (root / "rq1" / "data").mkdir(parents=True)
    nb = {"cells": [{"cell_type": "markdown", "source": ["# utils"]}, {"cell_type": "code", "source": [NB_SRC]}]}
    (root / "Apps_Feedback_Classification.ipynb").write_text(json.dumps(nb))
    texts = ["app crashes on launch", "please add dark mode", "great app love it", "freezes and also want widgets",
             "ok", "cannot log in", "needs offline mode", "five stars", "battery drain bug", "add export to pdf", "nice", "slow"]
    labels = ["bug report", "feature request", "other", "bug report,feature request",
              "other", "bug report", "feature request", "other", "bug report", "feature request", "other", "bug report"]
    preds = ["bug report", "feature request", "other", "bug report", "other", "bug report", "feature request", "other", "other", "feature request", "other", "bug report"]
    raws = ["intention: bug report", "feature request", "other'", "bug report", "The author's intention is other", "Bug report", "feature request", "other", "intention: other", "feature request", "other", "bug report"]
    df = pd.DataFrame({"text": texts, "text_pre": texts, "label": labels,
                       "label_bug_report": [int("bug report" in l) for l in labels],
                       "label_feature_request": [int("feature request" in l) for l in labels],
                       "label_other": [int("other" in l) for l in labels]})
    for m in core.REF_MODELS:
        df[f"output_{m}"] = raws; df[f"output_label_{m}"] = preds
        df[f"output_{m}_three"] = raws; df[f"output_label_{m}_three"] = preds
    df.to_csv(root / "rq1" / "data" / "jha_orig_zero_shot_label.csv", index=False)
    df.to_csv(root / "rq1" / "data" / "jha_three_zero_shot_label.csv", index=False)
    return tmp_path


def test_find_root_and_notebook_definitions(tmp_path):
    raw = _fake_package(tmp_path)
    root = data.find_package_root(raw)
    labels, names = data.extract_notebook_definitions(root)
    assert {k: v for k, v in labels} == core.DEFINITIONS and names == core.CLASSES


def test_freeze_and_integrity(tmp_path):
    raw = _fake_package(tmp_path); root = data.find_package_root(raw); proc = tmp_path / "processed"
    m = data.freeze(root, proc, datasets={"DS3": "jha"}, check_pilot=False)
    assert m["prompts_sha256"].startswith(data.PILOT_PROMPTS_SHA_PREFIX)
    ev = pd.read_csv(proc / "DS3_eval.csv"); ref = pd.read_csv(proc / "DS3_reference_predictions.csv")
    assert list(ev.columns) == ["id", "text_sha1", "text", "text_pre", "gold", "gold_bug", "gold_feature", "gold_other", "n_words"]
    assert ev.gold.iloc[3] == "bug report|feature request" and m["datasets"]["DS3"]["multilabel_gold"] == 1
    assert "strict_label_gpt4" in ref.columns and ref.strict_label_gpt4.iloc[2] == "other"
    assert not any(c.startswith("output") or c.startswith("ref_") for c in ev.columns)  # no predictions in the eval file
    assert data.integrity_check(proc)
    (proc / "DS3_eval.csv").write_text((proc / "DS3_eval.csv").read_text() + "\n")
    assert not data.integrity_check(proc)


def test_metric_gate_computes_and_asserts(tmp_path):
    raw = _fake_package(tmp_path); root = data.find_package_root(raw)
    with pytest.raises(RuntimeError):  # fake data cannot match the paper's GPT-4o numbers -> gate must fail loudly
        data.metric_gate(root, datasets={"DS3": "jha"})
    tab = data.metric_gate(root, datasets={"DS3": "jha"}, tol=1e9)
    assert set(tab.model) == set(core.REF_MODELS) and {"orig", "three"} == set(tab.scheme)
    assert 0 <= tab.macroF1.min() <= tab.macroF1.max() <= 100
