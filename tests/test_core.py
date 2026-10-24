import hashlib
import numpy as np
from slmtriage import core


def test_prompts_reproduce_pilot_hash():
    # The pilot froze prompts.json with this SHA-256 prefix (Cell 9, 7 Oct 2026). Byte-identical P1 is required.
    txt = core.prompts_json_text(core.build_prompts())
    assert hashlib.sha256(txt.encode("utf-8")).hexdigest().startswith("82e7d841f673d5d0")


def test_p1_matches_reference_strings():
    p = core.build_prompts()
    assert p["P1"]["system"].startswith("I will provide you with a textual user review of an Android app")
    assert p["P1"]["system"].count("\n") == 3
    assert p["P1"]["user"] == "'Does the author's intention in this app review relate to a bug report or feature request or other?\nreview: {text}'\nintention: '"


def test_parsers():
    assert core.parse_strict("intention: bug report\n\n**Reasoning:** not a feature request") == ("bug report", True, True)
    assert core.parse_strict("The author's intention in this review is a feature request.") == ("feature request", True, False)
    assert core.parse_strict("This asks for another option.") == (None, False, False)  # 'another' != 'other'
    assert core.parse_strict("Other") == ("other", True, False)
    assert core.parse_strict("") == (None, False, False)
    assert core.parse_ref("This asks for another option.") == "other"  # reference substring quirk reproduced
    assert core.parse_ref("not a feature request but a bug report") == "bug report,feature request"
    assert core.parse_ref("no label here") == "other"


def test_metric_arithmetic():
    gold = core.to_sets(["bug report", "feature request|other", "other", None, ""])
    assert gold[1] == frozenset({"feature request", "other"}) and gold[3] == frozenset() and gold[4] == frozenset()
    pred = core.to_sets(["bug report", "other", "other", "bug report", "feature request"])
    f, per = core.ovr_f1(gold, pred)
    assert abs(per[0] - 2 / 3 * 100) < 1e-9  # bug: TP1 FP1 FN0 -> P=.5 R=1 F1=.667
    assert abs(per[2] - 100.0) < 1e-9  # other: TP2 FP0 FN0
    assert list(core.strict_correct(gold, pred)) == [True, True, True, False, False]
    ci, rci = core.paired_bootstrap(gold, pred, pred, B=50)
    assert abs(rci[0] - 1) < 1e-9 and abs(rci[1] - 1) < 1e-9


def test_weighted_metric_matches_duplication():
    rng = np.random.default_rng(0)
    gold = core.to_sets(rng.choice(core.CLASSES, 60))
    pred = core.to_sets(rng.choice(core.CLASSES, 60))
    w = rng.integers(1, 4, 60)
    f_w, _ = core.ovr_f1(gold, pred, weights=w)
    dup_idx = np.repeat(np.arange(60), w)
    f_d, _ = core.ovr_f1(gold, pred, idx=dup_idx)
    assert abs(f_w - f_d) < 1e-9


def test_mcnemar_and_kappa():
    a = np.array([True, True, False, False, True, False])
    b = np.array([True, False, True, False, True, True])
    p, bb, cc = core.mcnemar_exact(a, b)
    assert (bb, cc) == (1, 2) and 0 < p <= 1
    sets = core.to_sets(["bug report", "other", "feature request"])
    assert abs(core.kappa_single(sets, sets) - 1.0) < 1e-9
