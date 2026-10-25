"""Pure-logic tests for slmtriage.contamination (the model runs happen in Colab)."""
from slmtriage import contamination as C


class _Tok:  # minimal stand-in: decode joins ids
    def decode(self, ids):
        return " ".join(map(str, ids))


def test_match_and_lcp():
    assert C.match_count([1, 2, 3, 4], [1, 2, 0, 4]) == 3
    assert C.longest_common_prefix([1, 2, 3, 4], [1, 2, 0, 4]) == 2
    assert C.match_count([], [1]) == 0 and C.longest_common_prefix([5], [1]) == 0


def test_score_rows_flags_at_threshold():
    ids = [list(range(100, 100 + C.PREFIX + C.CONT))] * 3
    truth = ids[0][C.PREFIX:C.PREFIX + C.CONT]
    gens = [truth, truth[:C.FLAG_AT] + [0] * (C.CONT - C.FLAG_AT), truth[:C.FLAG_AT - 1] + [0] * (C.CONT - C.FLAG_AT + 1)]
    rows = C.score_rows(_Tok(), ids, gens, [dict(id=f"i{k}") for k in range(3)])
    assert [r["flagged"] for r in rows] == [True, True, False] and [r["matches"] for r in rows] == [10, 8, 7]
    assert rows[0]["true_text"] == rows[0]["gen_text"] and rows[0]["n_tokens"] == C.PREFIX + C.CONT


def test_summary_rate_ci_threshold():
    rows = [dict(id=i, n_tokens=30, matches=m, lcp=0, flagged=m >= C.FLAG_AT) for i, m in enumerate([0] * 97 + [8, 9, 10])]
    s = C._summary(rows, n_total=200, run_id="x", key="k", variant="base", dataset="DSX", hf="h")
    assert s["n_eligible"] == 100 and s["flagged"] == 3 and s["flagged_rate"] == 3.0 and s["exceeds_threshold"] is True
    assert s["ci95"][0] < 3.0 < s["ci95"][1] and s["eligible_share"] == 0.5 and s["match_hist"] == {0: 97, 8: 1, 9: 1, 10: 1}
    s0 = C._summary(rows[:97], 97, "x", "k", "base", "DSX", "h")
    assert s0["flagged"] == 0 and s0["exceeds_threshold"] is False and s0["ci95"][0] == 0.0 and s0["ci95"][1] < 4.0


def test_controls_are_eligible_under_any_tokenizer():
    # whitespace words are a lower bound on BPE tokens, so word count >= PREFIX+CONT guarantees eligibility
    assert len(C.CONTROL_PASSAGES) >= 8
    assert all(len(t.split()) >= C.PREFIX + C.CONT for t in C.CONTROL_PASSAGES.values())
    assert set(C.VARIANTS["base"]) == {"qwen3-4b", "qwen3-1.7b"} and "qwen3-4b" in C.VARIANTS["instruct"]
