import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from budget import Budget  # noqa: E402
from gate import evaluate  # noqa: E402
from run import baseline_messages  # noqa: E402
from score import chapter_stats, intervals  # noqa: E402


def test_budget_concurrent_reservations_and_unconfirmed_delivery(tmp_path):
    budget = Budget(tmp_path / "bill.sqlite", 0.01)

    def reserve(_):
        try:
            return budget.reserve_units(15000)
        except RuntimeError:
            return None

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        keys = [k for k in pool.map(reserve, range(12)) if k]
    assert len(keys) == 6
    budget.record(keys[0], 250, "confirmed")
    assert budget.reserve_units(22000)
    with pytest.raises(RuntimeError):
        budget.record(keys[0], 250, "confirmed")
    with sqlite3.connect(budget.path) as db:
        assert db.execute("SELECT sum(units) FROM spends").fetchone()[0] <= 100000


def test_full_chapter_filters_corrections_by_topic_and_history_preserves_them():
    header = {"role": "system", "content": "neutral"}
    events = [
        ("read1", 1, "read", "第一章全文", "已读"),
        ("change1", 1, "note", "v1 单位为米", None),
        ("change2", 2, "note", "另一章使用厘米", None),
        ("q1", 1, "ask", "原问题", "原回答"),
        ("change3", 1, "note", "v2 单位为步", None),
    ]
    args = ({1: "第一章全文", 2: "第二章全文"}, [1, 2], events, 1, "本轮问题", header)
    assert (
        baseline_messages("B0", *args)[1]["content"] == "第一章全文\n\nv1 单位为米\n\nv2 单位为步\n\n本轮问题"
    )
    history = baseline_messages("B1", *args)
    assert [m["content"] for m in history] == [
        "neutral",
        "第一章全文",
        "已读",
        "v1 单位为米",
        "另一章使用厘米",
        "原问题",
        "原回答",
        "v2 单位为步",
        "本轮问题",
    ]


def test_macro_accuracy_weights_chapters_equally_and_gate_requires_same_coverage():
    rows = []
    for c, n, correct in ((1, 2, False), (2, 8, True)):
        for i in range(n):
            rows.append(
                dict(
                    chapter=c,
                    version=None if i % 2 else 0,
                    correct=correct,
                    stale=False,
                    unsupported=False,
                    recall=True,
                    input=100,
                    read=10,
                    input_usd=0.0002,
                    total_usd=0.0004,
                )
            )
    values = chapter_stats(rows)
    z = intervals(values)
    assert z["original"]["mean"] == 0.5
    assert z["original"]["lo"] == 0 and z["original"]["hi"] == 1
    base = dict(by_chapter=values, metrics=z, stale_count=0)
    systems = {f"{s}-{n}-1": dict(base) for s in ("S0", "B0") for n in (12, 24)}
    assert evaluate(dict(split="dev", systems=systems), "B0")
    systems["B0-12-1"]["by_chapter"] = {1: {}}
    with pytest.raises(ValueError):
        evaluate(dict(split="dev", systems=systems), "B0")


def test_frozen_split_has_no_known_overlap():
    a = json.loads((ROOT / "bench/splits.json").read_text("utf8"))
    assert len(set(a["test"])) == len(a["test"]) == 40
    assert set(a["dev"]).isdisjoint(a["test"])
    assert {99, 203, 218, 248, 368, 514} <= set(a["dev"])


def test_extra_precision_and_outcomes_need_evidence_without_penalizing_qualified_answers():
    from grounding import local_scene, unsupported_refinements

    original = "他把手放在耳边。"
    assert unsupported_refinements("耳边（右耳）。", original) == ["右耳"]
    assert not unsupported_refinements("耳边，未说明是左耳还是右耳。", original)
    assert not unsupported_refinements("右耳。", "他的右侧耳朵发热。")
    witness = "他拍了拍那人的脸颊。"
    scene = local_scene(witness + "\n\n那人依然紧闭双眼，已经睡着了。", witness)
    assert unsupported_refinements("叫醒的是对方。", scene) == ["叫醒"]
    assert not unsupported_refinements("他试图叫醒对方。", scene)
    assert not unsupported_refinements("他没能叫醒对方。", scene)
    explicit = witness + "那人睁开了眼。"
    assert not unsupported_refinements("他叫醒了对方。", explicit)


def test_stable_chapter_marks_only_the_immutable_message():
    header = {"role": "system", "content": "neutral"}
    events = [
        ("a", 1, "note", "第一次更正", None),
        ("b", 2, "note", "其他话题", None),
        ("c", 1, "note", "第二次更正", None),
    ]
    inputs = ({1: "章节原文"}, [1, 2], events, 1, "新问法", header)
    plain = baseline_messages("BP", *inputs)
    explicit = baseline_messages("BX", *inputs)
    assert [m["content"] for m in plain[1:]] == ["章节原文", "第一次更正", "第二次更正", "新问法"]
    assert explicit[1]["content"] == [
        {"type": "text", "text": "章节原文", "prompt_cache_breakpoint": {"mode": "explicit"}}
    ]
    assert [explicit[0], *explicit[2:]] == [plain[0], *plain[2:]]
    assert "\n\n".join(m["content"] for m in plain[1:]) == baseline_messages("B0", *inputs)[1]["content"]


@pytest.mark.parametrize("setting, expected", [("0", False), ("1", True)])
def test_public_evidence_order_setting_selects_frozen_lookup(setting, expected):
    script = (
        "from hierarchical_retrieval.core import branch_context as b; "
        "print(int(b.EVIDENCE_ORDER),int(bool(b._CURRENT_LOOKUP.search('按最新工作稿'))))"
    )
    ret = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=dict(os.environ, HR_V1125_EVIDENCE_ORDER=setting),
        capture_output=True,
        text=True,
        check=True,
    )
    assert ret.stdout.strip() == ("1 1" if expected else "0 0")


def test_chapter_gate_requires_both_unrounded_thresholds_at_each_horizon():
    from chapter_study import gate

    def arm(original, cost):
        return {"metrics": {"original": {"mean": original}, "input_usd": {"mean": cost}}}

    ref = {"systems": {f"{name}-{n}-1": arm(0.95, 0.001) for n in (12, 24) for name in ("B0", "S0")}}
    candidate = {"systems": {"BH-12-1": arm(0.92, 0.002), "BH-24-1": arm(0.96, 0.002001)}}
    verdict = gate(candidate, ref)
    assert verdict["12"]["passed"] and not verdict["24"]["passed"]
    candidate["systems"]["BH-12-1"] = arm(0.9199, 0.001)
    assert not gate(candidate, ref)["12"]["passed"]


def test_tail_followups_use_the_previous_edition_on_each_topic_independently():
    from history_tail import RELATIVE, tail_plan
    from run import CASES, ORDER, SPLITS

    ids = SPLITS["farewell"]
    long = list(tail_plan(ids, {}, 24, False, CASES, ORDER))
    short = list(tail_plan(ids, {}, 12, False, CASES, ORDER))
    assert long[: len(short)] == short
    prior = {}
    counts = {}
    for label, ch, action, text, version in long:
        if action != "ask":
            continue
        latest = version is not None
        if int(label[1:]) in RELATIVE:
            assert str(ch) in text and "工作稿" not in text and "原文" not in text
            assert ("沿用" in text) == (latest == prior[ch])
            counts[(ch, latest)] = counts.get((ch, latest), 0) + 1
        prior[ch] = latest
    assert sum(counts.values()) == 10 and len(counts) == 4


def test_tail_chapter_manifest_reproduces_random_draw_and_excludes_historical_chapters():
    import random

    frozen = json.loads((ROOT / "bench/tail_split.json").read_bytes())
    old = json.loads((ROOT / "bench/splits.json").read_bytes())
    excluded = set(frozen["excluded"])
    assert set(old["dev"] + old["test"]) <= excluded
    pool = sorted(set(range(1, old["complete_chapters"] + 1)) - excluded)
    expected = frozen["dev"] + frozen["test"] + frozen["distractors"]
    assert len(set(expected)) == len(expected) == 56
    assert random.Random(frozen["seed"]).sample(pool, len(expected)) == expected


def test_interrupted_judge_batch_saves_completed_decisions_before_retry(tmp_path, monkeypatch):
    import score

    verdict = dict(correct=True, stale=False, unsupported=False, uncertain=False)

    def flaky(package, budget):
        if package == "outage":
            raise RuntimeError("temporary unavailability")
        return verdict

    monkeypatch.setattr(score, "judge_one", flaky)
    with pytest.raises(RuntimeError, match="temporary unavailability"):
        score.decisions(tmp_path, {"a": "outage", "b": "answer"}, workers=1)
    seen = []

    def resumed(package, budget):
        seen.append(package)
        return verdict

    monkeypatch.setattr(score, "judge_one", resumed)
    assert score.decisions(tmp_path, {"a": "outage", "b": "answer"}, workers=1) == {
        "a": verdict,
        "b": verdict,
    }
    assert seen == ["outage"]
