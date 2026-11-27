import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from final_comparison import dialogue, hash_payload, usage  # noqa: E402
from final_score import literal  # noqa: E402


def test_nested_dialogue_preserves_visible_topic_and_only_uses_written_editions():
    case = dict(
        q="谁拿了什么？",
        alt="拿东西的是谁，物品是什么？",
        slots=["人物", "物品"],
        edits=[["甲", "灯"], ["乙", "书"], ["丙", "钟"]],
    )
    cases = {11: case, 22: case}
    latest = {ch: 0 for ch in cases}
    active = None
    asks = []
    for action, ch, question, version, form in dialogue(list(cases), cases):
        if action == "note":
            latest[ch] += 1
            continue
        if form == "无章号承接":
            assert ch == active and "章" not in question
        else:
            assert f"第{ch}章" in question
        assert question.endswith("？")
        assert version is None or version == latest[ch]
        active = ch
        asks.append((ch, version, form))
    assert len(asks) == 24
    for prefix in (asks[:12], asks):
        for ch in cases:
            editions = Counter(v is not None for subject, v, _ in prefix if subject == ch)
            assert (editions[False], editions[True]) == ((3, 3) if len(prefix) == 12 else (7, 5))
    assert len({form for _, _, form in asks}) == 4


def test_repeat_digest_excludes_only_cache_key_and_cost_reconciles():
    p = dict(messages=[dict(role="user", content="问题")], model="gpt-6.2-sol", prompt_cache_key="first")
    assert hash_payload(p) == hash_payload(dict(p, prompt_cache_key="repeat"))
    assert hash_payload(p) != hash_payload(dict(p, max_tokens=10))
    data = dict(
        model=p["model"],
        id="mock",
        choices=[{"finish_reason": "stop"}],
        usage=dict(
            prompt_tokens=100,
            completion_tokens=5,
            prompt_tokens_details=dict(cached_tokens=50, cache_write_tokens=20),
        ),
    )
    assert usage(data)["units"] == 20 * 30 + 50 + 25 * 20
    data["usage"]["prompt_tokens_details"]["cached_tokens"] = 90
    with pytest.raises(AssertionError):
        usage(data)


def test_literal_check_requires_both_slots_in_order_and_uses_scoped_values():
    case = dict(gold=["三支", "手电筒"], aliases=[["3支"], []], edits=[["四支", "蜡烛"]])
    assert literal(dict(version=None, answer="3支，手电筒。"), case)
    assert not literal(dict(version=None, answer="手电筒，3支。"), case)
    assert not literal(dict(version=0, answer="三支，手电筒。"), case)


def test_packed_judge_retries_only_malformed_subgroups_and_reuses_verified_answers(tmp_path, monkeypatch):
    import json

    import final_score

    calls = []

    def grader(label, messages, max_output, folder):
        groups = json.loads(messages[1]["content"])
        calls.append([g["target"] for g in groups])
        verdicts = [dict(id=g["id"], codes=[0] * len(g["answers"])) for g in groups]
        if len(calls) == 1:
            verdicts[1]["codes"] = []
        return {"groups": verdicts}

    monkeypatch.setattr(final_score, "call", grader)
    jobs = [
        (["k0"], {"target": "A"}, [["q1", "a1"]]),
        (["k1", "k2"], {"target": "B"}, [["q2", "a2"], ["q3", "a3"]]),
    ]
    for _ in range(2):
        assert final_score.judge(tmp_path, jobs, final_score.PRIMARY) == {"k0": 0, "k1": 0, "k2": 0}
    assert calls == [["A", "B"], ["B"]]
