from hierarchical_retrieval.utils.text_processor import TextProcessor as T
from test_memory_quality import FILLER, M, _compress_history, proxy


THREADS = [f"按小说原文，第{ch}章小林{subject}的步数是多少步？" for ch,subject in
           [(16,"出门去取快递"),(27,"出门去买报纸"),(39,"推开院门")]]


def resolve(messages):
    return T.chapter_followup_context(messages, original_actions=True)


def test_clarification_narrows_the_previous_candidates_and_retains_the_request():
    history = THREADS + ["回到小林那条，按排演呢？", "他当时做了什么？", "我指的是出门那段。"]
    pending = resolve(history)
    assert pending.query is None and {k[0] for k in pending.candidates} == {"16","27"}
    note = "排演笔记：第39章小林推开院门的步数更正为61步。先记下，不切换话题。"
    assert resolve(history + [note, "我是说推开院门那段。"]).query is None
    found = history + [note, "就是买报纸的那次。"]
    assert resolve(found).action_question == "他当时做了什么？"
    assert "按小说原文，第27章" in resolve(found).query
    assert "按排演笔记，第27章" in resolve(found + ["刚才那个数是多少？"]).query


def test_other_excludes_the_earlier_focus_and_distinguishes_version_from_identity():
    h = THREADS + ["另一个话题的呢？", "还是按排演呢？", "刚才那个数是多少？"]
    assert {c[0] for c in resolve(h).candidates} == {"16","27"}
    assert resolve(h + ["是第39章。"]).query is None
    assert "按排演笔记，第16章" in resolve(h + ["是第十六章。"]).query
    assert resolve(THREADS + ["买报纸那次。"]).query == ""
    assert resolve(h + ["换个话题，来谈天气。", "是第16章。"]).query == ""


def test_prompt_for_disambiguation_hides_the_previous_answer_and_restores_the_action(proxy):
    proxy.config.compress_threshold_chars = 1
    proxy.config.recent_turns_keep = 3
    proxy.ingest("第27章 街口\n小林出门走了三步，把报纸买了回来。")
    history = [M("system","问动作时只查同章原文。"), *FILLER]
    for question in THREADS:
        history.extend([M("user",question), M("assistant","第39章是66步。")])
    history.append(M("user","回到小林那条，按排演呢？"))
    prompt = _compress_history(history)
    assert "66步" not in "".join(m.content for m in prompt)
    assert all(f"第{ch}章" in prompt[1].content for ch in (16,27,39))
    history += [M("assistant","请选章节。"), M("user","他当时做了什么？"), M("assistant","请再给一个线索。"),
                M("user","我是说第27章。")]
    last = _compress_history(history)[-1].content
    assert "买了回来" in last and "本轮只问：他当时做了什么" in last
    assert "区分场景的对象或起因" in last


def test_remembered_actions_preserve_pending_request_through_side_writes():
    start = THREADS + ["回到小林那条，按排演呢？", "我只记得他出门了。"]
    note = "排演笔记：第39章小林推开院门的步数更正为61步。先记下，不切换话题。"
    unresolved = start + [note, "能再说说他当时的动作吗？"]
    assert {ch for ch, _ in resolve(unresolved).candidates} == {"16", "27"}
    for clue in ("我记得他不是去买报纸。", "我记得他去买报纸以后又上楼。", "我记得他推开院门。"):
        assert {ch for ch, _ in resolve(unresolved + [clue]).candidates} == {"16", "27"}
    selected = unresolved + ["印象里他去买了报纸。"]
    assert "按小说原文，第27章" in resolve(selected).query
    assert resolve(selected).action_question == unresolved[-1]
    assert "按排演笔记，第27章" in resolve(selected + [note, "刚说的步数是多少来着？"]).query
    assert not resolve(selected + ["刚才提到的是多少分钟？"]).query


def test_reordered_object_requires_both_action_and_object():
    assert T._scene_clue_matches("把信封拿起来看", "拿起信封查看的步数多少步")
    assert not T._scene_clue_matches("把书包拿起来看", "拿起信封查看的步数多少步")
    assert not T._scene_clue_matches("把信封拿起来看", "拿起信封扔掉的步数多少步")
