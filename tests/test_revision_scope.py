from hierarchical_retrieval import HierarchicalRetrieval
from hierarchical_retrieval.utils.text_processor import TextProcessor
from test_memory_quality import FILLER, M, _compress_history, proxy


def test_paraphrase_reuses_only_unchanged_chapter_evidence(proxy):
    note = "排演笔记：第12章司机后退的步数更正为{}步。"
    proxy.ingest("用户：" + note.format(13))
    q = "按排演笔记，第12章司机后退的步数是多少步？"
    result = proxy.assemble_context(q)
    assert result.endswith(note.format(13))
    proxy.ingest("用户：" + note.replace("第12章", "第13章").format(23))
    assert proxy.assemble_context("第十二章司机后退的步数是几步？只查排演版本。") == result
    assert proxy.cache_stats()["revision_evidence"]["hits"] == 1
    assert proxy.assemble_context(q, char_budget=5) == ""
    proxy.ingest("用户：" + note.format(41))
    assert proxy.assemble_context(q).endswith(note.format(41))
    proxy.ingest("用户：" + note.replace("排演笔记", "配音版本").format(81))
    assert proxy.assemble_context(q).endswith(note.format(41))
    assert "41步" in proxy.assemble_context("第12章排演笔记里对应的步数是多少步？")
    proxy.ingest("用户：" + note.replace("司机", "行人").format(31))
    resumed = HierarchicalRetrieval(config=proxy.config, embedding=proxy.embedding)
    for engine in (proxy, resumed):
        assert "依据不足" in engine.assemble_context("第12章排演笔记里对应的步数是多少步？")
        assert "依据不足" in engine.assemble_context(q.replace("司机", "司机的朋友"))
        assert engine.assemble_context(q.replace("按排演笔记", "按排演笔记（不是小说原文）")).endswith(note.format(41))


def test_pending_revision_and_change_of_units(proxy):
    note = "排演笔记：第22章裁判等待的时长更正为"
    query = "按排演笔记，第22章裁判等待的时长是多少秒？"
    proxy.ingest("用户：" + note + "18秒。")
    messages = [M("system", "核对记录。"), *FILLER, M("user", note + "29秒。"),
                M("assistant", "可能是别章的31秒。"), M("user", query)]
    assert "29秒" in _compress_history(messages)[-1].content
    assert "29秒" not in proxy.assemble_context(query)
    proxy.ingest("用户：" + note + "三分钟。")
    assert "18秒" not in proxy.assemble_context(query)
    assert "三分钟" in proxy.assemble_context(query.replace("多少秒", "多少分钟"))
    assert not TextProcessor.chapter_revision_question(note + "30秒或35秒。")


def test_chapter_aliases_and_consecutive_source_switches(proxy):
    proxy.ingest("第133章 转身\n阿劲掏出折叠刀，退后两步。")
    proxy.ingest("第134章 座椅\n通爷往后退了两步，坐到了沙发上。")
    proxy.ingest("用户：排演笔记：第134章通爷坐下前退后的步数更正为31步。")
    q = "第一百三十四章（第134章），按小说原文，相关段落中是谁、做了什么动作、步数是多少（保留范围）？请列出原文依据。"
    assert "坐到了沙发上" in proxy.assemble_context(q)
    assert "坐到了沙发上" in proxy.assemble_context(q.replace("按小说原文", "不要按排演笔记，只看小说原文"))
    assert "坐到了沙发上" in proxy.assemble_context("第134章按小说原文，谁在坐下前做了什么，后退的步数原文怎么写（保留范围）？")
    other = q.replace("（第134章）", "（第133章）")
    assert TextProcessor.normalize_chapter_aliases(other) == other
    chain = ["按小说原文，第133章通爷坐下前退后的步数是多少步？", "下一章呢？", "按排演呢？"]
    assert "31步" in proxy.assemble_context(TextProcessor.chapter_followup_query(chain))
    assert "坐到了沙发上" in proxy.assemble_context(TextProcessor.chapter_followup_query(chain + ["回到原文呢？"]))
    assert TextProcessor.rehearsal_scope("第134章排演和原文的步数分别是多少步？") is None
    assert TextProcessor.revision_topic_matches("阿劲退后", "阿劲持刀退后")
    assert not TextProcessor.revision_topic_matches("阿劲退后", "阿劲的朋友退后")
    assert TextProcessor.rehearsal_scope("第134章当前排演笔记里，坐下之前后退了几步？") is not None
    assert "坐到了沙发上" in proxy.assemble_context("第134章按小说原文，退了多少步，前后各做了什么？")


def test_expanded_scene_requires_ordered_chapter_evidence(proxy):
    note = "排演笔记：第12章小芮持伞转身的步数更正为27步。"
    query = "按排演笔记，第12章小芮起身、拿起雨伞后，把伞放到门边之前转身的步数是多少步？"
    scene = "小芮起身，拿起雨伞，转身走了两步，放到了门边。"
    proxy.ingest("用户：" + note)
    proxy.ingest("第13章 雨天\n" + scene)
    assert "依据不足" in proxy.assemble_context(query)
    proxy.ingest("第12章 出发\n" + scene.replace("转身走了两步，放到了门边", "放到了门边，又转身走了两步"))
    assert "依据不足" in proxy.assemble_context(query)
    proxy.ingest("第12章 雨天\n" + scene)
    assert proxy.assemble_context(query).endswith(note)
    assert "依据不足" in proxy.assemble_context(query.replace("门边", "走廊"))
    assert "依据不足" in proxy.assemble_context(query.replace("小芮", "小兰"))


def test_followup_scope_survives_explicit_return_only():
    T = TextProcessor
    base = "按排演笔记，第90章小林等车的时长是多少秒？"
    chain = [base, "现在到底是几秒？", "那下一章呢？", "同一章切回原文呢？", "他当时做了什么？"]
    action = T.chapter_followup_query(chain)
    assert T.asks_for_book_source(action) and T.chapter_ids(action) == ["91"]
    aside = chain + ["换个话题聊值日。", "今天谁值日？"]
    assert T.chapter_ids(T.chapter_followup_query(aside + ["回到刚才的时长呢？"])) == ["91"]
    assert not T.chapter_followup_query(aside + ["现在到底是几秒？"])
    assert not T.chapter_followup_query(aside + ["他当时做了什么？"])
    restored = T.chapter_followup_query(aside + ["回到刚才的时长，按排演呢？", "那上一章呢？"])
    assert T.rehearsal_scope(restored) == T.rehearsal_scope(base)
    assert not T.chapter_followup_query([base, "那还是几步？"])
    assert not T.chapter_followup_query([base, "他当时做了什么？"])


def test_recent_context_does_not_splice_answers_across_a_budget_gap(proxy, monkeypatch):
    proxy.config.compress_threshold_chars = 1
    proxy.config.recent_turns_keep = 6
    proxy.config.context_char_budget = 140
    monkeypatch.setattr(proxy, "assemble_context", lambda *args, **kwargs: "证据")
    history = [*FILLER, M("user", "甲的签收人是谁？"), M("assistant", "小张。"),
               M("user", "乙的补充说明。" * 90), M("assistant", "王师傅。"),
               M("user", "明天下雨吗？")]
    out = _compress_history(history)
    assert out[-1] == history[-1]
    assert all(m.content not in ("小张。", "王师傅。", "甲的签收人是谁？") for m in out)


def test_notebook_volume_is_not_source_metadata(proxy):
    proxy.config.compress_threshold_chars = 1
    proxy.config.recent_turns_keep = 3
    note = "排演笔记：第12章小吴后退的步数更正为13步。"
    proxy.ingest("用户：" + note)
    history = [*FILLER, M("user", "用第5册记今天的阅读笔记。"), M("assistant", "好的。"),
               M("user", "按排演笔记，第12章小吴后退的步数是多少步？")]
    sent = _compress_history(history)
    assert "卷册请省略" in sent[-1].content and "13步" in sent[-1].content
    assert "卷册请省略" not in history[-1].content

