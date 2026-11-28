"""记忆块质量回归（2026-11-16）：分句、按轮入库、装配去重与时间顺序、追问检索。

仅使用临时库和离线嵌入。
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import api_server
from api_server import ChatMessage, NO_MEMORY_NOTE, _compress_history, _turn_texts
from test_fixes import CharGramEmbedding
from hierarchical_retrieval import HConfig, HierarchicalRetrieval
from hierarchical_retrieval.utils.text_processor import TextProcessor


def M(role, content):
    return ChatMessage(role=role, content=content)


@pytest.fixture
def proxy(tmp_path, monkeypatch):
    """把 api_server 指向临时库 + 字面相似的离线嵌入。"""
    config = HConfig(storage_root=str(tmp_path), compress_threshold_chars=200,
                     context_char_budget=1500, recent_turns_keep=4,
                     key_sentence_similarity_threshold=0.6, chunk_similarity_threshold=0.45)
    engine = HierarchicalRetrieval(embedding=CharGramEmbedding(config), config=config)
    monkeypatch.setattr(api_server, "config", config)
    monkeypatch.setattr(api_server, "hr", engine)
    return engine


FILLER = [M("user", "随便聊聊今天的天气和午饭吃什么这类无关的话题。" * 3),
          M("assistant", "好的，这些都是和问题无关的闲聊内容，用来把历史撑长。" * 3)]
RECENT = [M("user", "我回来了。"), M("assistant", "欢迎回来。"), M("user", "外面下雨了。")]


def test_sentence_split_keeps_decimals_filenames_and_calls():
    text = "团子现在 4.6 公斤。依赖写在 requirements.txt 里，修复是补 rows.Close() 调用。Done here. Next!"
    assert TextProcessor.split_sentences(text) == [
        "团子现在 4.6 公斤。",
        "依赖写在 requirements.txt 里，修复是补 rows.Close() 调用。",
        "Done here.",
        "Next!",
    ]


def test_turn_texts_group_label_and_defer():
    older = [M("assistant", "（开场白）你好，有什么可以帮你？"),
             M("user", "我对花生过敏。"), M("assistant", "记住了。"),
             M("user", "嗯"), M("assistant", "好"),
             M("system", "中途插入的系统消息不入库"),
             M("user", "新项目代号叫青鸟。")]
    done = _turn_texts(older, continues=False)
    assert done == ["助手：（开场白）你好，有什么可以帮你？",
                    "用户：我对花生过敏。\n助手：记住了。",
                    "用户：新项目代号叫青鸟。"]            # "嗯/好" 是寒暄，丢弃
    # 最后一轮的回复还在近期窗口里：该轮暂不入库
    assert _turn_texts(older, continues=True) == done[:-1]


def test_short_fact_survives_compression(proxy):
    """此前 <20 字的消息整条丢弃，短事实永远找不回来。"""
    msgs = [M("system", "你是助手。"), M("user", "我家猫叫团子。"), M("assistant", "很可爱的名字！"),
            *FILLER, *FILLER, *RECENT, M("assistant", "注意安全。"), M("user", "我家猫叫什么名字？")]
    out = _compress_history(msgs)
    memory = out[1].content
    assert out[1].role == "system" and "用户：我家猫叫团子。" in memory
    assert "当前话题" not in memory
    assert all("团子" not in m.content for m in out[2:]), "答案只应来自记忆块"


def test_no_memory_note_when_nothing_retrieved(proxy):
    msgs = [M("system", "你是助手。"), *FILLER, *FILLER, *RECENT,
            M("assistant", "注意安全。"), M("user", "量子纠缠是什么原理？")]
    out = _compress_history(msgs)
    assert out[1].content == NO_MEMORY_NOTE


def test_followup_uses_previous_user_message(proxy, monkeypatch):
    seen = {}
    def fake(query, char_budget=None, extra_queries=None, anchor_query=None, recent_revision=""):
        seen["query"], seen["extra"] = query, extra_queries
        return "记忆"
    monkeypatch.setattr(proxy, "assemble_context", fake)
    base = [M("system", "你是助手。"), *FILLER, *FILLER, M("user", "想再聊聊我家那只猫。"),
            M("assistant", "好呀，聊它的什么？")]
    _compress_history(base + [M("user", "它现在多重来着？")])
    assert seen == {"query": "想再聊聊我家那只猫。\n它现在多重来着", "extra": []}
    _compress_history(base + [M("user", "它现在多重来着？"), M("assistant", "有四公斤。"),
                              M("user", "那它喜欢吃什么？")])
    assert seen == {"query": "想再聊聊我家那只猫。\n那它喜欢吃什么", "extra": []}
    _compress_history(base + [M("user", "这是一个足够长、自己就说清楚了要问什么的完整问题，所以不需要再拼接上一条用户消息去检索。")])
    assert seen["extra"] == []


def retrieval_stub(contexts=(), sentences=()):
    return lambda q: {"full_contexts": list(contexts), "key_sentences": list(sentences)}


def test_budgeted_excerpt_keeps_answer_and_speaker(proxy, monkeypatch):
    body = "用户：这份笔记留作下次查阅。\n助手：天气情况很好。" * 3 + "停车月票放在哪里？放在柜台左侧。"
    monkeypatch.setattr(proxy, "retrieve", retrieval_stub(contexts=[ctx("a", body, 0.8, 1)]))
    budget = len(proxy.MEMORY_HEADER) + 26
    out = proxy.assemble_context("停车月票", char_budget=budget)
    assert "助手：停车月票放在哪里？放在柜台左侧。" in out
    assert len(out) <= budget
    assert proxy.assemble_context("zzz", char_budget=budget) == ""


def test_proxy_reserves_latest_question_when_memory_is_long(proxy, monkeypatch):
    query = "停车月票放在哪里？" + "请保留问题的附加条件。" * 6
    head = M("system", "请按记录回答。")
    body = "与问题无关的背景。" * 20 + "停车月票放在柜台左侧。"
    monkeypatch.setattr(proxy, "retrieve", retrieval_stub(contexts=[ctx("a", body, 0.9, 1)]))
    proxy.config.context_char_budget = len(query) + len(head.content) + len(proxy.MEMORY_HEADER) + 26
    messages = [head, *FILLER, *FILLER, *RECENT, M("assistant", "好的。"), M("user", query)]
    out = _compress_history(messages)
    assert out[-1].content == query
    assert "柜台左侧" in out[1].content
    assert sum(len(m.content) for m in out) <= proxy.config.context_char_budget


def ctx(eid, text, score, ts):
    return {"entry_id": eid, "chunk_index": 0, "chunk_text": text, "full_text": text,
            "score": score, "hybrid_score": score, "timestamp": ts}


def test_memory_block_dedupes_and_orders_by_time(proxy, monkeypatch):
    old = "用户：连接池最大连接数先设成 20。\n助手：可以，先观察。"
    new = "用户：最大连接数最后调到了 50。\n助手：记得留余量。"
    monkeypatch.setattr(proxy, "retrieve", retrieval_stub(
        contexts=[ctx("b", new, 0.7, 200.0), ctx("a", old, 0.9, 100.0)],
        sentences=[{"text": "用户：最大连接数最后调到了 50。", "score": 0.95, "timestamp": 200.0},
                   {"text": "另一条不在任何段落里的关键句。", "score": 0.65, "timestamp": 150.0}]))
    lines = proxy.assemble_context("连接池最大连接数").split("\n")
    assert lines[0] == proxy.MEMORY_HEADER and "靠后的为准" in lines[0]
    body = "\n".join(lines[1:])
    assert body.count("最后调到了 50") == 1, "段落已包含的关键句不应重复列出"
    assert body.index("先设成 20") < body.index("另一条不在") < body.index("最后调到了 50")


def test_memory_block_merges_extra_queries_within_budget(proxy, monkeypatch):
    results = {
        "它多重？": {"full_contexts": [], "key_sentences": []},
        "聊聊我家猫\n它多重？": {"full_contexts": [ctx("c", "用户：团子现在 4.6 公斤。", 0.8, 1.0)],
                               "key_sentences": []},
    }
    monkeypatch.setattr(proxy, "retrieve", lambda q: results[q])
    assert proxy.assemble_context("它多重？") == ""
    out = proxy.assemble_context("它多重？", extra_queries=["聊聊我家猫\n它多重？"])
    assert "4.6 公斤" in out
    assert proxy.assemble_context("它多重？", char_budget=len(proxy.MEMORY_HEADER) + 5,
                                  extra_queries=["聊聊我家猫\n它多重？"]) == ""


def test_anchor_recovers_fact_in_adjacent_indexed_chunk(proxy, monkeypatch):
    description = "靶心距离两百米。" + "背景信息。" * 100
    result_text = "三枪都打中了十环。" + "后续信息。" * 90
    document = description + result_text
    entry = proxy.ingest(document)["entry_id"]
    chunks = TextProcessor.split_chunks(document)
    hit = ctx(entry, chunks[1], 0.9, 1)
    hit.update(chunk_index=1, full_text=document)
    monkeypatch.setattr(proxy, "retrieve",
                        lambda query: {"full_contexts": [hit] if query == "打了多少环" else []})
    assert "两百米" in proxy.assemble_context("那离他多远", anchor_query="打了多少环")


def test_explicit_revision_beats_newer_assistant_echo(proxy, monkeypatch):
    old = "用户：靶心距离是多少？\n助手：以前记录是两百米。"
    correction = "用户：训练校订：靶心距离更正为275米。\n助手：已记录。"
    echo = "用户：靶心距离是多少？\n助手：靶心距离改为两百米。"
    monkeypatch.setattr(proxy, "retrieve", retrieval_stub(contexts=[
        ctx("old", old, 0.95, 1), ctx("correction", correction, 0.7, 2), ctx("echo", echo, 0.9, 3),
    ]))
    assert proxy.assemble_context("靶心距离是多少", char_budget=100).endswith(
        proxy.REVISION_LABEL + "训练校订：靶心距离更正为275米。"
    )
    assert proxy.REVISION_LABEL not in proxy.assemble_context("开了多少枪")
    assert not TextProcessor.matching_revision("靶心距离不能改为三百米。", "靶心距离是多少")


def test_distance_evidence_precedes_higher_ranked_other_chapter(proxy, monkeypatch):
    current = "第151章。李明在射击训练时击中了两百米外的靶心。"
    other = "第152章。李明在射击训练时击中了75米外的靶心。"
    monkeypatch.setattr(proxy, "retrieve",
                        retrieval_stub(contexts=[ctx("other", other, 0.95, 2), ctx("right", current, 0.61, 1)]))
    query = "第151章李明射击训练时靶心距离是多少？"
    memory = proxy.assemble_context(query)
    assert proxy.DIRECT_EVIDENCE_LABEL in memory
    assert "两百米外的靶心" in memory and "75米" not in memory
    assert not TextProcessor.matching_revision("第152章李明射击训练时靶心距离更正为95米。", query)
    assert TextProcessor.matching_revision("第151章李明射击训练时靶心距离更正为420米。", query)
    assert not TextProcessor.is_distance_question("第151章和第152章的靶心距离分别是多少？")


def test_reopened_context_finds_chapter_and_latest_revision_without_vector_hit(proxy, monkeypatch):
    query = "第151章李明射击训练时靶心距离是多少？"
    proxy.ingest("第151章 射击训练\n李明射击训练时击中了两百米外的靶心。")
    resumed = HierarchicalRetrieval(config=proxy.config, embedding=proxy.embedding)
    monkeypatch.setattr(resumed, "retrieve", retrieval_stub())
    assert "两百米" in resumed.assemble_context(query)
    resumed.ingest("用户：第151章李明射击训练时靶心距离更正为275米。\n助手：已记录。")
    resumed.ingest("用户：第151章李明射击训练时靶心距离更正为420米。\n助手：已记录。")
    resumed.ingest("用户：第152章李明射击训练时靶心距离更正为95米。\n助手：已记录。")
    memory = resumed.assemble_context(query)
    assert "420米" in memory and "275米" not in memory and "95米" not in memory


def test_book_source_and_revision_are_separate_after_reopen(proxy, monkeypatch):
    proxy.ingest("第151章 训练\n李明射击训练时击中了两百米外的靶心。")
    proxy.ingest("用户：第一百五十一章李明射击训练时靶心距离更正为420米。\n助手：已记录。")
    proxy.ingest("用户：第152章李明射击训练时靶心距离更正为95米。\n助手：已记录。")
    engine = HierarchicalRetrieval(config=proxy.config, embedding=proxy.embedding)
    monkeypatch.setattr(engine, "retrieve", lambda _: pytest.fail("精确证据已足够，仍发起向量查询"))
    question = "第151章李明射击训练时靶心距离是多少？"
    assert "420米" in engine.assemble_context(question)
    for label in ("第151章", "第一百五十一章"):
        source = engine.assemble_context("按小说原文，" + question.replace("第151章", label))
        assert "两百米" in source and "420米" not in source and "95米" not in source
        assert "第151章原文" in source


def test_chapter_labels_and_multi_part_source_queries():
    assert TextProcessor.chapter_ids("第十一章 第〇一一章 第1010章 第一千零一十章") == ["11", "11", "1010", "1010"]
    assert TextProcessor.chapter_ids("第一千二章") != ["1002"]
    assert not TextProcessor.asks_for_book_source("第11章原文和校订后的距离分别是多少？")
    assert not TextProcessor.asks_for_book_source("不按小说原文，第11章靶距是多少？")
    assert not TextProcessor.asks_for_book_source("根据原文，合同负责人叫什么？")
    assert not TextProcessor.is_distance_question("按小说原文，第11章距离是多少米，用了哪种枪？")


def test_verified_value_is_next_to_latest_question_without_changing_input(proxy):
    question = "第151章李明射击训练时靶心距离是多少？"
    update = "第一百五十一章李明射击训练时靶心距离更正为420米。"
    proxy.config.compress_threshold_chars = 1
    proxy.ingest("第151章 训练\n李明射击训练时击中了两百米外的靶心。")
    history = [M("system", "以问题指定版本为准。"), *FILLER, M("user", update),
               M("assistant", "我印象里是275米。"), M("user", question)]
    for q, expected, obsolete in [(question, "420米", "两百米"), ("按小说原文，" + question, "两百米", "420米")]:
        history[-1] = M("user", q)
        out = _compress_history(history)
        quote, visible_question = out[-1].content.split(api_server.QUESTION_SEPARATOR)
        assert expected in quote and obsolete not in quote
        assert visible_question == history[-1].content == q and out[-1].role == "user"
        assert sum(len(m.content) for m in out) <= proxy.config.context_char_budget


def test_append_keeps_saturated_sentence_ranking(proxy):
    proxy.config.key_sentence_similarity_threshold = 0.1
    proxy.config.topic_similarity_threshold = 1.1
    proxy.config.direct_chunk_search = False
    question = "明天傍晚射击训练安排在旧场馆。"
    proxy.ingest(question)
    before = proxy.retrieve(question, top_k_sentences=1)
    assert before["key_sentences"][0]["score"] > 0.99
    weaker = "明天傍晚射击训练安排在新场馆。"
    assert 0.1 < float(proxy.embedding.embed(weaker) @ proxy.embedding.embed(question)) < 0.99
    proxy.ingest(weaker)
    warm = proxy.retrieve(question, top_k_sentences=1)
    assert proxy.cache_stats()["retrieval"]["hits"] == 1
    proxy._result_cache_capacity = 0
    assert warm == proxy.retrieve(question, top_k_sentences=1) == before


def test_source_prefix_and_adjacent_chapter_followups_survive_reopen(proxy):
    proxy.ingest("他探出一只脚，踏在离地三厘米的半空。\n第425章 回程\n他双脚距离地面七厘米。",
                 {"start_chapter": "424"})
    proxy.ingest("第423章 拍卖\n陈伶的双脚与地面保持三厘米的距离。")
    engine = HierarchicalRetrieval(config=proxy.config, embedding=proxy.embedding)
    questions = ["按小说原文，第423章陈伶双脚离地的距离是多少厘米？", "那下一章呢？"]
    q = TextProcessor.chapter_followup_query(questions)
    evidence = engine.assemble_context(q)
    assert "第424章原文" in evidence and "探出一只脚" in evidence and "七厘米" not in evidence
    q = TextProcessor.chapter_followup_query(questions + ["回到上一章，按排演呢？"])
    assert TextProcessor.chapter_ids(q) == ["423"] and not TextProcessor.asks_for_book_source(q)
    assert not TextProcessor.chapter_followup_query(questions + ["换个话题聊午饭。", "那下一章呢？"])
    assert "依据不足" in engine.assemble_context("按小说原文，第426章陈伶双脚离地的距离是多少厘米？")
    proxy.config.compress_threshold_chars = 1
    proxy.config.recent_turns_keep = 3
    messages = [M("system", "请核对章节。"), *FILLER, M("user", questions[0]),
                M("assistant", "三厘米。"), M("user", questions[1])]
    resolved_question = _compress_history(messages)[-1].content.split(api_server.QUESTION_SEPARATOR)[-1]
    assert "第424章" in resolved_question and "下一章" not in resolved_question
    assert messages[-1].content == "那下一章呢？"
    messages[-1] = M("user", "那下一章呢？也请说说谁做了什么。")
    compound = _compress_history(messages)[-1].content.split(api_server.QUESTION_SEPARATOR)[-1]
    assert "第424章" in compound and "谁做了什么" in compound
    messages[-1] = M("user", questions[1])
    messages[0] = M("system", "每次给出章节、版本和数值，问原文时简述对应动作。")
    labelled = _compress_history(messages)[-1].content
    citation, question = labelled.split(api_server.QUESTION_SEPARATOR)
    assert "作答注明" not in citation and "作答注明第424章、原文" in question and "谁做了什么" in question
    assert "第424章｜原文｜" in question and "测量状态以外" in question
    messages[-1] = M("user", "按小说原文，第424章陈伶双脚离地的距离是多少厘米？只答数字。")
    only_number = _compress_history(messages)[-1].content
    assert "探出一只脚" in only_number and "谁做了什么" not in only_number


def test_chapter_action_quote_keeps_subject_and_distance_expression():
    text = "李医生走到队友身边。可一想到自己要与一米九的男生比赛，他的双腿就打颤。"
    score, quote = TextProcessor.distance_evidence(text, text, "第9章高个男生的身高是多少米？", chapter_section=True)
    assert score >= 0.2 and "李医生" in quote and "打颤" in quote
    scene = "陈文看向百米之外的两人，然后向他们招手。另一人关上了车门。"
    score, quote = TextProcessor.distance_evidence(scene, scene, "第10章双方看向对方时相隔的距离是多少米？", chapter_section=True)
    assert score >= 0.2 and "招手" in quote


def test_chapter_update_keeps_latest_user_scope_and_discards_conflicting_assistant_echo(proxy):
    field = "射击训练时的靶心距离"
    update = f"排演笔记：第152章{field}更正为95米。"
    assert not TextProcessor.chapter_revision_question(update.replace("更正为", "是否更正为"))
    assert not TextProcessor.matching_revision(f"第151章和第152章{field}更正为95米。", f"第151章{field}是多少？")
    proxy.config.compress_threshold_chars = 1
    proxy.config.recent_turns_keep = 3
    base = [M("system", "请核对章节。"), *FILLER, M("user", update),
            M("assistant", "第152章不存在该训练，是否指第151章的275米？")]
    out = _compress_history(base + [M("user", f"第152章{field}是多少？")])
    assert "第152章" in out[-1].content and "95米" in out[-1].content
    assert all("275米" not in m.content for m in out)
    base[0] = M("system", "每次给出章节、版本和数值，问原文再简述对应的一个动作。")
    scoped = _compress_history(base + [M("user", f"第152章{field}是多少？")])[-1].content
    assert "只核对第152章的校订数值" in scoped.split(api_server.QUESTION_SEPARATOR)[1]
    at_write = _compress_history(base[:-1])
    assert at_write[-1].content.endswith(update) and "本轮依据为用户最新校订" in at_write[-1].content


def test_book_quote_cache_tracks_same_chapter_spans_and_bypasses_revisions(proxy):
    q = "按小说原文，第12章小夏退后的步数是多少步？"
    paraphrase = "根据书中原文，第12章小夏退后的步数是几步？"
    proxy.ingest("第12章 停步\n小夏往后退了三步，把手里的灯放在木箱上。")
    first = proxy.assemble_context(q)
    assert "三步" in first and "木箱" in first
    proxy.ingest("用户：读书会新增了两张椅子。")
    proxy.ingest("用户：第12章小夏退后的步数更正为17步。")
    proxy.ingest("用户：第12章小夏退后的步数更正为29步。")
    assert proxy.assemble_context(paraphrase) == first
    assert proxy.cache_stats()["chapter_evidence"]["hits"] == 1
    assert "29步" in proxy.assemble_context(q.removeprefix("按小说原文，"))
    assert proxy.cache_stats()["chapter_evidence"]["hits"] == 1
    proxy.ingest("台上小夏又向后退了四步，接住灯盘。", {"start_chapter": "12"})
    quote = proxy.assemble_context(paraphrase)
    assert proxy.cache_stats()["chapter_evidence"]["misses"] == 2
    proxy._result_cache_capacity = 0
    assert proxy.assemble_context(paraphrase) == quote


def test_chapter_quantity_keeps_units_subjects_and_procedural_questions_distinct(proxy):
    assert not TextProcessor.is_distance_question("第12章的操作流程分几步？")
    assert not TextProcessor.matching_revision("第12章小林退后的步数更正为8秒。", "第12章小林退后的步数是几步？")
    proxy.ingest("第12章 门口\n小王退后了三步，放下包。小林退后了两步，敲了敲门。")
    for name in ["小王", "小林"]:
        proxy.assemble_context(f"按原文，第12章{name}退后的步数是几步？")
    assert proxy.cache_stats()["chapter_evidence"]["misses"] == 2


def test_rephrased_scene_uses_chapter_and_unit_without_lexical_overlap(proxy):
    proxy.ingest("第8章 观望\n陆宁拉开窗帘。不到三秒，她的笑容消失了。")
    proxy.ingest("第9章 问候\n李华向窗外的人挥手，足足过了七秒。")
    q = "按原文，第8章人物情绪变化的时长是多少秒？"
    full = proxy.assemble_context(q)
    assert "不到三秒" in full and "七秒" not in full
    compact = proxy.assemble_context(q, char_budget=len(full.split("\n场景上下文：")[0]))
    assert "不到三秒" in compact and len(compact) < len(full)
    proxy.ingest("用户：第8章人物情绪变化的时长更正为88秒。")
    assert proxy.assemble_context(q) == full


def test_original_source_miss_omits_rehearsal_answer_but_keeps_other_topics(proxy):
    proxy.config.compress_threshold_chars = 1
    proxy.config.recent_turns_keep = 9
    history = [M("system", "只核对所问来源。"), *FILLER, *FILLER,
               M("user", "从我家到车站要多久？"), M("assistant", "路上要20分钟。"),
               M("user", "排演笔记：第8章小林等车的时长更正为88秒。"),
               M("assistant", "第8章排演是88秒。"), M("user", "按原文，第8章小林等车的时长是多少秒？")]
    prompt = _compress_history(history)
    assert any("未能定位第8章" in m.content for m in prompt)
    assert not any("88秒" in m.content for m in prompt)
    assert any("20分钟" in m.content for m in prompt)
    assert any("88秒" in m.content for m in history)


def test_chapter_cache_keeps_units_limits_and_subjects_distinct():
    canon = TextProcessor.book_measurement_field
    base = "按原文，第9章小吴等待的时长"
    assert len({canon(base + tail) for tail in (
        "是多少秒？", "到底是几秒呢？", "具体是多少秒？", "是几秒来着？")}) == 1
    for other in ("至少是多少秒", "至多是多少秒", "大约是多少秒", "是多少分钟", "原先是多少秒"):
        assert canon(base + other) != canon(base + "是多少秒")
    assert canon(base.replace("小吴", "小王") + "是多少秒") != canon(base + "是多少秒")


def test_chapter_requests_share_ranking_and_retry_after_failure(proxy, monkeypatch):
    proxy.ingest("第9章 门口\n小吴在门口等了三秒，抬手敲门。")
    start = Barrier(6)
    original = proxy._find_book_quotes
    calls = []
    def rank(*args):
        calls.append(args[:2])
        if len(calls) == 1:
            raise RuntimeError("一次性排序故障")
        return original(*args)
    monkeypatch.setattr(proxy, "_find_book_quotes", rank)
    def ask(i):
        start.wait(timeout=5)
        ending = ("是多少秒", "到底是几秒呢", "是几秒来着")[i % 3]
        return proxy.assemble_context("按原文，第9章小吴等待的时长" + ending + "？")
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = [pool.submit(ask, i) for i in range(6)]
        answers = [job.result(10) for job in pending if job.exception(10) is None]
    assert len(answers) == 5 and len(set(answers)) == 1 and "三秒" in answers[0]
    assert len(calls) == 2 and not proxy._book_inflight
    stats = proxy.cache_stats()["chapter_evidence"]
    assert (stats["hits"], stats["misses"]) == (4, 2)


def test_warmed_chapter_quotes_include_appended_contiguous_source_only(proxy):
    left = "第8章 灯会\n阿宁翻开了卡片。"
    right = "她高兴了三秒，转身挥手。"
    q = "按原文，第8章翻开卡片后高兴的时长是多少秒？"
    proxy.ingest(right, {"document": "灯会", "offset": len(left), "start_chapter": "8"})
    assert "阿宁" not in proxy.assemble_context(q)
    proxy.ingest(left, {"document": "灯会", "offset": 0})
    text = proxy.assemble_context(q)
    assert "阿宁翻开了卡片。她高兴了三秒" in text
    assert proxy.cache_stats()["chapter_evidence"]["misses"] == 2
    proxy.ingest("来自另一份文档。", {"document": "剧本", "offset": len(left) + len(right), "start_chapter": "8"})
    proxy.ingest("来自不连续的位置。", {"document": "灯会", "offset": 400, "start_chapter": "8"})
    text = proxy.assemble_context(q)
    assert "另一份" not in text and "不连续" not in text
