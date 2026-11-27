import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from hierarchical_retrieval.core.topic_workspace import TopicConflict, TopicWorkspaceStore


def turn(store, topic, text, *, workspace="group", **kwargs):
    before = store.snapshot(workspace, topic)
    return store.append_turn(workspace, topic, text, "收到。", expected_revision=before["revision"], **kwargs)


def test_position_resume_and_incidental_mentions(tmp_path):
    s = TopicWorkspaceStore(tmp_path)
    assert s.snapshot("group", "论文")["revision"] == 0 and not s.list_topics("group")["topics"]
    assert s.route("group", "聊聊论文。现在看第三节。")["topic"] == "论文"
    assert s.route("group", "换个话题聊代码。现在看看接口")["topic"] == "代码"
    turn(s, "论文", "当前进度：论文 / 第三节 / 末段；待改结尾。", source="原稿")
    turn(s, "代码", "当前进度：工具 / 验证接口。")
    assert s.route("group", "接着写第三节")["topic"] == "论文"
    assert s.route("group", "这个接口是否需要在论文里引用？")["topic"] == "代码"
    assert s.route("group", "回到手头的事情，现在该核什么？")["topic"] == "代码"
    assert s.route("group", "顺口说论文的数字改成95，先记一下，继续代码。")["status"] == "note"
    s.add_note("group", "论文", "步数更正为95。", source="校订")
    assert s.list_topics("group")["current"] == "代码"
    assert s.back("group")["current"] == "论文"
    assert (
        TopicWorkspaceStore(tmp_path).snapshot("group", "论文")["messages"][-1]["content"] == "步数更正为95。"
    )


def test_clarification_resume_persistent_and_idempotent(tmp_path):
    s = TopicWorkspaceStore(tmp_path)
    for name in ("论文", "工具", "展览"):
        s.open_topic("group", name, position="审稿表 / 第二节")
    q = "  回到之前那个审稿表，接下来核什么？  "
    choice = s.route("group", q, request_id="once", fingerprint="x")
    assert choice["status"] == "clarify" and set(choice["candidates"]) == {"论文", "工具", "展览"}
    s.add_note("group", "论文", "旁线登记码更新为T42。")
    fresh = TopicWorkspaceStore(tmp_path)
    selected = fresh.route("group", "我说的是展览。")
    assert selected["topic"] == "展览" and "接下来核什么" in selected["question"]
    turn(
        fresh,
        "展览",
        "我说的是展览。",
        wire_content=selected["question"],
        expected_workspace_revision=selected["workspace_revision"],
    )
    assert s.list_topics("group")["pending"] is None
    assert s.route("group", q, request_id="once", fingerprint="x") == choice
    assert s.list_topics("group")["pending"] is None
    with pytest.raises(TopicConflict):
        s.replay("group", "once", q, fingerprint="different")


def test_note_clarification_archives_original_correction_without_moving_focus(tmp_path):
    s = TopicWorkspaceStore(tmp_path)
    for name in ("论文", "代码"):
        s.open_topic("group", name)
    q = "先记一下，编号更正为BETA-7，不切换话题。"
    assert s.route("group", q)["status"] == "clarify"
    selected = s.route("group", "是论文。")
    assert selected["status"] == "note" and "BETA-7" in selected["question"]
    s.add_note("group", "论文", "是论文。", wire_content=selected["question"], request_id="retry")
    assert s.replay("group", "retry", "是论文。")["topic"] == "论文"
    result = s.search("group", "论文", "BETA-7")
    assert result[0]["message_id"] == 1
    assert "BETA-7" in s.source_message("group", "论文", 1)["content"]
    assert not s.list_topics("group")["pending"] and s.list_topics("group")["current"] == "代码"


def test_branch_search_and_expansion_ignore_rendered_evidence_and_adjacent_workspaces(tmp_path):
    s = TopicWorkspaceStore(tmp_path)
    messages = [
        {"role": "assistant", "content": "归档前的助手提醒。"},
        {"role": "user", "content": "原文登记码是alpha-17。"},
    ]
    assert s.bootstrap("group", "论文", messages)
    assert not s.bootstrap("group", "代码", messages)
    assert s.snapshot("group", "论文")["messages"] == messages
    turn(s, "论文", "校订登记码为alpha-31。", wire_content="prompt里的外来词phantom。", source="审稿单")
    turn(s, "论文", "另一用户的密语是zebra-9。", workspace="neighbor")
    turn(s, "代码", "无关登记码为zebra-9。")
    assert s.search("group", "论文", "zebra-9") == s.search("group", "论文", "phantom") == []
    hits = s.search("group", "论文", "登记码", limit=1)
    assert hits[0]["message_id"] == 3 and hits[0]["source"] == "审稿单"
    assert s.source_message("group", "论文", 3)["content"] == "校订登记码为alpha-31。"
    assert s.source_message("group", "论文", 1)["role"] == "assistant"
    with pytest.raises(KeyError):
        s.source_message("group", "论文", 10000)
    # Simulate a pre-index database and rebuild only the requested branch.
    with sqlite3.connect(s.path) as db:
        for table in ("topic_terms", "topic_quotes", "topic_message_refs"):
            db.execute(f"DELETE FROM {table} WHERE workspace='group' AND topic='论文'")
    assert TopicWorkspaceStore(tmp_path).search("group", "论文", "登记码", limit=1) == hits


def test_conflicting_writes_and_identical_concurrent_retries(tmp_path):
    s = TopicWorkspaceStore(tmp_path)

    def send(_):
        s.append_turn("w", "A", "q", "a", expected_revision=0, request_id="unique", fingerprint="context")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(send, range(4)))
    assert len(s.snapshot("w", "A")["messages"]) == 2
    with pytest.raises(TopicConflict):
        s.append_turn("w", "A", "q2", "a2", expected_revision=0)
    rev = s.list_topics("w")["workspace_revision"]
    s.open_topic("w", "B")
    with pytest.raises(TopicConflict):
        s.append_turn("w", "A", "q3", "a3", expected_revision=1, expected_workspace_revision=rev)


def test_open_ending_in_remember_does_not_turn_into_side_note(tmp_path):
    store = TopicWorkspaceStore(tmp_path)
    turn(store, "代码", "检查接口。")
    q = "开一个话题：论文。第二节讲误差，先记下。"
    assert store.route("group", q, topic="论文")["status"] == "ready"
    assert store.route("group", q)["topic"] == "论文"
    assert store.route("group", "当前进度：接口 / 路由；先记一下。", topic="代码")["status"] == "ready"


def test_attachment_preserves_original_reference_and_navigation(tmp_path):
    store = TopicWorkspaceStore(tmp_path)
    store.open_topic("task", "书", position="第二章")
    store.open_topic("task", "论文")
    store.route("task", "回到上次那个话题？")
    pending = store.list_topics("task")["pending"]
    for _ in range(2):
        result = store.attach_source(
            "task", "书", "原文中的暗号是棕熊。", source="第二章原文", request_id="quote1"
        )
        assert "messages" not in result and result["revision"] == 1
    assert store.list_topics("task")["current"] == "论文" and store.list_topics("task")["pending"] == pending
    msg = store.snapshot("task", "书")["messages"][0]
    assert msg["wire_content"].startswith("引用资料") and msg["content"] == "原文中的暗号是棕熊。"
    assert store.source_message("task", "书", 1)["content"] == msg["content"]
    assert store.search("task", "书", "棕熊")[0]["kind"] == "source"


def test_attachment_indexes_middle_fact_and_backfills_legacy_attachment(tmp_path):
    store = TopicWorkspaceStore(tmp_path)
    repeated = "会议总结要优先盘点设备，排班留至工作日处理。" * 3100
    needle = "ORBIT-7632"
    original = repeated[:33000] + "登记码为" + needle + "。" + repeated[33000:]
    store.attach_source("archive", "手册", original, source="手册A", request_id="book-a")
    result = store.search("archive", "手册", needle)
    assert result and result[0]["message_id"] == 1 and needle in result[0]["quote"]
    assert (
        result[0]["quote"] in original and store.source_message("archive", "手册", 1)["content"] == original
    )
    # Prior versions have message refs and selected key sentences but lack the
    # complete attachment index. No further write should be needed to recover.
    with sqlite3.connect(store.path) as db:
        ids = [r[0] for r in db.execute("SELECT id FROM topic_quotes WHERE instr(quote,?)>0", (needle,))]
        db.executemany("DELETE FROM topic_terms WHERE quote_id=?", ((i,) for i in ids))
        db.executemany("DELETE FROM topic_quotes WHERE id=?", ((i,) for i in ids))
        db.execute("DELETE FROM topic_source_coverage")
    reopened = TopicWorkspaceStore(tmp_path)
    assert reopened.search("archive", "手册", needle)[0]["message_id"] == 1
    assert not reopened.search("elsewhere", "手册", needle)


def test_attachment_keeps_identifier_spanning_long_unpunctuated_window(tmp_path):
    store = TopicWorkspaceStore(tmp_path)
    content = "甲" * 32765 + "HARBOR-3038" + "乙" * 32768
    store.attach_source("w", "抄本", content, source="原始抄本")
    hits = store.search("w", "抄本", "HARBOR-3038")
    assert hits and "HARBOR-3038" in hits[0]["quote"] and hits[0]["quote"] in content


def test_past_reference_uses_literal_evidence_and_clarifies_collisions(tmp_path):
    s = TopicWorkspaceStore(tmp_path)
    turn(s, "卷甲", "他解开青隼封签，然后拉下铜栓。")
    turn(s, "卷乙", "他们发现蓝色瓷瓶，然后转动铁轮。")
    assert s.route("group", "青隼封签后来怎样了？")["topic"] == "卷乙"
    assert s.route("group", "上次聊的那个青隼封签后来怎样了？")["topic"] == "卷甲"
    assert s.route("group", "那次提到的蓝色瓷瓶，后来呢？")["topic"] == "卷乙"
    assert s.route("group", "先前作业里提到的青隼封签后来怎样了？")["topic"] == "卷甲"
    # Repeated mentions never break a tie between two independent branches.
    for _ in range(3):
        turn(s, "卷甲", "青隼封签的印记仍然完好。")
    turn(s, "卷乙", "另一份青隼封签很轻。")
    pending = s.route("group", "上次聊的那个青隼封签后来怎样了？")
    assert pending["status"] == "clarify" and set(pending["candidates"]) == {"卷甲", "卷乙"}
    assert s.route("group", "那个相关的事情。")["status"] == "clarify"
    selected = s.route("group", "我指的是拉下铜栓的那段。")
    assert selected["topic"] == "卷甲" and "青隼封签后来怎样" in selected["question"]


def test_past_reference_ignores_other_workspaces_and_weak_fragments(tmp_path):
    s = TopicWorkspaceStore(tmp_path)
    turn(s, "卷甲", "此处有青隼，但封签已经取走。当前进度没有更新。")
    turn(s, "卷乙", "青隼封签在另一工作区。", workspace="foreign")
    q = "先前提到的青隼封签后来怎么样了？"
    assert s.route("group", q)["status"] == "clarify"
    assert s.route("group", "我指的是当前进度。")["status"] == "clarify"
    assert s.route("foreign", q)["topic"] == "卷乙"
    assert s.route("group", "是卷甲。")["topic"] == "卷甲"


def test_past_reference_in_empty_workspace_recovers_on_next_new_chat(tmp_path):
    s = TopicWorkspaceStore(tmp_path)
    q = "上次聊的那个青隼封签后来怎样了？"
    missing = s.route("empty", q, request_id="recall", fingerprint="r")
    assert missing["status"] == "clarify" and "还没有话题" in missing["clarification"]
    assert not s.list_topics("empty")["pending"]
    assert s.route("empty", "现在帮我写一份摘要。")["topic"] == "默认话题"
    assert s.route("empty", q, request_id="recall", fingerprint="r") == missing


def test_prefix_purchases_are_scoped_to_recent_real_turns(tmp_path):
    store = TopicWorkspaceStore(tmp_path)
    turn(store, "港口", "进入长任务", provider_cache=("profile", 100, 2, 0, 4))

    def busy(topic="设备", signature="profile", start=99):
        return store.has_active_prefix_peer("group", topic, signature, start, 150)

    assert busy() and not busy("港口") and not busy(signature="other") and not busy(start=100)
    assert not store.has_active_prefix_peer("another", "设备", "profile", 99, 150)
    for i in range(6):
        store.add_note("group", "设备", f"勘误{i}")
    for i in range(3):
        turn(store, "设备", f"核对{i}")
    assert busy()
    turn(store, "设备", "继续独立处理")
    assert not busy()
