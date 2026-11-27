from hierarchical_retrieval.core.chapter_history import chapter_history


def event(kind, user, assistant=None):
    return {"kind": kind, "user": user, "assistant": assistant}


def test_chapter_history_evicts_oldest_exchange_and_preserves_verbatim_notes():
    original = "第12章 窗前\n他把书放回架上。"
    second = "第13章 灯下\n她离开了。"
    events = [
        event("source", original + "\n" + second),
        event("note", "初稿：放在窗前"),
        event("turn", "第一次怎么写？", "放在窗前。"),
        event("note", "修订：放在桌上"),
        event("turn", "现在呢？", "现在放在桌上。"),
    ]
    system = [{"role": "system", "content": "核对版本"}]
    question = "第十二章怎么写的？"
    whole = chapter_history("12", events, system, question, 1000)
    expected = [
        "核对版本",
        original,
        "初稿：放在窗前",
        "第一次怎么写？",
        "放在窗前。",
        "修订：放在桌上",
        "现在呢？",
        "现在放在桌上。",
        question,
    ]
    assert [m["content"] for m in whole["messages"]] == expected
    assert whole["cache_breakpoints"] == [1]
    compact = chapter_history("12", events, system, question, whole["chars"] - 1)
    assert [m["content"] for m in compact["messages"]] == expected[:3] + expected[5:]
    for m in (whole, compact):
        assert sum(len(i["content"]) for i in m["messages"]) == m["chars"]
    mandatory = sum(map(len, (expected[0], original, expected[2], expected[5], question)))
    assert chapter_history("12", events, system, question, mandatory)["chars"] == mandatory
    assert chapter_history("12", events, system, question, mandatory - 1) is None


def test_chapter_history_ambiguous_source_and_unpaired_input_fall_back():
    source = event("source", "第8章 回城\n行人已经离开。")
    assert chapter_history("8", [source, event("source", "第8章 修订\n他还在。")], [], "第8章？", 500) is None
    assert chapter_history("8", [source, event("user", "散落的一问")], [], "第8章？", 500) is None
    assert chapter_history("9", [source], [], "第9章？", 500) is None


def test_proxy_sends_only_its_topic_history_and_can_disable_chapter_routing(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import api_server
    from fastapi.testclient import TestClient
    from hierarchical_retrieval.topic_chat import workspace_store

    calls = []

    def send(url, *, json, **kwargs):
        calls.append(json)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "读到了。"}, "finish_reason": "stop"}]},
        )

    monkeypatch.setenv("HR_WHOLE_CHAPTER_HISTORY", "1")
    monkeypatch.setenv("HR_OPENAI_CACHE_BREAKPOINTS", "1")
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server.config, "chat_backend_url", "https://api.openai.com/v1")
    monkeypatch.setattr(api_server, "_chat_backend_is_openai", lambda: True)
    monkeypatch.setattr(api_server, "CHAT_BACKEND_OPENAI", "1")
    monkeypatch.setattr(api_server, "API_TOKEN", "")
    monkeypatch.setattr(api_server.requests, "post", send)
    store = workspace_store(str(tmp_path))
    for topic in ("场景甲", "场景乙"):
        store.attach_source("对稿", topic, "第九章 " + topic + "\n他们重新落座。", source="底稿")
    with TestClient(api_server.app) as client:

        def chat(topic, question):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-6.2-sol",
                    "topic_workspace": {"workspace_id": "对稿", "topic": topic},
                    "messages": [
                        {"role": "system", "content": "按指定版本。"},
                        {"role": "user", "content": question},
                    ],
                },
            )
            assert response.status_code == 200
            return response.json()["topic_workspace"]

        assert chat("场景甲", "第九章谁落座？")["prefix_policy"] == "whole_chapter_history"
        first_key = calls[-1]["prompt_cache_key"]
        chat("场景乙", "第九章讲了什么？")
        assert calls[-1]["prompt_cache_key"] != first_key
        store.add_note("对稿", "场景甲", "甲稿改为起身。")
        chat("场景甲", "第九章现在呢？")
        assert calls[-1]["prompt_cache_key"] == first_key
        assert calls[-1]["messages"][1] == calls[0]["messages"][1]
        assert [m["content"] for m in calls[-1]["messages"][2:]] == [
            "第九章谁落座？",
            "读到了。",
            "甲稿改为起身。",
            "第九章现在呢？",
        ]
        monkeypatch.setenv("HR_CHAPTER_QA_WINDOW", "0")
        chat("场景甲", "第九章工作稿讲了什么？")
        assert [m["content"] for m in calls[-1]["messages"][2:]] == [
            "甲稿改为起身。",
            "第九章工作稿讲了什么？",
        ]
        monkeypatch.setenv("HR_CHAPTER_QA_WINDOW", "all")
        chat("场景甲", "第九章原文和稿子各怎么写？")
        assert calls[-1]["messages"][1] == calls[0]["messages"][1]
        assert calls[-1]["prompt_cache_key"] == first_key
        assert "第九章谁落座？" in [m["content"] for m in calls[-1]["messages"][2:]]
        monkeypatch.setenv("HR_WHOLE_CHAPTER_HISTORY", "0")
        chat("场景甲", "再读第九章。")
        assert "prompt_cache_key" not in calls[-1]


def test_qa_window_preserves_source_edits_and_requested_earlier_answers():
    source = "第九章 风起\n她把窗户关上。"
    events = [
        event("source", source),
        event("turn", "第九章窗边是谁？", "她。"),
        event("note", "第一稿：改为开窗。"),
        event("turn", "我刚刚说错了，位置改为门口，记下了吗？", "已记下。"),
        event("turn", "门的原色是蓝色。", "收到。"),
        event("note", "工作稿第二次更正：停在帘边。"),
        event("turn", "第九章现在停在哪儿？", "帘边。"),
        event("turn", "还是刚才那一版？", "是。"),
    ]
    question = "第九章上次那两项是什么？"

    def body(q, keep):
        return [m["content"] for m in chapter_history("9", events, [], q, 2000, qa_window=keep)["messages"]]

    full = body(question, None)
    recent = body(question, 1)
    assert recent == [
        source,
        "第一稿：改为开窗。",
        events[3]["user"],
        "已记下。",
        events[4]["user"],
        "收到。",
        events[5]["user"],
        events[7]["user"],
        "是。",
        question,
    ]
    assert body(question, 2) == [*recent[:-3], events[6]["user"], "帘边。", *recent[-3:]]
    assert body(question, 0) == [*recent[:-3], question]
    retrospective = "第九章第一轮你怎么回答的？"
    assert body(retrospective, 1) == [*full[:-1], retrospective]
    assert chapter_history("9", events, [], question, len(source), qa_window=0) is None
