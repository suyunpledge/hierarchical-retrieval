import json

import api_server
import pytest
from fastapi.testclient import TestClient
from hierarchical_retrieval.topic_chat import workspace_store


def test_branch_proxy_cache_replay_navigation_and_sources(monkeypatch, tmp_path):
    calls = []

    def upstream(url, json, **kwargs):
        calls.append(json["messages"])

        class Reply:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"message": {"content": "记录完成"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 39, "prompt_tokens_details": {"cached_tokens": 21}},
                }

        return Reply()

    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server, "_chat_backend_is_openai", lambda: True)
    monkeypatch.setattr(api_server.requests, "post", upstream)
    monkeypatch.setattr(api_server, "API_TOKEN", "")
    with TestClient(api_server.app) as c:

        def chat(text, **opts):
            return c.post(
                "/v1/chat/completions",
                headers={"X-Topic-Workspace": "one"},
                json={
                    "topic_workspace": {"workspace_id": "one", **opts},
                    "messages": [{"role": "user", "content": text}],
                },
            )

        first = chat("停在：第三节。校样编码为青鹭。", topic="校样", request_id="x")
        assert first.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 21
        assert chat("停在：第三节。校样编码为青鹭。", topic="校样", request_id="x").json()["topic_workspace"][
            "replay"
        ]
        assert chat("不同的问题", topic="校样", request_id="x").status_code == 409
        chat("索引仍待压测。", topic="代码").raise_for_status()
        chat("校样编码更正为苍鹭。", topic="校样", action="note", request_id="note").raise_for_status()
        assert c.get("/topic-workspaces/one").json()["current"] == "代码" and len(calls) == 2
        chat("接着写第三节。", topic="校样").raise_for_status()
        assert calls[-1][: len(calls[0])] == calls[0]
        assert "苍鹭" in str(calls[-1]) and "索引仍待压测" not in str(calls[-1])
        listing = c.post("/topic-workspaces/one/back").json()
        assert listing["current"] == "代码"
        hits = c.get("/topic-workspaces/one/search", params={"topic": "校样", "query": "苍鹭"}).json()
        hit = next(h for h in hits if "苍鹭" in h["quote"])
        url = "/topic-workspaces/{}/source"
        params = {"topic": "校样", "message_id": hit["message_id"]}
        assert "苍鹭" in c.get(url.format("one"), params=params).json()["content"]
        assert c.get(url.format("other"), params=params).status_code == 404


def test_interrupted_stream_does_not_advance_topic(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server, "API_TOKEN", "")
    monkeypatch.setattr(api_server, "_chat_backend_is_openai", lambda: True)
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "半句"}}]})]

    class Reply:
        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def iter_lines(self, **_):
            return iter(lines)

    def upstream(url, json, **kwargs):
        assert json["stream_options"] == {"include_usage": True}
        return Reply()

    monkeypatch.setattr(api_server.requests, "post", upstream)
    body = {
        "topic_workspace": {"workspace_id": "one", "topic": "论文", "request_id": "retry"},
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "继续第三节。"}],
    }
    with TestClient(api_server.app) as c:
        path = "/v1/chat/completions"
        assert "incomplete_stream" in c.post(path, json=body).text
        store = workspace_store(str(tmp_path))
        assert store.snapshot("one", "论文")["revision"] == 0
        lines.extend(["data: {", "data: [DONE]"])
        assert "invalid_stream" in c.post(path, json=body).text
        assert store.snapshot("one", "论文")["revision"] == 0
        usage = {
            "prompt_tokens": 42,
            "completion_tokens": 5,
            "total_tokens": 47,
            "prompt_tokens_details": {"cached_tokens": 32},
        }
        lines[1:] = [
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: " + json.dumps({"choices": [], "usage": usage}),
            "data: [DONE]",
        ]
        events = [
            json.loads(line[6:])
            for line in c.post(path, json=body).text.splitlines()
            if line.startswith("data: {")
        ]
        assert events[-2]["topic_workspace"]["revision"] == 1
        assert events[-1]["choices"] == [] and events[-1]["usage"] == usage
        assert store.snapshot("one", "论文")["revision"] == 1
        assert "半句" in c.post(path, json=body).text and store.snapshot("one", "论文")["revision"] == 1


def test_existing_client_transcript_imported_only_once(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server, "API_TOKEN", "")
    older = [{"role": "user", "content": "第一轮原文"}, {"role": "assistant", "content": "第一轮回答"}]
    req = {
        "messages": [*older, {"role": "user", "content": "补记新内容。"}],
        "topic_workspace": {"workspace_id": "late", "topic": "资料", "action": "note", "request_id": "n"},
    }
    with TestClient(api_server.app) as c:
        for _ in range(2):
            c.post("/v1/chat/completions", json=req).raise_for_status()
        s = workspace_store(str(tmp_path))
        assert [m["content"] for m in s.snapshot("late", "已有对话")["messages"]] == [
            x["content"] for x in older
        ]
        assert s.snapshot("late", "资料")["revision"] == 1


def test_l1_ingest_attaches_original_only_to_chosen_branch(monkeypatch, tmp_path):
    from test_fixes import CharGramEmbedding

    from hierarchical_retrieval import HConfig, HierarchicalRetrieval

    config = HConfig(storage_root=str(tmp_path))
    engine = HierarchicalRetrieval(embedding=CharGramEmbedding(config), config=config)
    monkeypatch.setattr(api_server, "config", config)
    monkeypatch.setattr(api_server, "hr", engine)
    monkeypatch.setattr(api_server, "API_TOKEN", "")
    original = "展品用了靛蓝绸布。验收编码为青隼。"
    with TestClient(api_server.app) as client:
        base = "/topic-workspaces/reader"
        client.post(base + "/open", json={"topic": "论文"}).raise_for_status()
        raw = client.post("/ingest", json={"text": original})
        raw.raise_for_status()
        entry_id = raw.json()["entry_id"]
        attachment = {"topic": "展览", "entry_id": entry_id}
        for _ in range(2):
            added = client.post(base + "/attach", json=attachment)
            added.raise_for_status()
            assert added.json()["revision"] == 1
        assert client.get(base).json()["current"] == "论文"
        hits = client.get(base + "/search", params={"topic": "展览", "query": "青隼"}).json()
        assert hits[0]["kind"] == "source" and hits[0]["source"] == entry_id
        source = client.get(base + "/source", params={"topic": "展览", "message_id": hits[0]["message_id"]})
        assert source.json()["content"] == original
        assert client.get(base + "/search", params={"topic": "论文", "query": "青隼"}).json() == []
        assert (
            client.post(base + "/attach", json={**attachment, "entry_id": "cc_000000000000"}).status_code
            == 404
        )
        assert client.post(base + "/attach", json={**attachment, "entry_id": "../x"}).status_code == 422


@pytest.mark.parametrize(
    "target,protocol,switch,model,workspace,expected_mark,official",
    [
        ("https://api.openai.com", True, "auto", "gpt-6.2-sol", True, True, True),
        ("https://api.openai.com:443/v1/", True, "auto", "gpt-6.2-astra-2026-10-01", True, True, True),
        ("https://api.openai.com/v1", True, "0", "gpt-6.2-sol", True, False, True),
        ("https://api.openai.com", True, "auto", "other-model", True, False, True),
        ("https://api.openai.com", True, "1", "gpt-6.2-sol", False, False, True),
        ("https://api.openai.com.example", True, "auto", "gpt-6.2-sol", True, False, False),
        ("http://api.openai.com", True, "auto", "gpt-6.2-sol", True, False, False),
        ("http://localhost:8181", True, "1", "hosted-model", True, True, False),
        ("http://localhost:8181", False, "1", "gpt-6.2-sol", True, False, False),
    ],
)
def test_openai_breakpoints_and_credentials_stay_on_supported_requests(
    monkeypatch, tmp_path, target, protocol, switch, model, workspace, expected_mark, official
):
    calls = []
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server.config, "context_char_budget", 5500)
    monkeypatch.setattr(api_server.config, "chat_backend_url", target)
    monkeypatch.setattr(api_server, "_chat_backend_is_openai", lambda: protocol)
    monkeypatch.setattr(api_server, "API_TOKEN", "client-token")
    monkeypatch.setenv("HR_OPENAI_API_KEY", "provider-token")
    monkeypatch.setenv("HR_OPENAI_CACHE_BREAKPOINTS", switch)

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            if protocol:
                return {"choices": [{"message": {"content": "收到。"}}]}
            return {"message": {"content": "收到。"}}

    def upstream(url, json, **options):
        calls.append(json)
        if official:
            assert url == "https://api.openai.com/v1/chat/completions"
            assert options["headers"] == {"Authorization": "Bearer provider-token"}
            assert options["allow_redirects"] is False
        else:
            assert "headers" not in options
        if protocol:
            token_limit = (
                "max_completion_tokens" if official and model.startswith("gpt-6.2-") else "max_tokens"
            )
            assert json[token_limit] == 80
        else:
            assert json["options"]["num_predict"] == 80
        assert "prompt_cache_options" not in json
        return Reply()

    monkeypatch.setattr(api_server.requests, "post", upstream)
    original = "入库材料，全文逐字登记。" * 220
    headers = {"Authorization": "Bearer client-token"}
    with TestClient(api_server.app, headers=headers) as client:

        def chat(text, turn):
            req = {"model": model, "max_tokens": 80, "messages": [{"role": "user", "content": text}]}
            if workspace:
                req["topic_workspace"] = {"workspace_id": "w", "topic": "账目", "request_id": turn}
            response = client.post("/v1/chat/completions", json=req)
            response.raise_for_status()
            return response.json()

        chat(original, "first")
        initial = calls[0]["messages"]

        def markers(messages):
            return [(i, m["content"][0]) for i, m in enumerate(messages) if isinstance(m["content"], list)]

        first_marks = markers(initial)
        assert bool(first_marks) == expected_mark
        assert len(first_marks) <= 3
        for i, part in first_marks:
            assert initial[i]["role"] == "user"
            assert part == {
                "type": "text",
                "text": part["text"],
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        if workspace:
            chat("继续登记。", "next")
            assert markers(calls[-1]["messages"]) == first_marks
            assert chat(original, "first")["topic_workspace"]["replay"]
            assert len(calls) == 2
            archived = workspace_store(str(tmp_path)).snapshot("w", "账目")["messages"][0]
            assert archived.get("wire_content", archived["content"]) == original


def test_official_stream_and_model_catalog_use_only_configured_upstream_key(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server.config, "chat_backend_url", "https://api.openai.com/v1")
    monkeypatch.setattr(api_server, "_chat_backend_is_openai", lambda: True)
    monkeypatch.setattr(api_server, "API_TOKEN", "client-token")
    monkeypatch.setenv("HR_OPENAI_API_KEY", "stream-test-key")
    calls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "gpt-6.2-sol"}]}

        def iter_lines(self, **_):
            yield 'data: {"choices":[{"delta":{"content":"收到。"},"finish_reason":"stop"}]}'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":70,"prompt_tokens_details":{"cached_tokens":0}}}'
            yield "data: [DONE]"

    def upstream(url, **kwargs):
        if url.endswith("/chat/completions"):
            assert kwargs["json"]["stream_options"] == {"include_usage": True}
        assert kwargs["headers"] == {"Authorization": "Bearer stream-test-key"}
        assert kwargs["allow_redirects"] is False
        calls.append(url)
        return Reply()

    monkeypatch.setattr(api_server.requests, "post", upstream)
    monkeypatch.setattr(api_server.requests, "get", upstream)
    with TestClient(api_server.app, headers={"Authorization": "Bearer client-token"}) as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "gpt-6.2-sol"
        stream = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6.2-sol",
                "stream": True,
                "topic_workspace": {"workspace_id": "stream", "topic": "账目"},
                "messages": [{"role": "user", "content": "核对结余。"}],
            },
        )
        assert "topic_workspace" in stream.text and "[DONE]" in stream.text
        assert '"usage":' not in stream.text
    assert calls == ["https://api.openai.com/v1/models", "https://api.openai.com/v1/chat/completions"]


def test_idle_branch_clock_is_durable_and_only_real_prefixes_refresh_it(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hierarchical_retrieval import topic_chat

    now = [10_000.0]
    calls = []

    def backend(url, json, **kwargs):
        calls.append(json)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": "收到"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100},
            },
        )

    monkeypatch.setattr(topic_chat, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server.config, "context_char_budget", 5100)
    monkeypatch.setattr(api_server.config, "chat_backend_url", "https://api.openai.com/v1")
    monkeypatch.setattr(api_server, "CHAT_BACKEND_OPENAI", "1")
    monkeypatch.setattr(api_server.requests, "post", backend)
    monkeypatch.setattr(api_server, "API_TOKEN", "")

    with TestClient(api_server.app) as client:

        def chat(q, label, *, head="工作台", action="chat", topic="校样"):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-6.2-sol",
                    "topic_workspace": {
                        "workspace_id": "w",
                        "topic": topic,
                        "action": action,
                        "request_id": label,
                    },
                    "messages": [{"role": "system", "content": head}, {"role": "user", "content": q}],
                },
            )
            response.raise_for_status()
            return response.json()["topic_workspace"]

        opening = "原始封签为青隼。" + ("需要分别核实每条登记。" * 160)
        chat(opening, "read")
        now[0] += 1790
        chat("以下是附件，读完仅回答已读。\n" + ("另一份材料。" * 350), "another")
        assert opening[:100] not in str(calls[-1])
        assert calls[-1]["prompt_cache_options"] == {"mode": "explicit"}
        assert "prompt_cache_options" not in calls[0]
        chat("另一支记录。", "switch", topic="设备")
        chat("更正：当前封签为苍鹭。", "note", action="note")
        assert chat(opening, "read")["replay"]
        assert len(calls) == 3
        topic_chat.workspace_store.cache_clear()
        snap = topic_chat.workspace_store(str(tmp_path)).snapshot("w", "校样")
        assert snap["provider_cache"]["sent_at"] == 10000
        assert snap["messages"][0]["content"] == opening
        now[0] += 10
        cold = chat("按最新更正，当前封签是什么？", "cold")
        assert cold["prefix_policy"] == "blocks_0" and cold["opening_chars"] == 0
        assert "苍鹭" in str(calls[-1]) and opening[:200] not in str(calls[-1])
        assert chat("那么最早的封签呢？", "continue")["prefix_policy"] == "blocks_0"
        assert chat("请概括整篇记录。", "broad")["prefix_policy"] == "full"
        now[0] += 3
        warm = chat("现在用哪种封签？", "warm")
        assert warm["prefix_policy"] == "blocks_2" and warm["opening_chars"] > 0
        assert (
            topic_chat.workspace_store(str(tmp_path)).snapshot("w", "校样")["provider_cache"]["sent_at"]
            == now[0]
        )
        assert chat("封签呢？", "new-instructions", head="新审阅台")["prefix_policy"] == "blocks_0"


def test_repeated_cold_returns_purchase_once_per_branch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hierarchical_retrieval import topic_chat

    now = [1000.0]
    sent = []

    def respond(url, json, **kwargs):
        sent.append(json)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": "已经核对。"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 950, "prompt_tokens_details": {"cached_tokens": 0}},
            },
        )

    monkeypatch.setattr(topic_chat, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server.config, "context_char_budget", 5500)
    monkeypatch.setattr(api_server.config, "chat_backend_url", "https://api.openai.com/v1")
    monkeypatch.setattr(api_server, "CHAT_BACKEND_OPENAI", "1")
    monkeypatch.setattr(api_server, "API_TOKEN", "")
    monkeypatch.setattr(api_server.requests, "post", respond)
    with TestClient(api_server.app) as client:

        def ask(text, key, system="校样台", topic="港口", action="chat", remaining=None):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-6.2-sol",
                    "topic_workspace": {
                        "workspace_id": "desk",
                        "topic": topic,
                        "action": action,
                        "request_id": key,
                        **({"remaining_topic_questions": remaining} if remaining is not None else {}),
                    },
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                },
            )
            r.raise_for_status()
            return r.json()["topic_workspace"]

        def state():
            return topic_chat.workspace_store(str(tmp_path)).snapshot("desk", "港口")["provider_cache"]

        for key, text in (
            ("one", "码头巡查登记。" * 350),
            ("two", "以下是新材料，读完仅回复已读。\n" + "设备巡查记录。" * 380),
        ):
            ask(text, key, system="先前的校样台")
        ask("更正：港口编号为K98。", "notice", action="note")
        for i in range(6):
            question = "请核对目前的港口编号？"
            assert ask(question, str(i))["opening_chars"] == 0
            before = state()
            assert ask(question, str(i))["replay"]
            ask("切回设备，读表。", "other" + str(i), topic="设备")
            ask("更正：港口编号为K98。", "note" + str(i), action="note")
            assert state() == before
            now[0] += 1
        topic_chat.workspace_store.cache_clear()
        first = ask("港口编号现在是什么？", "promote")
        assert first["prefix_policy"] == "blocks_1" and first["opening_chars"] > 0
        assert state()["tail_floor"] > 0
        topic_chat.workspace_store.cache_clear()
        next_ask = ask("再核对一次。", "no-read")
        assert next_ask["opening_chars"] > 0
        assert state()["prefix_blocks"] == 0
        for i in range(7):
            assert ask("目前用什么编号？", "miss" + str(i))["opening_chars"] == 0
        ask("请确认。", "reset", system="另一种固定指令")
        assert state()["missed_tokens"] >= 0
        assert ask("再查一次。", "short-plan", system="新一轮", remaining=6)["opening_chars"] == 0
        assert ask("进入长任务。", "long-plan", system="新一轮", remaining=12)["opening_chars"] > 0
        assert ask("进入长任务。", "long-plan", system="新一轮", remaining=11)["replay"]


def test_provisional_opening_requires_a_provider_write_or_possible_reuse():
    from hierarchical_retrieval.topic_chat import _AWAITING_REUSE, _KEEP_COMPACT, _prefix_state

    def purchase(details, target=1):
        usage = {"prompt_tokens": 990, "prompt_tokens_details": details}
        return _prefix_state(target, 550, target, True, usage, 1700, 50, 550)

    assert purchase({"cached_tokens": 0, "cache_write_tokens": 0}) == (0, _KEEP_COMPACT)
    assert purchase({"cached_tokens": 0}) == (1, _AWAITING_REUSE)
    assert purchase({"cached_tokens": 0, "cache_write_tokens": 966}) == (1, _AWAITING_REUSE)
    assert purchase({"cached_tokens": 800, "cache_write_tokens": 0}) == (1, _AWAITING_REUSE)
    assert purchase({"cached_tokens": 0, "cache_write_tokens": 0}, target=2) == (2, _AWAITING_REUSE)
