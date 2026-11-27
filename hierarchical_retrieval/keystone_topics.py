"""Register branch navigation tools with a host-owned Keystone registry and store."""

import hashlib
import json
import os
from pathlib import Path


def keystone_workspace_id(home, workspace, session_path=None):
    """Stable across CLI starts in the same profile/project, isolated for explicit session logs."""
    paths = (home, workspace) if session_path is None else (home, workspace, session_path)
    canonical = [os.path.normcase(str(Path(path).expanduser().resolve())) for path in paths]
    digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"keystone-{digest}"


def register_topic_tools(registry, store, workspace_id):
    """Bind the workspace server-side. Importing this module does not require Keystone."""
    from keystone.policy import ToolEffect
    from keystone.tools import ToolResult, ToolSpec

    def listing():
        return {**store.list_topics(workspace_id), "workspace_id": workspace_id}

    def open_existing(topic):
        if topic not in {b["topic"] for b in listing()["topics"]}:
            raise ValueError("未找到该话题，请先查看目录。")
        return store.open_topic(workspace_id, topic)

    def note(topic, text, source=None, request_id=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请提供要保存的原文。")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("source 须为非空字符串。")
        if request_id is not None and (not isinstance(request_id, str) or not 1 <= len(request_id) <= 200):
            raise ValueError("request_id 须为 1 至 200 字符。")
        return store.add_note(
            workspace_id, topic, text, source=source, include_messages=False,
            request_id=f"keystone-note:{request_id}" if request_id else None,
            fingerprint=json.dumps([topic, source], ensure_ascii=False),
        )

    def back():
        return store.back(workspace_id)

    def search(topic, query, limit=5):
        return {"topic": topic, "matches": store.search(workspace_id, topic, query, limit)}

    def source(topic, message_id, offset=0, limit=1800):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 5000:
            raise ValueError("offset 须为非负整数，limit 须在 1 至 5000 之间")
        item = store.source_message(workspace_id, topic, message_id)
        content = item["content"]
        if offset > len(content):
            raise ValueError("offset 超出原文长度")
        end = min(len(content), offset + limit)
        return {
            **item,
            "topic": topic,
            "content": content[offset:end],
            "offset": offset,
            "chars": len(content),
            "next_offset": end if end < len(content) else None,
        }

    def wrap(fn):
        def invoke(args, ctx):
            try:
                result = fn(**args)
            except (ValueError, KeyError, TypeError) as exc:
                return ToolResult(False, error=str(exc))
            return ToolResult(True, json.dumps(result, ensure_ascii=False))

        return invoke

    text = {"type": "string"}
    number = {"type": "integer"}
    for name, desc, fn, schema, effect in [
        ("topic_list", "列出当前聊天工作区的话题、停留位置及活动分支。", listing, {}, ToolEffect.READ),
        (
            "topic_open",
            "打开目录中已经存在的话题，恢复其停留位置。",
            open_existing,
            {"topic": text},
            ToolEffect.WRITE,
        ),
        (
            "topic_note",
            "在指定话题保存一条可回查的进度或更正；可新建话题，保持当前焦点。",
            note,
            {"topic": text, "text": text, "source": text, "request_id": text},
            ToolEffect.WRITE,
        ),
        ("topic_back", "返回前一个话题。", back, {}, ToolEffect.WRITE),
        (
            "topic_search",
            "只在指定话题的关键句中检索，返回匹配原文的消息编号。",
            search,
            {
                "topic": text,
                "query": text,
                "limit": {"type": "integer", "description": "可省略，默认5，范围1至50"},
            },
            ToolEffect.READ,
        ),
        (
            "topic_source",
            "用话题和消息编号展开原始发言。长原文可按返回的下一偏移量继续读。",
            source,
            {
                "topic": text,
                "message_id": number,
                "offset": {"type": "integer", "description": "可省略，默认0"},
                "limit": {"type": "integer", "description": "可省略，默认1800，最大5000"},
            },
            ToolEffect.READ,
        ),
    ]:
        registry.register(
            ToolSpec(name, desc, wrap(fn), read_only=effect == ToolEffect.READ, schema={
                "type": "object", "properties": schema,
                "required": [key for key in schema if key in {"topic", "query", "message_id", "text"}],
                "additionalProperties": False,
            }),
            trusted=True,
            effect=effect,
        )
