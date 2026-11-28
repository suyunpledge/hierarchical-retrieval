"""Topic-aware chat adapter. The workspace owns history; the client supplies the current turn."""

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class TopicMode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str = Field(min_length=1, max_length=256)
    topic: Optional[str] = Field(default=None, min_length=1, max_length=256)
    action: Literal["chat", "note"] = "chat"
    history_topic: Optional[str] = Field(default=None, min_length=1, max_length=256)
    remaining_topic_questions: Optional[int] = Field(default=None, ge=1, strict=True)
    position: Optional[str] = Field(default=None, max_length=1024)
    source: Optional[str] = Field(default=None, max_length=1024)
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=256)


class TopicLocation(BaseModel):
    topic: str = Field(min_length=1, max_length=256)
    position: Optional[str] = Field(default=None, max_length=1024)
    source: Optional[str] = Field(default=None, max_length=1024)


@lru_cache(maxsize=4)
def workspace_store(root):
    from .core.topic_workspace import TopicWorkspaceStore

    return TopicWorkspaceStore(root)


def workspace_call(method, *args, **kwargs):
    from .core.topic_workspace import TopicConflict

    try:
        return method(*args, **kwargs)
    except TopicConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@dataclass
class BranchTurn:
    messages: list
    metadata: dict
    local_answer: Optional[str] = None
    persist: Optional[Callable[[str, Optional[dict]], tuple[dict, str]]] = None
    cache_breakpoints: list[int] = field(default_factory=list)
    one_time_document: bool = False
    cache_evidence_prefix_only: bool = False

    prompt_cache_key: Optional[str] = None

    def openai_cache_messages(self):
        """Annotate only the outbound copy. The persisted transcript stays plain text."""
        if not self.cache_breakpoints:
            return self.messages
        marked = set(self.cache_breakpoints)
        return [
            {
                **message,
                "content": [
                    {
                        "type": "text",
                        "text": message["content"],
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            }
            if index in marked
            else message
            for index, message in enumerate(self.messages)
        ]

    def finish(self, answer, usage=None):
        if self.persist is not None:
            snapshot, answer = self.persist(answer, usage)
            self.metadata["revision"] = snapshot["revision"]
            self.metadata["position"] = snapshot.get("position")
            self.metadata["breadcrumb"] = snapshot.get("breadcrumb")
        return answer


# The proxy requests the provider's default 30-minute writes. An idle branch
# may outlive that period in SQLite. Its evidence remains available for lookup.
_CACHE_WINDOW_SECONDS = 30 * 60
_BROAD_READING = re.compile(
    r"全文|逐(?:句|段|字)|通篇|梳理|概括|总结|翻译|续写|复述|"
    r"\b(?:summarize|summarise|translate|rewrite|full text|continue writing)\b",
    re.I,
)


# Stored in missed_tokens once an adaptive purchase has been attempted.
_AWAITING_REUSE = -1
_KEEP_COMPACT = -2


def _prefix_state(blocks, prior, target, promoting, usage, chars, fixed_chars, threshold):
    """Decide the next opening and carry only measured, avoidable input forward."""
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    reads = details.get("cached_tokens") if isinstance(details, dict) else None
    total = usage.get("prompt_tokens")
    valid = type(reads) is int and type(total) is int and 0 <= reads <= total and total > 0
    if promoting:
        # A provider-confirmed zero write/read cannot seed the provisional block.
        # Older providers without write telemetry still get one chance at reuse.
        writes = details.get("cache_write_tokens") if isinstance(details, dict) else None
        if target == 1 and valid and reads == 0 and type(writes) is int and writes == 0:
            return 0, _KEEP_COMPACT
        return blocks, _AWAITING_REUSE if blocks else _KEEP_COMPACT
    if blocks:
        if prior == _AWAITING_REUSE:
            if not valid:
                return blocks, prior
            if reads == 0:
                return 0, _KEEP_COMPACT
        return blocks, 0
    if target != 0 or prior < 0 or not valid:
        return blocks, prior
    if reads * 2 >= total:
        return blocks, 0
    # Approximate the share attributable to history. User input and fixed
    # instructions would still have to be provided with either representation.
    avoided = (total - reads) * max(0, chars - fixed_chars) // max(1, chars)
    # Require six genuine paid visits before assuming another few will
    # amortize an explicit write. Notes, retries and hot reads earn no credit.
    credit = min((threshold + 5) // 6, avoided)
    return blocks, min(threshold, prior + credit)


def prepare_turn(mode, incoming, model, root, budget, *, temperature=None, max_tokens=None, cache_scope="", chapter_cache=False):
    from .core.branch_context import EVIDENCE_ORDER, build_branch_context, chapter_prefix_layout

    if not incoming or incoming[-1].role != "user" or not incoming[-1].content.strip():
        raise HTTPException(422, "话题聊天需要以本轮用户消息结尾")
    store = workspace_store(root)
    wid = mode.workspace_id
    user = incoming[-1].content
    system = []
    for m in incoming[:-1]:
        if m.role not in ("system", "developer"):
            break
        system.append(m.model_dump())
    fingerprint = None
    meta = {"workspace_id": wid}
    if mode.request_id:
        identity = [
            model,
            temperature,
            max_tokens,
            system,
            user,
            mode.model_dump(exclude={"request_id", "remaining_topic_questions"}),
        ]
        fingerprint = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        replay = workspace_call(store.replay, wid, mode.request_id, user, fingerprint=fingerprint)
        if replay is not None:
            meta.update(
                status=replay.get("status", "replay"),
                topic=replay["topic"],
                revision=replay["revision"],
                replay=True,
            )
            if "candidates" in replay:
                meta["candidates"] = replay["candidates"]
            return BranchTurn([], meta, replay["assistant"])

    # Full-history clients may enable topic management in a running chat. Preserve
    # those unlabelled turns once in their own branch before routing the new turn.
    if len(incoming) > len(system) + 1:
        prior = incoming[len(system) : -1]
        if any(m.role not in ("user", "assistant") for m in prior):
            raise HTTPException(422, "话题聊天仅支持开头的系统指令及其后的用户/助手文本")
        directory = workspace_call(store.list_topics, wid)
        if not directory["topics"] and not directory["pending"]:
            workspace_call(
                store.bootstrap, wid, mode.history_topic or "已有对话", [m.model_dump() for m in prior]
            )
    route = workspace_call(
        store.route, wid, user, topic=mode.topic, request_id=mode.request_id, fingerprint=fingerprint,
        allow_reading=mode.action != "note",
    )
    meta.update(status=route["status"], topic=route.get("topic"))
    if route["status"] == "clarify":
        meta["candidates"] = route["candidates"]
        return BranchTurn(
            [], meta, route.get("clarification") or ("你想接着聊哪个话题？" + "、".join(route["candidates"]))
        )
    topic = route["topic"]
    common = dict(
        expected_workspace_revision=route["workspace_revision"],
        request_id=mode.request_id,
        fingerprint=fingerprint,
        source=mode.source,
    )
    if mode.action == "note" or route["status"] == "note":
        ack = f"已记入「{topic}」，当前话题未切换。"
        snapshot = workspace_call(
            store.add_note,
            wid,
            topic,
            user,
            acknowledgement=ack,
            wire_content=route["question"],
            include_messages=False,
            **common,
        )
        meta.update(status="note", revision=snapshot["revision"])
        return BranchTurn([], meta, ack)
    from .core.branch_context import _focused_chapter
    from .core.chapter_history import chapter_history

    named = None
    if (
        chapter_cache
        and os.getenv("HR_WHOLE_CHAPTER_HISTORY", "1") == "1"
        and not (route.get("reading") or mode.position or mode.source)
    ):
        named = _focused_chapter(user)
        # A clarification can prepend an older question. Both must still refer
        # to a single chapter before replacing retrieval with its full text.
        if named and route["question"] != user.strip() and _focused_chapter(route["question"]) != named:
            named = None
    snapshot = workspace_call(store.snapshot, wid, topic, include_events=bool(named))
    assembled = None
    if named and not snapshot.get("position"):
        setting = os.getenv("HR_CHAPTER_QA_WINDOW", "2")
        window = {"0": 0, "1": 1, "2": 2, "all": None}.get(setting, 2)
        assembled = chapter_history(
            named, snapshot["events"], system, route["question"], budget, qa_window=window,
            recover_edition=os.getenv("HR_CHAPTER_EDITION_ANCHOR", "0") == "1",
        )
    reading = route.get("reading")
    if reading and snapshot["revision"] != reading["branch_revision"]:
        raise HTTPException(409, "所选话题的依据已更新，请重试本轮请求")
    previous = snapshot["provider_cache"]
    sent_at = time.time()
    signature = None
    warm = False
    prefix_blocks = None
    # Require repeated paid history on this branch; a single large attachment
    # or two short detours should not purchase an otherwise cold prefix.
    threshold = max(512, budget // 10)
    missed = 0
    promoting = False
    tail_floor = 0
    track_demand = str(model).lower().split("-20")[0] == "gpt-6.2-sol"
    chapter_prefix = bool(
        cache_scope
        and track_demand
        and os.getenv("HR_IMPLICIT_FOLLOWUPS") == "1"
        and chapter_prefix_layout(snapshot["messages"], route["question"])
    )
    if cache_scope and assembled is None:
        profile_fields = [model, system, budget, cache_scope]
        if chapter_prefix:
            profile_fields.append("chapter-prefix:1450")
        if not EVIDENCE_ORDER:
            profile_fields.append("evidence-order:off")
        profile = json.dumps(profile_fields, ensure_ascii=False)
        signature = hashlib.sha256(profile.encode()).hexdigest()
        warm = (
            previous is not None
            and previous["signature"] == signature
            and 0 <= sent_at - previous["sent_at"] < _CACHE_WINDOW_SECONDS
        )
        if snapshot["messages"] and not _BROAD_READING.search(route["question"]):
            prefix_blocks = min(previous["prefix_blocks"], 2) if warm else 0
            missed = previous["missed_tokens"] if warm and track_demand else 0
            if warm and prefix_blocks:
                tail_floor = previous["tail_floor"]
            remaining = mode.remaining_topic_questions
            sustained = missed >= threshold if remaining is None else remaining >= 8 and missed >= 0
            if (
                track_demand
                and prefix_blocks == 0
                and sustained
                and (
                    remaining is not None
                    or not workspace_call(
                        store.has_active_prefix_peer,
                        wid,
                        topic,
                        signature,
                        sent_at - _CACHE_WINDOW_SECONDS,
                        sent_at,
                    )
                )
            ):
                # For an open-ended conversation, risk one opening block first.
                prefix_blocks = 1 if remaining is None else 2
                promoting = True
                # Anchor the new suffix at the most recent exchange. Earlier
                # evidence still participates in checkpointing and source lookup.
                history = snapshot["messages"]
                tail_floor = len(history) - 1
                if tail_floor and history[-1]["role"] == "assistant" and history[-2]["role"] == "user":
                    tail_floor -= 1
    built = assembled or workspace_call(
        build_branch_context,
        system=system,
        topic=topic,
        messages=snapshot["messages"],
        question=route["question"],
        char_budget=budget,
        prefix_blocks=prefix_blocks,
        compact_lookup=bool(cache_scope) and track_demand,
        chapter_prefix=chapter_prefix,
        tail_floor=tail_floor,
        allow_document_receipt=bool(signature) and not mode.source and not mode.position,
        position=mode.position if mode.position is not None else snapshot.get("position") or "",
        source=mode.source if mode.source is not None else snapshot.get("source") or "",
        search=lambda query, limit: store.search(wid, topic, query, limit=limit),
        **({"reading": reading} if reading else {}),
    )
    cache_record = None
    cache_key = None
    if assembled:
        identity = json.dumps([wid, topic, named, model, system], ensure_ascii=False, sort_keys=True)
        cache_key = hashlib.sha256(identity.encode()).hexdigest()
        meta.update(prefix_policy="whole_chapter_history")
    if signature:
        # A request that omits a cached opening cannot renew its longer prefix.
        # Keep the last actual transmission time when that prefix may still be warm.
        renewal_blocks = min(previous["prefix_blocks"], 2) if warm else 0
        if "one_time_document" not in built and built["prefix_blocks"] >= renewal_blocks:
            cache_record = (signature, sent_at, built["prefix_blocks"])
        meta.update(
            opening_chars=built["opening_chars"],
            prefix_policy=(
                "full" if prefix_blocks is None or prefix_blocks == 3 else f"blocks_{prefix_blocks}"
            ),
        )
    meta.update(checkpoint=built["checkpoint"], recalled=built["recalled"], chars=built["chars"])

    def persist(answer, usage):
        record = cache_record
        if record and track_demand:
            blocks, demand = _prefix_state(
                built["prefix_blocks"],
                missed,
                prefix_blocks,
                promoting,
                usage,
                built["chars"],
                len(route["question"]) + sum(len(m["content"]) for m in system),
                threshold,
            )
            record = (signature, sent_at, blocks, demand, tail_floor if blocks else 0)
        saved = workspace_call(
            store.append_turn,
            wid,
            topic,
            user,
            answer,
            expected_revision=snapshot["revision"],
            provider_cache=record,
            reading_focus=reading["focus"] if reading else None,
            include_messages=False,
            position=mode.position,
            wire_content=built["wire_user"],
            **common,
        )
        if mode.request_id:
            committed = workspace_call(store.replay, wid, mode.request_id, user, fingerprint=fingerprint)
            saved["revision"] = committed["revision"]
            answer = committed["assistant"]
        return saved, answer

    return BranchTurn(
        built["messages"],
        meta,
        persist=persist,
        cache_breakpoints=built.get("cache_breakpoints", []),
        one_time_document=built.get("one_time_document", False),
        cache_evidence_prefix_only=built.get("cache_evidence_prefix_only", False),
        prompt_cache_key=cache_key,
    )
