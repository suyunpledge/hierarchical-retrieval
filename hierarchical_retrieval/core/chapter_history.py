"""Assemble one topic's dated edits and exchanges behind an immutable chapter."""

import re
from collections import deque

from ..utils.text_processor import TextProcessor
from .branch_context import _document_receipt

# Keep declarative turns and edits in mixed question/update messages. Notes have
# their own event kind and never enter the exchange eviction queue.
_EDIT = re.compile(r"更正一下|纠正一下|口误|(?:说|写|记)错了|改为|改成|应为|(?:更正|修订|订正)[：:]")
_EARLIER_ANSWERS = re.compile(
    r"上上(?:轮|次)|第[一二三四五六七八九十百0-9]+(?:问|轮)|"
    r"最(?:早|初).{0,8}(?:回答|问过|说过)|(?:以前|此前|之前).{0,8}(?:答过|回答|回复)|"
    r"(?:回顾|汇总|列出|比较).{0,10}(?:问答|回答|对话|回复)"
)


def is_ordinary_question(text):
    return text.rstrip().endswith(("?", "？")) and not _EDIT.search(text)


def chapter_history(chapter, events, system, question, budget, *, qa_window=None):
    if qa_window is not None and (type(qa_window) is not int or qa_window < 0):
        raise ValueError("qa_window must be a nonnegative integer or None")
    compact = qa_window is not None
    keep_all = bool(_EARLIER_ANSWERS.search(question))

    original = None
    exchanges = deque()
    messages = []
    size = sum(len(item["content"]) for item in system) + len(question)
    for event in events:
        kind, user, answer = event["kind"], event["user"], event["assistant"]
        if kind == "source" or (kind == "turn" and _document_receipt(user)):
            for number, start, end in TextProcessor.chapter_spans(user):
                if number == chapter:
                    text = user[start:end].strip()
                    if original is not None and original != text:
                        return None
                    original = text
            continue
        if kind == "note":
            messages.append([{"role": "user", "content": user}])
            size += len(user)
        elif kind == "turn" and isinstance(answer, str):
            if not compact or is_ordinary_question(user):
                exchanges.append(len(messages))
            messages.append([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
            size += len(user) + len(answer)
        else:
            # An imported partial exchange has no reliable pair boundary.
            return None
    if not original:
        return None
    size += len(original)
    while exchanges and (size > budget or (compact and not keep_all and len(exchanges) > qa_window)):
        index = exchanges.popleft()
        size -= sum(len(m["content"]) for m in messages[index])
        messages[index] = []
    if size > budget:
        return None
    return {
        "messages": [
            *system,
            {"role": "user", "content": original},
            *(m for group in messages for m in group),
            {"role": "user", "content": question},
        ],
        "chars": size,
        "wire_user": question,
        "checkpoint": "",
        "recalled": [],
        "cache_breakpoints": [len(system)],
        "cache_evidence_prefix_only": True,
    }
