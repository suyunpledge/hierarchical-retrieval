"""R3 paired calls via the production topic proxy, with identical-input cache-key replay."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import requests
from budget import Budget
from diverse_questions import ROOT, chapters, fingerprint
from fixtures import load_provider, record
from run import MODEL

RESULT = ROOT / "results/20261127-final-comparison"
HEADER = "读书会校样。按本轮所问的版本，依顺序简短答出两项；保留原文与排演工作稿的区别。"
ARMS = ("S0", "BT", "B0", "repeat")
GPT_LIMIT = 6.7


def load_bank(folder, book):
    protocol = json.loads((RESULT / "protocol.json").read_bytes())
    for name in ("selection.json", "fixtures.lock", "audits.json"):
        if fingerprint(folder / name) != protocol["sha256"][name]:
            raise ValueError("Unregistered question bank")
    s = json.loads((folder / "selection.json").read_bytes())
    lock = json.loads((folder / "fixtures.lock").read_bytes())
    if hashlib.sha256(book.read_text("utf-8-sig")[:2_000_000].encode()).hexdigest() != s["corpus_sha256"]:
        raise ValueError("Wrong corpus")
    cases = {}
    for ch in s["chapters"]:
        p = folder / "fixtures" / f"{ch}.json"
        if fingerprint(p) != lock[str(ch)]:
            raise ValueError("Changed frozen question")
        cases[ch] = json.loads(p.read_bytes())
    return s, cases


def edit(ch, f, revision):
    fields = "；".join(f"{k}改为“{v}”" for k, v in zip(f["slots"], f["edits"][revision]))
    return f"关于第{ch}章的排演工作稿，第{revision + 1}次更正。问题“{f['q']}”涉及的两项分别更正。{fields}。小说原文另存。"


def dialogue(ids, cases):
    for block in range(6):
        for side, ch in enumerate(ids):
            f = cases[ch]
            if block in (1, 3):
                # A note may be filed on the inactive topic without moving the cursor.
                yield "note", ch, edit(ch, f, 1 if block == 1 else 2), None, None
            # Each topic appears as a pair. An unnamed continuation uses the visible active topic.
            for offset in (0, 1):
                named = not offset or block % 2 == 1
                v = (0, 1, 1, 2, 2, 2)[block]
                first_draft = block in (1, 3, 5)
                draft = (
                    first_draft if not offset else (not first_draft if block in (2, 3, 5) else first_draft)
                )
                phrase = f["alt" if (block + side + offset) % 2 else "q"]
                lead = f"第{ch}章，" if named else "接着刚才那个场景，"
                if not offset:
                    ask = ("按工作稿最后一次更正，" if draft else "按未改动的原文，") + phrase
                    form = "更正后直接问" if draft else "点章点版"
                else:
                    ask = ("换到另一版，" if draft != first_draft else "沿用上一问选的版本，") + phrase
                    form = "点章承接" if named else "无章号承接"
                yield "ask", ch, lead + ask.rstrip("。 .？?") + "？", v if draft else None, form


def official_provider():
    provider = load_provider("gpt")
    if provider["base"].rstrip("/") not in (
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/chat/completions",
        "https://api.openai.com/v1/responses",
    ):
        raise ValueError("Expected the official GPT endpoint")
    return provider


def post_gpt(provider, payload):
    return requests.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer " + provider["key"]},
        timeout=90,
    )


def billed(send, payload, budget):
    key = budget.reserve(payload)
    r = send(payload)
    budget.settle(key, r)
    return r


def usage(data):
    u = data["usage"]
    d = u["prompt_tokens_details"]
    n = u["prompt_tokens"]
    read = d["cached_tokens"]
    write = d["cache_write_tokens"]
    assert data["model"].split("-20")[0] == MODEL and 0 <= read + write <= n
    assert not d.get("cache_write_tokens_details", {}).get("ttl_12h_tokens", 0)
    return dict(
        id=data["id"],
        input=n,
        read=read,
        write=write,
        output=u["completion_tokens"],
        units=20 * (n - read - write) + read + 25 * write,
        finish=data["choices"][0]["finish_reason"],
    )


def ask_payload(provider, body, budget):
    for attempt in range(4):
        try:
            r = billed(lambda p: post_gpt(provider, p), body, budget)
            if r.status_code not in (429, 502, 503, 504):
                r.raise_for_status()
                return r.json()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 3:
                raise
        time.sleep(1 + 2**attempt)
    raise RuntimeError("Upstream unavailable")


def worker(folder, book, index, arm):
    quota = json.loads((folder / "quota-check.json").read_bytes())
    if not quota.get("both_calibrated") or quota.get("primary_balance_covers_cny", 0) < 7.2:
        raise ValueError("Missing R3 scoring preflight")
    selection, cases = load_bank(folder, book)
    ids = selection["chapters"][2 * index : 2 * index + 2]
    noise = [selection["distractors"][str(ch)] for ch in ids]
    docs = chapters(book, [*ids, *noise])
    provider = official_provider()
    budget = Budget(folder / "gpt.sqlite", GPT_LIMIT)
    digest = hashlib.sha256()
    for f in [ROOT / "api_server.py", *sorted((ROOT / "hierarchical_retrieval").rglob("*.py"))]:
        digest.update(f.relative_to(ROOT).as_posix().encode() + bytes.fromhex(fingerprint(f)))
    budget.pin("production", digest.hexdigest())
    budget.pin("protocol", fingerprint(RESULT / "protocol.json"))
    path = folder / f"{arm}-{index:02}"
    path.mkdir(exist_ok=True)
    trace = path / "trace.json"
    seed = folder / f"seed-{index:02}"
    if not trace.exists():
        if arm not in ("seed", "repeat") and not (path / "db").exists():
            shutil.copytree(seed / "db", path / "db")
        record(dict(salt=uuid.uuid4().hex, rows=[]), trace)
    state = json.loads(trace.read_bytes())

    def add(row):
        state["rows"].append(row)
        record(state, trace)

    if arm == "repeat":
        primary = json.loads((folder / f"BT-{index:02}" / "trace.json").read_bytes())
        for row in primary["rows"][len(state["rows"]) :]:
            r = dict(row)
            if row["usage"]:
                body = json.loads((folder / f"BT-{index:02}" / f"wire-{row['id']}.json").read_bytes())
                assert row["digest"] == hash_payload(body)
                body["prompt_cache_key"] = hashlib.sha256(
                    (state["salt"] + str(row["chapter"])).encode()
                ).hexdigest()
                assert body["prompt_cache_key"] != row["cache_key"] and hash_payload(body) == row["digest"]
                data = ask_payload(provider, body, budget)
                r.update(
                    answer=data["choices"][0]["message"]["content"],
                    usage=usage(data),
                    cache_key=body["prompt_cache_key"],
                )
            add(r)
        return
    os.environ.update(
        HR_STORAGE_ROOT=str(path / "db"),
        HR_CHAT_BACKEND="https://api.openai.com/v1",
        HR_CHAT_BACKEND_OPENAI="1",
        HR_OPENAI_API_KEY=provider["key"],
        HR_OPENAI_CACHE_BREAKPOINTS="1",
        HR_CONTEXT_CHAR_BUDGET="5500",
        HR_WHOLE_CHAPTER_HISTORY="1" if arm == "BT" else "0",
        HR_CHAPTER_QA_WINDOW="2",
        HR_CHAPTER_EDITION_ANCHOR="0",
    )
    sys.path.insert(0, str(ROOT))
    import api_server
    from fastapi.testclient import TestClient

    if arm == "seed":
        events = [
            ("read", ch, "以下为读书会原文，读完仅回答已读。\n" + docs[t], None, None)
            for targets in (ids, noise)
            for ch, t in zip(ids, targets)
        ]
        events += [("note", ch, edit(ch, cases[ch], 0), None, None) for ch in ids]
    else:
        events = list(dialogue(ids, cases))
    notes = {ch: ([edit(ch, cases[ch], 0)] if arm != "seed" else []) for ch in ids}
    history = {ch: [] for ch in ids}
    active = None
    real_send = requests.post
    with TestClient(api_server.app) as client:
        if api_server.API_TOKEN:
            client.headers["Authorization"] = "Bearer " + api_server.API_TOKEN
        for i, (action, ch, q, v, form) in enumerate(events):
            cached = state["rows"][i] if i < len(state["rows"]) else None
            named = form != "无章号承接"
            if action == "ask" and not named and active != ch:
                raise AssertionError("Unnamed question has no visible topic")
            if cached is None:
                wirelog = {}

                def observe(url, *, json, **kwargs):
                    assert url == "https://api.openai.com/v1/chat/completions" and json["model"] == MODEL
                    body = dict(json)
                    if arm == "B0" and action == "ask":
                        chosen = ch if named else active
                        body["messages"] = [
                            body["messages"][0],
                            dict(role="user", content="\n\n".join([docs[chosen], *notes[chosen], q])),
                        ]
                        body.pop("prompt_cache_options", None)
                    body["prompt_cache_key"] = hashlib.sha256((state["salt"] + str(ch)).encode()).hexdigest()
                    bp = any(
                        isinstance(m["content"], list)
                        and any("prompt_cache_breakpoint" in z for z in m["content"])
                        for m in body["messages"]
                    )
                    if (
                        arm == "BT"
                        and bp
                        and action == "ask"
                        and isinstance(body["messages"][1]["content"], list)
                        and body["messages"][1]["content"][0]["text"] == docs[ch]
                    ):
                        msgs = body["messages"]
                        assert msgs[-1]["content"] == q
                        turns = history[ch]
                        last_two = [i for i, event in enumerate(turns) if event[0] == "ask"][-2:]
                        expected = [
                            event for i, event in enumerate(turns) if event[0] == "note" or i in last_two
                        ]
                        serialized = [dict(role="user", content=notes[ch][0])]
                        for kind, query, ans in expected:
                            serialized.append(dict(role="user", content=query))
                            if kind == "ask":
                                serialized.append(dict(role="assistant", content=ans))
                        assert msgs[2:-1] == serialized
                        wirelog["chapter_prefix"] = True
                    if arm == "BT" and action == "ask":
                        record(body, path / f"wire-{i}.json")
                    wirelog.update(digest=hash_payload(body), cache_key=body["prompt_cache_key"])
                    r = billed(lambda p: real_send(url, json=p, **kwargs), body, budget)
                    if r.ok:
                        wirelog["usage"] = usage(r.json())
                    return r

                meta = dict(
                    workspace_id="reading",
                    request_id=f"{'seed' if arm == 'seed' else 'q'}-{i}",
                    action="note" if action == "note" else "chat",
                )
                if named:
                    meta["topic"] = f"校样{ch}"
                body = dict(
                    model=MODEL,
                    max_tokens=160,
                    messages=[dict(role="system", content=HEADER), dict(role="user", content=q)],
                    topic_workspace=meta,
                )
                for attempt in range(4):
                    with patch.object(api_server.requests, "post", observe):
                        r = client.post("/v1/chat/completions", json=body)
                    if r.status_code not in (429, 502, 503, 504):
                        break
                    time.sleep(2**attempt)
                r.raise_for_status()
                obj = r.json()
                route = obj["topic_workspace"]
                answer = obj["choices"][0]["message"]["content"]
                if route.get("prefix_policy") == "whole_chapter_history":
                    assert wirelog.get("chapter_prefix")
                if (
                    action in ("ask", "read")
                    and not wirelog.get("usage")
                    and route.get("status") != "clarify"
                ):
                    raise ValueError("Missing provider receipt")
                cached = dict(
                    id=i,
                    chapter=ch,
                    action=action,
                    question=q,
                    version=v,
                    form=form,
                    answer=answer,
                    usage=None,
                    topic=route.get("topic"),
                    status=route.get("status"),
                    **{k: v for k, v in wirelog.items() if k != "usage"},
                )
                cached["usage"] = wirelog.get("usage")
                add(cached)
            if action == "note":
                notes[ch].append(q)
                history[ch].append(("note", q, None))
            if action == "ask":
                active = ch
                history[ch].append(("ask", q, cached["answer"]))


def hash_payload(body):
    material = {k: v for k, v in body.items() if k != "prompt_cache_key"}
    return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run_all(folder, book, nworkers):
    load_bank(folder, book)
    if not (folder / "quota-check.json").exists():
        from final_score import preflight

        record(preflight(folder), folder / "quota-check.json")

    def run(job):
        i, arm = job
        trace = folder / f"{arm}-{i:02}" / "trace.json"
        if trace.exists():
            rows = json.loads(trace.read_bytes())["rows"]
            if len(rows) == (6 if arm == "seed" else 28):
                return
        logfile = folder / f"{arm}-{i:02}.log"
        for _ in range(3):
            with logfile.open("a", encoding="utf8") as log:
                res = subprocess.run(
                    [sys.executable, __file__, str(folder), str(book), "--pair", str(i), "--arm", arm],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONIOENCODING="utf8"),
                )
            if res.returncode == 0:
                return
            if "budget reached" in logfile.read_text("utf8", errors="replace")[-3500:].lower():
                break
        raise RuntimeError("See " + logfile.name)

    with ThreadPoolExecutor(nworkers) as pool:
        for phase in (
            [(i, "seed") for i in range(20)],
            [(i, ("S0", "BT", "B0")[(i + j) % 3]) for i in range(20) for j in range(3)],
            [(i, "repeat") for i in range(20)],
        ):
            list(pool.map(run, phase))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("book", type=Path)
    p.add_argument("--pair", type=int)
    p.add_argument("--arm", choices=("seed", *ARMS))
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    if a.pair is None:
        run_all(a.folder, a.book, a.workers)
    else:
        worker(a.folder, a.book, a.pair, a.arm)
