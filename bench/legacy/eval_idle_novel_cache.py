"""Live, paired idle-topic tests using official GPT-6.2-sol usage and an isolated SQLite store."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from eval_e2e import load_provider
from fastapi.testclient import TestClient

# Each tuple fixes the subject, the original witnesses and both fictional rehearsal revisions.
CASES = {
    438: (
        472,
        "齐夏让周六和罗十一分别进哪个房间，按这两人顺序报房名或房号",
        (("二号", "2号"), ("四号", "4号")),
        ("兰亭", "荷亭"),
        ("竹亭", "松亭"),
    ),
    566: (
        693,
        "许流年说最初的计划叫什么，是谁杀死了最初的裁判",
        (("女娲游戏",), ("天龙",)),
        ("青穗计划", "乌雀"),
        ("白塔计划", "灰鹭"),
    ),
    344: (
        348,
        "齐夏捧着的小石头是什么颜色，形状像什么",
        (("翠绿", "绿色"), ("牙齿", "牙")),
        ("琥珀色", "杏核"),
        ("乳白色", "水滴"),
    ),
    345: (
        352,
        "云瑶回到天堂口时，门口的两个守卫是谁",
        (("老吕",), ("小眼镜",)),
        ("孟秋", "乔冬"),
        ("陆青", "安禾"),
    ),
    373: (
        604,
        "林檎和章晨泽各自手中的签名是什么，按这两人顺序报告",
        (("春暖花开日",), ("暴雨滚瓢泼",)),
        ("青藤过桥", "银杏入河"),
        ("晚灯映水", "初雪照城"),
    ),
    638: (
        605,
        "第二轮地猴打头发的是什么牌，齐夏拿到的牌面写了什么",
        (("暗牌",), ("处暑",)),
        ("铜牌", "白露"),
        ("银牌", "谷雨"),
    ),
}
SPLITS = {"dev": (344, 345), "holdout": (373, 638), "confirmation": (438, 566)}
MODEL = "gpt-6.2-sol"


def score(ch, version, ans):
    _, _, orig, former, new = CASES[ch]
    needles = orig if version == "原文" else [(k,) for k in new]
    forbidden = (*former, *new) if version == "原文" else (*former, *(w[0] for w in orig))
    positions = [min((ans.index(w) for w in alts if w in ans), default=-1) for alts in needles]
    ok = all(offset >= 0 for offset in positions)
    if ch in (373, 438):  # These questions explicitly request the two people in order.
        ok = ok and positions[0] < positions[1]
    stale = any(w in ans for w in forbidden)
    ok = ok and not stale
    return ok, stale


def chapters(book, ids):
    corpus = book.read_text(encoding="utf-8-sig")[:2_000_000]
    headers = list(re.finditer(r"(?m)^[　 \t]*第\s*(\d+)\s*章[^\n]*", corpus))
    result = {
        int(m[1]): corpus[m.start() : n.start()].strip()
        for m, n in zip(headers, headers[1:])
        if int(m[1]) in ids
    }
    assert result.keys() == set(ids)
    return result


def record(state, path):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf8")
    temp.replace(path)


def run(args):
    sys.path.insert(0, str(args.code.resolve()))
    db = args.folder / "db"
    args.folder.mkdir(parents=True, exist_ok=True)
    prov = load_provider("gpt")
    assert prov["base"].rstrip("/") == "https://api.openai.com/v1"
    os.environ.update(
        HR_STORAGE_ROOT=str(db),
        HR_OPENAI_API_KEY=prov["key"],
        HR_CHAT_BACKEND="https://api.openai.com/v1",
        HR_CHAT_BACKEND_OPENAI="1",
        HR_OPENAI_CACHE_BREAKPOINTS="1",
        HR_CONTEXT_CHAR_BUDGET="5500",
    )
    import api_server
    from hierarchical_retrieval.core import branch_context

    assert Path(api_server.__file__).resolve().parent == args.code.resolve()
    ids = SPLITS[args.split]
    docs = chapters(args.book, [*ids, *(CASES[ch][0] for ch in ids)])
    for ch in ids:
        assert all(any(w in docs[ch] for w in alternatives) for alternatives in CASES[ch][2])
        assert all(w not in docs[ch] for w in (*CASES[ch][3], *CASES[ch][4]))
    progress = args.folder / "trace.json"
    if progress.exists():
        state = json.loads(progress.read_text("utf8"))
        assert state["split"] == args.split
    else:
        assert args.phase == "seed"
        state = dict(
            split=args.split,
            mode=args.mode,
            nonce=uuid.uuid4().hex[:18],
            cold_nonce=uuid.uuid4().hex[:18],
            chapters={str(ch): hashlib.sha256(doc.encode()).hexdigest() for ch, doc in docs.items()},
            turns=[],
        )
    nonce = state["cold_nonce"] if args.cold and args.phase == "probe" else state["nonce"]
    sysmsg = "读书档案 " + nonce + "。请按指定章节和版本作答，只答所问的两项。原文和排演更正分开。"

    def reading(ch):
        return "以下为读书会原文，读完仅回答已读。\n" + docs[ch]

    upstream = api_server.requests.post
    raw_usage = {}

    def observed(url, *, json, **kw):
        assert url == "https://api.openai.com/v1/chat/completions"
        assert json["model"] == MODEL
        response = upstream(url, json=json, **kw)
        response.raise_for_status()
        data = response.json()
        assert data["model"] == MODEL or data["model"].startswith(MODEL + "-")
        assert data.get("service_tier", "default") in ("default", "standard")
        usage = data["usage"]
        detail = usage["prompt_tokens_details"]
        inputs, reads, writes = usage["prompt_tokens"], detail["cached_tokens"], detail["cache_write_tokens"]
        ttl = detail.get("cache_write_tokens_details", {})
        assert ttl.get("ttl_12h_tokens", 0) == 0 and ttl.get("ttl_30m_tokens", writes) == writes
        normal = inputs - reads - writes
        assert normal >= 0
        msgs = [
            dict(
                role=m["role"],
                content=m["content"]
                if isinstance(m["content"], str)
                else "".join(p["text"] for p in m["content"]),
            )
            for m in json["messages"]
        ]
        opening = reading(ch)
        kept = 0
        for m in msgs[2:]:
            if m["role"] != "user" or not opening.startswith(m["content"], kept):
                break
            kept += len(m["content"])
        raw_usage.update(
            input=inputs,
            read=reads,
            write_30m=writes,
            ordinary=normal,
            uncached=inputs - reads,
            output=usage["completion_tokens"],
            input_usd=(normal * 2 + reads * 0.1 + writes * 2.5) / 1e6,
            opening_chars=kept,
            chars=sum(len(m["content"]) for m in msgs),
            model=data["model"],
            tier=data.get("service_tier"),
            response_id=data["id"],
            finish_reason=data["choices"][0]["finish_reason"],
        )
        return response

    def call(ch_num, text, label, *, note=False, version=None):
        nonlocal ch
        ch = ch_num
        if any(r["label"] == label and r["chapter"] == ch for r in state["turns"]):
            return
        prior = [r for r in state["turns"] if r["chapter"] == ch and r["usage"]]
        gap = time.time() - prior[-1]["ended"] if prior else None
        raw_usage.clear()
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "max_tokens": 360,
                "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": text}],
                "topic_workspace": {
                    "workspace_id": "reading",
                    "topic": f"读书线{ch}",
                    "action": "note" if note else "chat",
                    "request_id": f"{ch}:{label}",
                },
            },
        )
        r.raise_for_status()
        d = r.json()
        assert note == (not raw_usage)
        ans = d["choices"][0]["message"]["content"]
        ok, stale = None, None
        if version:
            ok, stale = score(ch, version, ans)
        row = dict(
            label=label,
            chapter=ch,
            phase=args.phase,
            ended=time.time(),
            gap_seconds=gap,
            version=version,
            answer=ans if version else None,
            correct=ok,
            wrong_version=stale,
            usage=dict(raw_usage),
            workspace=d.get("topic_workspace", {}),
        )
        state["turns"].append(row)
        record(state, progress)
        print(args.folder.name, ch, label, raw_usage.get("input"), raw_usage.get("read"), ok, flush=True)

    ch = ids[0]
    original_build = branch_context.build_branch_context

    def policy(**kw):
        if args.phase == "probe" and args.mode.startswith("blocks"):
            kw["prefix_blocks"] = int(args.mode[-1])
        return original_build(**kw)

    with (
        TestClient(api_server.app) as c,
        patch.object(api_server.requests, "post", observed),
        patch.object(branch_context, "build_branch_context", policy),
    ):
        if api_server.API_TOKEN:
            c.headers["Authorization"] = "Bearer " + api_server.API_TOKEN
        if args.phase == "seed":
            for ch in ids:
                call(ch, reading(ch), "首次阅读")
            for ch in ids:
                call(ch, reading(CASES[ch][0]), "阅读同线下一章")
            for ch in ids:
                _, subject, _, first, last = CASES[ch]
                for i, pair in enumerate((first, last), 1):
                    call(
                        ch,
                        f"第{ch}章排演版第{i}次更正。关于“{subject}”，两项依次改为“{pair[0]}”和“{pair[1]}”。保留小说原文。",
                        f"勘误{i}",
                        note=True,
                    )
        else:
            assert len([r for r in state["turns"] if r["phase"] == "seed"]) == 8
            for i, (ch, version) in enumerate(
                [
                    (ids[0], "排演"),
                    (ids[1], "原文"),
                    (ids[0], "原文"),
                    (ids[1], "排演"),
                    (ids[0], "排演"),
                    (ids[1], "原文"),
                ]
            ):
                subject = CASES[ch][1]
                q = (
                    f"回到第{ch}章。只按"
                    + ("工作稿最后一次更正" if version == "排演" else "小说原始正文")
                    + f"回答：{subject}？"
                )
                call(ch, q, f"回问{i + 1}", version=version)
            state["probe_cold_namespace"] = args.cold
            record(state, progress)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["seed", "probe"])
    p.add_argument("split", choices=SPLITS)
    p.add_argument("mode", choices=["before", "adaptive", "blocks0", "blocks1", "blocks2", "blocks3"])
    p.add_argument("code", type=Path)
    p.add_argument("book", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("--cold", action="store_true")
    run(p.parse_args())
