"""Run paired cold topic returns on private, cloned archives through official GPT-6.2-sol."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from json import dumps
from pathlib import Path
from unittest.mock import patch

import requests
from eval_idle_novel_cache import MODEL, chapters, load_provider, record
from fastapi.testclient import TestClient

# Subject, alternate phrasing, original field variants, working-copy revisions, and distractor chapter.
CASES = {
    362: (
        "苏闪这支签上写的五个字是什么，签下箭头朝哪边",
        "苏闪拿到的签文是哪五个字，箭头指向哪边",
        (("万民齐赈灾",), ("左",)),
        (("清泉映明灯", "北方"), ("暮雨过山城", "东南"), ("孤舟归柳岸", "东北"), ("晨星照野渡", "南方")),
        512,
    ),
    658: (
        "最先站起来看牌的是谁，他看的是谁面前的牌",
        "谁第一个起身查看牌面，被他查看的是谁的牌",
        (("陈俊南",), ("地猴",)),
        (("季禾", "铜雀"), ("苏桐", "金雕"), ("郑琴", "黑雁"), ("叶笙", "青鹿")),
        711,
    ),
    197: (
        "谁和云瑶组队，路过哪个生肖的游戏场地",
        "云瑶当天的搭档是谁，经过的是哪位生肖的场地",
        (("老吕",), ("人兔",)),
        (("朱青", "沙雀"), ("方泉", "白鹿"), ("郑松", "山猫"), ("唐泉", "灰狐")),
        546,
    ),
    215: (
        "谁扶着方向盘驾车，车里与他同行的人是谁",
        "谁在开车，和他一起坐车的是谁",
        (("楚天秋",), ("云瑶",)),
        (("许桐", "沈晴"), ("秦桥", "杜江"), ("陈辰", "陆辰"), ("任泉", "丁雨")),
        682,
    ),
    323: (
        "钟震脱下了哪件衣服，叠好放在哪里",
        "钟震叠好什么衣物，又放在什么位置",
        (("西装", "西服"), ("座椅", "椅子", "座位", "椅面", "椅背")),
        (("蓝衫", "竹架"), ("白袍", "窗台"), ("黑褂", "矮柜"), ("紫衫", "木凳")),
        401,
    ),
    617: (
        "地虎把哪件衣物铺在了哪里",
        "地虎脱下什么铺开，铺在什么位置",
        (("西装", "西服"), ("地上", "地面", "地板")),
        (("灰袍", "石阶"), ("金褂", "茶几"), ("布衫", "草垫"), ("青袍", "藤床")),
        579,
    ),
    266: (
        "齐夏扔掉了手里的什么，又坐到了哪里",
        "齐夏丢开什么东西，接着坐在什么位置",
        (("扇子", "折扇", "扇"), ("桌面", "桌上", "桌子")),
        (("木牌", "石凳"), ("铜铃", "矮墙"), ("布袋", "木箱"), ("竹筹", "窗台")),
        232,
    ),
    476: (
        "按许流年的计划，要向谁传递消息，消息中的厉害人物叫什么",
        "许流年要让谁听到消息，消息里那位厉害人物姓甚名谁",
        (("齐夏",), ("文巧云",)),
        (("沈棠", "白舟"), ("陆川", "程秋"), ("林溪", "孟桥"), ("洛青", "温宁")),
        404,
    ),
}
SPLITS = {
    "development": (323, 617),
    "validation": (266, 476),
    "heldout": (197, 215),
    "replication": (362, 658),
}
ORDER = (0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1)
# Issue a note on the other topic where possible. Notes do not count as model questions.
CHANGES = {5: (0, 1), 8: (1, 1), 14: (1, 2), 16: (0, 2), 20: (0, 3), 23: (1, 3)}
LATEST = {1, 2, 5, 6, 7, 8, 11, 12, 14, 16, 18, 19, 20, 22, 23, 24}
SYSTEM = "。请按指定章节和版本作答。只答所问的两项，顺序按问题，原文和工作稿分别保存。"


def grade(answer, chapter, version, cases=CASES):
    original, revisions = cases[chapter][2:4]
    if chapter == 362:
        for direction in ("南", "北"):
            for wording in (
                "朝" + direction,
                "向" + direction,
                "往" + direction,
                direction + "边",
                "正" + direction,
            ):
                answer = answer.replace(wording, direction + "方")
    options = original if version is None else [(item,) for item in revisions[version]]
    forbidden = [t for i, pair in enumerate(revisions) if i != version for t in pair]
    if version is not None:
        forbidden += [v for group in original for v in group]
    stale = any(v in answer for v in forbidden)
    offsets = [min((answer.index(t) for t in alts if t in answer), default=-1) for alts in options]
    return 0 <= offsets[0] < offsets[1] and not stale, stale


def plan(ids, docs, limit, seed, cases=CASES, order=ORDER):
    versions = [0, 0]

    def note(branch, rev, label):
        ch = ids[branch]
        pair = cases[ch][3][rev]
        return (
            label,
            ch,
            "note",
            f"关于第{ch}章的排演工作稿，第{rev + 1}次更正：“{cases[ch][0]}”这两项依次改成“{pair[0]}”和“{pair[1]}”。保留小说原始正文。",
            None,
        )

    if seed:
        for stage in ("opening", "distractor"):
            for ch in ids:
                doc = docs[ch if stage == "opening" else cases[ch][-1]]
                yield stage + str(ch), ch, "read", "以下为读书会原文，读完仅回答已读。\n" + doc, None
        for b in range(2):
            yield note(b, 0, "initial" + str(b))
        return
    for i, b in enumerate(order[:limit], 1):
        if i in CHANGES:
            other, rev = CHANGES[i]
            versions[other] = rev
            yield note(other, rev, f"rev{i}")
        ch = ids[b]
        current = i in LATEST
        phrasing = cases[ch][(i // 3) % 2]
        target = (
            ("按排演工作稿最后一次更正", "只取工作稿现在生效的版本")
            if current
            else ("按小说原始正文", "切回未改动的小说原文")
        )
        lead = ("接回刚才那条阅读线", "换回这一章", "再核一次", "接着问刚才这两项")[i % 4]
        yield (
            f"q{i}",
            ch,
            "ask",
            f"{lead}，第{ch}章。{target[i % 2]}，{phrasing}？",
            versions[b] if current else None,
        )


def run(
    args,
    *,
    cases=CASES,
    splits=SPLITS,
    order=ORDER,
    blind=False,
    grader=grade,
    witnesses=None,
    planner=plan,
    prompt_builder=None,
    billing=None,
    reference=None,
):
    folder = args.traces / args.name
    folder.mkdir(parents=True, exist_ok=True)
    seed = args.mode == "seed"
    ids = splits[args.split]
    docs = chapters(args.book, [*ids, *(cases[ch][-1] for ch in ids)])
    for ch in ids:
        assert all(any(v in docs[ch] for v in group) for group in cases[ch][2])
        for pair in cases[ch][3]:
            assert all(v.lstrip("-").isdecimal() or v not in docs[ch] for v in pair)
    path = folder / "trace.json"
    if not path.exists():
        if not seed:
            source = args.traces / ("seed-" + args.split)
            assert len(json.loads((source / "trace.json").read_text("utf8"))["rows"]) == 6
            shutil.copytree(source / "db", folder / "db")
        record(
            dict(
                nonce=uuid.uuid4().hex,
                mode=args.mode,
                split=args.split,
                limit=args.limit,
                chapters={ch: hashlib.sha256(v.encode()).hexdigest() for ch, v in docs.items()},
                rows=[],
            ),
            path,
        )
    state = json.loads(path.read_text("utf8"))
    assert (state["mode"], state["split"], state["limit"]) == (args.mode, args.split, args.limit)
    prov = load_provider("gpt")
    # Forge profiles may store either the API root or a complete OpenAI endpoint.
    base = prov["base"].rstrip("/")
    assert base in {
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/responses",
        "https://api.openai.com/v1/chat/completions",
    }
    prov = dict(prov, base="https://api.openai.com/v1")
    os.environ.update(
        HR_STORAGE_ROOT=str(folder / "db"),
        HR_OPENAI_API_KEY=prov["key"],
        HR_CHAT_BACKEND=prov["base"],
        HR_CHAT_BACKEND_OPENAI="1",
        HR_OPENAI_CACHE_BREAKPOINTS="1",
        HR_CONTEXT_CHAR_BUDGET="5500",
    )
    sys.path.insert(0, str(args.code.resolve()))
    import api_server
    from hierarchical_retrieval.core import branch_context

    assert Path(api_server.__file__).resolve().parent == args.code.resolve()
    from hierarchical_retrieval.core import topic_workspace

    from hierarchical_retrieval import topic_chat

    for module in (topic_chat, topic_workspace, branch_context):
        assert Path(module.__file__).resolve().is_relative_to(args.code.resolve()), module.__file__
    original_build = branch_context.build_branch_context

    reference_builder = None
    if reference:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "hierarchical_retrieval.core._frozen_context", reference
        )
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        reference_builder = frozen.build_branch_context

    def build(**kw):
        if args.mode in ("two", "zero"):
            kw["prefix_blocks"] = {"two": 2, "zero": 0}[args.mode]
            kw.pop("tail_floor", None)  # Fixed arms never start a new adaptive suffix.
        result = original_build(**kw)
        if reference_builder and result != reference_builder(**kw):
            raise ValueError("Assembled context differs from the frozen comparison arm")
        return result

    send = api_server.requests.post
    receipt = {}

    events = []
    audit_chapter = args.mode in ("BH", "BT", "BD", "BZ")
    if prompt_builder or audit_chapter:
        seed_rows = {
            row["id"]: row
            for row in __import__("json").loads(
                (args.traces / ("seed-" + args.split) / "trace.json").read_text("utf8")
            )["rows"]
        }
        for key, c, act, content, _ in planner(ids, docs, args.limit, True, cases, order):
            events.append((key, c, act, content, seed_rows[key]["answer"]))
        history = {row["id"]: row for row in state["rows"]}
        for key, c, act, content, _ in planner(ids, docs, args.limit, False, cases, order):
            if key not in history:
                break
            events.append((key, c, act, content, history[key]["answer"]))

    def observe(url, *, json, **kw):
        assert url == "https://api.openai.com/v1/chat/completions" and json["model"] == MODEL
        if prompt_builder and action == "ask":
            json = dict(
                json, messages=prompt_builder(args.mode, docs, ids, events, ch, q, json["messages"][0])
            )
        if prompt_builder and action == "ask" and args.mode in ("BX", "BG"):
            json["prompt_cache_options"] = {"mode": "explicit"}
        if getattr(args, "capture_layout", False) and action == "ask":
            receipt["wire_parts"] = [
                hashlib.sha256(dumps(message, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                for message in json["messages"]
            ]
            receipt["cache_mode"] = json.get("prompt_cache_options", {}).get("mode", "auto")
        if audit_chapter and action == "ask":
            from chapter_study import audit_layout

            if args.split.startswith("tail-"):
                from history_tail import audit_tail

                receipt.update(audit_tail(json, docs[ch], ch, events, q, state["nonce"], args.mode))
            else:
                audit_layout(json, docs[ch], ch, events, q, state["nonce"])
            receipt["topic_key"] = json["prompt_cache_key"]
        reservation = billing.reserve(json) if billing else None
        try:
            r = send(url, json=json, **kw)
        except requests.RequestException as error:
            print("upstream connection error", type(error).__name__, flush=True)
            raise
        if billing:
            billing.settle(reservation, r)
        if not r.ok:
            print("upstream HTTP", r.status_code, flush=True)
        if r.ok:
            obj = r.json()
            assert obj["model"].split("-20")[0] == MODEL and obj.get("service_tier", "default") in (
                "default",
                "standard",
            )
            u = obj["usage"]
            det = u["prompt_tokens_details"]
            n, c, w = u["prompt_tokens"], det["cached_tokens"], det["cache_write_tokens"]
            assert 0 <= c + w <= n and det.get("cache_write_tokens_details", {}).get("ttl_12h_tokens", 0) == 0
            if witnesses and action == "ask":
                messages = json["messages"]

                def content(m):
                    value = m["content"]
                    return value if isinstance(value, str) else "".join(part["text"] for part in value)

                receipt["witness_user_messages"] = [
                    i
                    for i, message in enumerate(messages)
                    if message["role"] == "user" and witnesses[ch] in content(message)
                ]
            receipt.update(
                response_id=obj["id"],
                input=n,
                read=c,
                write=w,
                ordinary=n - c - w,
                output=u["completion_tokens"],
                cost_units=20 * (n - c - w) + c + 25 * w,
                finish=obj["choices"][0]["finish_reason"],
            )
        return r

    with (
        TestClient(api_server.app) as client,
        patch.object(api_server.requests, "post", observe),
        patch.object(branch_context, "build_branch_context", build),
    ):
        if api_server.API_TOKEN:
            client.headers["Authorization"] = "Bearer " + api_server.API_TOKEN
        for label, ch, action, q, ver in planner(ids, docs, args.limit, seed, cases, order):
            if any(r["id"] == label for r in state["rows"]):
                continue
            if args.through and label.startswith(("q", "rev")) and int(label.lstrip("qrev")) > args.through:
                break
            receipt.clear()
            body = {
                "model": MODEL,
                "max_tokens": 400,
                "messages": [
                    {"role": "system", "content": "档案 " + state["nonce"] + SYSTEM},
                    {"role": "user", "content": q},
                ],
                "topic_workspace": {
                    "workspace_id": "reading",
                    "topic": f"校样{ch}",
                    "action": "note" if action == "note" else "chat",
                    "request_id": label,
                },
            }
            if args.mode == "planned" and action == "ask":
                number = int(label[1:]) - 1
                body["topic_workspace"]["remaining_topic_questions"] = order[number : args.limit].count(
                    ids.index(ch)
                )
            if blind:
                assert "remaining_topic_questions" not in body["topic_workspace"]
            for attempt in range(3):
                r = client.post("/v1/chat/completions", json=body)
                if r.status_code not in (502, 503, 504, 429) or attempt == 2:
                    break
                time.sleep(2**attempt)
            r.raise_for_status()
            obj = r.json()
            meta = obj["topic_workspace"]
            assert bool(receipt) == (action != "note")
            answer = obj["choices"][0]["message"]["content"]
            valid, stale = grader(answer, ch, ver, cases) if action == "ask" else (None, None)
            state["rows"].append(
                dict(
                    id=label,
                    chapter=ch,
                    action=action,
                    version=ver,
                    question=q if action != "read" else None,
                    answer=answer if action in ("read", "ask") else None,
                    correct=valid,
                    wrong_version=stale,
                    usage=dict(receipt),
                    opening=meta.get("opening_chars"),
                    policy=meta.get("prefix_policy"),
                    recalled=meta.get("recalled", []),
                )
            )
            if prompt_builder or audit_chapter:
                events.append((label, ch, action, q, answer))
            record(state, path)
            print(args.name, label, ch, receipt.get("read"), valid, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("code", type=Path)
    p.add_argument("book", type=Path)
    p.add_argument("traces", type=Path)
    p.add_argument("name")
    p.add_argument("split", choices=SPLITS)
    p.add_argument("mode", choices=["seed", "zero", "two", "before", "after", "planned"])
    p.add_argument("limit", type=int, choices=[12, 24])
    p.add_argument("--through", type=int)
    run(p.parse_args())
