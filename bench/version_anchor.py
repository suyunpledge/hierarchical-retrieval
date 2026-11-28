"""Paired, horizon-blind edition return study with independent checks of the outbound messages."""

import argparse
import hashlib
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

from run import ROOT, edition_plan

RESULTS = ROOT / "results/20261127-version-anchor"
# Both topics get a fresh explicit instruction before the next sustained chain.
EXPLICIT = {1, 2, 3, 4, 13, 16, 20}


def paired_chapters(split):
    side = {"anchor-dev": "dev", "anchor-test": "test"}[split]
    ids = json.loads((ROOT / "bench/anchor_split.json").read_bytes())[side]
    return {f"anchor-{side}-{i // 2:02}": ids[i : i + 2] for i in range(0, len(ids), 2)}


def verify_lock(folder):
    protocol = json.loads((RESULTS / "protocol.json").read_bytes())
    for path, key in (
        (folder / "fixtures.lock", "fixture_lock_sha256"),
        (ROOT / "bench/anchor_split.json", "manifest_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError("Fresh chapter manifest or locked question fixtures changed")


def anchor_plan(ids, docs, limit, seed, cases, order):
    previous = {}
    for label, ch, action, question, version in edition_plan(ids, docs, limit, seed, cases, order):
        if action == "ask":
            i = int(label[1:])
            latest = version is not None
            if i not in EXPLICIT:
                same = ("沿用上一次选的版本", "照上一轮选的那版", "还是刚才那一版")
                switch = ("换到上一次没选的那个版本", "切到另一版", "这次用和上一问相反的版本")
                wording = (same if previous[ch] == latest else switch)[i % 3]
                question = f"回到第{ch}章，{wording}。上次问的两项分别是什么？"
            previous[ch] = latest
        yield label, ch, action, question, version


def audit_anchor(payload, chapter, ch, events, question, nonce, mode):
    from eval_long_prefix_return import SYSTEM
    from hierarchical_retrieval.core.chapter_history import is_ordinary_question

    msgs = payload["messages"]
    assert msgs[0] == {"role": "system", "content": "档案 " + nonce + SYSTEM}
    assert msgs[1] == {
        "role": "user",
        "content": [{"type": "text", "text": chapter, "prompt_cache_breakpoint": {"mode": "explicit"}}],
    }
    assert payload["prompt_cache_key"] and payload["prompt_cache_options"] == {"mode": "explicit"}
    assert msgs[-1] == {"role": "user", "content": question}
    groups = []
    ask_indices = []
    ordinary = []
    edits = 0
    for turn, side, action, q, a in events:
        if side != ch or action not in ("ask", "note"):
            continue
        group = [{"role": "user", "content": q}]
        if action == "ask":
            ask_indices.append(len(groups))
            group.append({"role": "assistant", "content": a})
            if mode == "BH" or is_ordinary_question(q):
                ordinary.append(len(groups))
        else:
            edits += 1
        groups.append((turn, group))
    kept = {i: pair for i, (_, pair) in enumerate(groups)}
    budget = 5500 - len(msgs[0]["content"]) - len(chapter) - len(question)

    def size(selected):
        return sum(len(m["content"]) for group in selected.values() for m in group)

    queue = ordinary.copy()
    while queue and (size(kept) > budget or mode != "BH" and len(queue) > 2):
        del kept[queue.pop(0)]
    restored = {}
    if question.startswith("回到第") and mode != "BH":
        for i in reversed(ask_indices):
            label, pair = groups[i]
            explicit = int(label[1:]) in EXPLICIT
            if i not in kept:
                restored[i] = pair if explicit else pair[:1]
            if explicit:
                break
    need = bool(restored)
    if mode == "BA" and size(kept) + size(restored) <= budget:
        kept.update(restored)
    else:
        restored = {}
    expected = [m for i in sorted(kept) for m in kept[i]]
    assert expected == msgs[2:-1], "Unexpected actual chapter, correction or QA layout"
    return dict(
        needs_anchor=need,
        restored_turns=[groups[i][0] for i in sorted(restored)],
        restored_exchanges=sum(len(g) == 2 for g in restored.values()),
        restored_instructions=sum(len(g) == 1 for g in restored.values()),
        restored_chars=size(restored),
        verbatim_notes=edits,
    )


def score_study(folder, split, book):
    from score import analyze, decisions
    from tail_review import mimo_judge

    exposure = {}
    for p in folder.glob(f"BA-{split}-*-r1/trace.json"):
        state = json.loads(p.read_bytes())
        for row in state["rows"]:
            if row["action"] == "ask":
                exposure[(state["limit"], row["chapter"], row["id"])] = row["usage"]["needs_anchor"]
    strata = defaultdict(lambda: defaultdict(lambda: {"complete": 0, "asks": 0}))

    def observe(k, r, correct, *_):
        if r["question"].startswith("回到第"):
            need = exposure[k[1], r["chapter"], r["id"]]
            d = strata["-".join(map(str, k))]["missing" if need else "already_present"]
            d["asks"] += 1
            d["complete"] += int(correct)

    result = analyze(
        folder,
        split,
        book=book,
        judge_workers=8,
        on_scored=observe,
        review=partial(decisions, reviewer=mimo_judge, cache="mimo-decisions.sqlite"),
    )
    result["scorer"] = "frozen rule, grounding veto and blind MiMo v2.6 Flash"
    result["version_evidence"] = strata
    return result


if __name__ == "__main__":
    from study import launch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("book", type=Path)
    parser.add_argument("split", choices=("anchor-dev", "anchor-test"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    verify_lock(args.folder)
    if args.split == "anchor-test" and not (args.folder / "anchor-dev-aggregate.json").exists():
        parser.error("Score development before opening the new holdout")
    launch(args.folder, args.book, args.split, args.workers)
    result = score_study(args.folder, args.split, args.book)
    (args.folder / f"{args.split}-aggregate.json").write_text(json.dumps(result, indent=2), "utf8")
    print(args.split, result["unique_receipts"], "GPT answers verified and scored", flush=True)
