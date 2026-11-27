"""Paired, horizon-blind evaluation of the answer history after the chapter breakpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from run import ROOT, edition_plan

RESULTS = ROOT / "results/20261127-history-tail"
RELATIVE = {5, 6, 9, 10, 13, 14, 17, 18, 21, 22}


def paired_chapters(split):
    side = {"tail-dev": "dev", "tail-test": "test"}[split]
    ids = json.loads((ROOT / "bench/tail_split.json").read_bytes())[side]
    return {f"tail-{side}-{i // 2:02}": ids[i : i + 2] for i in range(0, len(ids), 2)}


def verify_lock(folder):
    protocol = json.loads((RESULTS / "protocol.json").read_bytes())
    for path, key in (
        (folder / "fixtures.lock", "fixture_lock_sha256"),
        (ROOT / "bench/tail_split.json", "manifest_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError("Fresh chapter manifest or frozen questions changed")


def tail_plan(ids, docs, limit, seed, cases, order):
    previous = {}
    for label, ch, action, question, version in edition_plan(ids, docs, limit, seed, cases, order):
        if action == "ask":
            latest = version is not None
            if int(label[1:]) in RELATIVE:
                relation = "沿用上一次选的版本" if previous[ch] == latest else "换到上一次没选的那个版本"
                question = f"回到第{ch}章，{relation}。上次问的两项，现在分别是什么？"
            previous[ch] = latest
        yield label, ch, action, question, version


def audit_tail(payload, chapter, ch, events, question, nonce, mode):
    """Read the actual wire tail as a chronological subsequence of the real local transcript."""
    from eval_long_prefix_return import SYSTEM

    messages = payload["messages"]
    assert messages[0] == {"role": "system", "content": "档案 " + nonce + SYSTEM}
    assert messages[1] == {
        "role": "user",
        "content": [{"type": "text", "text": chapter, "prompt_cache_breakpoint": {"mode": "explicit"}}],
    }
    assert payload["prompt_cache_key"] and payload["prompt_cache_options"] == {"mode": "explicit"}
    assert messages[-1] == {"role": "user", "content": question}
    tail = messages[2:-1]
    offset = notes = seen = kept = kept_chars = history_chars = 0
    flags = []
    for _, side, action, raw, answer in reversed(events):
        if side != ch or action not in ("ask", "note"):
            continue
        msg = {"role": "user", "content": raw}
        if action == "note":
            assert tail[len(tail) - offset - 1 : len(tail) - offset] == [msg]
            offset += 1
            notes += 1
            continue
        pair = [msg, {"role": "assistant", "content": answer}]
        end = len(tail) - offset
        sent = end >= 2 and tail[end - 2 : end] == pair
        seen += 1
        history_chars += len(raw) + len(answer)
        flags.append(sent)
        if sent:
            offset += 2
            kept += 1
            kept_chars += len(raw) + len(answer)
    assert offset == len(tail)
    if mode == "BH":
        from chapter_study import audit_layout

        audit_layout(payload, chapter, ch, events, question, nonce)
    else:
        window = int(os.environ["HR_CHAPTER_QA_WINDOW"])
        if window:
            assert all(flags[:window])
    return dict(
        available_qa=seen,
        retained_qa=kept,
        retained_qa_chars=kept_chars,
        available_qa_chars=history_chars,
        verbatim_notes=notes,
    )


def choose(dev):
    scores = dev["systems"]
    ranked = []
    for window, arm in ((1, "BT"), (2, "BD")):
        differences = []
        cost = 0
        for n in (12, 24):
            s, b = (scores[f"{name}-{n}-1"] for name in (arm, "BH"))
            differences.extend(
                s["metrics"][m]["mean"] - b["metrics"][m]["mean"] for m in ("original", "latest")
            )
            cost += s["input_usd"]
        floor = min(differences)
        eligible = floor >= -0.030000000001
        ranked.append(((not eligible, 0 if eligible else -floor, cost, window), window))
    return min(ranked)[1]


def main():
    from score import analyze
    from study import launch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("book", type=Path)
    parser.add_argument("split", choices=("tail-dev", "tail-test"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--judge-workers", type=int, choices=range(1, 13), default=4)
    args = parser.parse_args()
    folder, split = args.folder, args.split
    verify_lock(folder)
    if split == "tail-test":
        dev = json.loads((folder / "tail-dev-aggregate.json").read_bytes())
        if json.loads((folder / "tail-choice.json").read_bytes()) != {"window": choose(dev)}:
            raise ValueError("Development selection differs from the registered rule")
    launch(folder, args.book, split, args.workers)
    result = analyze(folder, split, book=args.book, judge_workers=args.judge_workers)
    (folder / (split + "-aggregate.json")).write_text(json.dumps(result, indent=2), "utf8")
    if split == "tail-dev":
        (folder / "tail-choice.json").write_text(json.dumps({"window": choose(result)}), "utf8")
    print(split, result["unique_receipts"], "verified and scored", flush=True)


if __name__ == "__main__":
    main()
