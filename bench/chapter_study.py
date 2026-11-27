"""Run and score N1 using the sealed baseline prompts and isolated copies of its initial workspaces."""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from zipfile import ZipFile

from run import CHAPTER_SPLITS, ROOT, study_pairs, verify_chapter_lock

BASELINE = ROOT / "results/20261127-evaluation-baselines"


def audit_layout(payload, chapter_text, chapter, events, question, nonce):
    """Verify the actual outgoing API payload against the independent dialogue script."""
    from eval_long_prefix_return import SYSTEM

    messages = payload["messages"]
    assert messages[0] == {"role": "system", "content": "档案 " + nonce + SYSTEM}
    assert messages[1] == {
        "role": "user",
        "content": [{"type": "text", "text": chapter_text, "prompt_cache_breakpoint": {"mode": "explicit"}}],
    }
    assert payload["prompt_cache_options"] == {"mode": "explicit"} and payload["prompt_cache_key"]
    pairs = []
    for _, ch, action, raw, answer in events:
        if ch == chapter and action in ("note", "ask"):
            chunk = [{"role": "user", "content": raw}]
            if action == "ask":
                chunk.append({"role": "assistant", "content": answer})
            pairs.append(chunk)
    current = {"role": "user", "content": question}
    remaining = 5500 - len(messages[0]["content"]) - len(chapter_text) - len(question)
    for pair in pairs:
        if sum(len(m["content"]) for group in pairs for m in group) <= remaining:
            break
        if len(pair) == 2:
            pair.clear()
    assert messages[2:] == [*(m for group in pairs for m in group), current]


def prepare(folder, original, split):
    verify_chapter_lock(original)
    folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(BASELINE / "sealed-traces.zip") as frozen:
        if not (folder / "fixtures.lock").exists():
            shutil.copytree(original / "fixtures", folder / "fixtures", dirs_exist_ok=True)
            shutil.copyfile(original / "fixtures.lock", folder / "fixtures.lock")
        verify_chapter_lock(folder)
        for pair in study_pairs(split):
            name = "seed-" + pair
            source = original / name
            if (source / "trace.json").read_bytes() != frozen.read(name + "/trace.json"):
                raise ValueError("Baseline seed receipts have changed")
            if not (folder / name).exists():
                shutil.copytree(source, folder / name)
        if not (folder / "decisions.sqlite").exists():
            # Only identical judge packages can reuse a decision.
            with (
                sqlite3.connect(original / "decisions.sqlite") as src,
                sqlite3.connect(folder / "decisions.sqlite") as dst,
            ):
                src.backup(dst)


def verify_sessions(folder, split):
    def script(state):
        return [(r["chapter"], r["action"], r["id"], r["version"], r["question"]) for r in state["rows"]]

    keys_seen = set()
    with ZipFile(BASELINE / "sealed-traces.zip") as frozen:
        for pair in study_pairs(split):
            for n in (12, 24):
                suffix = f"-{pair}-{n}-r1/trace.json"
                actual = json.loads((folder / ("BH" + suffix)).read_bytes())
                baseline = json.loads(frozen.read("B0" + suffix))
                if script(actual) != script(baseline):
                    raise ValueError("Scored questions differ from original comparison")
                rows = [r for r in actual["rows"] if r["action"] == "ask"]
                by_chapter = {}
                for row in rows:
                    key = row["usage"]["topic_key"]
                    chapter = row["chapter"]
                    if by_chapter.setdefault(chapter, key) != key or row["policy"] != "whole_chapter_history":
                        raise ValueError("The chapter route or cache identifier changed")
                keys = set(by_chapter.values())
                if len(keys) != 2 or keys & keys_seen:
                    raise ValueError("Cache identifier shared across topics or sessions")
                keys_seen.update(keys)


def gate(result, reference):
    out = {}
    for length in (12, 24):
        c = result["systems"][f"BH-{length}-1"]["metrics"]
        b = reference["systems"][f"B0-{length}-1"]["metrics"]
        s = reference["systems"][f"S0-{length}-1"]["metrics"]
        delta = c["original"]["mean"] - b["original"]["mean"]
        ratio = c["input_usd"]["mean"] / s["input_usd"]["mean"]
        out[str(length)] = dict(
            original_delta_pp=100 * delta,
            input_cost_ratio=ratio,
            passed=delta >= -0.03 - 1e-12 and ratio <= 2 + 1e-12,
        )
    return out


def main():
    from score import analyze
    from study import launch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("original", type=Path, help="Original, locked five-system study directory")
    p.add_argument("book", type=Path)
    p.add_argument("split", choices=CHAPTER_SPLITS)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.split == "chapter-test" and not (a.folder / "chapter-dev-aggregate.json").exists():
        p.error("Complete development before opening sealed evaluation")
    prepare(a.folder, a.original, a.split)
    launch(a.folder, a.book, a.split, a.workers)
    verify_sessions(a.folder, a.split)
    result = analyze(a.folder, a.split, book=a.book)
    if a.split == "chapter-test":
        result["gate"] = gate(result, json.loads((BASELINE / "test.json").read_bytes()))
    (a.folder / (a.split + "-aggregate.json")).write_text(json.dumps(result, indent=2), "utf8")
    print(a.split, result["unique_receipts"], "receipts verified and scored", flush=True)


if __name__ == "__main__":
    main()
