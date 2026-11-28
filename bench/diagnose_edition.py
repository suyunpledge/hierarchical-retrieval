"""Reconcile the previous 24-question regression against its stored blind verdicts and wire counters."""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from run import ROOT
from score import analyze

sys.path.insert(0, str(ROOT))
from hierarchical_retrieval.core.chapter_history import is_ordinary_question  # noqa: E402


def diagnose(folder, book):
    verdicts = {}
    with sqlite3.connect(folder / "mimo-decisions.sqlite") as db:
        saved = {k: json.loads(v) for k, v in db.execute("select key,v from reviews")}

    def offline(folder, packages, cap, *, workers):
        return {key: saved[key] for key in packages}

    def remember(k, row, correct, initial, judge, veto):
        if k[0] in ("BT", "BH") and k[1] == 24 and row["question"].startswith("回到第"):
            verdicts[k[0], row["chapter"], row["id"]] = {
                "credited": bool(correct),
                "rule": initial["pair_correct"],
                "review": judge,
                "veto": bool(veto),
            }

    analyze(folder, "tail-test", book=book, review=offline, on_scored=remember)
    issues = []
    missing = 0
    totals = defaultdict(int)
    for f in sorted(folder.glob("BT-tail-test-*-24-r1/trace.json")):
        history = defaultdict(list)
        for row in json.loads(f.read_bytes())["rows"]:
            if row["action"] != "ask":
                continue
            c, t = row["chapter"], row["id"]
            turns = history[c]
            if ("BT", c, t) in verdicts:
                old = [v for v in turns if is_ordinary_question(v["question"])][:-2]
                kept = [v for v in turns if v not in old]
                assert len(kept) == row["usage"]["retained_qa"]
                assert (
                    sum(len(v["question"]) + len(v["answer"]) for v in kept)
                    == row["usage"]["retained_qa_chars"]
                )
                index = max(i for i, v in enumerate(turns) if not v["question"].startswith("回到第"))
                missing += int(any(v in old for v in turns[index:]))
                before, full = [verdicts[arm, c, t] for arm in ("BT", "BH")]
                totals["asks"] += 1
                totals["two_complete"] += int(before["credited"])
                totals["full_complete"] += int(full["credited"])
                if not (before["credited"] and full["credited"]):
                    issues.append(
                        dict(
                            chapter=c,
                            turn=t,
                            kept=[v["id"] for v in kept],
                            omitted=[v["id"] for v in old],
                            explicit=turns[index]["id"],
                            previous=before,
                            full=full,
                        )
                    )
            turns.append(row)
    return dict(totals, missing_version_chain=missing, errors=issues)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("book", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(diagnose(a.folder, a.book), indent=2), "utf8")
