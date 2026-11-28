"""Recompute paired quality and cache economics from private traces, without making API calls."""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from score import analyze


def summarize(folder, book, split):
    aggregate = json.loads((folder / f"{split}-aggregate.json").read_bytes())
    with sqlite3.connect(f"file:{(folder / 'mimo-decisions.sqlite').as_posix()}?mode=ro", uri=True) as db:
        saved = {k: json.loads(v) for k, v in db.execute("SELECT key,v FROM reviews")}

    def offline(_folder, packages, cap, *, workers):
        return {k: saved[k] for k in packages}

    credited = {}

    def observe(key, row, correct, *_):
        credited[key[0], key[1], row["chapter"], row["id"]] = int(correct)

    replay = analyze(folder, split, book=book, review=offline, on_scored=observe)
    if json.loads(json.dumps(replay["systems"])) != aggregate["systems"]:
        raise ValueError("Cached verdict replay differs from the published scores")

    audit = Path(__file__).resolve().parents[1] / "results/20261127-version-anchor/question-audit.json"
    audited_chapters = set(json.loads(audit.read_bytes())["include"]) if split == "anchor-test" else set()
    sensitivity = defaultdict(Counter)
    group = defaultdict(Counter)
    pairs = defaultdict(lambda: defaultdict(list))
    missing = {}
    for path in sorted(folder.glob(f"*-{split}-*-r1/trace.json")):
        state = json.loads(path.read_bytes())
        arm, n = state["mode"], state["limit"]
        counts = group[arm, n]
        for row in state["rows"]:
            if row["action"] != "ask":
                continue
            ch, turn, u = row["chapter"], row["id"], row["usage"]
            ok = credited[arm, n, ch, turn]
            edition = "original" if row["version"] is None else "latest"
            counts.update(asks=1, complete=ok, hits=int(u["read"] > 0))
            counts[edition] += 1
            counts[edition + "_complete"] += ok
            counts.update({k: u[k] for k in ("input", "read", "write", "ordinary", "output", "cost_units")})
            if arm == "BA":
                counts["need"] += int(u["needs_anchor"])
                counts["restored"] += int(bool(u["restored_turns"]))
                for k in ("restored_exchanges", "restored_instructions", "restored_chars", "verbatim_notes"):
                    counts[k] += u[k]
                missing[n, ch, turn] = u["needs_anchor"]
            if ch in audited_chapters:
                stats = sensitivity[f"{arm}-{n}-1"]
                stats[edition] += 1
                stats[edition + "_complete"] += ok
                if row["question"].startswith("回到第"):
                    stats["relative"] += 1
                    stats["relative_complete"] += ok
            if row["question"].startswith("回到第"):
                pairs[n][state["split"]].append((arm, ch, turn, ok))

    out = {}
    for (arm, n), c in sorted(group.items()):
        name = f"{arm}-{n}-1"
        a = aggregate["systems"][name]
        assert c["asks"] == a["asks"] and c["complete"] == a["complete"]
        assert abs(c["cost_units"] / 1e7 - a["input_usd"]) < 1e-9
        assert abs((c["cost_units"] + c["output"] * 100) / 1e7 - a["total_usd"]) < 1e-9
        out[name] = dict(
            c,
            edition_followups=a["edition_followups"],
            evidence=aggregate["version_evidence"][name],
            input_usd_per_ask=a["input_usd"] / c["asks"],
            total_usd_per_complete=a["total_usd"] / c["complete"],
            original_chapter_mean=a["metrics"]["original"]["mean"],
            latest_chapter_mean=a["metrics"]["latest"]["mean"],
        )

    changes = {}
    for n, sessions in pairs.items():
        strata = defaultdict(Counter)
        clusters = []
        for records in sessions.values():
            scores = {(a, c, t): value for a, c, t, value in records}
            delta = 0
            count = 0
            for a, c, t, ok in records:
                if a != "BA":
                    continue
                diff = ok - scores["BT", c, t]
                delta += diff
                count += 1
                s = strata["missing" if missing[n, c, t] else "already_present"]
                s["gained" if diff == 1 else "lost" if diff == -1 else "tied"] += 1
            clusters.append((delta, count))
        data = np.array(clusters)
        draw = np.random.default_rng(20261127).integers(len(data), size=(10000, len(data)))
        totals = data[draw].sum(axis=1)
        bounds = np.quantile(totals[:, 0] / totals[:, 1], [0.025, 0.975])
        changes[n] = dict(strata=strata, pairs=len(data), edition_diff_interval=bounds.tolist())
    return dict(split=split, systems=out, paired_changes=changes, question_audit_subset=sensitivity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("book", type=Path)
    parser.add_argument("split", choices=("anchor-dev", "anchor-test"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = summarize(args.folder, args.book, args.split)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
