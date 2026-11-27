"""Audit the alternating-edition trials from their receipts and frozen questions."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from eval_long_prefix_return import record
from eval_version_detail import CASES, SPLITS

HELD = ("shoulder-talk", "comfort", "pockets")
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text("utf8"))


def assess(row, audited, decisions=None):
    ch, ver, s = row["chapter"], row["version"], row["answer"].strip()
    question = row["question"]
    key = json.dumps([ch, ver, s], ensure_ascii=False)
    decision = (decisions or {}).get(key)
    groups, versions = CASES[ch][2:4]
    if ver is None:
        # Any additional factual assertion was separately reviewed in context.
        correct = all(any(x in s for x in group) for group in groups)
        if ch in (203, 246):
            correct = correct and "手背" not in s
        extra = audited.get(str(ch), {}).get(s)
        detail_prompt = question.count("？") == 2
        missing_detail_reply = detail_prompt and not re.search(
            r"未|没|无法|不(?:能|确定|分|清楚|明确)|难以", s
        )
        return dict(
            pair_correct=(decision["correct"] if decision else correct) and not extra,
            stale=False,
            unsupported=bool(extra),
            unsupported_kind=extra,
            extra_specificity=extra == "extra_specificity",
            literal_recalled=bool(row["usage"]["witness_user_messages"]),
            detail_prompt=detail_prompt,
            missing_detail_reply=missing_detail_reply,
        )
    pair = versions[ver]
    correct = all(x in s for x in pair) and s.index(pair[0]) < s.index(pair[1])
    rest = s
    for expected in pair:
        rest = rest.replace(expected, " ")
    alternatives = [v for group in groups for v in group] + [v for p in versions[:ver] for v in p]
    # A separately labelled original is not asserted as the working value.
    if correct and ("原文为" in rest or re.search(r"小说原文[（(：:]|小说原始正文[（(：:]", rest)):
        rest = re.sub(r"原文为[^。；\n]*|小说原(?:始正文|文)[（(：:][^\n]*", " ", rest)
    stale = any(v in rest for v in alternatives)
    if decision:
        correct, stale = decision["correct"], decision["stale"]
    return dict(pair_correct=correct and not stale, stale=stale)


def digest(rows):
    totals = Counter()
    for r in rows:
        g = r["audit"]
        u = r["usage"]
        v = r["version"]
        totals.update(
            calls=1,
            original=int(v is None),
            latest=int(v is not None),
            latest_correct=int(v is not None and g["pair_correct"]),
            stale=g["stale"],
            original_correct=int(v is None and g["pair_correct"]),
            unsupported=g.get("unsupported", False),
            extra_specificity=g.get("extra_specificity", False),
            recalled=g.get("literal_recalled", False),
            detail_prompts=g.get("detail_prompt", False),
            missing_detail_reply=g.get("missing_detail_reply", False),
            read_calls=u["read"] > 0,
            **{k: u[k] for k in ("input", "read", "write", "ordinary", "output", "cost_units")},
        )
    totals["uncached"] = totals["input"] - totals["read"]
    return dict(totals)


def run(root, target, partial=False):
    locked = read(root / "v4/protocol.json")
    aud = read(root / "v4/judgments.json")
    results, receipts, charges = {}, {}, {}

    def bill(group, u):
        key = u["response_id"]
        assert (
            key
            and key not in receipts
            and u["cost_units"] == 20 * (u["input"] - u["read"] - u["write"]) + u["read"] + 25 * u["write"]
        )
        receipts[key] = True
        bucket = charges.setdefault(group, Counter())
        bucket.update(calls=1, **{k: u[k] for k in ("input", "read", "write", "output", "cost_units")})

    # Save every billable call and keep development separate from the holdout.
    dev_traces = {}
    for area in ("", "final", "v3", "v4"):
        for f in sorted((root / area).glob("*/trace.json")):
            name = f.parent.name
            t = read(f)
            split = t["split"]
            if split not in SPLITS or (t["mode"] == "seed" and name != "seed-" + split):
                continue
            if t["mode"] == "seed" and (area == "v3" or area == "v4" and split not in HELD):
                continue
            if area == "v4" and split in HELD:
                group = "held_seed" if t["mode"] == "seed" else "held"
            else:
                group = {
                    "": "pilot",
                    "final": "second_wave",
                    "v3": "third_revision",
                    "v4": "regression_retest",
                }[area]
            for r in t["rows"]:
                if r["usage"]:
                    bill(group, r["usage"])
            if group == "held":
                # Include interruptions and process restarts in the live wall-clock span.
                folder_stat = f.parent.stat()
                t["observed_session_seconds"] = max(
                    0, round(f.stat().st_mtime - getattr(folder_stat, "st_birthtime", folder_stat.st_ctime))
                )
                assert partial or sum(r["action"] == "ask" for r in t["rows"]) == t["limit"]
                for r in t["rows"]:
                    if r["action"] == "ask":
                        r["audit"] = assess(r, aud["unsupported_by_chapter"], aud["review_overrides"])
                results[name] = t
            else:
                dev_traces[(area or "pilot") + "/" + name] = t
    assert partial or len(results) == 24
    def completed(name):
        state=results.get(name)
        return state is not None and sum(row["action"]=="ask" for row in state["rows"])==state["limit"]

    paired={name for name in results if completed(name) and completed(name.replace("-before-","-after-") if "-before-" in name else name.replace("-after-","-before-"))}
    summaries = {}
    for n in (12, 24):
        for arm in ("before", "after"):
            rows = [
                r
                for name,t in results.items()
                if name in paired and t["mode"] == arm and t["limit"] == n
                for r in t["rows"]
                if r["action"] == "ask"
            ]
            assert partial or len(rows) == 6 * n
            summaries[f"{n}-{arm}"] = digest(rows)

    def aligned(row):
        return row["id"], row["action"], row["chapter"], row["version"], row["question"]

    audited_strings = set()
    for name, t in results.items():
        arm = t["mode"]
        other = "after" if arm == "before" else "before"
        peer = results.get(name.replace("-" + arm + "-", "-" + other + "-"))
        these = list(map(aligned, t["rows"]))
        if peer is not None:
            theirs=list(map(aligned,peer["rows"]))
            common=min(len(these),len(theirs))
            assert these[:common]==theirs[:common] and t["chapters"]==peer["chapters"] and t["nonce"]!=peer["nonce"]
        short=results.get(name.replace("-24-","-12-"))
        if short is not None:
            prefix=list(map(aligned,short["rows"]))
            common=min(len(these),len(prefix))
            assert prefix[:common]==these[:common]
        first=next((row for row in t["rows"] if row["action"]=="ask"),None)
        if first is not None: assert first["usage"]["read"]==0
        for row in t["rows"]:
            if row["action"] == "ask":
                assert (int(row["id"][1:]) % 2 == 1) == (row["version"] is not None)
                audited_strings.add((row["chapter"], str(row["version"]), row["answer"]))
    checksum = hashlib.sha256(json.dumps(sorted(audited_strings), ensure_ascii=False).encode()).hexdigest()
    assert checksum == aud["answers_sha256"], "Complete the masked answer audit before publication"
    probes = {}
    for path, label in (
        ("probes.json", "tail_only"),
        ("dev.json", "history_ablation"),
        ("final/probes.json", "without_scene_bridge"),
        ("v3/probes.json", "clause_bridge"),
        ("v4/probes.json", "final"),
    ):
        probes[label] = read(root / path)
        for u in probes[label]:
            bill(label, u)
    recon = {
        k: read(root / p)
        for k, p in [
            ("historical", "before.json"),
            ("first_tail", "after.json"),
            ("without_scene_bridge", "final/after.json"),
            ("clause_bridge", "v3/after.json"),
            ("final", "v4/after.json"),
        ]
    }
    code = locked["code"]["ship"]
    for name, expected in code.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    record(
        dict(
            model="gpt-6.2-sol",
            endpoint="https://api.openai.com/v1/chat/completions",
            service_tier="Standard",
            price_usd_per_million=dict(ordinary=2, read=0.10, write_30m=2.50),
            complete=not partial,
            completed_paired_sessions=sorted(paired),
            previous_wave=read(root / "v4/previous_wave.json"),
            protocol=locked,
            corpus=read(root / "v4/corpus.json"),
            source="十日终焉 · first 2,000,000 normalized characters",
            historical=recon,
            historical_probes=probes,
            summary=summaries,
            all_input_charges=charges,
            answer_review=aud,
            heldout=results,
            development=dev_traces,
        ),
        target,
    )
    print(json.dumps(dict(summary=summaries, charges=charges), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--partial",action="store_true")
    a = p.parse_args()
    run(a.root, a.out, a.partial)
