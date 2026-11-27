"""Separate same-rubric sensitivity scoring and bounds for an interrupted primary judge."""

import argparse
import json
import sqlite3
import time
from functools import partial
from pathlib import Path

import requests
from fixtures import load_provider
from score import JUDGE, analyze, decisions

MODEL = "mimo-v2.6-flash"


def mimo_judge(package, budget):
    provider = load_provider("mimo")
    base = provider["base"].rstrip("/")
    if base != "https://token-plan-cn.xiaomimimo.com/v1" or provider["model"] != MODEL:
        raise ValueError("Expected the registered prepaid MiMo endpoint and model")
    body = dict(
        model=MODEL,
        messages=[
            dict(role="system", content=JUDGE),
            dict(role="user", content=json.dumps(package, ensure_ascii=False)),
        ],
        temperature=0,
        max_tokens=220,
        thinking={"type": "disabled"},
        response_format={"type": "json_object"},
    )
    for attempt in range(3):
        reservation = budget.reserve(body)
        response = requests.post(
            base + "/chat/completions",
            json=body,
            timeout=90,
            headers={"Authorization": "Bearer " + provider["key"]},
        )
        if response.ok:
            data = response.json()
            if data["model"] != MODEL:
                raise ValueError("Unexpected independent judge model")
            usage = data["usage"]
            budget.record(reservation, 0, "tokenplan-" + data["id"])
            with sqlite3.connect(budget.path.parent / "mimo-usage.sqlite", timeout=30) as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS usage(id TEXT PRIMARY KEY, input INTEGER, output INTEGER)"
                )
                db.execute(
                    "INSERT INTO usage VALUES(?,?,?)",
                    (data["id"], usage["prompt_tokens"], usage["completion_tokens"]),
                )
            try:
                value = json.loads(data["choices"][0]["message"]["content"])
                fields = ("correct", "stale", "unsupported", "uncertain")
                if all(type(value.get(k)) is bool for k in fields):
                    return {k: value[k] for k in fields}
            except (ValueError, TypeError):
                pass
        elif response.status_code in (400, 401, 402, 403, 404, 429):
            budget.record(reservation, 0, "rejected-" + reservation)
            if response.status_code != 429:
                raise PermissionError(f"MiMo judge rejected HTTP {response.status_code}")
        time.sleep(attempt + 1)
    raise RuntimeError("MiMo verdict unavailable")


def bounds(folder, book):
    with sqlite3.connect(folder / "decisions.sqlite") as db:
        known = {k: json.loads(v) for k, v in db.execute("SELECT key,v FROM reviews")}
    result = {}
    for label in ("lower", "upper"):
        counts = []

        def fill(folder, packages, cap, *, workers):
            missing = set(packages) - known.keys()
            counts.append(dict(distinct=len(packages), missing=len(missing)))
            unknown = dict(
                correct=label == "upper", uncertain=label == "lower", unsupported=False, stale=False
            )
            return {k: known.get(k, unknown) for k in packages}

        a = analyze(folder, "tail-test", book=book, review=fill)
        result[label] = dict(
            systems={
                k: {m: s["metrics"][m]["mean"] for m in ("original", "latest")}
                for k, s in a["systems"].items()
            },
            distinct_review_packages=counts,
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("book", type=Path)
    p.add_argument("--bounds-only", action="store_true")
    args = p.parse_args()
    if not args.bounds_only:
        analysis = analyze(
            args.folder,
            "tail-test",
            book=args.book,
            judge_workers=6,
            review=partial(decisions, reviewer=mimo_judge, cache="mimo-decisions.sqlite"),
        )
        analysis["scorer"] = (
            "same frozen rule and grounding veto + independent MiMo v2.6 Flash, conservative agreement"
        )
        analysis["protocol_status"] = "sensitivity analysis, registered DeepSeek judge incomplete"
        (args.folder / "tail-test-mimo.json").write_text(json.dumps(analysis, indent=2), "utf8")
    (args.folder / "deepseek-bounds.json").write_text(
        json.dumps(bounds(args.folder, args.book), indent=2), "utf8"
    )
