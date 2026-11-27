"""Evaluate the registered development gate; a positive result does not deploy changes."""

import argparse
import json
from pathlib import Path


def evaluate(report, candidate, baseline="S0", repeat=1):
    if report.get("split") != "dev":
        raise ValueError("Use a development aggregate")
    failures = []
    for n in (12, 24):
        base = report["systems"][f"{baseline}-{n}-1"]
        test = report["systems"][f"{candidate}-{n}-{repeat}"]
        if set(base["by_chapter"]) != set(test["by_chapter"]):
            raise ValueError("Different development coverage")
        for metric in ("original", "latest", "recall"):
            if test["metrics"][metric]["lo"] + 1e-12 < base["metrics"][metric]["mean"] - 0.05:
                failures.append(f"{n}: {metric} lower limit")
        if test["metrics"]["unsupported"]["hi"] > base["metrics"]["unsupported"]["mean"] + 0.05 + 1e-12:
            failures.append(f"{n}: unsupported upper limit")
        if test["stale_count"] > base["stale_count"]:
            failures.append(f"{n}: stale count")
    return failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("aggregate", type=Path)
    p.add_argument("candidate")
    p.add_argument("--repeat", type=int, default=1)
    a = p.parse_args()
    reasons = evaluate(json.loads(a.aggregate.read_text("utf8")), a.candidate, repeat=a.repeat)
    print("HOLD: " + ", ".join(reasons) if reasons else "PASS")
    raise SystemExit(bool(reasons))
