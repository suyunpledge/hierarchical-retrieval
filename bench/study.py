"""Run independent sessions concurrently, saving sealed traces outside the project."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fixtures import prepare
from run import CHAPTER_SPLITS, MODEL, ROOT, STAGE_SPLITS, TAIL_SPLITS, study_arms, study_pairs


def lock_cases(folder, book):
    ids = json.loads((ROOT / "bench/splits.json").read_text("utf8"))["test"]
    files = [folder / "fixtures" / f"{ch}.json" for ch in ids]
    for attempt in range(6):
        if all(f.exists() for f in files):
            break
        try:
            prepare(folder, book)
        except Exception as exc:
            print("Fixture preparation retry", type(exc).__name__, flush=True)
            time.sleep(min(2**attempt, 15))
    if not all(f.exists() for f in files):
        raise RuntimeError("Sealed fixtures incomplete")
    lock = {str(ch): hashlib.sha256(f.read_bytes()).hexdigest() for ch, f in zip(ids, files)}
    path = folder / "fixtures.lock"
    if path.exists() and json.loads(path.read_text("utf8")) != lock:
        raise RuntimeError("Sealed cases changed")
    path.write_text(json.dumps(lock, indent=2), "utf8")


def launch(folder, book, split, workers):
    pairs = study_pairs(split)
    if split == "test":
        lock_cases(folder, book)

    def job(item):
        pair, system, n, repeat, seed = item
        name = "seed-" + pair if seed else f"{system}-{pair}-{n}-r{repeat}"
        trace = folder / name / "trace.json"
        for attempt in range(5):
            completed = 0
            if trace.exists():
                rows = json.loads(trace.read_text("utf8"))["rows"]
                completed = sum(r["action"] == "ask" for r in rows)
                if (seed and len(rows) == 6) or (not seed and completed == n):
                    return
            cmd = [
                sys.executable,
                str(ROOT / "bench/run.py"),
                "--split",
                split,
                "--system",
                system,
                "--questions",
                str(n),
                "--repeat",
                str(repeat),
                "--book",
                str(book),
                "--pair",
                pair,
                "--model",
                MODEL,
                "--output",
                str(folder),
            ]
            midpoint = split in STAGE_SPLITS and split not in (*CHAPTER_SPLITS, *TAIL_SPLITS) and not seed and n == 24 and completed < 12
            if midpoint:
                cmd.extend(["--through", "12"])
            reference = folder / ("frozen-" + system + ".py")
            if reference.is_file() and not seed:
                cmd.extend(["--reference", str(reference)])
            if seed:
                cmd.append("--seed")
            with (folder / (name + ".log")).open("a", encoding="utf8") as log:
                code = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1"),
                ).returncode
            if code == 0:
                if midpoint:
                    continue
                return
            if (
                "Experiment budget reached"
                in (folder / (name + ".log")).read_text("utf8", errors="replace")[-2800:]
            ):
                raise RuntimeError("Budget cap reached")
            time.sleep(2**attempt)
        raise RuntimeError("Session incomplete; check transport diagnostics")

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(job, [(pair, "S0", 12, 1, True) for pair in pairs]))
        arms = study_arms(split)
        systems = list(arms)
        work = []
        for n in (12, 24):
            for i, pair in enumerate(pairs):
                for system in systems[i % len(systems) :] + systems[: i % len(systems)]:
                    work.extend((pair, system, n, repeat, False) for repeat in arms[system])
        list(pool.map(job, work))
    print(split, "completed", len(work), "sessions", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("book", type=Path)
    p.add_argument("split", choices=["dev", "test", *STAGE_SPLITS])
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    launch(a.folder, a.book, a.split, a.workers)
