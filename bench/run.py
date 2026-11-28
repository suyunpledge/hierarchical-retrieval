"""Compare named-chapter systems on registered dialogues using the same live GPT transport."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "bench" / "legacy"
sys.path.insert(0, str(LEGACY))
from eval_idle_novel_cache import MODEL  # noqa: E402
from eval_version_detail import CASES, ORDER, SPLITS, WITNESSES, edition_plan, grade, run  # noqa: E402

DEV_PAIRS = ("farewell", "gesture-rule")


REGRESSION_PAIRS = ("comfort", "pockets", "shoulder-talk")


OUTCOME_PAIRS = {
    "pull-eyes": (807, 182),
    "door-depart": (484, 537),
    "motion-card": (542, 374),
    "radio-look": (206, 510),
}
CHAPTER_SPLITS = ("chapter-dev", "chapter-test")
ANCHOR_SPLITS = ("anchor-dev", "anchor-test")
TAIL_SPLITS = ("tail-dev", "tail-test", *ANCHOR_SPLITS)
SEALED_SPLITS = ("test", "chapter-test", "tail-test", "anchor-test")
STAGE_SPLITS = ("regression", "prefix", "breakpoint", "outcome", *CHAPTER_SPLITS, *TAIL_SPLITS)


def verify_chapter_lock(folder):
    protocol = json.loads((ROOT / "results/20261127-evaluation-baselines/protocol.json").read_text("utf8"))
    for path, expected in ((folder / "fixtures.lock", "locked_questions_sha256"),
                           (ROOT / "bench/splits.json", "split_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[expected]:
            raise ValueError("The original baseline lock is required")


def outcome_cases():
    return {int(k): v for k, v in json.loads((ROOT / "bench/outcome_cases.json").read_text("utf8")).items()}


def study_pairs(split):
    if split in ANCHOR_SPLITS:
        from version_anchor import paired_chapters

        return list(paired_chapters(split))
    if split in TAIL_SPLITS:
        from history_tail import paired_chapters

        return list(paired_chapters(split))
    if split == "outcome":
        return tuple(OUTCOME_PAIRS)
    if split in SEALED_SPLITS:
        return [f"sealed-{i:02d}" for i in range(20)]
    if split == "breakpoint":
        return ("door-outcome", "reply-outcome")
    return REGRESSION_PAIRS if split == "regression" else DEV_PAIRS


def study_arms(split):
    if split in ANCHOR_SPLITS:
        return {arm: (1,) for arm in ("S0", "BH", "BT", "BA")}
    if split in TAIL_SPLITS:
        arms = ("S0", "BH", "BT", "BD", "BZ") if split == "tail-dev" else ("S0", "BH", "BT")
        return {arm: (1,) for arm in arms}
    if split in CHAPTER_SPLITS:
        return {"BH": (1,)}
    if split == "outcome":
        return {arm: (1, 2) for arm in ("BP", "BX", "BG")}
    if split == "regression":
        return {"S0": (1, 2), "S1": (1, 2)}
    if split == "breakpoint":
        return {"BP": (1, 2), "BX": (1, 2)}
    if split == "prefix":
        return {"B0": (1, 2), "BP": (1, 2)}
    if split == "dev":
        return {"B0": (1,), "S0": (1, 2)}
    return {name: (1,) for name in ("S0", "B0", "S1", "B1", "B2")}


def baseline_messages(system, docs, ids, events, chapter, question, header):
    if system in ("B0", "BP", "BX", "BG"):
        notes = [m for _, ch, action, m, _ in events if ch == chapter and action == "note"]
        if system in ("BP", "BX", "BG"):
            if system == "BG":
                from hierarchical_retrieval.core.branch_context import revision_outcome_suffix

                question += revision_outcome_suffix(question)
            messages = [
                header,
                *({"role": "user", "content": entry} for entry in (docs[chapter], *notes, question)),
            ]
            if system == "BP":
                return messages
            from hierarchical_retrieval.topic_chat import BranchTurn

            return BranchTurn(messages, {}, cache_breakpoints=[1]).openai_cache_messages()
        body = "\n\n".join([docs[chapter], *notes, question])
        return [header, {"role": "user", "content": body}]
    if system == "B1":
        messages = [header]
        for _, _, action, content, answer in events:
            messages.append({"role": "user", "content": content})
            if action != "note":
                messages.append({"role": "assistant", "content": answer})
        return [*messages, {"role": "user", "content": question}]
    raise ValueError(system)


def worker(args):
    data = json.loads((ROOT / "bench/splits.json").read_text("utf8"))
    cases = dict(CASES)
    pairs = dict(SPLITS)
    witnesses = dict(WITNESSES)
    planner = edition_plan
    if args.split in TAIL_SPLITS or args.split in SEALED_SPLITS:
        from eval_version_detail import DETAIL

        if args.split == "chapter-test":
            verify_chapter_lock(args.output)
        if args.split in TAIL_SPLITS:
            if args.split in ANCHOR_SPLITS:
                from version_anchor import anchor_plan as tail_plan
                from version_anchor import paired_chapters, verify_lock
            else:
                from history_tail import paired_chapters, tail_plan, verify_lock

            verify_lock(args.output)
            pairs = paired_chapters(args.split)
            if args.pair not in pairs:
                raise ValueError("Unregistered fresh chapter pair")
            planner = tail_plan
        else:
            ids = data["test"]
            index = int(args.pair.removeprefix("sealed-"))
            if index < 0 or index >= len(ids) // 2:
                raise ValueError("Invalid sealed pair")
            pairs[args.pair] = ids[2 * index : 2 * index + 2]
        lock = json.loads((args.output / "fixtures.lock").read_text("utf8"))
        for ch in pairs[args.pair]:
            f = args.output / "fixtures" / f"{ch}.json"
            if hashlib.sha256(f.read_bytes()).hexdigest() != lock[str(ch)]:
                raise ValueError("Fixture changed")
            fixture = json.loads(f.read_text("utf8"))
            cases[ch] = fixture["case"]
            witnesses[ch] = fixture["witness"]
            DETAIL[ch] = fixture["detail"]
    elif args.split == "outcome":
        from eval_version_detail import DETAIL

        pairs.update(OUTCOME_PAIRS)
        fixtures = outcome_cases()
        chapters = pairs[args.pair]
        if args.pair not in OUTCOME_PAIRS or set(chapters) & set(data["dev"] + data["test"]):
            raise ValueError("Previously used outcome chapters")
        for ch in chapters:
            cases[ch] = fixtures[ch]["case"]
            witnesses[ch] = fixtures[ch]["witness"]
            DETAIL[ch] = fixtures[ch]["detail"]
    elif args.pair not in study_pairs(args.split) or not set(pairs[args.pair]) <= set(data["dev"]):
        raise ValueError("Unregistered development chapters")
    corpus = args.book.read_text("utf-8-sig")[:2_000_000]
    if hashlib.sha256(corpus.encode()).hexdigest() != data["corpus_sha256"]:
        raise ValueError("Corpus fingerprint differs from the registered split.")
    study_split = args.split
    os.environ["HR_WHOLE_CHAPTER_HISTORY"] = "1" if args.system in ("BH","BT","BD","BZ","BA") else "0"
    window = {"BT": 1, "BD": 2, "BZ": 0}.get(args.system, "all")
    if study_split == "tail-test" and args.system == "BT":
        window = json.loads((args.output / "tail-choice.json").read_bytes())["window"]
    if study_split in ANCHOR_SPLITS and args.system in ("BT","BA"):
        window = 2
    os.environ["HR_CHAPTER_QA_WINDOW"] = str(window)
    os.environ["HR_CHAPTER_EDITION_ANCHOR"] = "1" if args.system == "BA" else "0"
    args.capture_layout = study_split == "outcome"
    args.code = ROOT
    os.environ["HR_V1125_EVIDENCE_ORDER"] = "0" if args.system == "S1" and not args.seed else "1"
    os.environ["HR_GROUND_REVISION_OUTCOMES"] = "0"  # Select the treatment once in the outbound layout.
    args.traces = args.output
    args.mode = "seed" if args.seed else args.system
    args.name = (
        "seed-" + args.pair if args.seed else f"{args.system}-{args.pair}-{args.questions}-r{args.repeat}"
    )
    args.split = args.pair
    args.limit = args.questions
    from budget import Budget

    args.traces.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.traces / "budget.sqlite", 8 if study_split in STAGE_SPLITS else 25)
    source_files = [
        args.code / "api_server.py",
        *sorted((args.code / "hierarchical_retrieval").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for file in source_files:
        digest.update(
            file.relative_to(args.code).as_posix().encode()
            + b"\0"
            + hashlib.sha256(file.read_bytes()).digest()
        )
    budget.pin("source-S1" if args.mode == "S1" else "source-S0", digest.hexdigest())
    budget.pin("chapter-split", hashlib.sha256((ROOT / "bench/splits.json").read_bytes()).hexdigest())
    if study_split in TAIL_SPLITS:
        for name in ("history_tail.py", "tail_split.json", "fixtures.py", "score.py", "run.py",
                     "legacy/eval_version_detail.py", "legacy/eval_long_prefix_return.py"):
            budget.pin(name, hashlib.sha256((ROOT / "bench" / name).read_bytes()).hexdigest())
        if study_split in ANCHOR_SPLITS:
            for name in ("version_anchor.py", "anchor_split.json", "tail_review.py"):
                budget.pin(name, hashlib.sha256((ROOT / "bench" / name).read_bytes()).hexdigest())
        budget.pin("tail-fixtures", hashlib.sha256((args.output / "fixtures.lock").read_bytes()).hexdigest())
        if study_split == "tail-test":
            budget.pin("tail-choice", hashlib.sha256((args.output / "tail-choice.json").read_bytes()).hexdigest())
    if study_split in CHAPTER_SPLITS:
        for name in ("score.py", "run.py", "chapter_study.py", "legacy/eval_long_prefix_return.py", "legacy/eval_version_detail.py"):
            budget.pin(name, hashlib.sha256((ROOT / "bench" / name).read_bytes()).hexdigest())
    if study_split == "outcome":
        for name in ("outcome_cases.json", "run.py", "legacy/eval_long_prefix_return.py"):
            budget.pin(name, hashlib.sha256((ROOT / "bench" / name).read_bytes()).hexdigest())
    builder = None
    if not args.seed:
        if args.system in ("B0", "B1", "BP", "BX", "BG"):
            builder = baseline_messages
        elif args.system == "B2":
            from rag import ParagraphSearch

            builder = ParagraphSearch(args.output / "embeddings.sqlite")
    run(
        args,
        billing=budget,
        cases=cases,
        splits=pairs,
        order=ORDER,
        blind=True,
        grader=grade,
        witnesses=witnesses,
        planner=planner,
        prompt_builder=builder,
        reference=getattr(args, "reference", None),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("S0", "S1", "B0", "B1", "B2", "BP", "BX", "BG", "BH", "BT", "BD", "BZ", "BA"), required=True)
    parser.add_argument("--split", choices=("dev", "test", *STAGE_SPLITS), required=True)
    parser.add_argument("--questions", type=int, choices=(12, 24), required=True)
    parser.add_argument("--model", choices=[MODEL], required=True)
    parser.add_argument("--book", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "hr-chapter-baselines-20261127",
        help="Isolated trace directory with a shared budget ledger",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pair")
    parser.add_argument("--through", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.split in TAIL_SPLITS:
        if not args.seed and args.system not in study_arms(args.split):
            parser.error("The requested arm is outside the registered fresh-chapter split.")
    elif args.system in ("BT", "BD", "BZ", "BA"):
        parser.error("Use a registered history-tail split for this comparison arm.")
    if args.book is None:
        paths = list((Path.home() / "Desktop/本地工作流/番茄爆款语料库").glob("十日终焉*.txt"))
        if len(paths) != 1:
            parser.error("Pass --book with the registered UTF-8 novel.")
        args.book = paths[0]
    path = args.output.resolve()
    if any(part.startswith(("hierarchical_api_storage", "hierarchical_storage")) for part in path.parts):
        parser.error("Choose a separate evaluation output directory.")
    if args.pair:
        worker(args)
        return
    from study import lock_cases

    args.output.mkdir(exist_ok=True, parents=True)
    if args.split == "test":
        lock_cases(args.output, args.book)
    for pair in study_pairs(args.split):
        cmd = [sys.executable, __file__, *sys.argv[1:], "--pair", pair]
        subprocess.run([*cmd, "--seed"], check=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
