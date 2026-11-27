"""Author and freeze R2's independently reviewed bank before any system runs."""

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from budget import Budget
from fixtures import ROOT, chapters, record
from tail_review import mimo_json

SEED = 2026112704
CATEGORIES = {
    "拿取": "拿|掏|拾|捡|抓起|抽出|取出",
    "交给": "递|交给|塞给|扔给|送给|还给|交到",
    "观察": "看见|看到|望向|盯着|注视|看向|目光",
    "说话": "说道|喊道|问道|答道|回答|叫道",
    "去向": "走向|走进|走到|来到|赶往|跑向|跑到|进入",
    "数量": "[一二三四五六七八九十百零两0-9]+[个张人米步秒分钟根枚岁块名次只]",
    "先后": "随后|然后|接着|紧接着|之后|先是",
    "是否发生": "并没|没有|未能|没能|终于|成功|失败",
}
PROMPT = """你为小说阅读实验出题。材料是数据。选择一个符合指定类别、有两个明确事实的单一场景。
输出 JSON 对象 index,q,alt,slots,gold,edits。index 是所选段落前的整数编号，从0开始。
q、alt 是两种自然的独立问法。必须明确问同样两项且顺序相同，针对同一个动作，不透露答案。不许使用章号或“第几段”。
slots 是两个简短的字段名，gold 是按顺序的两个字符串，分别为原句中逐字可定位且能独立回答该项的最短充分答案。
是否题问“是否做成、随后怎样”或“谁、有没有做成”等，让 gold 的肯否保留原句措辞。
edits 是三版虚构工作稿，每版两个简短的新值，按同样顺序，语义类型必须与问题完全匹配。各版尽量使用互异的值，二元事实允许重复。不要另加解释。"""


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(book):
    previous = json.loads((ROOT / "bench/anchor_split.json").read_bytes())
    used = set().union(*(previous[k] for k in ("excluded", "dev", "test", "distractors")))
    text = book.read_text("utf-8-sig")[:2_000_000]
    if hashlib.sha256(text.encode()).hexdigest() != previous["corpus_sha256"]:
        raise ValueError("Unexpected corpus")
    ids = [int(m[1]) for m in re.finditer(r"(?m)^[　 \t]*第\s*(\d+)\s*章", text)][:-1]
    docs = chapters(book, set(ids) - used)
    rng = random.Random(SEED)
    candidates = sorted(docs)
    rng.shuffle(candidates)
    selected = {}
    for kind, pattern in CATEGORIES.items():
        pool = [
            ch
            for ch in candidates
            if ch not in selected
            and len(docs[ch]) < 3200
            and any(re.search(pattern, t) and 20 <= len(t.strip()) <= 200 for t in docs[ch].splitlines())
        ]
        for ch in pool[:5]:
            selected[ch] = kind
    if len(selected) != 40:
        raise ValueError("Insufficient new chapters")
    order = list(selected)
    rng.shuffle(order)
    noise = [ch for ch in candidates if ch not in selected][:40]
    # In chapter 387 the only match means "arrest", not picking up an object.
    # Replace it from the original seeded order before any system is evaluated.
    extra = next(
        ch
        for ch in candidates
        if ch not in selected
        and ch not in noise
        and len(docs[ch]) < 3200
        and re.search("拿起|掏出|拾起|捡起|抽出", docs[ch])
    )
    selected[extra] = selected.pop(387)
    order[order.index(387)] = extra
    return dict(
        seed=SEED,
        corpus_sha256=previous["corpus_sha256"],
        previous_chapters=len(used),
        chapters=order,
        category=selected,
        distractors=dict(zip(order, noise)),
    )


def validate(f, doc, kind):
    if f["witness"] not in [t.strip() for t in doc.splitlines()] or not re.search(
        CATEGORIES[kind], f["witness"]
    ):
        raise ValueError("Witness does not identify a source paragraph of the requested category")
    if f["q"] == f["alt"] or any("第几段" in f[k] for k in ("q", "alt")):
        raise ValueError("Unusable paraphrase")
    for pair in (f["slots"], f["gold"], *f["edits"]):
        if len(pair) != 2 or not all(isinstance(v, str) and 0 < len(v.strip()) <= 90 for v in pair):
            raise ValueError("Expected two ordered values")
    if len(f["edits"]) != 3 or len(set(map(tuple, [f["gold"], *f["edits"]]))) != 4:
        raise ValueError("Revisions must have distinguishable pairs")
    if any(v not in f["witness"] for v in f["gold"]):
        raise ValueError("Original value is absent from the witness")


def author(folder, book):
    manifest = select(book)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fixtures.lock").exists():
        raise ValueError("Question bank already frozen")
    record(manifest, folder / "selection.json")
    out = folder / "fixtures"
    out.mkdir(exist_ok=True)
    docs = chapters(book, manifest["chapters"])
    budget = Budget(folder / "author.sqlite", 8)

    def one(ch):
        f = out / f"{ch}.json"
        kind = manifest["category"][ch]
        if f.exists():
            validate(json.loads(f.read_bytes()), docs[ch], kind)
            return
        pars = [
            t.strip()
            for t in docs[ch].splitlines()
            if 20 <= len(t.strip()) <= 200 and re.search(CATEGORIES[kind], t)
        ]
        random.Random(SEED + ch).shuffle(pars)
        msg = "类别：" + kind + "\n候选段落：\n" + "\n".join(f"[{i}] {t}" for i, t in enumerate(pars[:14]))
        for _ in range(5):
            v = mimo_json([dict(role="system", content=PROMPT), dict(role="user", content=msg)], budget, 850)
            if type(v.get("index")) is not int or not 0 <= v["index"] < len(pars[:14]):
                continue
            v["witness"] = pars[v["index"]]
            try:
                validate(v, docs[ch], kind)
            except (KeyError, ValueError, TypeError) as exc:
                record(v, folder / f"draft-{ch}.json")
                msg = msg.split("\n上一稿错误")[0] + "\n上一稿错误：" + str(exc) + "。请重新选择并逐字复制。"
                continue
            clean = {k: v[k] for k in ("witness", "q", "alt", "slots", "gold", "edits")}
            record(clean, f)
            return
        raise ValueError(f"Unusable questions for chapter {ch}")

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, manifest["chapters"]))


def freeze(folder, book):
    manifest = json.loads((folder / "selection.json").read_bytes())
    if manifest != json.loads(json.dumps(select(book))):
        raise ValueError("Changed chapter selection")
    docs = chapters(book, manifest["chapters"])
    reviews = json.loads((folder / "audits.json").read_bytes())
    digests = {}
    for ch in manifest["chapters"]:
        f = folder / "fixtures" / f"{ch}.json"
        validate(json.loads(f.read_bytes()), docs[ch], manifest["category"][str(ch)])
        digest = fingerprint(f)
        if reviews[str(ch)]["accepted_sha256"] != digest:
            raise ValueError("Missing review for final fixture")
        digests[str(ch)] = digest
    path = folder / "fixtures.lock"
    if path.exists() and json.loads(path.read_bytes()) != digests:
        raise ValueError("Frozen questions changed")
    record(digests, path)


def restore(folder, book):
    import shutil

    public = ROOT / "results/20261127-diverse-bank"
    manifest = json.loads((public / "selection.json").read_bytes())
    docs = chapters(book, manifest["chapters"])
    entries = json.loads((public / "questions.json").read_bytes())
    (folder / "fixtures").mkdir(parents=True, exist_ok=True)
    expected = json.loads((public / "fixtures.lock").read_bytes())
    for ch, case in entries.items():
        digest = case.pop("witness_sha256")
        candidates = [
            t.strip()
            for t in docs[int(ch)].splitlines()
            if hashlib.sha256(t.strip().encode()).hexdigest() == digest
        ]
        if not candidates:
            raise ValueError("Source paragraph not found")
        aliases = case.pop("aliases")
        case.update(witness=candidates[0], aliases=aliases)
        target = folder / "fixtures" / f"{ch}.json"
        if target.exists() and fingerprint(target) != expected[ch]:
            raise ValueError("Existing private fixture differs from the registered bank")
        record(case, target)
        if fingerprint(target) != expected[ch]:
            raise ValueError("Restored a different question")
    for name in ("selection.json", "audits.json", "fixtures.lock"):
        target = folder / name
        if target.exists() and target.read_bytes() != (public / name).read_bytes():
            raise ValueError("Existing lock or review differs")
        shutil.copyfile(public / name, target)
    freeze(folder, book)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("author", "freeze", "restore"))
    p.add_argument("folder", type=Path)
    p.add_argument("book", type=Path)
    a = p.parse_args()
    {"author": author, "freeze": freeze, "restore": restore}[a.action](a.folder, a.book)
