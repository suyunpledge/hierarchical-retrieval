# -*- coding: utf-8 -*-
"""
端到端效果评测：长对话 → /v1 代理的历史压缩 → 真实对话模型作答 → 判对错。

衡量的是"压缩之后模型还答不答得出旧对话里的事实"，而不是检索指标本身。

用法（需本机嵌入服务 127.0.0.1:8101）:
    python bench/legacy/eval_e2e.py                       # 只算证据命中（不调对话模型）
    python bench/legacy/eval_e2e.py --models deepseekflash,gpt,mimo
    python bench/legacy/eval_e2e.py --models deepseekflash --full   # 另跑"不压缩"上限对照

指标:
    evidence  压缩后的 prompt 里含有标准答案（与对话模型无关）
    answer    对话模型的回答含有标准答案；negative 题为"没有编造"
    chars     发给模型的 prompt 平均字符数

对话模型的地址与密钥在运行时从 ~/.forge（forge.patch.json + secrets.json）
按 modelLabel 读取，不写入本仓库，也不打印。
"""

import argparse
import json
import os
import shutil
import time
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import requests


def load_provider(label: str) -> dict:
    home = Path(os.environ.get("FORGE_HOME") or Path.home() / ".forge")
    rows = json.loads((home / "forge.patch.json").read_text(encoding="utf-8"))
    secrets = json.loads((home / "secrets.json").read_text(encoding="utf-8"))
    for row in rows:
        cfg = row.get("config", {})
        if cfg.get("modelLabel") == label and row["id"] in secrets:
            base = cfg["baseURL"].rstrip("/")
            return {"label": label, "base": base, "model": cfg["model"], "key": secrets[row["id"]]}
    raise SystemExit(f"~/.forge 中找不到 modelLabel={label} 的服务商或其密钥")


def chat(provider: dict, messages: list[dict]) -> str:
    url = provider["base"] + "/chat/completions"
    headers = {"Authorization": f"Bearer {provider['key']}"}
    payload = {"model": provider["model"], "messages": messages, "temperature": 0}
    last = ""
    for attempt in range(5):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=180)
            if r.status_code == 400 and "temperature" in payload:
                payload.pop("temperature")      # 部分模型不接受自定义 temperature
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""
        except (requests.RequestException, KeyError, ValueError) as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
    return f"[ERROR] {last}"


DENIALS = ("没有", "没说", "没提", "没告诉", "还没", "不知道", "不记得", "未提", "并未", "无法", "不清楚")


def contains_all(text: str, groups: list[list[str]]) -> bool:
    low = text.lower()
    return all(any(alt.lower() in low for alt in group) for group in groups)


def judge(question: dict, answer: str) -> bool:
    if answer.startswith("[ERROR]"):
        return False
    if question["type"] == "negative":
        # 从未提过的事：答"没说过/不知道"才算对（顺带提到相关旧事实不算错）
        return any(w in answer for w in DENIALS)
    return contains_all(answer, question["all"])


def build_cases(data: dict) -> list[tuple[dict, list[dict]]]:
    # 把无关的长问答均匀穿插进历史，模拟真实对话里夹杂的其他话题
    history, filler = [], list(data.get("filler", []))
    step = max(2, (len(data["history"]) // max(len(filler), 1)) // 2 * 2)
    for i in range(0, len(data["history"]), step):
        history += data["history"][i:i + step]
        if filler:
            history += filler.pop(0)
    cases = []
    for q in data["questions"]:
        turns = history + q.get("recent", data["default_recent"]) + [["user", q["q"]]]
        messages = [{"role": "system", "content": data["system"]}]
        messages += [{"role": r, "content": c} for r, c in turns]
        cases.append((q, messages))
    return cases


def compress_all(cases, embed_url: str, overrides: dict, distractors: bool = False) -> list[list[dict]]:
    """用 api_server._compress_history（临时库 + 真实嵌入）压缩每个用例。"""
    root = tempfile.mkdtemp(prefix="hr_e2e_")
    os.environ["HR_STORAGE_ROOT"] = root
    os.environ["HR_EMBED_URL"] = embed_url
    import logging
    import api_server
    logging.disable(logging.WARNING)
    from hierarchical_retrieval import HierarchicalRetrieval
    cfg = api_server.config
    cfg.storage_root = root
    cfg.embedding_base_url = embed_url
    for k, v in overrides.items():
        setattr(cfg, k, v)
    api_server.hr = HierarchicalRetrieval(config=cfg)
    if distractors:
        # 库里预置无关的旧对话，模拟"不是空库"的真实情况
        # （c07 也有一只叫团子的猫，会与本评测的事实冲突，排除）
        extra = json.loads((Path(__file__).parent / "eval_recall_data.json").read_text(encoding="utf-8"))
        for conv in extra["conversations"]:
            if conv["id"] != "c07":
                api_server.hr.ingest(conv["text"], {"source": "distractor"})
    try:
        out = []
        for _q, messages in cases:
            msgs = [api_server.ChatMessage(**m) for m in messages]
            out.append([{"role": m.role, "content": m.content}
                        for m in api_server._compress_history(msgs)])
        return out
    finally:
        api_server.hr = None
        shutil.rmtree(root, ignore_errors=True)


def summarize(name: str, cases, prompts, answers=None) -> dict:
    rows = []
    for i, (q, _m) in enumerate(cases):
        text = "\n".join(m["content"] for m in prompts[i][:-1])   # 不含问题本身
        row = {"id": q["id"], "type": q["type"],
               "evidence": contains_all(text, q["all"]) if q["all"] else None}
        if answers is not None:
            row["ok"] = judge(q, answers[i])
            row["answer"] = answers[i][:160]
        rows.append(row)
    pos = [r for r in rows if r["evidence"] is not None]
    out = {
        "condition": name,
        "chars": round(sum(sum(len(m["content"]) for m in p) for p in prompts) / len(prompts)),
        "evidence": f"{sum(r['evidence'] for r in pos)}/{len(pos)}",
        "evidence_missing": [r["id"] for r in pos if not r["evidence"]],
    }
    if answers is not None:
        out["answer"] = f"{sum(r['ok'] for r in rows)}/{len(rows)}"
        by_type = {}
        for r in rows:
            ok, n = by_type.get(r["type"], (0, 0))
            by_type[r["type"]] = (ok + r["ok"], n + 1)
        out["by_type"] = {t: f"{ok}/{n}" for t, (ok, n) in by_type.items()}
        out["wrong"] = {r["id"]: r["answer"] for r in rows if not r["ok"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-url", default="http://127.0.0.1:8101")
    ap.add_argument("--models", default="", help="逗号分隔的 forge modelLabel，如 deepseekflash,gpt,mimo")
    ap.add_argument("--full", action="store_true", help="另跑不压缩的完整历史作为上限")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖 HConfig 字段（JSON 值），可重复")
    ap.add_argument("--distractors", action="store_true", help="库里预置 15 段无关旧对话")
    ap.add_argument("--dump", help="把压缩后的 prompt 写到该 JSON 文件")
    args = ap.parse_args()

    data = json.loads((Path(__file__).parent / "eval_e2e_data.json").read_text(encoding="utf-8"))
    cases = build_cases(data)
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    conditions = {"proxy": compress_all(cases, args.embed_url, overrides, args.distractors)}
    if args.full:
        conditions["full"] = [m for _q, m in cases]
    if args.dump:
        Path(args.dump).write_text(json.dumps(
            {q["id"]: p for (q, _m), p in zip(cases, conditions["proxy"])},
            ensure_ascii=False, indent=1), encoding="utf-8")

    report = []
    labels = [m for m in args.models.split(",") if m]
    if not labels:
        report = [summarize(name, cases, prompts) for name, prompts in conditions.items()]
    for label in labels:
        provider = load_provider(label)
        for name, prompts in conditions.items():
            with ThreadPoolExecutor(max_workers=4) as ex:
                answers = list(ex.map(lambda p: chat(provider, p), prompts))
            row = summarize(name, cases, prompts, answers)
            row["model"] = provider["model"]
            report.append(row)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
