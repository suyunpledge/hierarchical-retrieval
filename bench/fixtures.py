"""Author unseen chapter questions without inspecting individual test results."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from pathlib import Path

import requests
from budget import Budget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench/legacy"))
from eval_e2e import load_provider  # noqa: E402
from eval_idle_novel_cache import chapters, record  # noqa: E402


def deepseek(messages, budget, max_output):
    provider = load_provider("deepseekflash")
    if provider["base"].rstrip("/") not in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
        raise ValueError("Unexpected DeepSeek endpoint")
    body = dict(
        model="deepseek-flash",
        messages=messages,
        max_tokens=max_output,
        temperature=0,
        thinking={"type": "disabled"},
        response_format={"type": "json_object"},
    )
    upper = len(json.dumps(body, ensure_ascii=False).encode()) + 512
    key = budget.reserve_units(3 * upper + 12 * max_output)
    res = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": "Bearer " + provider["key"]},
        json=body,
        timeout=90,
    )
    if not res.ok:
        if res.status_code in (400, 401, 402, 403, 404, 429):
            budget.record(key, 0, "rejected-" + key)
        if res.status_code == 402:
            raise PermissionError("DeepSeek balance exhausted (HTTP 402)")
        raise RuntimeError(f"DeepSeek HTTP {res.status_code}")
    data = res.json()
    u = data["usage"]
    # Reserve/account at peak rates even off-peak, rounding cached units upward.
    hit = u.get("prompt_cache_hit_tokens", 0)
    budget.record(
        key, 3 * (u["prompt_tokens"] - hit) + (hit * 6 + 99) // 100 + 12 * u["completion_tokens"], data["id"]
    )
    return json.loads(data["choices"][0]["message"]["content"])


def prepare(folder, book, *, split=None, cap=25):
    if split is None:
        split = json.loads((ROOT / "bench/splits.json").read_text("utf8"))
    if hashlib.sha256(book.read_text("utf-8-sig")[:2_000_000].encode()).hexdigest() != split["corpus_sha256"]:
        raise ValueError("wrong book")
    ids = split["test"]
    docs = chapters(book, ids)
    out = folder / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    distractors = split.get("distractors") or random.Random(split["seed"] + 1).sample(split["dev"], len(ids))
    budget = Budget(folder / "budget.sqlite", cap)
    replacements = {
        "人名": ["程梧", "孟薇", "卢菁", "顾梨", "陈穗", "林樾", "赵莳", "苏禾"],
        "部位": ["左肩胛", "右脚踝", "左手腕内侧", "右手掌心", "右耳垂", "左小臂", "右手肘", "左眉梢"],
        "物品": ["铜哨", "木尺", "纸灯笼", "红绳结", "青瓷碗", "油纸伞", "竹书签", "蓝布袋"],
        "动作": [
            "向左挥手",
            "双手抱膝",
            "转身鞠躬",
            "抬头望窗",
            "起身拍掌",
            "低头摆手",
            "俯身拾笔",
            "转头闭眼",
        ],
        "位置": ["石阶旁", "南窗外", "桥头边", "走廊尽头", "木柜底下", "红门后面", "楼梯转角", "北墙边上"],
        "数量": ["十九", "三十七", "四十八", "七十六", "二十二", "五十四", "六十三", "八十一"],
        "其他": [
            "青石标记",
            "秋叶标记",
            "春水标记",
            "长风标记",
            "竹影标记",
            "湖光标记",
            "松涛标记",
            "远山标记",
        ],
    }

    def one(item):
        i, ch = item
        path = out / f"{ch}.json"
        if path.exists():
            return
        paragraphs = [t.strip() for t in docs[ch].splitlines() if 18 <= len(t.strip()) <= 160]
        rng = random.Random(i + 804)
        rng.shuffle(paragraphs)
        paragraphs = paragraphs[:28]
        instruction = """从编号片段选一个自然场景，设计两种问法询问同样两条事实。只输出 JSON 对象。index 是所选段编号(int)。gold 是两个字符串数组，每组第一项为该段连续出现的简短答案，可有同义答案。question 与 alternate 用不同措辞，给出定位线索，两者都按相同顺序问两项，不要泄露答案。types 是所问两项的类别，从人名、部位、物品、动作、位置、数量、其他选择。优先选择能清楚定位的动作与人物或物品。"""
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "\n".join(f"[{j}] {t}" for j, t in enumerate(paragraphs))},
        ]
        for attempt in range(8):
            row = deepseek(messages, budget, 650)
            try:
                j = row["index"]
                assert isinstance(j, int) and 0 <= j < len(paragraphs)
                w = paragraphs[j]
                g = row["gold"]
                q = row["question"].rstrip("？?")
                a = row["alternate"].rstrip("？?")
                assert 4 <= len(q) <= 150 and 4 <= len(a) <= 150 and q != a
                assert len(g) == 2 and all(
                    isinstance(group, list)
                    and group
                    and any(isinstance(t, str) and t and t in w for t in group)
                    for group in g
                )
                g = [[t.strip() for t in group if isinstance(t, str) and t.strip()] for group in g]
                assert all(not any(len(t) > 1 and t in q + a for t in group) for group in g)
                available = [v for key in row["types"] for v in [replacements[key]]]
                assert len(available) == 2
                choices = []
                for group in available:
                    candidates = [
                        v
                        for v in group
                        if v not in docs[ch] and v not in q + a and not any(v in other for other in choices)
                    ]
                    choices.append(rng.sample(candidates, 4))
                rev = list(map(list, zip(*choices)))
                record(
                    dict(case=[q, a, g, rev, distractors[i]], witness=w, detail="两项都请按这处原句核实"),
                    path,
                )
                return
            except (KeyError, ValueError, TypeError, AssertionError, IndexError):
                messages = messages[:2] + [
                    {"role": "assistant", "content": json.dumps(row, ensure_ascii=False)},
                    {
                        "role": "user",
                        "content": "换一个片段设计题目，确保 gold 两个数组各有一个答案逐字出现在编号对应原句中，问法不泄露答案。两种类别只能用上面所列名称。",
                    },
                ]
        raise ValueError("Could not validate a sealed fixture")

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(one, enumerate(ids)))
    print("Validated sealed fixtures:", len(ids))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("book", type=Path)
    a = p.parse_args()
    prepare(a.folder, a.book)
