from hierarchical_retrieval.core.chapter_history import chapter_history


def event(kind, user, assistant=None):
    return {"kind": kind, "user": user, "assistant": assistant}


def test_budgeted_edition_chain_restores_only_missing_instructions_in_time_order():
    source = "第九章 灯下\n她把书放回窗边。"
    events = [
        event("source", source),
        event("turn", "第九章以前有几本书？", "一册。"),
        event("turn", "第九章，按小说原文，她做了什么？", "把书放回窗边。"),
        event("note", "修订一：改为合上手稿。"),
        event("turn", "第九章，切到另一版，两项呢？", "合上手稿。"),
        event("note", "修订二：改为翻开手稿。"),
        event("turn", "第九章，沿用上次的版本呢？", "翻开手稿。"),
        event("turn", "第九章，换到另一版呢？", "把书放回窗边。"),
    ]

    def build(q, *, enabled=True, budget=1000, history=events):
        return chapter_history("9", history, [], q, budget, qa_window=2, recover_edition=enabled)

    q = "回到第九章，照上一轮选的那版，两项是什么？"
    base, recovered = build(q, enabled=False), build(q)
    texts = [m["content"] for m in recovered["messages"]]
    assert texts == [
        source,
        events[2]["user"],
        events[2]["assistant"],
        events[3]["user"],
        events[4]["user"],
        events[5]["user"],
        events[6]["user"],
        events[6]["assistant"],
        events[7]["user"],
        events[7]["assistant"],
        q,
    ]
    assert "合上手稿。" not in texts and events[1]["user"] not in texts
    assert (
        recovered["cache_breakpoints"] == base["cache_breakpoints"]
        and recovered["messages"][0] == base["messages"][0]
    )
    assert build(q, budget=recovered["chars"] - 1)["messages"] == base["messages"]
    for standalone in ["第九章按最新稿，动作呢？", "第九章原文和工作稿各怎么写？", "第九章人物是谁？"]:
        assert build(standalone) == build(standalone, enabled=False)
    for anchor in ["第八章，按原文答？", "第九章，原文与工作稿的差别？", "第九章她穿什么颜色？"]:
        alternate = [*events[:2], event("turn", anchor, "答。"), *events[3:]]
        assert build(q, history=alternate) == build(q, history=alternate, enabled=False)
    assert build(q, history=events[:5]) == build(q, enabled=False, history=events[:5])
