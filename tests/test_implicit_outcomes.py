from hierarchical_retrieval.core.branch_context import build_branch_context
from hierarchical_retrieval.core.topic_workspace import TopicWorkspaceStore


def test_scene_then_edition_survives_aside_restart_and_never_uses_the_aside_as_selection(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HR_IMPLICIT_FOLLOWUPS", "1")
    store = TopicWorkspaceStore(tmp_path)
    for name, chapter, scene in (
        ("花园", 18, "伸手抚摸栏杆，推开了一扇门"),
        ("小楼", 21, "左顾右盼之后，推开了屏风"),
    ):
        store.attach_source(
            "w", name, f"第{chapter}章 一刻\n" + ("窗外一片宁静。" * 45) + "\n" + scene, source="读书会"
        )
        store.add_note(
            "w",
            name,
            f"关于第{chapter}章的更正：“{scene[:6]}那一幕，推的对象和结果”这两项依次改成“木柜”和“发出轻响”。",
        )
    store.open_topic("w", "小楼")

    def ask(text, **kwargs):
        return store.route("w", text, **kwargs)

    first = ask("两段我记混了。那他到底推开了没有？", request_id="a")
    assert first["status"] == "clarify" and len(first["candidates"]) == 2
    assert ask("两段我记混了。那他到底推开了没有？", request_id="a") == first
    picked = ask("就是伸手抚摸栏杆那段。")
    assert picked["candidates"] == ["花园"] and "后来改过" in picked["clarification"]
    store.add_note("w", "小楼", "补充排练时间。")
    route = TopicWorkspaceStore(tmp_path).route("w", "用我们后来改过的那份。")
    assert route["topic"] == "花园" and route["reading"]["focus"]["edition"] == "latest"
    assert "发出轻响" in route["reading"]["evidence"] and len(route["reading"]["evidence_ids"]) == 1
    store.append_turn(
        "w",
        "花园",
        "用我们后来改过的那份。",
        "收到",
        expected_revision=route["reading"]["branch_revision"],
        reading_focus=route["reading"]["focus"],
    )
    assert ask("那他到底推开了没有？")["reading"]["focus"] == route["reading"]["focus"]
    # An ordinary user correction supersedes the note, including when no later note exists.
    store.append_turn(
        "w", "花园", "把木柜改为门帘。", "收到", expected_revision=store.snapshot("w", "花园")["revision"]
    )
    new = ask("回到伸手抚摸栏杆那段，照后来改过的讲。他推开了吗？")
    assert new["topic"] == "花园" and not new["reading"]["evidence"]
    assert not any(t["topic"].endswith("那段") for t in store.list_topics("w")["topics"])


def test_shared_clue_does_not_exclude_branch_where_it_occurs_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("HR_IMPLICIT_FOLLOWUPS", "1")
    s = TopicWorkspaceStore(tmp_path)
    for n in (1, 2):
        s.attach_source(
            "w",
            str(n),
            f"第{n}章 夜\n" + ("微风吹过。" * 65) + "\n" + "站到窗户前面，推门。\n" * n,
            source="书",
        )
    route = s.route("w", "回到站到窗户前面那段，他推开了吗？")
    assert route["status"] == "clarify" and set(route["candidates"]) == {"1", "2"}


def test_compact_evidence_cannot_renew_an_omitted_opening():
    text = "第一批资料。" * 60
    args = dict(
        system=[],
        topic="记录",
        messages=[{"role": "user", "content": text}, {"role": "assistant", "content": "收到"}],
        question="他打开了吗？",
        char_budget=4500,
    )
    original = build_branch_context(**args)
    assert original["opening_chars"] and not original["cache_breakpoints"]
    built = build_branch_context(**args, reading=dict(evidence="只听到一声响。", evidence_ids=[1], suffix=""))
    assert built["recalled"] == [1] and built["wire_user"] == built["messages"][-1]["content"]
    assert built["opening_chars"] == built["prefix_blocks"] == 0


def test_change_scene_while_clarifying_edition_and_reject_repeated_clue(tmp_path, monkeypatch):
    monkeypatch.setenv("HR_IMPLICIT_FOLLOWUPS", "1")
    store = TopicWorkspaceStore(tmp_path)
    scenes = [
        "他伸手拂去灰尘，将门推开。",
        "他快步穿过大厅，将窗推开。",
        "他看了看门锁后推门，她看了看门锁后关门。",
    ]
    for chapter, scene in enumerate(scenes, 1):
        topic = f"记录{chapter}"
        store.attach_source(
            "w", topic, f"第{chapter}章 阅读\n" + "晚风渐渐停了。" * 50 + "\n" + scene, source="书"
        )
        store.add_note("w", topic, "排演时已经修订。")
    store.route("w", "两段我记混了。那他到底推开了没有？")
    picked = store.route("w", "就是伸手拂去灰尘那段。")
    assert picked["candidates"] == ["记录1"]
    # Reader corrects the scene in response to the edition question.
    corrected = store.route("w", "我指的是快步穿过大厅那幕，照最初读到的说。")
    assert corrected["topic"] == "记录2" and corrected["reading"]["evidence"] == scenes[1]
    repeated = store.route("w", "回到看了看门锁那段，照最初读到的说。他推开了吗？")
    assert repeated["status"] == "clarify"


def test_verified_evidence_keeps_warm_span_without_search_and_preserves_extra_material():
    chapter = "风从窗前经过。\n" * 260
    messages = [
        {"role": "user", "content": chapter},
        {"role": "assistant", "content": "已读"},
        {"role": "user", "content": "旁线补记。"},
    ]
    args = dict(
        system=[],
        topic="记录",
        messages=messages,
        question="他推开了吗？",
        char_budget=4000,
        reading=dict(evidence="推后只听到两声响。", evidence_ids=[1], suffix=""),
    )

    def unused(*args):
        raise AssertionError("Grounded answer should not repeat retrieval")

    for count in (0, 1, 2):
        out = build_branch_context(**args, prefix_blocks=count, search=unused)
        assert bool(out["cache_breakpoints"]) == bool(count)
        assert out["cache_evidence_prefix_only"] == bool(count)
        assert out["recalled"] == [1] and "两声响" in out["messages"][-1]["content"]
        assert out["chars"] == sum(len(m["content"]) for m in out["messages"]) <= 4000
        assert all(i < len(out["messages"]) - 1 for i in out["cache_breakpoints"])
    out = build_branch_context(**args, prefix_blocks=0, evidence="另附访谈中对该动作的记录。")
    assert "另附访谈" in out["wire_user"] and not out.get("cache_evidence_prefix_only")


def test_pronoun_source_quotes_antecedents_in_same_chapter_and_retains_pending_edition(tmp_path, monkeypatch):
    monkeypatch.setenv("HR_IMPLICIT_FOLLOWUPS", "1")
    s = TopicWorkspaceStore(tmp_path)
    paragraphs = ["阿青和小林站起来。", "老李指着门外。", "他推开二人，快步离开大厅。"]
    s.attach_source(
        "w", "读书", "第18章 晚上\n" + "窗外有雨。" * 65 + "\n" + "\n\n".join(paragraphs), source="书"
    )
    s.add_note("w", "读书", "排演时已经更正。")
    picked = s.route("w", "回到快步离开大厅那段，那他推开了吗？")
    assert picked["status"] == "clarify" and "最初读到" in picked["clarification"]
    generic = s.route("w", "就是他推东西的那一下。")
    assert "最初读到" in generic["clarification"] and generic["candidates"] == ["读书"]
    resolved = s.route("w", "照最初读到的讲。")
    assert resolved["reading"]["evidence"] == "\n".join(paragraphs)
    assert resolved["reading"]["evidence_ids"] == [1]
    s.append_turn(
        "w",
        "读书",
        "照最初读到的讲。",
        "是阿青和小林。",
        expected_revision=resolved["reading"]["branch_revision"],
        reading_focus=resolved["reading"]["focus"],
    )
    s.add_note(
        "w", "读书", "第18章第2次更正：“快步离开大厅那一幕，推的对象和结果”这两项依次改成“木板”和“轻响”。"
    )
    switched = s.route("w", "还是那个人，这次照后来改过的讲。他推开了吗？")["reading"]
    assert switched["focus"]["edition"] == "latest" and len(switched["evidence_ids"]) == 2
    assert "阿青和小林" in switched["evidence"].split("此幕当前更正：")[0]
    assert "轻响" in switched["evidence"].split("此幕当前更正：")[1]


def test_changed_revision_only_replaces_the_uncached_tail():
    from hierarchical_retrieval.core.branch_context import chapter_prefix_layout

    chapter = "第91章 约定\n" + ("树下的众人都望向小屋。\n" * 220)
    anchor = "小林停在门口。\n他推开门走了进去。"
    args = dict(system=[], topic="小屋", char_budget=5500, compact_lookup=True, chapter_prefix=True)
    assert chapter_prefix_layout([], chapter)
    assert not chapter_prefix_layout([], "一段没有章节的对话。" * 300)
    intake = build_branch_context(**args, messages=[], question=chapter)
    args["messages"] = [{"role": "user", "content": chapter}, {"role": "assistant", "content": "已读"}]
    base = dict(evidence_ids=[1, 3], trim_opening=True, identity_source=anchor, suffix="")

    def ask(note, blocks=2):
        return build_branch_context(
            **args,
            question="那句里的他指谁，推开了吗？",
            prefix_blocks=blocks,
            reading=dict(base, evidence=anchor + "\n" + note, selected_note=note),
        )

    first, second = ask("先改为木板。"), ask("再改为铁门。")
    cut = first["cache_breakpoints"][-1] + 1
    assert first["prefix_blocks"] == 1
    assert first["messages"][:cut] == second["messages"][:cut]
    assert (
        intake["messages"][intake["cache_breakpoints"][0]] == first["messages"][first["cache_breakpoints"][0]]
    )
    assert first["messages"][cut - 1]["content"].endswith(anchor)
    assert "木板" not in str(second["messages"]) and "铁门" in second["messages"][-1]["content"]
    assert anchor in second["wire_user"] and second["chars"] == sum(
        len(m["content"]) for m in second["messages"]
    )
    cold = ask("再改为铁门。", 0)
    assert not cold["cache_breakpoints"] and anchor in cold["messages"][-1]["content"]


def test_clarification_can_also_ask_who_performed_the_action(tmp_path, monkeypatch):
    monkeypatch.setenv("HR_IMPLICIT_FOLLOWUPS", "1")
    store = TopicWorkspaceStore(tmp_path)
    text = "第32章 交接\n" + "天色暗了。" * 80 + "\n小方扶着雨伞。\n他推开木门，随后站在台阶上。"
    store.attach_source("w", "花房", text, source="书")
    store.add_note(
        "w", "花房", "第32章更正：“站在台阶上那一幕，推的对象和结果”这两项依次改成“屏风”和“发出轻响”。"
    )
    route = store.route("w", "回到站在台阶上那段，他推开了吗？")
    assert route["status"] == "clarify"
    follow = store.route("w", "用后来改过的那份。顺带说清推门的他指谁。")["reading"]
    assert "指谁" in follow["grounded_question"] and "小方" in follow["identity_source"]
    assert "发出轻响" in follow["selected_note"] and follow["focus"]["edition"] == "latest"
