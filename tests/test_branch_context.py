import pytest
from hierarchical_retrieval.core.branch_context import build_branch_context


def msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def build(history, question, **options):
    return build_branch_context(
        system=[msg("system", "请简短作答。")],
        topic="论文",
        messages=history,
        question=question,
        **options,
    )


def archive(first=None, count=20):
    # Keep the tested old record behind the pinned branch opening and outside its tail.
    h = [
        msg("user", "历史稿件的分档说明。" * 55),
        msg("assistant", "已记录。"),
        msg("user", first or "备案的取件暗号是雪貂三号。"),
        msg("assistant", "已记录。"),
    ]
    for i in range(count):
        h.extend(
            [
                msg("user", f"检查第{i}号图例的颜色与字号。"),
                msg("assistant", "已经对照底稿核实了长表格的边距。" * 3),
            ]
        )
    return h


def test_replayed_provider_prefix_survives_side_note_and_return_to_branch():
    original = "第三节还需要改哪段？"
    first = build([], original, evidence="草稿标记为只改结尾。", position="第三节结尾")
    history = [msg("user", original, wire_content=first["wire_user"]), msg("assistant", "更新最后一句。")]
    side_note = msg("user", "旁线补记：实验编号更正为 R-72。")
    second = build(history + [side_note], "继续吧。", position="第三节结尾")
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert second["wire_user"] == "继续吧。"
    assert second["messages"][-2] == side_note
    assert second["chars"] == sum(len(m["content"]) for m in second["messages"])
    assert history[0]["content"] == original


def test_checkpoint_retrieves_exact_old_quote_with_index_and_is_stable_across_appends():
    h = archive()
    calls = []

    def search(question, limit):
        calls.append(question)
        return [
            {"message_id": 3, "role": "user", "quote": "取件暗号是雪貂三号。"},
            {"message_id": 3, "role": "user", "quote": "另一个话题的错误暗号。"},
        ]

    q = "取件暗号是什么？"
    # Test a boundary where the next turn still fits this checkpoint.
    for offset in range(0, 8, 2):
        history = h[: len(h) - offset]
        initial = build(history, q, char_budget=820, search=search)
        follow = history + [msg("user", q, wire_content=initial["wire_user"]), msg("assistant", "雪貂三号。")]
        again = build(follow, "然后呢？", char_budget=820)
        if initial["checkpoint"] == again["checkpoint"]:
            break
    assert initial["checkpoint"] > 0 and initial["recalled"] == [3]
    assert "取件暗号是雪貂三号。" in initial["wire_user"] and "错误暗号" not in initial["wire_user"]
    assert initial["messages"] == again["messages"][: len(initial["messages"])]
    assert initial["chars"] <= 820 and again["chars"] <= 820
    assert calls and all(c == q for c in calls)


def test_later_user_correction_blocks_obsolete_tail_injection():
    h = archive("取件暗号为雪貂。")
    h.append(msg("user", "更正：取件暗号为白鹤。"))
    r = build(h, "取件暗号是什么？", char_budget=760)
    assert r["checkpoint"] > 0 and 3 not in r["recalled"]
    assert "白鹤" in "".join(m["content"] for m in r["messages"])
    old = build(h, "最早的取件暗号是什么？", char_budget=760)
    assert "雪貂" in old["wire_user"]


def test_current_question_and_system_are_preserved_for_oversized_history():
    q = "a" * 89
    r = build([msg("user", "b" * 10000), msg("assistant", "c" * 10000)], q, char_budget=100)
    assert r["messages"][-1]["content"] == q and r["chars"] <= 100
    with pytest.raises(ValueError):
        build([], q, char_budget=91)
    with pytest.raises(ValueError):
        build([msg("system", "domain data")], "短问")


def test_index_errors_propagate_and_short_warm_turns_skip_index():
    def broken(q, limit):
        raise RuntimeError("index unavailable")

    assert build([], "短问", search=broken)["checkpoint"] == 0
    with pytest.raises(RuntimeError, match="index unavailable"):
        build(archive(), "查取件暗号", char_budget=820, search=broken)


def test_different_excerpts_from_same_source_remain_retrievable():
    h = archive("登记地点在白楼。备用取件号为884。")
    previous = "原文#3 用户：登记地点在白楼。\n\n本轮问题：\n登记地点呢？"
    h.extend([msg("user", "登记地点呢？", wire_content=previous), msg("assistant", "白楼。")])

    def search(question, limit):
        return [{"message_id": 3, "role": "user", "quote": "备用取件号为884。"}]

    r = build(h, "备用取件号是什么？", char_budget=820, search=search)
    assert r["recalled"] == [3] and "884" in r["wire_user"]


def test_location_can_return_to_an_earlier_value():
    h = []
    for loc in ("摘要", "附录"):
        pack = build(h, "继续", position=loc)
        h += [msg("user", "继续", wire_content=pack["wire_user"]), msg("assistant", "已处理。")]
    assert "话题进度：摘要" in build(h, "回来后继续", position="摘要")["wire_user"]


def test_page_appends_keep_the_exact_provider_opening_across_multiple_folds():
    budget = 2600
    history = []
    packs = []
    for n in range(10):
        page = f"交接第{n}页。\n" + "\n".join(f"档号{n}-{j}：完好，留档。" for j in range(63))
        pack = build(history, page, char_budget=budget)
        packs.append(pack)
        history += [msg("user", page, wire_content=pack["wire_user"]), msg("assistant", f"第{n}页已记录。")]

    folded = [pack for pack in packs if pack["checkpoint"]]
    assert len({pack["checkpoint"] for pack in folded}) >= 3
    first = packs[0]["messages"]
    assert all(pack["messages"][: len(first)] == first for pack in folded)
    assert all(pack["messages"][len(first)] == history[1] for pack in folded)
    for pack in packs:
        assert pack["chars"] == sum(len(m["content"]) for m in pack["messages"]) <= budget


def test_oversized_opening_still_recalls_middle_and_corrections_follow_the_origin():
    original = "本轮巡查。" * 300 + "仓门卡号为M-8214。" + "当班登记。" * 300
    history = [msg("user", original), msg("assistant", "已归档。")]
    history += archive("内页记录。", count=12)
    history.append(msg("user", "更正：仓门卡号为M-7731。"))

    def search(query, limit):
        return [{"message_id": 1, "role": "user", "quote": "仓门卡号为M-8214。"}]

    current = build(history, "仓门卡号？", char_budget=900, search=search)
    old = build(history, "更正前原始仓门卡号？", char_budget=900, search=search)
    pinned = old["messages"][2]["content"]
    assert pinned == current["messages"][2]["content"] and original.startswith(pinned[:300])
    assert "M-8214" not in pinned and "M-8214" in old["wire_user"] and 1 in old["recalled"]
    assert "M-8214" not in current["wire_user"] and "M-7731" in str(current["messages"][3:])


def test_long_fixed_system_and_almost_full_question_take_precedence():
    system = [msg("developer", "S" * 800), msg("system", "T" * 119)]
    q = "Q" * 80
    pack = build_branch_context(system=system, topic="新建", messages=archive(), question=q, char_budget=999)
    assert pack["messages"] == [*system, msg("user", q)]
    assert pack["chars"] == 999


def test_changing_page_lengths_only_reuse_segments_sent_on_the_first_turn():
    budget = 3000
    original = "".join(f"正文第{i}项已登记并复核。\n" for i in range(140))
    first = build([], original, char_budget=budget)
    chunks = first["messages"][2:]
    assert 1 < len(chunks) <= 3 and "".join(m["content"] for m in chunks) == original
    h = [msg("user", original), msg("assistant", "已归档。")] + archive(count=60)
    for size in [300, 800, 1100, 1400, 1650]:
        page = ("新页。" * size)[:size]
        result = build(h, page, char_budget=budget)
        assert result["messages"][:3] == first["messages"][:3]
        got = []
        for m in result["messages"][2:]:
            if len(got) == len(chunks) or m != chunks[len(got)]:
                break
            got.append(m)
        assert got and got == chunks[: len(got)]
        assert result["wire_user"].endswith(page) and result["chars"] <= budget


def test_explicit_current_does_not_reinject_two_obsolete_full_field_matches():
    h = archive("回收台卡材质为旧铜，编号为TM-431。", count=9)
    h += [msg("user", "更正：回收台卡材质改成紫绒，编号改为TM-537。")]
    h += [msg("user", "验收时间待定。" * 20), msg("assistant", "收到。")] * 8
    h += [msg("user", "第二次更正：回收台卡材质改成蓝绒，编号改为BR-7823。")]
    h += [msg("user", "值班人员已完成日常巡查。" * 9), msg("assistant", "确认。")] * 3
    matches = [
        {"message_id": i + 1, "role": "user", "quote": m["content"]}
        for i, m in enumerate(h)
        if "回收台卡" in m["content"]
    ]

    def search(q, limit):
        return matches

    now = build(h, "回收台卡目前用的材质和编号呢？", char_budget=800, search=search)
    assert (
        "BR-7823" in str(now["messages"])
        and "TM-431" not in now["wire_user"]
        and "TM-537" not in now["wire_user"]
    )
    original = build(h, "回收台卡原始材质和原始编号呢？", char_budget=800, search=search)
    assert "TM-431" in str(original["messages"])
    h[matches[-1]["message_id"] - 1] = msg("user", "第二次更正：回收台卡材质改成蓝绒。")
    # A material-only correction does not supersede the former registration number.
    matches[-1]["quote"] = h[matches[-1]["message_id"] - 1]["content"]
    partial = build(h, "回收台卡当前用的材质和编号呢？", char_budget=800, search=search)
    assert "TM-537" in str(partial["messages"])


def test_labeled_question_after_long_paste_guides_recall_without_changing_input():
    h = archive("港口封存戳是SEA-420。", count=65) + [msg("user", "更正：港口封存戳改为SEA-820。")]
    asked = "你能从原始工作册找出港口封存戳吗？"
    page = "接货时检查过标签，仓管签字。" * 140 + "本轮问题：" + asked
    calls = []

    def search(q, limit):
        calls.append(q)
        return [{"message_id": 3, "role": "user", "quote": "港口封存戳是SEA-420。"}] if q == asked else []

    # Keep the target itself out of the pinned opening and outside the fallback scan.
    h[:2] = [msg("user", "值班表。" * 600), msg("assistant", "已读。")]
    pack = build(h, page, char_budget=3000, search=search)
    assert calls == [asked] and pack["wire_user"].endswith(page)
    assert 3 in pack["recalled"] and "SEA-420" in pack["wire_user"]


def test_breakpoints_match_only_actual_opening_spans_through_restart_and_folds():
    budget = 3300
    original = "".join(f"第{i:03d}项，了解工作环境后逐项检查表格。\n" for i in range(82))
    first = build([], original, char_budget=budget)
    opening = first["messages"][2:]
    assert first["cache_breakpoints"] and len(first["cache_breakpoints"]) <= 3
    assert all(first["messages"][i]["role"] == "user" for i in first["cache_breakpoints"])
    assert "".join(m["content"] for m in opening) == original
    entry = msg("user", original, wire_content=first["wire_user"])
    warm = build([entry, msg("assistant", "已归档。")], "下一步？", char_budget=budget)
    assert warm["cache_breakpoints"] == first["cache_breakpoints"]
    assert all(warm["messages"][i] in opening for i in warm["cache_breakpoints"])
    assert 2 in warm["cache_breakpoints"] and len(warm["cache_breakpoints"]) <= 3
    h = [entry, msg("assistant", "已归档。")] + archive(count=55)
    for size in [90, 900, 1700]:
        question = "检" * size
        resumed = build(h, question, char_budget=budget)
        points = resumed["cache_breakpoints"]
        assert points and len(points) <= len(opening) and points == sorted(set(points))
        assert all(resumed["messages"][i] == first["messages"][i] for i in points)
        assert len(resumed["messages"]) - 1 not in points
    assert build([msg("user", "短开场"), msg("assistant", "收到")], "继续")["cache_breakpoints"] == []
    assert build([msg("assistant", original)], "继续")["cache_breakpoints"] == []


def test_compact_prefix_keeps_neighboring_evidence_within_chapter():
    before = "第42章 留档\n" + "逐条登记当天清单。" * 200
    hit = "保管员开始第二次校验。"
    witness = "表盘上写着山岚。"
    doc = before + hit + "第一次先把清单铺开。第二次拿出了表盘。旁边两人各盖一个章。" + witness + "随后交班。"
    other = "第43章 重复校验\n" + "记录次日另一份清单。" * 180 + "表盘上写着珊瑚。"
    hist = [msg("user", doc), msg("assistant", "已读"), msg("user", other), msg("assistant", "已读")]

    def hits(q, n):
        return [
            {"message_id": 1, "role": "user", "quote": hit},
            {"message_id": 3, "role": "user", "quote": "表盘上写着珊瑚。"},
        ]

    pack = build(hist, "第42章第二次校验时表盘上写着什么？", char_budget=2600, search=hits, prefix_blocks=0)
    assert pack["opening_chars"] == 0 and witness in pack["wire_user"]
    assert "珊瑚" not in str(pack["messages"]) and pack["chars"] <= 2600
    comparison = build(
        hist, "对比第42章和下一章的校验，表盘分别写着什么？", char_budget=2600, search=hits, prefix_blocks=0
    )
    assert witness in str(comparison["messages"]) and "表盘上写着珊瑚。" in str(comparison["messages"])


def test_intake_receipt_sends_new_document_and_preserves_dependencies():
    hist = [msg("user", "首件工作单" * 210), msg("assistant", "已读")]
    doc = "新件工作单" * 230
    pure = "以下是新材料，读完仅回复已读。\n" + doc
    pack = build(hist, pure, allow_document_receipt=True, char_budget=4500)
    assert pack["messages"][-1]["content"] == pure and pack["wire_user"] == pure
    assert pack["opening_chars"] == 0 and pack["chars"] == sum(len(m["content"]) for m in pack["messages"])
    journal = "阅读值班日志，第2页，确认留档即可。\n" + doc
    assert build(hist, journal, allow_document_receipt=True, char_budget=4500)["opening_chars"] == 0
    for instruction in ["请核对上一份工作单，再仅回复已读。", "以下是引用说明，引用中写着“仅回复已读”。"]:
        assert "首件工作单" in str(
            build(hist, instruction + "\n" + doc, allow_document_receipt=True, char_budget=4500)["messages"]
        )
    assert "首件工作单" in str(
        build(hist, pure, evidence="核对上次的尺寸", allow_document_receipt=True, char_budget=4500)[
            "messages"
        ]
    )


def test_versioned_lookup_recovers_facts_when_top_matches_are_old_questions():
    original = (
        "第83章 港务\n" + "受理时先查主单。" * 110 + "货箱封签是青桐，签收员为林羽。" + "当日放行。" * 150
    )
    old = "第83章工作稿更正：货箱封签改为柏木，签收员改为沈舟。"
    recent = "第83章工作稿再次更正：货箱封签改为白石，签收员改为江竹。"
    other = "第84章 次日\n" + "另一份入港登记。" * 130
    q = "第83章。按小说原始正文，货箱封签和签收员是什么？"
    h = [
        msg("user", original),
        msg("assistant", "已读"),
        msg("user", other),
        msg("assistant", "已读"),
        msg("user", old),
        msg("user", recent),
    ]
    for _ in range(22):
        h += [msg("user", q), msg("assistant", "收到，晚点核对。")]

    def hits(query, n):
        return [
            {"message_id": i + 1, "role": "user", "quote": q} for i, m in enumerate(h) if m["content"] == q
        ][-n:]

    opts = dict(char_budget=2200, search=hits, prefix_blocks=0, compact_lookup=True)
    first = build(h, q, **opts)
    assert "青桐" in str(first["messages"]) and "林羽" in str(first["messages"])
    second = build(h, "第83章。按现在生效的工作稿，货箱封签和签收员是什么？", **opts)
    assert recent in str(second["messages"])


def test_single_provisional_block_keeps_appending_across_corrections():
    original = "第42章，开头存档。" * 600
    hist = [msg("user", original), msg("assistant", "已归档。")]
    for i in range(4):
        hist.extend([msg("user", f"历史勘误{i}。" * 80), msg("assistant", "已更新。")])
    floor = len(hist)
    opts = dict(char_budget=5500, prefix_blocks=1, tail_floor=floor, compact_lookup=True)
    prompt = []
    for i in range(5):
        hist.extend([msg("user", f"补充第{i}次记录。" * 35), msg("assistant", f"生效编号K{i}。")])
        pack = build(hist, "接着说，当前编号是什么？", **opts)
        assert pack["prefix_blocks"] == 1 and pack["chars"] <= 5500
        assert pack["messages"][: len(prompt)] == prompt
        assert hist[-2] in pack["messages"] and hist[-1] in pack["messages"]
        prompt = pack["messages"][:-1]


def test_exact_current_replacement_does_not_append_the_superseded_source():
    field = "入场和出场各收多少"
    question = "第42章。按最后一次更正，" + field + "？"
    source = "第42章。入场两颗，出场十颗。" * 90
    recent = f"第42章更正：“{field}”这两项依次改成“三颗”和“十二颗”。"
    h = [msg("user", source), msg("assistant", "已读。"), msg("user", recent)]
    opts = dict(char_budget=1800, prefix_blocks=0, compact_lookup=True)
    current = build(h, question, **opts)
    assert current["recalled"] == [3] and recent in str(current["messages"])
    assert recent not in current["wire_user"] and "三颗”和“十二颗" in current["wire_user"]
    assert "出场十颗" not in current["wire_user"]
    original = build(h, "第42章。按小说原始正文，" + field + "？", **opts)
    assert "出场十颗" in str(original["messages"])

    # A partial amendment, another requested field, or a subsequent revocation
    # requires ordinary retrieval. A note for a different chapter does not.
    from hierarchical_retrieval.core.branch_context import _exact_current_evidence

    other = msg("user", "第43章更正：三颗改为四颗。")
    assert _exact_current_evidence(h + [other], question, "42", 400)[1] == [3]
    for extra in (msg("user", "刚才的更正撤销。"), msg("user", f"第42章：“{field}”仅第一项改为五颗。")):
        assert _exact_current_evidence(h + [extra], question, "42", 400) is None
    assert _exact_current_evidence(h, question[:-1] + "，再说说规则是谁定的？", "42", 400) is None
    assert _exact_current_evidence(h, question, "42", 15) is None


def test_original_lookup_expands_only_a_unique_same_chapter_field_name():
    from hierarchical_retrieval.core.branch_context import _source_queries

    q = "第42章。切回小说原文，参加茶馆游戏，进场费和退场费各是多少颗道？"
    field = "茶馆游戏规则里入场和出场分别要几颗道"
    note = f"第42章更正：“{field}”这两项依次改成“八十”和“九十”。"
    hist = [msg("user", "第42章。茶馆向来按季收费。" * 100), msg("user", note)]
    expanded, _ = _source_queries(hist, q, "42")
    assert field in expanded and "八十" not in expanded and "九十" not in expanded
    assert _source_queries(hist, q, "43") == (q, None)
    competitor = msg("user", note.replace("入场和出场", "茶点和夜宵"))
    assert _source_queries(hist + [competitor], q, "42") == (q, None)
    assert _source_queries(hist, "第42章。切回小说原文，掌柜叫什么？", "42")[0].endswith("掌柜叫什么？")


def test_source_lookup_reopens_old_paste_after_its_index_quote_and_recent_tail_miss():
    heading = "第72章 旧仓库\n"
    opening = "韩秋生靠着门想了很久。" * 45
    witness = "韩秋生坐在台阶上，伸手从信封里抽出了一张借条。"
    page = heading + opening + "\n" + witness + "\n" + "来人对了一遍台账。" * 50
    field = "韩秋生坐在台阶上时从什么地方抽出了什么物件"
    revision = f"第72章更正：“{field}”这两项依次改成“柜门”和“报纸”。"
    history = [
        msg("user", "先看另一份资料。"),
        msg("assistant", "好。"),
        msg("user", page),
        msg("assistant", "已读。"),
    ]
    for i in range(40):
        history.extend([msg("user", f"整理第{i}栏。"), msg("assistant", "整理完毕。")])
    history.extend([msg("user", revision), msg("user", "继续整理。" * 150), msg("assistant", "已完成。")])

    def unrelated(query, limit):
        return [{"message_id": 1, "role": "user", "quote": "先看另一份资料。"}]

    question = "换回第72章。按小说原始正文，韩秋生落座后伸向哪里，又取出了哪件东西？"
    pack = build(history, question, char_budget=2100, prefix_blocks=0, compact_lookup=True, search=unrelated)
    assert witness in pack["wire_user"] and 3 in pack["recalled"]
    assert "报纸" not in pack["wire_user"] and pack["chars"] <= 2100


def test_source_ranking_keeps_scene_constraints_before_the_version_marker():
    from hierarchical_retrieval.core.branch_context import _source_excerpt

    archive = "第87章 两次交接\n" + "小宋在北桥递给同伴一只篮子。\n" + "小宋在码头递给同伴一个信封。"
    for location, witness in (("北桥", "篮子"), ("码头", "信封")):
        q = f"第87章在{location}的交接，按小说原文，小宋递给同伴什么？"
        assert witness in _source_excerpt(archive, q, 24)


def test_weak_paraphrase_keeps_primary_query_and_uses_only_one_verified_original_field():
    from hierarchical_retrieval.core.branch_context import _source_queries

    field = "凌伯从哪儿掏出来什么"
    question = "第79章。按小说原文，凌伯查看的饰物是什么？"
    amendment = f"第79章更正：“{field}”改为“白玉佩”。"
    history = [msg("user", amendment)]
    assert _source_queries(history, question, "79") == (question, field)
    history.append(msg("user", "第79章更正：“凌伯后来又去了哪里”改为“西站”。"))
    assert _source_queries(history, question, "79") == (question, None)


def test_rephrased_draft_turn_after_an_incorrect_reply_anchors_latest_scoped_note():
    field = "她选的两枚印章上刻的各是什么，保持顺序"
    q = "第51章。按最新工作稿，盒内剩下的两枚章印文字依次是什么？"
    revision = f"第51章更正：“{field}”这两项依次改成“川北”和“淮东”。"
    h = archive(count=12)
    h += [
        msg("user", revision.replace("川北", "西林").replace("淮东", "南山")),
        msg("user", revision),
        msg("user", q),
        msg("assistant", "西林和南山。"),
    ]
    opts = dict(char_budget=1900, prefix_blocks=0, compact_lookup=True)
    pack = build(h, q, **opts)
    assert "川北" in pack["wire_user"] and "淮东" in pack["wire_user"] and field in pack["wire_user"]
    assert "西林" not in pack["wire_user"] and pack["chars"] <= 1900

    from hierarchical_retrieval.core.branch_context import _exact_current_evidence

    assert _exact_current_evidence(h + [msg("user", "撤销刚才的工作稿。")], q, "51", 400) is None
    assert (
        _exact_current_evidence(
            h, q.replace("文字依次是什么", "文字依次是什么，另外盒子当时是谁拿的"), "51", 400
        )
        is None
    )
    assert _exact_current_evidence(h, q, "52", 400) is None


def test_original_turn_reanchors_visible_source_after_an_assistant_added_a_side():
    witness = "为示范接听状态，小洛伸手按着耳边。"
    document = (
        "第81章 演练记录\n"
        + "大家核查设备。" * 85
        + "\n小洛一开始曾用右手捂住右耳。\n"
        + witness
        + "\n"
        + "大家重新分组。" * 35
    )
    q = "第81章。按小说原文，小洛示范接听状态时按着什么位置？"
    history = [
        msg("user", document),
        msg("assistant", "已读。"),
        msg("user", "别组点名。" * 90),
        msg("assistant", "收到。"),
        msg("user", q, wire_content=witness + "\n" + q),
        msg("assistant", "右耳。"),
    ]
    pack = build(history, q, char_budget=2200, prefix_blocks=0, compact_lookup=True)
    assert 1 in pack["recalled"] and witness in pack["wire_user"] and "右耳" not in pack["wire_user"]
    assert "未交代" in pack["wire_user"] and pack["chars"] <= 2200


def test_weak_working_copy_rephrase_links_both_descriptions_to_same_archived_objects():
    field = "她拿走的两枚印分别刻着什么，按顺序"
    page = "第51章 清点\n" + "逐项登记。" * 120 + "\n她把桌子上仅剩的两枚印拿了过来，分别是「北」和「南」。"
    latest = f"第51章工作稿更正：“{field}”这两项依次改成“山影”和“泽光”。"
    hist = [msg("user", page), msg("assistant", "已读。"), msg("user", latest)] + archive(count=4)
    hist.append(msg("assistant", "北和南。"))
    question = "第51章。按最新工作稿，桌上剩下的两句印文依次是什么？"
    pack = build(hist, question, char_budget=2200, prefix_blocks=0, compact_lookup=True)
    assert "桌子上仅剩的两枚印拿了过来" in pack["wire_user"] and "山影" in pack["wire_user"]
    assert "「北」" not in pack["wire_user"] and len(pack["messages"]) == 3 and pack["recalled"] == [1, 3]


def test_single_action_verb_beats_generic_addressee():
    from hierarchical_retrieval.core.branch_context import _source_sentence_scores

    doc = "第77章 两次动作\n小陈赶快扶住她，跑到对方身边。\n小陈伸手拍了拍王琪的肩膀。"
    scores = _source_sentence_scores(doc, "第77章。按小说原文，小陈拍了谁，手落在对方哪个部位？")
    _, left, right = max(scores)
    assert "王琪的肩膀" in doc[left:right]


def test_two_remarks_in_one_scene_remain_connected_for_a_draft_question():
    from hierarchical_retrieval.core.branch_context import _field_scene

    quote = "顾宁先打了个哈欠，指着窗框问道：“昨天还疼吗？夜里睡好了吗？”"
    history = [msg("user", "第61章 探访\n" + "大家排队等叫号。" * 75 + "\n" + quote)]
    scene = _field_scene(history, "61", "打完哈欠问睡眠时，他指着哪里", "顾宁问昨天还疼吗的时候指着哪里")
    assert scene == (1, quote)


def test_field_and_rephrased_question_can_land_on_different_sentences_in_one_paragraph():
    from hierarchical_retrieval.core.branch_context import _source_excerpt

    scene = "“你先收好。”周师傅拍了拍阿唐的手，“出去再看。”"
    decoy = "那会儿交接还很忙。"
    raw = "\n".join((decoy, scene, "赵老师在别处敲响了铃。"))
    question = "交接后让对方出去再看的那会儿，他碰的是谁？"
    field = "周师傅提醒先收好时拍了谁，碰在什么地方"
    assert _source_excerpt(raw, question, 100).startswith(decoy)
    assert _source_excerpt(raw, question, 100, field=field).startswith(scene)
    assert len(_source_excerpt(raw, question, 14, field=field)) <= 14


def test_field_cannot_resolve_a_different_event_or_tied_source_paragraphs():
    from hierarchical_retrieval.core.branch_context import _agreeing_paragraph, _source_sentence_scores

    scene = "“你先收好。”周师傅拍了拍阿唐的手，“出去再看。”"
    field = "周师傅提醒先收好时拍了谁，碰在什么地方"
    q = "交接后让对方出去再看的那会儿，他碰的是谁？"
    tied = "\n".join((scene, scene.replace("阿唐", "小章")))
    assert _agreeing_paragraph(tied, q, field, _source_sentence_scores(tied, q)) is None
    different = scene + "\n夜间清点时，蒋师傅把木盒交给小方。"
    other = "夜间清点时，蒋师傅把木盒交给谁？"
    assert _agreeing_paragraph(different, other, field, _source_sentence_scores(different, other)) is None


def test_corroborating_field_keeps_the_observed_outcome_in_the_same_source_budget():
    from hierarchical_retrieval.core.branch_context import _source_excerpt

    scene = "“你先收好。”周师傅拍了拍阿唐的手，“出去再看。”"
    outcome = "阿唐仍未睁开眼睛。"
    doc = "那会儿交接还很忙。\n" + scene + "\n" + outcome
    q = "交接后让对方出去再看的那会儿，他碰的是谁？"
    field = "周师傅提醒先收好时拍了谁，碰在什么地方"
    assert _source_excerpt(doc, q, 100, field=field) == scene + "\n" + outcome
    assert _source_excerpt(doc, q, len(scene), field=field) == scene


def test_short_quoted_reply_keeps_its_same_line_attribution_within_existing_limit():
    from hierarchical_retrieval.core.branch_context import _source_excerpt

    cue = "沈宁停下核对档案编号，问门外是否已经送达。"
    reply = "“对。”门口的信使举起了收据。"
    page = cue + "\n\n" + reply + "\n\n王师傅走向码头。"
    q = "沈宁停下核对档案编号，问了什么，门口如何示意？"
    assert _source_excerpt(page, q, 100) == cue + "\n\n" + reply
    clipped = _source_excerpt(page, q, len(cue) + 6)
    assert len(clipped) <= len(cue) + 6 and "码头" not in clipped


def test_revision_outcome_guard_only_changes_the_live_tail_within_budget(monkeypatch):
    from hierarchical_retrieval.core.branch_context import revision_outcome_suffix

    history = archive(count=3)
    q = "第19章，按最新工作稿，更正里的车到站了没有？"
    monkeypatch.setenv("HR_GROUND_REVISION_OUTCOMES", "0")
    before = build(history, q)
    monkeypatch.setenv("HR_GROUND_REVISION_OUTCOMES", "1")
    after = build(history, q)
    assert after["wire_user"] == before["wire_user"]
    assert after["messages"][:-1] == before["messages"][:-1]
    assert after["cache_breakpoints"] == before["cache_breakpoints"]
    assert after["messages"][-1]["content"] == before["messages"][-1]["content"] + revision_outcome_suffix(q)
    assert after["chars"] == before["chars"] + len(revision_outcome_suffix(q))
    for different in (
        "第19章，按小说原始正文，到站了没有？",
        "比较原文和最新稿，是否到站？",
        "第19章，按最新工作稿，车辆在哪里？",
    ):
        assert not revision_outcome_suffix(different)
        assert build(history, different)["messages"][-1]["content"] == build(history, different)["wire_user"]
    budget = len("请简短作答。") + len(q)
    tiny = build(history, q, char_budget=budget)
    assert tiny["messages"][-1]["content"] == q
    assert tiny["chars"] <= budget
