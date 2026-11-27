
from hierarchical_retrieval.utils.text_processor import TextProcessor
from test_memory_quality import FILLER, M, _compress_history, proxy


def test_each_topic_keeps_its_edition_after_a_side_correction(proxy):
    a = "按小说原文，第41章小岚拿钥匙时退后的步数是多少步？"
    b = "按排演笔记，第42章余平开门后退的步数是多少步？"
    update = "排演笔记：第42章余平开门后退的步数更正为61步。继续刚才的话题。"
    history = [a, b, "回到小岚那条呢？", update]
    proxy.ingest("第41章 出发\n小岚拿着钥匙，退后了两步。")
    proxy.ingest("第42章 回来\n余平开了门，朝后退了三步。")
    proxy.ingest("用户：排演笔记：第42章余平开门后退的步数更正为17步。")
    proxy.ingest("用户：" + update)

    def evidence(turns):
        q = TextProcessor.resolve_chapter_followup(turns, original_actions=True)
        return proxy.assemble_context(q)

    assert "第41章原文" in evidence(history + ["他当时做了什么？"])
    assert "两步" in evidence(history + ["刚才那个数是多少？"])
    back = history + ["再看另一个话题呢？"]
    assert "61步" in evidence(back)
    assert "第42章原文" in evidence(back + ["他当时做了什么？"])
    number = evidence(back + ["他当时做了什么？", "刚才那个数是多少？"])
    assert "61步" in number and "17步" not in number
    assert not TextProcessor.chapter_followup_query([a, "午饭要吃什么？", "刚才那个数是多少？"])


def test_ambiguous_switch_requires_clarification_and_can_be_resolved(proxy):
    questions = ["按排演笔记，第41章小岚出门的步数是多少步？",
                 "按排演笔记，第42章小岚回来的步数是多少步？",
                 "按小说原文，第46章余平出门的步数是多少步？",
                 "回到小岚那条呢？"]
    for q in [questions, questions + ["他当时做了什么？"], questions + ["刚才那个数是多少？"],
              questions[:-1] + ["再看另一个话题呢？"]]:
        assert TextProcessor.resolve_chapter_followup(q) is None
    assert "第42章" in TextProcessor.chapter_followup_query(questions + ["再看第42章那段，按原文呢？"])
    relative = ["按排演笔记，第41章小岚的父亲出门的步数是多少步？", questions[2], questions[3]]
    assert TextProcessor.resolve_chapter_followup(relative) is None
    proxy.config.compress_threshold_chars = 1
    history = [M("system", "请核对章节。"), *FILLER]
    for q in questions[:-1]:
        history.extend([M("user", q), M("assistant", "这一题尚待核对。")])
    prompt = _compress_history(history + [M("user", questions[-1])])
    assert any("无法唯一确定" in m.content for m in prompt if m.role == "system")


def test_user_can_request_source_actions_without_changing_the_numeric_edition(proxy):
    proxy.config.compress_threshold_chars = 1
    proxy.ingest("第52章 到达\n胡杨拎起皮箱，向前走了两步。")
    proxy.ingest("用户：排演笔记：第52章胡杨拎着皮箱走的步数更正为32步。")
    history = [*FILLER, M("user", "问动作时只查同章原文。"), M("assistant", "好的。"),
               M("user", "按排演笔记，第52章胡杨拎着皮箱走的步数是多少步？"), M("assistant", "32步。"),
               M("user", "他当时做了什么？")]
    assert "本轮依据为小说原文" in _compress_history(history)[-1].content
    query = _compress_history(history + [M("assistant", "他拎起皮箱。"), M("user", "刚才那个数是多少？")])[-1].content
    assert "32步" in query and "本轮依据为用户最新校订" in query
