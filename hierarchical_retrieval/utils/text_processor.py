"""
文本处理工具
"""

import re
from typing import NamedTuple, Optional


class ChapterFollowup(NamedTuple):
    query: Optional[str]
    action_question: str = ""
    candidates: tuple[tuple[str, str], ...] = ()


class TextProcessor:
    """文本分割、清洗工具"""

    # 英文句点只在后面是空白或文本结尾时才算句末：否则 "4.6 公斤"、
    # "requirements.txt"、"rows.Close()" 会被从中间切开，产生
    # "现在 4." / "6 公斤确实偏重" 这类残缺的关键句。
    SENTENCE_DELIMITERS = re.compile(r"(?<=[。！？!?])|(?<=\.)(?=\s|$)")

    @staticmethod
    def normalize_query(text: str) -> str:
        """只统一传输换行与首尾空白，保留大小写、标点和内部缩进。"""
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    _PLAIN_QUESTION = re.compile(r"[一-鿿A-Za-z0-9 ]+")
    _INTERROGATIVE = re.compile(r"什么|怎么|如何|多少|多久|多大|多重|几点|哪|谁|要不要|吗$")
    _PARAPHRASE_TEXT = re.compile(r"[一-鿿A-Za-z0-9 _./-]+")
    _EQUIVALENT_ENDINGS = (
        ("叫什么名字", "叫什么"),
        ("叫什么", "叫什么"),
        ("在哪里", "在哪里"),
        ("在哪儿", "在哪里"),
        ("金额是多大", "金额是多少"),
        ("金额是多少", "金额是多少"),
    )

    @classmethod
    def retrieval_query(cls, text: str) -> str:
        """普通中文疑问句的等价问法共享缓存，保留主体、否定、数字和条件。"""
        text = cls.normalize_query(text)
        body = text[:-1] if text.endswith(("?", "？")) else text
        if (len(body) >= 8 and cls._PARAPHRASE_TEXT.fullmatch(body)
                and cls._INTERROGATIVE.search(body)):
            if body.startswith("请问"):
                body = body[2:].lstrip()
                text = body
            for ending, canonical in cls._EQUIVALENT_ENDINGS:
                if len(body) >= len(ending) + 4 and body.endswith(ending):
                    return body[:-len(ending)] + canonical
        if text.endswith(("?", "？")):
            body = text[:-1]
            if cls._PLAIN_QUESTION.fullmatch(body) and cls._INTERROGATIVE.search(body):
                return body
        return text

    _REVISION = re.compile(r"更正为|修正为|更新为|调整为|改为|改成")
    _CHAPTER = re.compile(r"第\s*([0-9零〇一二两三四五六七八九十百千万]+)\s*章")
    _CHAPTER_ALIAS = re.compile(_CHAPTER.pattern + r"\s*[（(]\s*" + _CHAPTER.pattern + r"\s*[）)]")
    _CHAPTER_HEADING = re.compile(r"^\s*第\s*([0-9零〇一二两三四五六七八九十百千万]+)\s*章[^\n]*", re.MULTILINE)
    _DISTANCE_QUESTION = re.compile(r"距离[^。！？?，,\n]{0,12}(?:多少|几米|多远)|多远|(?:多少|几)(?:厘米|毫米|公里|千米|米)|(?:身高|高度).*?多高")
    _DISTANCE_VALUE = re.compile(r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万点]+)\s*(?:千米|公里|厘米|毫米|公尺|米)")
    _SCENE_UNIT = re.compile(r"(?:多少|几)(步|秒钟?|分钟)")
    _QUANTITY_VALUE = re.compile(_DISTANCE_VALUE.pattern + r"|(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万点]+)\s*(?:步|秒钟?|分钟)")
    _DISTANCE_FOLLOWUP = re.compile(r"^(?:那|所以)?(?:最后|最终)?(?:确认)?(?:到底)?(?:是)?(?:几米|多少米|多远)[？?]?$")
    _MULTI_VALUE = re.compile(r"分别|各自|原先|原来|原著|相差|前后(?!退)|旧值|对比")
    _BOOK_SOURCE = re.compile(r"(?:按(?:照)?|根据|依照|只看|只问)(?:小说|书中|书里|书上)?(?:原文|原著|原书)|(?:小说|书中|书里)原文")
    _SOURCE_COMPARISON = re.compile(r"分别|各自|相差|前后(?!退)|旧值|对比|比较|两者")
    _REHEARSAL = re.compile(
        r"(?:请)?(?:只)?(?:按照?|根据|依照|核对|看|查(?:询|看)?)?(?:当前|最新)?(?:读书会)?"
        r"(?:排演(?:笔记)?(?:校订)?|校订)(?:记录|版本|版)?(?:中|里)?"
    )
    _EXCLUDED_REHEARSAL = re.compile(
        r"[（(]?(?:不要|不用|不是|别|不)(?:按照?|看|查|用)?(?:当前|最新)?"
        r"(?:排演(?:笔记)?(?:校订)?|校订)(?:版本|记录|版)?[）)]?"
    )
    _MEASURE_FIELD = re.compile(r"步数|靶距|距离|身高|高度|半径|直径|长度|间距|时长|用时|耗时|时间")
    _ACTION_SEQUENCE = re.compile(r"前后(?:各|分别)?(?:做了什么|发生了什么)(?:动作|事)?[？?。]?$")
    _HELD_OBJECT = re.compile(r"(?<=[一-鿿]{2})(?:持|握|举|拿|提)[一-鿿]{1,3}(?=(?:退后|后退|上前|转身|起身|坐下)$)")
    _CHINESE_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))

    @classmethod
    def asks_for_book_source(cls, query: str) -> bool:
        focus = cls._EXCLUDED_REHEARSAL.sub("", cls._ACTION_SEQUENCE.sub("", query.rsplit("\n", 1)[-1]))
        return bool(cls._BOOK_SOURCE.search(focus)
                    and (cls._CHAPTER.search(focus) or re.search(r"小说|书[中里上]|原著|原书", focus))
                    and not re.search(r"(?:不|别|不用|不要)(?:再)?(?:按|根据|只看|只问)", focus)
                    and not cls._SOURCE_COMPARISON.search(focus)
                    and not re.search(r"排演|校订|更正", focus))

    @classmethod
    def rehearsal_scope(cls, query: str):
        """Extract an explicit single-edition measurement request; keep unfamiliar topic words."""
        query = re.sub(r"[（(]?(?:不是|不用|不看|不按|不查)(?:小说)?(?:原文|原著|原书)[）)]?", "", query)
        chapters = cls.chapter_ids(query)
        if (len(chapters) != 1 or not cls._REHEARSAL.search(query)
                or not re.search(r"多少|几(?:步|米|厘米|秒|分钟)|多远|多高", query)
                or cls._MULTI_VALUE.search(query) or cls._SOURCE_COMPARISON.search(query)
                or re.search(r"原文|原书|书中|别|不要|不用|不按|不查|不看|不是", query)):
            return None
        field_text = cls._CHAPTER.sub("", cls._REHEARSAL.sub("", query))
        field = cls._MEASURE_FIELD.search(field_text)
        unit_match = re.search(r"(?:多少|几)\s*(厘米|毫米|千米|公里|米|步|秒钟?|分钟)", field_text)
        if not field and not unit_match:
            return None
        unit = unit_match[1] if unit_match else ""
        field_name = field[0] if field else ("步数" if unit == "步" else "")
        topic = re.sub(r"(?:到底|究竟|具体)?(?:是|为)?(?:多少|几)(?:厘米|毫米|千米|公里|米|步|秒钟?|分钟)?(?:呢|来着)?", "", field_text)
        topic = cls._MEASURE_FIELD.sub("", topic)
        topic = re.sub(r"[（(](?:含|包含|保留)(?:原文)?范围[）)]", "", topic)
        topic = re.sub(r"[\s，,。；;：:？?]|对应的?|的", "", topic).removesuffix("时")
        return chapters[0], field_name, unit, cls.scene_terms_text(topic)

    @classmethod
    def revision_topic_matches(cls, query_topic, stored_topic):
        if not query_topic or query_topic == stored_topic:
            return True
        if cls._HELD_OBJECT.sub("", stored_topic) == query_topic:
            return True
        # A shared subject can govern two alternative two-character verbs.
        left, sep, right = stored_topic.partition("或")
        return bool(sep and (query_topic == left or
                            (len(right) == 2 and len(left) > 2 and query_topic == left[:-2] + right)))

    @classmethod
    def witnessed_revision_scene(cls, topic, stored_topic, sections):
        """Ground extra before/after clauses in one sentence of the specified chapter."""
        subject = re.fullmatch(r"([一-鿿]{1,6}?)(?:持|握|举|拿|提)([一-鿿]{1,4}?)(退后|上前|转身|起身|坐下)", stored_topic)
        if not subject:
            return False
        actor, item, motion = subject.groups()
        scene = re.fullmatch(re.escape(actor) + r"(.{2,40}?)(?:之)?后把" + re.escape(item)
                             + r"(.{2,18}?)之前" + re.escape(motion), topic)
        if not scene:
            return False
        lead = scene[1].split("、")
        if not lead[-1].endswith(item):
            return False
        # Match each narrated action in its stated order. Only aspect markers,
        # possessive particles and first-person reflexives are optional.
        parts = [*lead, motion, scene[2]]
        order = r"[^。！？；\n不未没别]{0,16}?".join(re.escape(piece) for piece in parts)
        for section in sections:
            # A first-person passage also needs a chapter-local mention of the actor.
            if actor not in section:
                continue
            narration = re.sub(r"了|自己的|的", "", cls.scene_terms_text(section))
            if re.search(r"(?:" + re.escape(actor) + r"|我)" + order, narration):
                return True
        return False

    @classmethod
    def _chapter_number(cls, label: str) -> str:
        if label.isascii() and label.isdigit():
            return label.lstrip("0") or "0"
        if all(c in cls._CHINESE_DIGITS for c in label):
            return "".join(str(cls._CHINESE_DIGITS[c]) for c in label).lstrip("0") or "0"
        # Conventional chapter ordinals below ten thousand. Leave mixed or ambiguous labels literal.
        if (re.search(r"[千百][一二两三四五六七八九]$", label) or not re.fullmatch(
                            r"(?:[一二两三四五六七八九]千[零〇]?)?(?:[一二两三四五六七八九]百[零〇]?)?"
                            r"(?:[一二两三四五六七八九]?十)?[一二两三四五六七八九]?", label)):
            return label
        total = digit = 0
        for c in label:
            if c in "千百十":
                total += (digit or 1) * {"千": 1000, "百": 100, "十": 10}[c]
                digit = 0
            else:
                digit = cls._CHINESE_DIGITS[c]
        return str(total + digit)

    @classmethod
    def scene_unit(cls, query: str):
        units = list(cls._SCENE_UNIT.finditer(query))
        if len(units) == 1 and not re.search(r"步骤|流程|几步棋", query):
            return units[0]
        if not units and re.search(
                r"步数(?:[（(][^()（）]{1,12}[）)])?(?:(?:是|为)?多少|(?:原文)?(?:怎么写|如何表述))", query):
            return re.search(r"(步)数", query)
        return None

    @classmethod
    def is_distance_question(cls, query: str) -> bool:
        focus = cls._ACTION_SEQUENCE.sub("", query.rsplit("\n", 1)[-1])
        return bool((cls._DISTANCE_QUESTION.search(focus) or (cls._CHAPTER.search(focus) and cls.scene_unit(focus)))
                    and not re.search(r"为什么|为何|叫什么|哪种|什么类型|多少环|多少枪|几枪|几环", focus)
                    and (not cls._MULTI_VALUE.search(focus) or cls.asks_for_book_source(focus))
                    and len(set(cls.chapter_ids(focus))) <= 1)

    @classmethod
    def chapter_ids(cls, text: str) -> list[str]:
        return [cls._chapter_number(label) for label in cls._CHAPTER.findall(text)]

    @classmethod
    def normalize_chapter_aliases(cls, text: str) -> str:
        """Collapse a parenthesized second spelling only when both ordinals agree."""
        def canonical(match):
            left, right = cls.chapter_ids(match[0])
            return f"第{left}章" if left == right else match[0]

        return cls._CHAPTER_ALIAS.sub(canonical, text)

    @classmethod
    def contains_measurement(cls, text: str) -> bool:
        return bool(cls._QUANTITY_VALUE.search(text))

    @classmethod
    def competing_chapter_measurement(cls, answer: str, query: str, chapter: str) -> bool:
        mentioned = set(cls.chapter_ids(answer))
        if not mentioned - {chapter} or not cls._QUANTITY_VALUE.search(answer):
            return False
        subject = cls._CHAPTER.sub("", query)
        return (cls.lexical_overlap(subject, answer) >= 0.45 or
                (chapter in mentioned and bool(re.search(r"是不是|是否指|应当是|应该是", answer))))

    @classmethod
    def chapter_spans(cls, text: str, start_chapter=None) -> list[tuple[str, int, int]]:
        headings = list(cls._CHAPTER_HEADING.finditer(text))
        spans = [(cls._chapter_number(heading[1]), heading.start(),
                 headings[i + 1].start() if i + 1 < len(headings) else len(text))
                for i, heading in enumerate(headings)]
        prefix_end = spans[0][1] if spans else len(text)
        if prefix_end and isinstance(start_chapter, (str, int)) and re.fullmatch(r"[0-9零〇一二两三四五六七八九十百千万]+", str(start_chapter)):
            spans.insert(0, (cls._chapter_number(str(start_chapter)), 0, prefix_end))
        return spans

    @classmethod
    def chapter_revision_question(cls, text: str) -> str:
        """Turn a single unconditional, chapter-scoped measurement update into its lookup question."""
        if len(text) > 240 or not cls._REVISION.search(text):
            return ""
        sentence = cls.split_sentences(text)[0]
        change = cls._REVISION.search(sentence)
        if (not change or len(cls.chapter_ids(sentence)) != 1 or re.search(r"[?？]|如果|假如|假设|不要|暂不|是否", sentence)
                or not cls._QUANTITY_VALUE.match(sentence[change.end():].lstrip())):
            return ""
        subject = sentence[:change.start()].rsplit("：", 1)[-1]
        if not cls._CHAPTER.match(subject) or not re.search(r"(?:距离|靶距|身高|高度|半径|直径|长度|间距|步数|时长|用时|耗时|时间)$", subject):
            return ""
        value_text = sentence[change.end():].strip().rstrip("。.!！").rstrip()
        value_match = cls._QUANTITY_VALUE.fullmatch(value_text)
        if not value_match:
            return ""
        value = value_match[0]
        unit = re.search(r"(?:千米|公里|厘米|毫米|公尺|米|步|秒钟?|分钟)$", value)[0]
        question = subject + f"是多少{unit}？"
        return question if cls.matching_revision(text, question) else ""

    _MEASURE_REMINDER = re.compile(
        r"(?:那|所以)?(?:现在|目前)?(?:还是|最后确认|到底)?(?:是)?(?:几|多少)"
        r"(厘米|毫米|千米|公里|米|步|秒钟?|分钟)(?:来着|呢)?[？?]?"
    )
    _RESUME_TOPIC = re.compile(r"^(?:再)?回到(?:刚才|之前|前面)(?:说|问)?的(步数|时长|距离|靶距)[，,、 ]*")
    _SCENE_FOLLOWUP = re.compile(
        r"(?:那|那么)?(?:能否|能不能|能|可以)?(?:再)?(?:说说|讲讲|描述一下)?"
        r"(?:他|她|他们|她们|这个人|那个人)(?:当时|那时|那会儿|随后|接下来|接着)"
        r"(?:具体)?(?:做了什么(?:动作)?|的(?:具体)?动作)(?:吗|呢|来着)?[？?]?"
    )

    @classmethod
    def is_scene_followup(cls, text: str) -> bool:
        return bool(cls._SCENE_FOLLOWUP.fullmatch(text.strip()))

    _SCOPE_FOLLOWUP = re.compile(
        r"(?:那|再)?(?:回到|回|换到|看)?(?:上一章|下一章|这一章|这章|同一章|第[0-9零〇一二两三四五六七八九十百千万]+章)?"
        r"[，,的 ]*(?:那|再)?(?:切回|回到|按|看|只看)?(?:小说原文|原文|原书|排演笔记|排演|校订)?"
        r"(?:最后确认|到底)?(?:是多少(?:厘)?米|是几米|几米|多少米)?(?:呢|来着)?[？?]?"
    )
    _ACTION_CLAUSE = re.compile(
        r"[？?，,。]\s*((?:也|再|同时|顺便)?(?:请)?(?:说说|说明|简述|描述|交代)(?:一下)?"
        r"(?:谁做了什么|(?:本章|该章|人物|对应|具体|场景|的)*动作)[。？！?]?)$"
    )

    @classmethod
    def split_action_clause(cls, text: str) -> tuple[str, str]:
        match = cls._ACTION_CLAUSE.search(text)
        return (text[:match.start()], match[1]) if match else (text, "")

    _NUMBER_REFERENCE = re.compile(r"(?:那|所以)?(?:刚才|刚刚|前面|上面)(?:说的|的)?(?:那个|这个)?(?:数值|数字|数)(?:到底|究竟)?(?:是|为)?多少(?:来着|呢)?[？?]?")
    # Only short references with a stated unit may inherit an existing measure.
    _UNIT_REFERENCE = re.compile(
        r"(?:刚才|刚刚|刚|前面)(?:提到|说)(?:的)?(?:(?P<field>步数|时长|距离|靶距)(?:到底)?是?多少|"
        r"(?:到底)?是?多少(?P<unit>步|秒钟?|分钟|厘米|毫米|千米|米))(?:来着|呢)?[？?]?"
    )
    _RECALLED_CLUE = re.compile(r"(?:我(?:只)?记得|印象里|印象中)(?:是)?(?P<clue>[^？?。！!]{1,55})[。！!]?")
    _TOPIC_SWITCH = re.compile(
        r"(?:(?:再)?(?:回到|切回|换回|看|说回)(?P<other>另一个话题)|(?P<bare_other>另一个话题)(?:的)?|"
        r"(?:再)?(?:回到|切回|换回|看|说回)"
        r"(?P<subject>第[0-9零〇一二两三四五六七八九十百千万]+章|[一-鿿]{2,12}?)(?:那一?[条段]|的话题))"
        r"[，, ]*(?:(?:按|看|切回)(?P<edition>小说原文|原文|排演笔记|排演|校订))?(?:呢|来着)?[？?]?"
    )
    _KEEP_FOCUS = re.compile(r"(?:继续刚才的话题|(?:先记下[，,]?)?不切换话题|话题先不换)[。！!… ]*$")
    _CHOICE_REPLY = re.compile(
        r"(?:(?:我(?:说|指)的?是|我是说|就是|是|选)\s*)?"
        r"(?:(?P<chapter>第\s*[0-9零〇一二两三四五六七八九十百千万]+\s*章)|"
        r"(?P<clue>[一-鿿]{2,30}?)(?:的)?(?:那一?[条段次]|那件事))"
        r"[，, ]*(?:(?:还是|这次)?(?:按|看)(?P<edition>小说原文|原文|排演笔记|排演|校订))?[。？?！!]?"
    )

    @classmethod
    def chapter_followup_query(cls, messages: list[str], *, original_actions=False) -> str:
        return cls.resolve_chapter_followup(messages, original_actions=original_actions) or ""

    @classmethod
    def resolve_chapter_followup(cls, messages: list[str], *, original_actions=False) -> Optional[str]:
        return cls.chapter_followup_context(messages, original_actions=original_actions).query

    @classmethod
    def chapter_followup_context(cls, messages, *, original_actions=False) -> ChapterFollowup:
        """An unresolved choice carries its possible topics until the user distinguishes them."""
        if not messages:
            return ChapterFollowup("")
        last, _ = cls.split_action_clause(messages[-1].strip())
        eligible = (cls._CHOICE_REPLY.fullmatch(last) or cls._RECALLED_CLUE.fullmatch(last) or cls._UNIT_REFERENCE.fullmatch(last)
                    or cls._NUMBER_REFERENCE.fullmatch(last) or cls._TOPIC_SWITCH.fullmatch(last)
                    or cls._SCOPE_FOLLOWUP.fullmatch(last) or cls.is_scene_followup(last)
                    or cls._MEASURE_REMINDER.fullmatch(last) or cls._RESUME_TOPIC.match(last))
        if eligible and (original_actions or any(cls._TOPIC_SWITCH.fullmatch(t.strip())
                or cls._NUMBER_REFERENCE.fullmatch(t.strip()) or cls._KEEP_FOCUS.search(t)
                for t in reversed(messages))):
            return cls._tracked_chapter_query(messages, original_actions)
        return ChapterFollowup(cls._relative_chapter_query(messages), last if cls.is_scene_followup(last) else "")

    @classmethod
    def _scene_clue_matches(cls, clue, field):
        """Require every substantive clause to match the same candidate, in order."""
        def normalize(s):
            s = cls.scene_terms_text(s)
            s = re.sub(r"(?:往|向|朝)前(?:走|移动)?", "向前", s)
            s = re.sub(r"挥(?:舞|起)", "挥", s)
            s = s.replace("拉起", "拉").replace("查看", "看").replace("走近", "靠近")
            s = re.sub(r"那个|那一个|的那个动作$|(?:那次|那段)$|起来(?=看)|[了的很]", "", s)
            return s.strip()

        # A similarity score can make a shared verb look unique when the object
        # is unknown or negated. Keep such choices open.
        if re.search(r"不|并非|没|未|还是|或者|可能", clue):
            return False
        text = re.sub(r"^(?:他|她)?(?:先是|先|是)?(?:伸手(?=.{3,})|把)?", "", clue)
        # A 把 construction puts the object before its verb; topic labels often
        # put it after. Retain both the object and the remainder of the action.
        obj = re.match(r"^(?:他|她)?(?:是)?把(.{1,12}?)(拿|举|抓|放|拉|推|打开)(?:了)?(?:起来|起)?(.*)$", clue)
        label = normalize(field.split("多少", 1)[0])
        if obj:
            thing, verb, tail = obj.groups()
            target = normalize(thing)
            pattern = re.escape(verb) + r"(?:起)?" + re.escape(target) + re.escape(normalize(tail))
            if target and re.search(pattern, label):
                return True
        parts = re.split(r"(?:以后|之后)(?:还|又|再)?", text)
        # A later clause may fully identify the scene. Still require each
        # preceding action to be present instead of discarding a contradiction.
        offset = 0
        for part in parts:
            part = normalize(part)
            if not part:
                return False
            match = label.find(part, offset)
            if match < 0 or re.search(r"(?:不|没有|未曾|并非|没|未)$", label[:match]):
                return False
            offset = match + len(part)
        return True

    @classmethod
    def _tracked_chapter_query(cls, messages, original_actions):
        # Store the last selected edition independently for each chapter and field.
        # Replay only user turns; retrieved prose and model guesses cannot move focus.
        topics = {}
        active = None
        history = []
        pending = None
        selected_edition = ""
        action = ""
        resolved = ""
        for raw in messages:
            text, _ = cls.split_action_clause(raw.strip())
            update = cls.chapter_revision_question(text)
            resolved = ""
            if update and cls._KEEP_FOCUS.search(text):
                continue
            switch = cls._TOPIC_SWITCH.fullmatch(text)
            reply = cls._CHOICE_REPLY.fullmatch(text) if pending is not None and not switch else None
            recalled = cls._RECALLED_CLUE.fullmatch(text) if pending is not None else None
            unit_ref = cls._UNIT_REFERENCE.fullmatch(text)
            if switch:
                action = ""
                selected_edition = switch["edition"] or ""
                subject = switch["subject"] or ""
                chapters = cls.chapter_ids(subject)
                if switch["other"] or switch["bare_other"]:
                    candidates = [k for k in (pending if pending is not None else topics) if k != active]
                elif chapters:
                    candidates = [k for k in topics if k[0] == chapters[0]]
                else:
                    candidates = []
                    for key in topics:
                        field = key[1]
                        if not field.startswith(subject):
                            continue
                        remainder = field[len(subject):]
                        # “小林的时长” concerns 小林; “小林的父亲等车的时长” has another subject.
                        if not remainder.startswith("的") or cls._MEASURE_FIELD.match(remainder[1:]):
                            candidates.append(key)
                pending = candidates if topics else None
                active = None
            elif reply or recalled:
                if reply and reply["chapter"]:
                    ch = cls.chapter_ids(reply["chapter"])[0]
                    matches = [k for k in (pending or topics) if k[0] == ch]
                else:
                    clue = (recalled or reply)["clue"]
                    matches = [k for k in pending if cls._scene_clue_matches(clue, k[1])]
                # An unrecognised clue does not reopen unrelated topics.
                if matches:
                    pending = matches
                    if reply and reply["edition"]:
                        selected_edition = reply["edition"]
            elif pending is not None and re.fullmatch(r"(?:还是|只)?(?:按|看)(小说原文|原文|排演笔记|排演|校订)(?:呢)?[。？?]?", text):
                selected_edition = "原文" if "原文" in text else "排演"
            elif pending is not None and re.match(r"^(?:我指|我说的是|我是说|不是第|第[一二三四五六七八九0-9]+个|前者|后者)", text):
                pass  # The user is clarifying, but the reply has not identified a candidate.
            elif unit_ref:
                requested = unit_ref["field"]
                marker = requested or "多少" + unit_ref["unit"]
                candidates = pending if pending is not None else ([active] if active else [])
                matches = [key for key in candidates if marker in key[1]]
                if not matches and pending is None:
                    active = None
                    continue
                if matches and pending is not None:
                    pending = matches
                action = ""
            elif cls._NUMBER_REFERENCE.fullmatch(text) or (pending is not None and cls._MEASURE_REMINDER.fullmatch(text)):
                action = ""
            elif cls.is_scene_followup(text) and (pending is not None or
                    (active and (original_actions or "原文" in topics[active]))):
                action = text
            else:
                pending = None
                selected_edition = ""
                action = ""
                history.append(text)
                relative = cls._relative_chapter_query(history)
                base = cls.normalize_chapter_aliases(relative or update or text)
                chapters = cls.chapter_ids(base)
                if len(chapters) != 1 or not cls.is_distance_question(base):
                    active = None
                    continue
                field = cls.book_measurement_field(cls._REHEARSAL.sub("", base))
                active = (chapters[0], field)
                prefix = ("按小说原文，" if cls.asks_for_book_source(base) else "按排演笔记，"
                          if cls.rehearsal_scope(base) or (update and cls._REHEARSAL.match(text)) else "")
                topics[active] = prefix
                history[-1] = prefix + f"第{active[0]}章{field}"
                resolved = relative
                continue
            if pending is not None and len(pending) == 1:
                active = pending[0]
                pending = None
                if selected_edition:
                    topics[active] = "按小说原文，" if "原文" in selected_edition else "按排演笔记，"
            numeric_query = topics[active] + f"第{active[0]}章{active[1]}" if active else ""
            history.append(numeric_query or text)
            resolved = numeric_query
            if action and active:
                resolved = f"按小说原文，第{active[0]}章{active[1]}" if original_actions or "原文" in topics[active] else ""
        return ChapterFollowup(None if pending is not None else resolved, action, tuple(pending or ()))

    @classmethod
    def _relative_chapter_query(cls, messages: list[str]) -> str:
        """Resolve consecutive relative chapter and version references against the last user topic."""
        steps = []
        resume_field = ""
        skipped_asides = 0
        for text in reversed(messages):
            text, _ = cls.split_action_clause(text.strip())
            resume = cls._RESUME_TOPIC.match(text)
            if resume:
                if resume_field and resume_field != resume[1]:
                    return ""
                resume_field = resume[1]
                text = text[resume.end():]
                if text in ("", "？", "?", "呢", "呢？", "呢?"):
                    steps.append(text)
                    continue
                if not (cls._SCOPE_FOLLOWUP.fullmatch(text) or cls._MEASURE_REMINDER.fullmatch(text)):
                    return ""
            reminder = cls._MEASURE_REMINDER.fullmatch(text)
            if reminder or cls.is_scene_followup(text):
                steps.append(text)
                continue
            if (len(text) < 50 and cls._SCOPE_FOLLOWUP.fullmatch(text)
                    and re.search(r"章|原文|原书|排演|校订|几米|多少米", text)):
                steps.append(text)
                continue
            if not steps:
                return ""
            update = cls.chapter_revision_question(text)
            base = cls.normalize_chapter_aliases(update or text)
            chapters = cls.chapter_ids(base)
            if len(chapters) != 1 or not chapters[0].isascii() or not cls.is_distance_question(base):
                if resume_field and skipped_asides < 12 and not chapters and not cls._MEASURE_FIELD.search(base) and not cls.scene_unit(base):
                    skipped_asides += 1
                    continue
                return ""
            if resume_field and resume_field not in base:
                return ""
            chapter = int(chapters[0])
            original = cls.asks_for_book_source(base)
            rehearsal = bool(cls.rehearsal_scope(base) or (update and cls._REHEARSAL.match(text)))
            field = cls._BOOK_SOURCE.sub("", cls._CHAPTER.sub("", base)).removeprefix("请问")
            field = cls._REHEARSAL.sub("", field).strip(" ，,：:。？?") if rehearsal else field.strip(" ，,。？?")
            for step in reversed(steps):
                reminder = cls._MEASURE_REMINDER.fullmatch(step)
                if reminder:
                    unit = re.search(r"(?:多少|几)(厘米|毫米|千米|公里|米|步|秒钟?|分钟)", base)
                    if unit:
                        if unit[1].removesuffix("钟") != reminder[1].removesuffix("钟"):
                            return ""
                    elif reminder[1] != "米" or not cls._DISTANCE_QUESTION.search(base):
                        return ""
                if cls.is_scene_followup(step) and not original:
                    return ""
                explicit = cls.chapter_ids(step)
                if explicit:
                    if not explicit[0].isascii():
                        return ""
                    chapter = int(explicit[0])
                elif "上一章" in step:
                    chapter -= 1
                elif "下一章" in step:
                    chapter += 1
                if chapter < 1:
                    return ""
                if re.search(r"原文|原书", step):
                    original, rehearsal = True, False
                elif re.search(r"排演|校订", step):
                    original, rehearsal = False, True
            prefix = "按小说原文，" if original else "按排演笔记，" if rehearsal else ""
            return prefix + f"第{chapter}章" + field
        return ""

    @classmethod
    def stored_user_revision(cls, text: str) -> str:
        if text.startswith("用户："):
            line = text.split("\n", 1)[0].removeprefix("用户：")
            if cls._REVISION.search(line):
                return line
        return ""

    @classmethod
    def revision_query(cls, query: str, anchor: Optional[str]) -> str:
        """Resolve a short, chapter-scoped request for a distance through its known topic."""
        focus = query.rsplit("\n", 1)[-1]
        chapters = cls.chapter_ids(focus)
        if (anchor and chapters and chapters == cls.chapter_ids(anchor)
                and cls.is_distance_question(anchor)
                and cls._DISTANCE_FOLLOWUP.fullmatch(cls._CHAPTER.sub("", focus))):
            return anchor
        return focus

    @classmethod
    def matching_revision(cls, user_text: str, query: str) -> str:
        """Only treat explicit changes addressing the current question as revisions."""
        if not cls._REVISION.search(user_text):
            return ""
        chapters = cls.chapter_ids(query)
        if len(set(chapters)) > 1 or cls._MULTI_VALUE.search(query) or cls.asks_for_book_source(query):
            return ""
        terms = cls.query_terms(cls._CHAPTER.sub("", query))
        for sentence in reversed(cls.split_sentences(user_text)):
            change = cls._REVISION.search(sentence)
            if not change or sentence.endswith(("？", "?")) or sentence.startswith(("如果", "假如")):
                continue
            unit = cls.scene_unit(query)
            if unit:
                quantity = cls._QUANTITY_VALUE.match(sentence[change.end():].lstrip())
                if not quantity or (unit[1] == "步" and not quantity[0].endswith("步")) or (
                        unit[1] != "步" and not quantity[0].endswith(("秒", "秒钟", "分钟"))):
                    continue
            subject = sentence[:change.start()].rsplit("：", 1)[-1]
            if chapters and set(chapters) != set(cls.chapter_ids(subject)):
                continue
            if subject.endswith(("不", "别", "没", "没有", "不能", "是否", "未", "想", "建议")):
                continue
            subject_terms = cls.query_terms(cls._CHAPTER.sub("", subject))
            matched = len(terms & subject_terms)
            if matched >= 2 and matched >= 0.6 * max(len(terms), len(subject_terms)):
                return sentence
        return ""

    @classmethod
    def book_measurement_field(cls, query: str) -> str:
        """Canonical field used both for scoring and for cached original-book quotes."""
        field = cls._CHAPTER.sub("", cls._BOOK_SOURCE.sub("", cls._EXCLUDED_REHEARSAL.sub("", query))).strip(" ，,。？?")
        field = field.removeprefix("请问").strip(" ，,")
        # Only rewrite the final question phrase. Scope, conditions, ranges and
        # the unit remain part of the key; use the same field for fresh ranking.
        return re.sub(
            r"(?:到底|究竟|具体)?(?:是)?(?:多少|几)(厘米|毫米|千米|公里|米|步|秒钟?|分钟)(?:来着|呢)?$",
            r"多少\1", field,
        )

    @staticmethod
    def scene_terms_text(text: str) -> str:
        # Directional complements can precede or follow the verb in narration.
        return re.sub(r"(?:往|向|朝)?后退", "退后", text)

    @classmethod
    def scene_quantity_quotes(cls, section: str, unit: str):
        """Keep the measurement sentence and its immediate action/attribution context."""
        unit_pattern = "秒钟?" if unit.startswith("秒") else unit
        measure = re.compile(r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万点]+)\s*" + unit_pattern)
        sentences = [line.strip() for sent in cls.split_sentences(section)
                     for line in sent.splitlines() if line.strip()]
        for i, sentence in enumerate(sentences):
            if measure.search(sentence):
                # Short dialogue interjections should not push the actor's preceding
                # action out of the quotation. Keep a bounded character window.
                before = "".join(sentences[max(0, i - 16):i])[-200:]
                after = "".join(sentences[i + 1:i + 5])[:150]
                yield sentence, before + sentence + after

    @classmethod
    def distance_evidence(cls, text: str, original: str, query: str, *, chapter_section: bool = False) -> tuple[float, str]:
        """Find a short distance quote with its actor or target still visible."""
        focus = query.rsplit("\n", 1)[-1]
        chapters = set(cls.chapter_ids(focus))
        terms = cls.query_terms(cls._BOOK_SOURCE.sub("", cls._CHAPTER.sub("", focus))) - {"距离", "多远", "几米"}
        target = re.search(r"([一-鿿]{2})(?:距离|离)", focus)
        best = (0.0, "")
        previous = ""
        for sentence in cls.split_sentences(text):
            if not cls._DISTANCE_VALUE.search(sentence):
                previous = sentence
                continue
            if chapters:
                start = original.find(sentence)
                labels = cls.chapter_ids(sentence)
                if not labels and start >= 0:
                    labels = cls.chapter_ids(original[:start])
                if labels and labels[-1] not in chapters:
                    previous = sentence
                    continue
            hits = sum(term in sentence.lower() for term in terms)
            nearby = sum(term in previous.lower() and term not in sentence.lower() for term in terms)
            # A colloquial height (一米九) or a separation (百米之外) can be
            # expressed without the literal query words 身高 or 距离.
            measured_relation = (
                ("身高" in focus and re.search(r"米[一二三四五六七八九](?:的|左右|上下|[，。！？])", sentence))
                or (re.search(r"相距|相隔|间隔", focus) and re.search(r"米(?:之|开)?外", sentence))
            )
            relation_match = bool(chapters and measured_relation and any(
                term in sentence for term in terms - {"身高", "相距", "相隔", "距离", "间隔", "厘米", "毫米", "公里", "千米", "公尺"}
            ))
            # The measurement and its subject must describe the same scene. Chinese
            # often expresses 身高 as “三米多高”, and 离地 as “与地面保持…”.
            if chapters and (("身高" in focus and re.search(r"米(?:多)?高", sentence))
                             or ("离地" in focus and "脚" in sentence and re.search(r"离地|与地面", sentence))):
                if hits or nearby:
                    hits += 1
            if relation_match:
                hits += 1
            score = (hits + 0.5 * nearby) / max(1, len(terms)) + (0.15 if relation_match else 0)
            if target and target[1] in sentence:
                score += 0.20
            if hits >= 2 and score > best[0] and len(sentence) <= 200:
                needs_subject = chapter_section and re.match(r"^[“”「」\s]*(?:[他她](?:便|却|立刻|立即)|(?:可|但)?一想到自己)", sentence)
                quote = previous + sentence if (nearby or needs_subject) and len(previous + sentence) <= 240 else sentence
                best = (score, quote)
            previous = sentence
        return best

    @classmethod
    def evidence_excerpt(cls, text: str, query: str, limit: int, sentence_scores: dict) -> str:
        """从命中的段落引用完整原句，在预算允许时带上相邻上下文。"""
        terms = cls.query_terms(query)
        if limit <= 0:
            return ""
        # 保留原文中的空格和换行。截取某个发言中间的句子时补回说话人。
        parts = cls.SENTENCE_DELIMITERS.split(text)
        best = ""
        best_score = (0, 0.0)
        offset = 0
        for index, part in enumerate(parts):
            start = offset
            offset += len(part)
            sentence = part.strip()
            lower = sentence.lower()
            # 字面证据相同（含无重合）时，复用关键句检索的语义分数。
            score = (sum(term in lower for term in terms), sentence_scores.get(sentence, 0.0))
            if score <= best_score:
                continue
            line_start = text.rfind("\n", 0, start) + 1
            label = ""
            if not part.startswith(("\n", "\r")):
                for role in ("用户：", "助手："):
                    if text.startswith(role, line_start) and not sentence.startswith(role):
                        label = role
                        break
            if not sentence or len(label) + len(sentence) > limit:
                continue
            best = label + sentence
            best_score = score
            # 下一句常含问题的回答、数字的单位或决定的生效条件。
            if index + 1 < len(parts):
                pair = label + (part + parts[index + 1]).strip()
                if len(pair) <= limit:
                    best = pair
        return best

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """按句末标点分割句子"""
        raw = TextProcessor.SENTENCE_DELIMITERS.split(text)
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences

    @staticmethod
    def split_chunks(text: str, max_chars: int = 512, overlap: int = 32) -> list[str]:
        """按句子切块；仅超长句子硬切，所有块均不超过字符上限。"""
        if max_chars <= 0:
            raise ValueError("max_chars 必须是正整数")
        if overlap < 0 or overlap >= max_chars:
            raise ValueError("overlap 必须是非负整数且小于 max_chars")

        chunks: list[str] = []
        buffer = ""
        for sentence in TextProcessor.split_sentences(text):
            segments = (
                [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)]
                if len(sentence) > max_chars else [sentence]
            )
            for segment in segments:
                if not buffer:
                    buffer = segment
                    continue
                if len(buffer) + len(segment) <= max_chars:
                    buffer += segment
                    continue

                chunks.append(buffer)
                tail = ""
                if overlap:
                    tail_sentences: list[str] = []
                    tail_len = 0
                    for previous in reversed(TextProcessor.split_sentences(buffer)):
                        if tail_len + len(previous) > overlap:
                            break
                        tail_sentences.insert(0, previous)
                        tail_len += len(previous)
                    tail = "".join(tail_sentences)
                    if len(tail) + len(segment) > max_chars:
                        tail = ""
                buffer = tail + segment

        if buffer:
            chunks.append(buffer)
        return chunks

    # 关键词重合度：疑问/功能词不携带信息，先剔除，否则"什么时候"之类
    # 的二元组会让任何含"时候"的段落都得分。
    _QUERY_FILLERS = re.compile(
        r"为什么|怎么样|怎么|什么|多少|哪里|哪个|哪种|哪些|几点|多久|多大|"
        r"要不要|是不是|有没有|能不能|可不可以|可以|如何|现在|之前|以前|以后|"
        r"我们|你们|他们|这个|那个|一下|时候|的|了|吗|呢|吧|啊|是|有|在|和|与|"
        r"请问|还是|一个"
    )
    _ASCII_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
    _CJK_RUN = re.compile(r"[一-鿿]+")

    @staticmethod
    def query_terms(query: str) -> set[str]:
        """查询的检索词：ASCII 词（≥2 字符，小写）+ 中文二元组。"""
        terms = {
            t.lower() for t in TextProcessor._ASCII_TOKEN.findall(query) if len(t) >= 2
        }
        stripped = TextProcessor._QUERY_FILLERS.sub(" ", query)
        for run in TextProcessor._CJK_RUN.findall(stripped):
            if len(run) == 1:
                continue
            terms.update(run[i:i + 2] for i in range(len(run) - 1))
        return terms

    @staticmethod
    def lexical_overlap(query: str, text: str) -> float:
        """查询检索词在 text 中出现的比例（0~1）。

        用于对向量召回的段落做关键词校正：专名、数字、术语（如
        "bge-m3"、"设计文档"）在稠密向量里常被稀释，字面命中可以
        把这类"明明聊过却差一点过阈值"的段落拉回来。
        """
        terms = TextProcessor.query_terms(query)
        if not terms:
            return 0.0
        haystack = text.lower()
        return sum(1 for t in terms if t in haystack) / len(terms)

    @staticmethod
    def clean_text(text: str) -> str:
        """清理多余空白"""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
