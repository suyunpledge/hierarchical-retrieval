"""Resolve a scene and its reading edition from user-owned chapter sources and selections."""

import re

from .branch_context import _FIELD_REPLACEMENT, revision_outcome_suffix

_HEADER = re.compile(r"(?m)^[ \t　]*第\s*(\d+)\s*章[^\n]*")
_POLAR = re.compile(r"有没有|是否|了没有|了没[？?。]|了吗|了么|到底.{0,12}没")
_SOURCE = re.compile(r"(?:最初|当初|起初|最早|一开始)(?:我们)?(?:读|看|听)到|原书|原文|原始正文|最初那份")
_DRAFT = re.compile(
    r"(?:后来|刚才|刚刚|最后|我们|咱们).{0,5}改(?:过|好|定|的)|改过的|改后的|工作稿|排演稿|现行版本"
)
_MIXED_SCENE = re.compile(
    r"(?:两(?:边|幕|段|个场景)|这两).{0,20}(?:记混|弄混|分不清|其中一|都有人)|其中一幕.{0,20}分不清"
)
_MIXED_EDITION = re.compile(r"(?:两种说法|两个版本|两份).{0,20}(?:没选|不确定|分不清|记混)")
_ASK_IDENTITY = re.compile(
    r"指(?:的)?(?:是谁|谁)|(?:具体|到底|分别)?是谁|哪两(?:个)?人|指的是哪(?:个|位)|叫什么"
)
_CLUE = re.compile(
    r"(?:换[到回]|回到|切回|再说说|我(?:指|说)的是|就是|只记得是|记得是)"
    r"(.{2,64}?)(?:的?那(?:一)?(?:幕|段|次)|的那一下)(?=[，,。？！?\s]|$)"
)


def _catalog(db, workspace):
    # Only inspect document headers until a clue actually requires the source text.
    rows = db.execute(
        """SELECT topic, revision, substr(user,1,240) AS opening FROM topic_events
           WHERE workspace=? AND kind IN ('turn','source') AND length(user)>=240
           AND instr(substr(user,1,160),'章')>0 AND instr(user,char(10))>0""",
        (workspace,),
    )
    docs = {}
    for row in rows:
        header = _HEADER.search(row["opening"])
        if header:
            topic = row["topic"]
            docs[topic] = (row["revision"], header[1]) if topic not in docs else None
    return docs


def _scene(db, workspace, topic, ref, clue, *, antecedents=False, identify=False):
    if len(clue) < 4:
        return ""
    if not ref:
        return None
    row = db.execute(
        "SELECT user FROM topic_events WHERE workspace=? AND topic=? AND revision=? AND instr(user,?)>0",
        (workspace, topic, ref[0], clue),
    ).fetchone()
    if not row:
        return ""
    doc = row[0]
    if len(_HEADER.findall(doc)) != 1 or doc.count(clue) != 1:
        return None
    excerpts = [line.strip() for line in doc.splitlines() if clue in line and not _HEADER.match(line)]
    if not excerpts:
        return ""
    scene = excerpts[0]
    target = re.search(
        r"(?:推开|拉住|扶起|放下|抱住|拦下|叫住|抓住|撞到)(?:了)?(两人|二人|他们|她们|对方|那人|他|她)(?=[，。；！？、\s”]|$)",
        scene,
    )
    if identify or (antecedents and target):
        # A named identity question may be about the actor or the recipient.
        # In a plural scene an intervening speaker can separate the two names.
        before = doc[_HEADER.search(doc).end() : doc.index(scene)]
        plural = (
            bool(re.search(r"两人|二人|他们|她们", scene))
            if identify
            else target[1] in ("两人", "二人", "他们", "她们")
        )
        wanted = (4 if plural else 2) if identify else (2 if plural else 1)
        limit = 420 if identify else 280
        context = []
        size = 0
        for line in reversed(before.splitlines()):
            line = line.strip()
            if not line:
                continue
            size += len(line) + 1
            if size > limit:
                break
            context.append(line)
            if len(context) == wanted:
                break
        scene = "\n".join([*reversed(context), scene])
    return scene


def _focus(db, workspace, topic, source_revision):
    row = db.execute(
        """SELECT scene,edition FROM topic_reading_focus WHERE workspace=? AND topic=? AND source_revision=?
           AND revision=(SELECT max(revision) FROM topic_events WHERE workspace=? AND topic=? AND kind='turn')""",
        (workspace, topic, source_revision, workspace, topic),
    ).fetchone()
    return (row[0], row[1]) if row else ("", "")


def _current_note(db, workspace, topic, chapter, scene, question):
    # A shortcut needs a complete named replacement, scoped to this scene. A newer
    # partial amendment or revocation leaves ordinary branch retrieval in charge.
    row = db.execute(
        "SELECT coalesce(wire_content,user),revision FROM topic_events WHERE workspace=? AND topic=? "
        "AND kind='note' ORDER BY revision DESC LIMIT 1",
        (workspace, topic),
    ).fetchone()
    if not row:
        return "", None
    # Notes are not the only way a user can correct a transcript. A later
    # declarative amendment in ordinary chat must also go through retrieval.
    later = db.execute(
        "SELECT user FROM topic_events WHERE workspace=? AND topic=? AND revision>? AND kind='turn'",
        (workspace, topic, row[1]),
    )
    if any(re.search(r"改成|改为|替换为|作废|撤回|撤销|应为|以.{1,24}为准", u[0]) for u in later):
        return "", None
    note = row[0]
    fields = _FIELD_REPLACEMENT.findall(note)
    if len(note) > 480 or len(fields) != 1 or re.search(r"撤销|撤回|作废|取消|如果|假设", note):
        return "", None
    if set(re.findall(r"第\s*(\d+)\s*章", note)) != {chapter}:
        return "", None
    label = re.split(r"那一幕|那一段|那次|[，,：:]", fields[0], maxsplit=1)[0]
    action = re.search(r"推|拉|打开|关闭|移动|拿起|放下", question)
    scoped = action and action[0] in fields[0] and len(label) >= 4 and label in scene
    if not scoped or re.search(r"另外|顺便|除此|同时问|为什么", question):
        return "", None
    return note, row[1]


def resolve(db, workspace, text, names, current, pending):
    """Return None to delegate unrelated dialogue to the ordinary topic router."""
    managed = pending if pending and pending.get("reading") else {}
    clue_match = _CLUE.search(text)
    mixed = bool(_MIXED_SCENE.search(text))
    mixed_edition = bool(_MIXED_EDITION.search(text))
    original, latest = bool(_SOURCE.search(text)), bool(_DRAFT.search(text))
    if re.search(r"(?:不(?:是|要|看)|别按|并非).{0,8}(?:那段|那幕|那份|原文|改过|最初)", text):
        # A negated or explicitly disjunctive clue cannot identify a new scene or edition.
        clue_match, original, latest = None, False, False
        mixed = True
    named = [n for n in names if n in text]
    if pending and not managed:
        return None
    if _HEADER.search(text) or re.search(r"新建话题|换个话题|一起比较|分别说|分别回答|各自", text):
        return None
    if not (
        managed
        or _POLAR.search(text)
        and (clue_match or mixed or mixed_edition or re.search(r"他|她|那个人|这件事|那一幕", text))
    ):
        return None
    # An unrelated new question is a normal turn, even while a reading choice is open.
    if managed and not (
        clue_match
        or original
        or latest
        or mixed
        or mixed_edition
        or _POLAR.search(text)
        or named
        or re.search(r"那一下|想不起|记不清|不记得|记得|那段|那一幕", text)
    ):
        return None
    docs = _catalog(db, workspace)
    if not docs or (not managed and not mixed and not clue_match and current not in docs):
        return None
    pool = list(managed.get("candidates", docs))
    question = managed.get("question", text)
    edition = "original" if original and not latest else "latest" if latest and not original else ""
    unset = mixed_edition or (original and latest) or managed.get("edition_open", False)
    scene = managed.get("scene", "")
    clue = clue_match[1].strip() if clue_match else ""
    topic = managed.get("selected") or (None if managed else current)
    if mixed:
        topic, scene, pool = None, "", list(docs)
    elif clue:
        # A clarification can redirect the reader even while an edition choice is pending.
        pool = list(docs)
        matches = {t: _scene(db, workspace, t, docs.get(t), clue) for t in pool}
        hits = [t for t, value in matches.items() if value != ""]
        if len(hits) == 1 and matches[hits[0]]:
            topic, scene = hits[0], clue
        elif (
            not hits
            and managed.get("selected")
            and not re.search(r"换到|换回|切回|回到", text)
            and (
                action := re.fullmatch(
                    r"(?:他|她|那个人)?(?:伸手)?(推|拉|拿|放|碰)(?:了|过)?(?:东西|一下)", clue
                )
            )
            and action[1] in question
        ):
            pool = [topic]  # Repeating just the known action supplies no new scene or edition.
        else:
            topic, scene, pool = None, "", hits or pool
    elif len(named) == 1 and named[0] in pool:
        topic = named[0]
        if topic != managed.get("selected"):
            scene = ""
    elif _POLAR.search(text) and re.search(r"换到|切回|回到|换回", text) and not managed:
        topic = None
    if topic not in docs or not docs.get(topic):
        topic = None
    if topic:
        remembered, previous = _focus(db, workspace, topic, docs[topic][0])
        scene = scene or remembered
        edition = edition or ("" if unset else managed.get("edition") or previous)
        if scene and not edition and not unset:
            edited = db.execute(
                "SELECT 1 FROM topic_events WHERE workspace=? AND topic=? AND "
                "(kind='note' OR (kind='turn' AND (instr(user,'更正') OR instr(user,'改为') OR instr(user,'改成')))) LIMIT 1",
                (workspace, topic),
            ).fetchone()
            if not edited:
                edition = "original"
    if not topic or not scene or not edition:
        want = "edition" if topic and scene else "topic"
        if not scene and not clue and not named and not managed.get("selected"):
            topic = None
        choices = [topic] if topic else list(pool)
        state = dict(
            reading=True,
            question=question,
            candidates=choices,
            selected=topic,
            scene=scene,
            edition=edition,
            edition_open=bool(unset),
        )
        reply = (
            "这处场景已经确定。你想核对最初读到的情节，还是后来改过的说法？"
            if want == "edition"
            else "你指的是哪一幕？请给一个能区分场景的动作、人物或物件线索。"
        )
        return dict(chosen=None, pending=state, clarification=reply)
    clarification = next(
        (clause.strip() for clause in re.split(r"(?<=[。！？?])", text) if _ASK_IDENTITY.search(clause)), ""
    )
    if managed:
        # Preserve the requested action without repeating a now-resolved declaration of ambiguity.
        clauses = re.split(r"(?<=[。！？?])", question)
        if len(clauses) > 1 and _POLAR.search(clauses[-2] if not clauses[-1] else clauses[-1]):
            question = next(c for c in reversed(clauses) if c)
        if clarification and clarification not in question:
            question += " " + clarification
    identify = bool(_ASK_IDENTITY.search(question))
    rev, chapter = docs[topic]
    literal = _scene(
        db,
        workspace,
        topic,
        docs[topic],
        scene,
        antecedents=edition == "original",
        identify=identify and edition == "original",
    )
    source, revision = (
        (literal, rev)
        if edition == "original"
        else _current_note(db, workspace, topic, chapter, literal or "", question)
    )
    marker = "按小说原始正文" if edition == "original" else "按最新工作稿"
    request = f"第{chapter}章。{marker}，{scene}那一幕。{question}"
    revisions = {revision} if source else set()
    locator = literal if identify and edition == "original" else ""
    selected_note = source if edition == "latest" else ""
    if edition == "latest" and source and literal:
        locator = literal if scene not in source and len(literal) <= 300 else ""
        if identify or re.search(r"还是(?:那个人|同一人|他|她)|刚才那个人|同一个人", question):
            linked = _scene(db, workspace, topic, docs[topic], scene, antecedents=True, identify=identify)
            if linked and (identify or linked != literal) and len(linked) <= 600:
                locator = linked
        if locator:
            revisions.add(rev)
            source = f"同一幕的定位原句：{locator}\n此幕当前更正：{source}"

    refs = db.execute(
        "SELECT revision,message_id FROM topic_message_refs WHERE workspace=? AND topic=? AND role='user' AND revision IN (?,?) ORDER BY message_id",
        (workspace, topic, rev, revision),
    )
    evidence_ids = [message for source_rev, message in refs if source_rev in revisions]
    # The selected evidence already identifies the scene. Keep the user's actual
    # question for generation and the expanded search query for retrieval fallback.
    grounded_question = f"第{chapter}章，{marker}。{question}"
    scoped = bool(source and len(evidence_ids) == len(revisions) and len(source) <= 800)
    if identify and edition == "latest" and rev not in revisions:
        scoped = False
    rule = revision_outcome_suffix(request)
    if identify and rule:
        rule = "\n先按定位原句说明所问人物，再列生效更正的对象和结果。成败未明说时，只给更正写明的结果。"
    return dict(
        chosen=topic,
        question=request,
        grounded_question=grounded_question,
        focus=dict(source_revision=rev, scene=scene, edition=edition),
        evidence=source if scoped else "",
        identity_source=locator if scoped and identify else "",
        selected_note=selected_note if scoped and identify else "",
        trim_opening=scoped and (identify or edition == "latest"),
        evidence_ids=evidence_ids,
        suffix=rule,
    )
