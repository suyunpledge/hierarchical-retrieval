"""Build the provider prompt for an already selected, isolated chat branch."""

import os
import re
from collections import Counter, deque
from math import log1p

from ..utils.text_processor import TextProcessor

_DECISION = re.compile(
    r"更正|改成|改为|纠正|撤销|以.{1,20}为准|确认|决定|下一步|接下来|待办|"
    r"\b(?:correction|instead|decided|next step|confirmed)\b",
    re.I,
)
_SCOPED_READING_BLOCKS = 1
_ROLES = {"user": "用户", "assistant": "助手"}


def _clip(text, limit):
    if len(text) <= limit:
        return text
    if limit < 4:
        return text[:limit]
    left = (limit - 1) // 2
    return text[:left] + "…" + text[-(limit - left - 1) :]


def _units(messages, start=0):
    """A side note is one user turn; a reply and its question move together."""
    i = start
    while i < len(messages):
        stop = i + 1
        if messages[i]["role"] == "user" and stop < len(messages) and messages[stop]["role"] == "assistant":
            stop += 1
        yield i, stop
        i = stop


def _tail_start(messages, limit, start=0):
    # Leave room to grow after crossing a boundary. Re-evaluating this on
    # durable history reproduces the same boundary after a worker restart.
    tail = deque()
    used = 0
    for begin, end in _units(messages, start):
        cost = sum(len(m.get("wire_content") or m["content"]) for m in messages[begin:end])
        tail.append((end, cost))
        used += cost
        if used > limit:
            while tail and used > limit // 2:
                start, removed = tail.popleft()
                used -= removed
    return start


def _opening_spans(content, block):
    """Use a few stable boundaries in a long opening record; preserve every character."""
    start = 0
    if len(content) > block * 3 // 2:
        for _ in range(2):
            if len(content) - start <= block:
                break
            end = start + block
            # Prefer a complete paragraph or sentence near the end of this block.
            lo = start + block * 3 // 4
            for separator in ("\n", "。", ". "):
                split = content.rfind(separator, lo, end)
                if split >= 0:
                    end = split + len(separator)
                    break
            yield start, end
            start = end
    yield start, len(content)


def _render_opening(message, block):
    text = message["content"]
    return [{"role": "user", "content": text[a:b]} for a, b in _opening_spans(text, block)]


def _sent_opening(prompt, offset, text, block):
    """Inspect the final wire layout, including ordinary (non-breakpoint) segments."""
    marks = []
    chars = spans = 0
    for i, (begin, end) in enumerate(_opening_spans(text, block), offset):
        if i >= len(prompt):
            break
        message = prompt[i]
        if (
            message["role"] != "user"
            or len(message["content"]) != end - begin
            or not text.startswith(message["content"], begin)
        ):
            break
        spans += 1
        chars = end
        if len(text) >= max(768, block) and end - begin >= block // 2:
            marks.append(i)
    return marks, chars, (3 if chars == len(text) and chars else spans)


def _opening(messages, stop, limit, block):
    """Only retain complete spans that were actually sent at the start of this branch."""
    opening = []
    used = 0
    first_turn_end = next(_units(messages))[1]
    for i in range(stop):
        m = messages[i]
        content = m.get("wire_content") or m["content"]
        if i >= first_turn_end and used + len(content) > block:
            return opening, i, used
        spans = _opening_spans(content, block) if i == 0 and m["role"] == "user" else [(0, len(content))]
        for a, b in spans:
            if used + b - a > limit:
                return opening, i, used
            opening.append({"role": m["role"], "content": content[a:b]})
            used += b - a
    return opening, stop, used


def _document_receipt(question):
    instruction, sep, document = question.partition("\n")
    if not sep or len(document) < 700 or len(instruction) > 120:
        return False
    # Only accept a standalone intake instruction above a substantial document.
    # Everything in the document still goes to the model and the original archive.
    return bool(
        re.fullmatch(
            r"(?:(?:以下(?:是|为)|这是|(?:请)?阅读)[^，,。；;！？?\n]{0,50}[，,]\s*)?"
            r"(?:第\s*\d+\s*[页章][，,]\s*)?(?:(?:留作原始依据|留作存档)[，,]\s*)?"
            r"(?:确认(?:留档|归档)即可|(?:读完(?:后)?|阅读后)?(?:请)?(?:仅(?:需)?|只(?:需)?)(?:回复|回答)"
            r"[「“]?(?:已读|收到|已收到|已收录|已归档)[」”]?)[。.!！]?",
            instruction.strip(),
        )
    ) and not re.search(
        r"对照|核对|校对|比较|核查|更正|之前|此前|根据|结合|参照|沿用|上[一页文轮]", instruction
    )


def _focused_chapter(question):
    if len(question) > 1024 or re.search(
        r"上(?:一|个)?章|下(?:一|个)?章|前一章|后一章|相邻|两章|跨章|其他章|其它章|对照|对比|比较|"
        r"\b(?:previous|next|adjacent|compare|across chapters)\b",
        question,
        re.I,
    ):
        return None
    chapters = set(TextProcessor.chapter_ids(question))
    return next(iter(chapters)) if len(chapters) == 1 else None


def _checkpoint(messages, start, limit, first=0, chapter=None):
    if start <= first or limit <= 0:
        return ""
    # Include both the last exchange and the most recent explicit user
    # decision. A later side note should not erase a pending step completely.
    ids = set(range(max(first, start - 2), start))
    for i in range(start - 1, first - 1, -1):
        if messages[i]["role"] == "user" and _DECISION.search(messages[i]["content"]):
            ids.add(i)
            break
    if chapter is not None:
        scoped = set()
        for i in ids:
            source = (
                i - 1
                if i > 0 and messages[i]["role"] == "assistant" and messages[i - 1]["role"] == "user"
                else i
            )
            sections = TextProcessor.chapter_spans(messages[source]["content"])
            if not sections or any(ch == chapter for ch, _, _ in sections):
                scoped.add(i)
        ids = scoped
        if not ids:
            return ""
    title = f"本话题较早记录（到第{start}条，按原顺序摘录）"
    share = (limit - len(title)) // len(ids) - 1
    if share < 12:
        return ""
    lines = [title]
    for i in sorted(ids):
        m = messages[i]
        lines.append(_clip(f"#{i + 1} {_ROLES[m['role']]}：{m['content']}", share))
    return "\n".join(lines)


def _excerpt(text, question, terms, limit):
    quote = TextProcessor.evidence_excerpt(text, question, limit, {})
    if quote:
        return quote
    if len(text) <= limit:
        return text
    # A single unpunctuated log line can exceed the entire prompt budget.
    lowered = text.lower()
    pos = min((lowered.index(t) for t in terms if t in lowered), default=0)
    start = min(max(0, pos - limit // 3), max(0, len(text) - limit + 2))
    return ("…" if start else "") + text[start : start + limit - 2] + "…"


_ORIGINAL_LOOKUP = re.compile(r"(?:小说|原始|未改动的)(?:的)?(?:原始)?(?:正文|原文)|原文版本|更正前的原文")
# Read once per process, so a request cannot change modes midway through assembly.
EVIDENCE_ORDER = os.getenv("HR_V1125_EVIDENCE_ORDER", "1").strip().lower()
if EVIDENCE_ORDER not in {"0", "1", "true", "false", "on", "off"}:
    raise ValueError("HR_V1125_EVIDENCE_ORDER expects 0 or 1")
EVIDENCE_ORDER = EVIDENCE_ORDER in {"1", "true", "on"}
_CURRENT_LOOKUP = re.compile(
    r"最后一次更正|现在生效|最新(?:的)?(?:工作稿|排演稿|更正|勘误|版本)|现行版本"
    if EVIDENCE_ORDER
    else r"最后一次更正|现在生效|最新(?:的)?(?:更正|勘误|版本)|现行版本"
)


_POLAR_QUESTION = re.compile(r"有没有|是否|有无|能否|可曾|了[吗么]|没有(?=[，,、。？！?\s]|$)|([一-鿿])不\1")
_REVISION_OUTCOME_RULE = (
    "\n对更正过的项目，按生效更正逐项原样列值。是非问只在更正明说时判定，否则直接列出它写的结果。"
)


def revision_outcome_suffix(question):
    """Qualify a polar follow-up only on an explicitly requested current edition."""
    marker = _CURRENT_LOOKUP.search(question)
    if not marker or re.search(r"原文|原稿|原版|原始正文|对照|对比|比较", question):
        return ""
    return _REVISION_OUTCOME_RULE if _POLAR_QUESTION.search(question, marker.end()) else ""


_AMENDMENT = re.compile(r"更正|勘误|改为|改成|替换为|撤销|作废|取消|撤回|以.{1,20}为准")
_FIELD_REPLACEMENT = re.compile(
    r"[“「\"]([^”」\"\n]{4,160})[”」\"]\s*(?:这[两二三四五六七八九十]项)?(?:依次|分别|统一|均|都)?\s*(?:改为|改成|替换为)"
)


def _exact_current_evidence(messages, question, chapter, limit, visible=""):
    """A complete, explicitly named field replacement needs only its latest source."""
    version = _CURRENT_LOOKUP.search(question)
    if not version or chapter is None:
        return None
    field = re.sub(r"^(?:的)?(?:版本)?[，,:：。 \t]*", "", question[version.end() :]).rstrip("。！？? \t")
    for i in range(len(messages) - 1, max(-1, len(messages) - 65), -1):
        m = messages[i]
        raw = m["content"].strip()
        if m["role"] != "user" or raw.endswith(("?", "？")) or not _AMENDMENT.search(raw):
            continue
        sections = set(TextProcessor.chapter_ids(raw))
        if sections and chapter not in sections:
            continue
        # A newer unscoped, revoked, partial or unrelated change may supersede
        # this field. Fall back to ordinary retrieval instead of skipping it.
        if (
            sections != {chapter}
            or len(raw) > 420
            or re.search(r"撤销|作废|取消|撤回|假如|如果|假设|例如", raw)
        ):
            return None
        match = _FIELD_REPLACEMENT.search(raw)
        if not match:
            return None
        name = match[1].strip().rstrip("。！？?")
        if name == field:
            # Echo the entire replacement clause when its full source is already visible.
            clause = raw[match.end() :] if raw in visible else raw
            line = f"用户更正#{i + 1}（按项替换原值）：{clause}"
            if len(line) + 1 <= limit:
                return [line], [i + 1]
        elif (not EVIDENCE_ORDER or name not in field) and not re.search(
            r"另外|此外|再说|顺便|还要|以及|还想", field
        ):
            companion = None
            linked = False
            if EVIDENCE_ORDER:
                expanded, companion = _source_queries(messages, question, chapter, marker=version)
                linked = expanded.endswith(" " + name) or companion == name
            scene = None
            if not linked and _named_fields(messages, chapter) == {name if EVIDENCE_ORDER else match[1]}:
                scene = _field_scene(
                    messages, chapter, field, name if EVIDENCE_ORDER else match[1], independent=True
                )
            if linked or scene:
                # Preserve the field as well as the values on a paraphrase. The
                # reader can check the scope, even following a wrong assistant turn.
                line = f"用户更正#{i + 1}（仅适用于所列事项）：{raw[match.start() :]}"
                if len(line) + 1 <= limit:
                    if (
                        EVIDENCE_ORDER
                        and not scene
                        and (companion or re.search(r"问|说|喊|回答|告诉", field + name))
                    ):
                        scene = _field_scene(messages, chapter, field, name)
                    if scene:
                        source_id, clause = scene
                        evidence = f"场景原句#{source_id}（定位所问对象）：{clause}\n{line}"
                        if len(evidence) + 1 <= limit:
                            return [evidence], sorted([source_id, i + 1])
                        if not linked:
                            return None
                    return [line], [i + 1]
        return None
    return None


def _named_fields(messages, chapter):
    names = set()
    for message in messages[-64:]:
        raw = message["content"]
        if message["role"] == "user" and len(raw) <= 420 and set(TextProcessor.chapter_ids(raw)) == {chapter}:
            match = _FIELD_REPLACEMENT.search(raw)
            if match:
                names.add(match[1])
    return names


def _source_queries(messages, question, chapter, *, marker=None):
    """Recover a previous field name for source ranking without importing its revised value."""
    marker = marker or _ORIGINAL_LOOKUP.search(question)
    if chapter is None or not marker:
        return question, None
    focus = question[marker.end() :].lstrip("，,:：。 ")
    terms = TextProcessor.query_terms(focus)
    if len(terms) < 3:
        return question, None
    names = set()
    related = set()
    fields = _named_fields(messages, chapter)
    characters = set("".join(terms))
    for name in fields:
        field_terms = TextProcessor.query_terms(name)
        shared = characters & set("".join(field_terms))
        if len(shared) >= 2:
            related.add(name)
        common = terms & field_terms
        extra = shared - set("".join(common))
        if len(common) >= 3 or (len(common) >= 2 and len(extra) >= 2):
            names.add(name)
    if len(names) == 1:
        name = names.pop()
        if name not in focus:
            return question + " " + name, None
    # With weak overlap the field might refer to another action. Keep the
    # direct result and try a second source lookup only for an unambiguous label.
    if not names and len(fields) == 1 and related:
        return question, related.pop()
    return question, None


_SOURCE_SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?]*[”’」』]*")


def _source_sentence_scores(raw, query):
    """Score literal sentences within one verified chapter, independently of index sampling."""
    # Chapter and edition have already selected the source. Keep all other
    # constraints, including ones the speaker put before the edition marker.
    focus = _ORIGINAL_LOOKUP.sub(" ", TextProcessor._CHAPTER.sub(" ", query))
    # query_terms omits isolated Han characters. A verb before 了 or after
    # 被 can be the only distinction between two actions by the same person.
    verbs = set()
    if EVIDENCE_ORDER:
        verbs = set(re.findall(r"[一-鿿](?=了(?:谁|什么|哪|几|对方|自己))|(?<=被)[一-鿿](?=的|到)", focus))
        focus = re.sub(r"对方|哪个部位|什么部位|哪个位置|哪一侧", " ", focus)
    terms = TextProcessor.query_terms(focus)
    # Characters contribute weakly when an action is reworded. Exact terms and rare
    # details carry more weight than a character's name repeated throughout a chapter.
    chars = {c for t in terms for c in t if "\u4e00" <= c <= "\u9fff"} | verbs
    spans, frequencies = [], Counter()
    for m in _SOURCE_SENTENCE.finditer(raw):
        sentence = m[0].lower()
        matches = {t for t in terms if t in sentence} | (chars & set(sentence))
        spans.append((m.start(), m.end(), matches))
        frequencies.update(matches)
    weights = {
        t: log1p((len(spans) - count + 0.5) / (count + 0.5)) * (0.18 if t in chars and t not in verbs else 1)
        for t, count in frequencies.items()
    }
    return [
        (sum(weights[t] for t in sorted(matches)), left, right) for left, right, matches in spans if matches
    ]


def _agreeing_paragraph(raw, query, field, scores):
    """Use a named field only when the question independently retrieves the same short paragraph."""
    anchors = _source_sentence_scores(raw, field)
    if not scores or not anchors:
        return None
    peak, left, right = max(anchors, key=lambda x: x[0])
    begin = raw.rfind("\n", 0, left) + 1
    stop = raw.find("\n", right)
    end = stop if stop >= 0 else len(raw)
    if end - begin > 220 or any(s >= peak * 0.85 for s, a, _ in anchors if not begin <= a < end):
        return None
    paragraph = raw[begin:end]
    if not any(term in paragraph.lower() for term in TextProcessor.query_terms(query)):
        return None
    top = sorted(scores, key=lambda x: -x[0])[:3]
    if any(begin <= a < end and s >= 0.75 * top[0][0] for s, a, _ in top):
        return begin, end
    return None


def _field_scene(messages, chapter, focus, field, *, independent=False):
    """Find a literal scene cue shared by a paraphrase and a named amendment."""
    query_chars = set("".join(TextProcessor.query_terms(focus)))
    field_chars = set("".join(TextProcessor.query_terms(field)))
    # The opening document and recent pastes cover the common paths without
    # reranking an unbounded archive on each paraphrase.
    recent = range(len(messages) - 1, max(0, len(messages) - 65), -1)
    for idx in (*recent, 0):
        entry = messages[idx]
        raw = entry["content"]
        if entry["role"] != "user" or len(raw) < 500:
            continue
        for cid, begin, end in TextProcessor.chapter_spans(raw):
            if cid != chapter:
                continue
            excerpt = raw[begin:end]
            if independent:
                bounds = _agreeing_paragraph(excerpt, focus, field, _source_sentence_scores(excerpt, focus))
                if bounds:
                    return idx + 1, excerpt[bounds[0] : bounds[1]].strip()
                continue
            scores = _source_sentence_scores(excerpt, focus + " " + field)
            if not scores:
                continue
            _, a, b = max(scores, key=lambda item: item[0])
            left = excerpt.rfind("\n", 0, a) + 1
            endline = excerpt.find("\n", b)
            paragraph = excerpt[left : endline if endline >= 0 else len(excerpt)].strip()
            # Dialogue can identify one action by two consecutive remarks. A
            # clause before the first remark does not establish that identity.
            if (
                len(paragraph) <= 220
                and re.search(r"问|说|喊|回答|告诉", focus + field)
                and re.search(r"[“「]", paragraph)
            ):
                if (
                    len(query_chars.intersection(paragraph)) >= 2
                    and len(field_chars.intersection(paragraph)) >= 2
                ):
                    return idx + 1, paragraph
            # Prefer the action that introduces the value. Copy it literally
            # and keep quoted original values out of the working-copy answer.
            for clause in re.split(r"[，,。！？；：]", excerpt[a:b]):
                if 6 <= len(clause) <= 96 and not re.search(r"[「」“”]", clause):
                    if (
                        len(query_chars.intersection(clause)) >= 2
                        and len(field_chars.intersection(clause)) >= 2
                    ):
                        return idx + 1, clause
    return None


def _source_excerpt(raw, query, limit, *, field=None):
    scores = _source_sentence_scores(raw, query)
    if not scores:
        return _excerpt(raw, query, TextProcessor.query_terms(query), limit)
    bounds = _agreeing_paragraph(raw, query, field, scores) if field else None
    if bounds and bounds[1] - bounds[0] <= limit:
        left, right = bounds
    else:
        _, left, right = max(scores, key=lambda item: item[0])
        if right - left > limit:
            return _excerpt(raw[left:right], query, TextProcessor.query_terms(query), limit)
        # Restore the utterance and its attribution together when they fit.
        a = raw.rfind("\n", 0, left) + 1
        end = raw.find("\n", right)
        b = end if end >= 0 else len(raw)
        if b - a <= limit:
            left, right = a, b
    if right - left < 100:
        neighbor = _SOURCE_SENTENCE.search(raw, right)
        allowance = min(80, limit - (right - left))
        if neighbor and neighbor.end() - right <= allowance:
            following = neighbor.end()
            if neighbor[0].lstrip().startswith(("“", "「", "‘")) and neighbor[0].endswith(("”", "」", "’")):
                # A short reply can be followed by its speaker and gesture on
                # the same line. Keep the complete paragraph within the same cap.
                newline = raw.find("\n", following, right + allowance + 1)
                stop = newline if newline >= 0 else len(raw)
                if stop - right <= allowance:
                    following = stop
            right = following
    return raw[left:right].strip()


def _recall(messages, start, question, visible, limit, search, *, expand_indexed=False, version=None):
    # A pasted report can dwarf the actual question. When the user labels a short
    # final question, use it as the search query and keep the entire input on the wire.
    if len(question) > 1024:
        focus = re.search(r"本轮问题[：:]\s*([^\n]{2,160})\s*$", question)
        if focus:
            question = focus[1].strip()
    terms = TextProcessor.query_terms(question)
    if not terms or limit < 30 or not start:
        return [], []
    target_chapter = _focused_chapter(question) if expand_indexed else None
    if version == "current":
        exact = _exact_current_evidence(messages, question, target_chapter, limit, visible)
        if exact:
            return exact
    # Resolve search results back to durable, branch-local source messages.
    # The untrusted snippet is useful only if it occurs verbatim in the cited entry.
    found = {}
    if search is not None:
        for hit in search(question, 8):
            index = hit["message_id"] - 1
            quote = hit.get("quote", "")
            if (
                0 <= index < start
                and hit["role"] == messages[index]["role"]
                and quote
                and quote in messages[index]["content"]
            ):
                found.setdefault(index, quote)

    # If a later user correction covers the whole current question, injecting
    # an older quote at the tail would give stale evidence undue prominence.
    first = 0
    historical = re.search(
        r"最早|原定|原先|起初|更正前|原文|原版|原始|对比|曾经|以前|之前|旧值|original|before", question, re.I
    )
    if not historical:
        coverage = terms
        current = re.compile(r"(?:目前|现行|当前|现在)(?:生效|有效|使用|采用|用|对应)?")
        if current.search(question):
            # The temporal qualifier is not an evidence field. Insert whitespace so
            # Chinese character bigrams cannot bridge field names to the removed words.
            query = current.sub(" ", question)
            coverage = TextProcessor.query_terms(query) or terms
        # With an indexed lookup, check recent corrections plus its hits;
        # avoid re-scanning a years-long archive on every user question.
        scan = (
            range(len(messages) - 1, -1, -1)
            if search is None
            else sorted({*found, *range(max(0, start - 64), len(messages))}, reverse=True)
        )
        for i in scan:
            m = messages[i]
            if version and m["content"].rstrip().endswith(("?", "？")):
                continue
            if m["role"] == "user" and re.search(
                r"更正|改为|改成|以.{1,20}为准|corrected|instead", m["content"], re.I
            ):
                lower = m["content"].lower()
                if all(t in lower for t in coverage):
                    first = i
                    break

    candidates = []
    scan = (
        range(first, start)
        if search is None
        else sorted({i for i in found if i >= first}.union(range(max(first, start - 64), start)))
    )
    if version and first == 0:
        scan = sorted({0, *scan})
    if version == "original" and target_chapter and search is not None:
        # A document pasted into a normal turn has a sampled quote index. Its heading
        # still identifies the source after both the top hits and recent tail move on.
        archived = (
            i
            for i in range(max(0, start - 64))
            if messages[i]["role"] == "user"
            and len(messages[i]["content"]) >= 500
            and any(
                ch == target_chapter for ch, _, _ in TextProcessor.chapter_spans(messages[i]["content"][:256])
            )
        )
        scan = sorted({*scan, *archived})
    for i in scan:
        m = messages[i]
        text = m["content"]
        lower = text.lower()
        hits = sum(t in lower for t in terms)
        if not (hits or i in found) or not text:
            continue
        correction = m["role"] == "user" and bool(_DECISION.search(text))
        priority = 0
        if version:
            if m["role"] == "user" and text.rstrip().endswith(("?", "？")) and len(text) < 500:
                continue  # A past question has no asserted value.
            if (
                version == "original"
                and m["role"] == "user"
                and target_chapter in (ch for ch, _, _ in TextProcessor.chapter_spans(text))
                and len(text) >= 500
            ):
                priority = 2
            elif (
                version == "current" and correction and re.search(r"改为|改成|以.{1,35}为准|更正[：:]", text)
            ):
                priority = 2
        if text in visible and not (EVIDENCE_ORDER and priority == 2 and version == "original"):
            continue
        candidates.append((priority, i in found, hits, correction, m["role"] == "user", i))
    source_query, companion = (
        _source_queries(messages, question, target_chapter) if version == "original" else (question, None)
    )
    fields = (
        _named_fields(messages, target_chapter)
        if version == "original" and source_query == question and not companion
        else set()
    )
    independent_field = next(iter(fields)) if len(fields) == 1 else None
    lines, ids = [], []
    has_original = False
    for priority, _, _, _, _, i in sorted(candidates, reverse=True):
        m = messages[i]
        if (
            has_original
            and priority != 2
            and (EVIDENCE_ORDER or m["role"] == "assistant" or _AMENDMENT.search(m["content"]))
        ):
            continue
        label = f"原文#{i + 1} {_ROLES[m['role']]}："
        available = min(420, limit - len(label) - 1)
        if available < 20:
            break
        raw = m["content"]
        body = found.get(i, raw)
        if target_chapter is not None:
            sections = TextProcessor.chapter_spans(raw)
            if sections:
                relevant = [(a, b) for chapter, a, b in sections if chapter == target_chapter]
                if not relevant:
                    continue
                a, b = next(((a, b) for a, b in relevant if raw.find(body, a, b) >= 0), relevant[0])
                raw = raw[a:b]
                if body not in raw:
                    body = raw
        if expand_indexed and i in found:
            # The quote that identified the scene and the sentence that answers
            # a follow-up can be different. Look on both sides of the index hit.
            offset = raw.index(body)
            body = raw[max(0, offset - available) : offset + len(body) + available]
        if priority == 2 and version == "current" and len(raw) <= available:
            body = raw  # Indexed snippets can separate a correction from its new values.
        is_original = priority == 2 and version == "original"
        if is_original:
            quote = _source_excerpt(raw, source_query, available, field=independent_field)
            if companion:
                other = _source_excerpt(raw, source_query + " " + companion, available)
                if quote in other or (not EVIDENCE_ORDER and quote in visible):
                    quote = other
                elif (
                    other not in quote
                    and (EVIDENCE_ORDER or other not in visible)
                    and len(quote) + len(other) + 1 <= available
                ):
                    quote += "\n" + other
            has_original = bool(quote)
        else:
            quote = _excerpt(body, question, terms, available)
        if not is_original and expand_indexed and quote in body and i in found:
            # Keep the following local context as well: subjects, actions and their
            # outcomes are often split over several short dialogue paragraphs.
            begin = body.index(quote)
            end = min(len(body), begin + available, begin + len(quote) + 150)
            boundary = max((body.rfind(stop, begin + len(quote), end) for stop in "。！？\n"), default=-1)
            if boundary >= begin + len(quote):
                quote = body[begin : boundary + 1].strip()
        if not quote or (quote in visible and not (EVIDENCE_ORDER and is_original)):
            continue
        line = label + quote
        lines.append((i, line))
        ids.append(i + 1)
        limit -= len(line) + 1
        if len(lines) == 3:
            break
    return [v for _, v in sorted(lines)], sorted(ids)


def chapter_prefix_layout(messages, question):
    """Use a reproducible opening boundary only for long Chinese chapter material."""
    first = (messages[0].get("wire_content") or messages[0]["content"]) if messages else question
    return (
        (not messages or messages[0]["role"] == "user")
        and len(first) >= 1800
        and bool(re.search(r"(?m)^[ \t　]*第\s*\d+\s*章", first[:240]))
        and len(re.findall(r"[一-鿿]", first[:480])) >= 200
    )


def build_branch_context(
    *,
    system,
    topic,
    messages,
    question,
    char_budget=6000,
    evidence="",
    position="",
    source="",
    search=None,
    prefix_blocks=None,
    allow_document_receipt=False,
    compact_lookup=False,
    chapter_prefix=False,
    tail_floor=0,
    reading=None,
):
    """Compose chat messages; persist wire_user with its original human question.

    The budget counts Unicode content characters (including system and the
    current question). Provider framing and tokenization are outside it.
    The caller supplies history and any external evidence scoped to this topic.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("当前提问不能为空")
    if any(m["role"] not in _ROLES for m in messages):
        raise ValueError("话题历史只接受用户消息和助手消息")
    head = [{"role": m["role"], "content": m["content"]} for m in system]
    if any(m["role"] not in ("system", "developer") for m in head):
        raise ValueError("固定指令只接受 system 或 developer 角色")
    head_chars = sum(len(m["content"]) for m in head)
    if head_chars + len(question) > char_budget:
        raise ValueError("固定指令和本轮原始提问超过上下文字符预算")

    room = char_budget - head_chars
    free = room - len(question)
    label = "当前话题：" + topic
    if len(label) <= free:
        head.append({"role": "user", "content": label})
        room -= len(label)
        free -= len(label)

    # These usual-case shares depend on the branch budget rather than the
    # latest wording. An exceptionally long question has priority.
    if messages and allow_document_receipt and not evidence and _document_receipt(question):
        return {
            "messages": [*head, {"role": "user", "content": question}],
            "chars": char_budget - free,
            "checkpoint": len(messages),
            "one_time_document": head_chars + len(label) <= 256,
            "wire_user": question,
            "recalled": [],
            "cache_breakpoints": [],
            "opening_chars": 0,
            "prefix_blocks": 0,
        }

    opening_offset = len(head)
    opening_block = max(256, room // 5)
    if chapter_prefix:
        first = (messages[0].get("wire_content") or messages[0]["content"]) if messages else question
        # The old ~1,080-character span was too short to yield any cache reads
        # on the development chapters. Preserve a whole, repeatable opening.
        if len(first) > 2100:
            opening_block = max(opening_block, min(1450, (2 * len(first) - 1) // 3))
        else:
            opening_block = max(opening_block, 1450)
    history_limit = min(room * 3 // 5, free)
    # A cold branch can retrieve facts from disk. Spend its replay allowance on
    # recent dialogue and complete opening blocks only.
    opening_cap = history_limit
    if prefix_blocks is not None and (type(prefix_blocks) is not int or not 0 <= prefix_blocks <= 3):
        raise ValueError("prefix_blocks must be 0, 1, 2 or 3")
    bounded = prefix_blocks is not None and prefix_blocks < 3
    version = None
    if bounded:
        first = (messages[0].get("wire_content") or messages[0]["content"]) if messages else ""
        spans = list(_opening_spans(first, opening_block)) if first else []
        opening_cap = sum(b - a for a, b in spans[:prefix_blocks])
        recent_budget = room // 3 if prefix_blocks == 1 and tail_floor else room // 6
        if compact_lookup and _focused_chapter(question) is not None:
            original = _ORIGINAL_LOOKUP.search(question)
            latest = _CURRENT_LOOKUP.search(question)
            if bool(original) != bool(latest):
                version = "original" if original else "current"
        if prefix_blocks == 0 and version:
            # A grounded lookup names its source and version. Keep space for
            # fresh source excerpts and the last decision when older turns fold.
            recent_budget = room // 10
        history_limit = min(history_limit, opening_cap + recent_budget)
    if messages and reading and reading["evidence"] and not (source or position or evidence):
        ask = reading.get("grounded_question", question)
        wire = f"{ask}\n依据：{reading['evidence']}{reading['suffix']}"
        if len(wire) <= room:
            # Source selection already validated this single-scene answer. Only reuse
            # complete opening spans, and always reserve room for the current evidence.
            original = messages[0]
            first = original.get("wire_content") or original["content"]
            if chapter_prefix and bounded and reading.get("trim_opening"):
                # The verified source and any requested identities already fit in the tail.
                # Only reuse a span that was selected by the ordinary warm-prefix policy.
                retained = min(prefix_blocks, _SCOPED_READING_BLOCKS)
                opening_cap = sum(b - a for a, b in spans[:retained])
            prefix, _, _ = _opening(messages, 1, min(opening_cap, room - len(wire)), opening_block)
            prompt = [*head, *prefix]
            marks, opening_chars, sent_blocks = ([], 0, 0)
            if original["role"] == "user":
                marks, _, _ = _sent_opening(prompt, opening_offset, first, opening_block)
            if marks:
                prompt = prompt[: marks[-1] + 1]
                _, opening_chars, sent_blocks = _sent_opening(prompt, opening_offset, first, opening_block)
            else:
                prompt = list(system)
            archive_wire = wire
            anchor = reading.get("identity_source")
            if chapter_prefix and marks and anchor:
                # The literal and its antecedents are shared across original/draft
                # turns. Keep revisions and user phrasing after this second boundary.
                stable = "所问人物的定位原句：\n" + anchor
                note = reading.get("selected_note")
                end = ask + ("\n此幕当前更正：" + note if note else "") + reading["suffix"]
                if sum(len(m["content"]) for m in prompt) + len(stable) + len(end) <= char_budget:
                    marks = [*marks, len(prompt)]
                    prompt.append({"role": "user", "content": stable})
                    wire = end
            prompt.append({"role": "user", "content": wire})
            return {
                "messages": prompt,
                "chars": sum(len(m["content"]) for m in prompt),
                "wire_user": archive_wire,
                "recalled": reading["evidence_ids"],
                "checkpoint": len(messages),
                "opening_chars": opening_chars,
                "prefix_blocks": sent_blocks,
                "cache_breakpoints": marks,
                "cache_evidence_prefix_only": bool(marks),
            }
    start = _tail_start(messages, history_limit, tail_floor)
    consumed = 0
    snapshot_limit = 0
    if start:
        # Reuse the beginning that established this branch even across folds. Only the
        # already-folded portion may be pinned. Leave part of the same history budget
        # for recent turns as well as separate space for current evidence.
        # Keep some capacity for later evidence and corrections without resizing an
        # existing span according to the wording or length of the current question.
        opening_limit = min(
            history_limit - min(room // 12, history_limit // 5), free - min(room // 8, free // 3)
        )
        opening, consumed, opening_chars = _opening(
            messages, start, min(opening_cap, opening_limit), opening_block
        )
        head.extend(opening)
        free -= opening_chars
        # Reserve a reproducible checkpoint size before admitting variable recent
        # turns. Current questions keep their full text, and recall gets its own room.
        snapshot_limit = min(room // 12, free // 2)
        evidence_reserve = min(room // 10, free // 3)
        tail_limit = min(history_limit - opening_chars, free - snapshot_limit - evidence_reserve)
        start = _tail_start(messages, tail_limit, max(consumed, tail_floor))
    tail = [{"role": m["role"], "content": m.get("wire_content") or m["content"]} for m in messages[start:]]
    if not start and tail and tail[0]["role"] == "user":
        tail[:1] = _render_opening(tail[0], opening_block)
    free -= sum(len(m["content"]) for m in tail)
    if start == consumed:
        start = 0  # Every prior message is present in its original order.
    chapter = _focused_chapter(question) if bounded and start else None
    snapshot = _checkpoint(messages, start, snapshot_limit, consumed, chapter=chapter)
    if snapshot:
        head.append({"role": "user", "content": snapshot})
        free -= len(snapshot)
    visible = "\n".join(m["content"] for m in head + tail)
    notes = []
    suffix = "\n\n用户更正覆盖的项目只取新值。\n本轮问题：\n" if version == "current" else "\n\n本轮问题：\n"
    if version == "original":
        suffix = "\n\n只答所问的原文场景。区分尝试和结果，未交代的细节答未说明。\n本轮问题：\n"
    remaining = free - len(suffix)
    for name, value in (("话题进度", position), ("来源", source), ("本轮参考资料", evidence)):
        if not value:
            continue
        line = f"{name}：{value}"
        if name == "本轮参考资料":
            if line in visible:
                continue
        else:
            entries = [s for s in visible.splitlines() if s.startswith(name + "：")]
            if entries and entries[-1] == line:
                continue
        if remaining <= len(name) + 10:
            break
        line = _clip(line, remaining)
        notes.append(line)
        remaining -= len(line) + 1
    quotes, ids = _recall(
        messages, start, question, visible, remaining, search, expand_indexed=bounded, version=version
    )
    notes.extend(quotes)
    wire_user = "\n".join(notes) + suffix + question if notes else question
    # A previous model answer is not a source for an explicit edition lookup.
    # When it competes with a named replacement or a repeated scene, read that
    # source independently; keep ordinary conversational follow-ups on the branch.
    if EVIDENCE_ORDER and start and len(quotes) == 1 and not (evidence or position or source):
        line = quotes[0]
        scoped = (
            version == "current"
            and line.startswith(("用户更正#", "场景原句#"))
            and "仅适用于所列事项" in line
        )
        if version == "original" and ids:
            raw = messages[ids[0] - 1]["content"]
            literal = line.partition("：")[2].splitlines()[0]
            scoped = (
                len(literal) >= 12
                and literal in visible
                and literal in raw
                and len(raw) >= 500
                and messages[ids[0] - 1]["role"] == "user"
                and chapter in (c for c, _, _ in TextProcessor.chapter_spans(raw))
                and len(TextProcessor.query_terms(question[_ORIGINAL_LOOKUP.search(question).end() :])) >= 4
            )
        if scoped:
            rule = (
                "原句只用来辨认是哪件事。仅输出该事项在最新工作稿中的各项新值。若证据指向不同事项，说明依据不足。"
                if version == "current"
                else "只答所问的原文场景。按原句区分做了什么和是否做到，未交代的细节答未说明。"
            )
            isolated = question + "\n\n" + rule + "\n" + line
            if len(isolated) <= room:
                keep = opening_offset
                if version == "original" and any(literal in m["content"] for m in opening):
                    if sum(len(m["content"]) for m in opening) + len(isolated) <= room:
                        keep += len(opening)
                head, tail, wire_user = head[:keep], [], isolated
    current = {"role": "user", "content": wire_user}
    prompt = head + tail + ([current] if messages else _render_opening(current, opening_block))
    original = messages[0] if messages else current
    marks, opening_chars, sent_blocks = ([], 0, 0)
    if original["role"] == "user":
        marks, opening_chars, sent_blocks = _sent_opening(
            prompt, opening_offset, original.get("wire_content") or original["content"], opening_block
        )
    chars = sum(len(m["content"]) for m in prompt)
    if messages and os.getenv("HR_GROUND_REVISION_OUTCOMES", "0") == "1":
        suffix = revision_outcome_suffix(question)
        if suffix and chars + len(suffix) <= char_budget:
            # Apply only to this request; keep past wire turns free of edition-specific instructions.
            prompt[-1] = {**current, "content": wire_user + suffix}
            chars += len(suffix)
    return {
        "opening_chars": opening_chars,
        "prefix_blocks": sent_blocks,
        "messages": prompt,
        "cache_breakpoints": marks,
        "wire_user": wire_user,
        "checkpoint": start,
        "recalled": ids,
        "chars": chars,
    }
