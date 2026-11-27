"""Audit affirmative precision and completed actions against the scene being asked about."""

import re

_SIDE = re.compile(
    r"([左右])(?:边|侧)?(?:的)?(?:那只|那边的)?(耳(?:朵|边|垂)?|眼(?:睛)?|肩(?:膀)?|手(?:腕|掌|背)?|脚(?:踝)?|膝(?:盖)?|脸(?:颊)?)"
)
_OUTCOME = re.compile(r"叫醒|唤醒|推开|打开")
_DENIAL = re.compile(
    r"未说明|没有说明|不(?:能|可)确定|不能确认|不清楚|未交代|没有交代|不知|是否|哪(?:一)?侧|究竟|还是|无法|没有写|未写|没写|并非|不是"
)
_INTENT = re.compile(
    r"(?:没(?:有|能)?|未(?:能)?|无法|不能|想(?:要)?|打算|试图|尝试|试着|准备|要|能否|没有被|未被|想把.{0,8})$"
)
# A completed event can be expressed without repeating the verb used by the question.
_OUTCOME_PROOF = {
    "叫醒": r"叫醒|唤醒|醒了|醒来|醒过来|睁开.{0,4}眼",
    "唤醒": r"叫醒|唤醒|醒了|醒来|醒过来|睁开.{0,4}眼",
    "推开": r"推开|推了开|应声而开|门开了|打开",
    "打开": r"打开|开启|推开|拉开|门开了|敞开",
}


def local_scene(chapter, witness):
    pos = chapter.find(witness)
    if pos < 0:
        return witness
    before = re.split(r"\n\s*\n", chapter[:pos].rstrip())[-1]
    after = re.split(r"\n\s*\n", chapter[pos + len(witness) :].lstrip())[0]
    return before[-120:] + "\n" + witness + "\n" + after[:160]


def positive(text, match):
    around = re.split(r"[。！？；;\n]", text[: match.start()])[-1]
    clause = around + re.split(r"[。！？；;\n]", text[match.start() :], maxsplit=1)[0]
    return not _DENIAL.search(clause) and not _INTENT.search(around.rstrip())


def unsupported_refinements(answer, scene):
    problems = []
    for m in _SIDE.finditer(answer):
        if not positive(answer, m):
            continue
        side, part = m.groups()
        if not any(s[1] == side and s[2][0] == part[0] for s in _SIDE.finditer(scene)):
            problems.append(m[0])
    for m in _OUTCOME.finditer(answer):
        if not positive(answer, m):
            continue
        # A quoted user premise followed by an explicit qualification is not an assertion.
        if re.search(r"(?:未|没(?:有)?)(?:写|说明|交代).{0,12}(?:醒|开)|仍.{0,10}(?:闭|睡)|未必", answer):
            continue
        affirmative = [v for v in re.finditer(_OUTCOME_PROOF[m[0]], scene) if positive(scene, v)]
        if not affirmative and re.search(
            r"紧闭双眼|还在睡|仍在睡|睡着了|纹丝不动|没有推开|未能打开|依然紧闭", scene
        ):
            problems.append(m[0])
    return list(dict.fromkeys(problems))
