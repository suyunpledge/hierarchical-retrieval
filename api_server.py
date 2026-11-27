"""
分级检索 API 服务 (FastAPI)
将 HierarchicalRetrieval 暴露为 HTTP 接口，并提供 OpenAI 兼容的
/v1/chat/completions 对话代理：长历史在代理处被压缩为
"记忆块 + 近期对话"再转发给本地模型，从结构上杜绝上下文溢出。
"""
import sys
import os
import json
import time
import uuid
import logging
import re
import secrets
import threading
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Optional, List

# 确保能导入 hierarchical_retrieval
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

from hierarchical_retrieval import HConfig, HierarchicalRetrieval
from hierarchical_retrieval.topic_chat import TopicLocation, TopicMode, prepare_turn, workspace_call, workspace_store
from hierarchical_retrieval.utils.text_processor import TextProcessor
from hierarchical_retrieval.core.tool_receipts import ReceiptIdentityConflict, ReceiptScope, ToolReceipt, ToolReceiptStore

# ── 日志配置 ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI 应用 ──────────────────────────────────────
app = FastAPI(
    title="分级检索 API",
    description="基于本地嵌入模型（默认 bge-m3）的三级架构检索服务 (CloudCache + KeySentence + TopicDomain) + OpenAI 兼容上下文压缩代理",
    version="0.6.0",
)

# 允许跨域（方便前端或测试工具调用）。
# B3 修复：删去 allow_credentials——它与 allow_origins=["*"] 的组合违反
# CORS 规范（等于向任意来源放行带凭证的请求）。默认仅本机监听，
# 通配来源即可；确需凭证白名单时改为显式 origins 列表。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局配置 ──────────────────────────────────────────
STORAGE_ROOT = os.environ.get("HR_STORAGE_ROOT", "./hierarchical_api_storage")
EMBED_URL = os.environ.get("HR_EMBED_URL", os.environ.get("HR_OLLAMA_URL", "http://localhost:11434"))
EMBED_BACKEND = os.environ.get("HR_EMBED_BACKEND", "auto")  # auto | ollama | openai(llama.cpp)
CHAT_MODEL = os.environ.get("HR_CHAT_MODEL", "")  # /v1 代理的回退对话模型
# HR_EMBED_MODEL / HR_EMBED_DIM / HR_TOPIC_*（安装器注入）：切换嵌入模型预设
# （bge-m3 ↔ nomic-embed-text）与相似度阈值；不设置时保持下面 config 里的
# bge-m3 默认值，行为与不带补丁的版本完全一致。
EMBED_MODEL = os.environ.get("HR_EMBED_MODEL", "")
EMBED_DIM_RAW = os.environ.get("HR_EMBED_DIM", "")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


THRESH_TOPIC_MIN = _env_float("HR_TOPIC_MIN_SIM", 0.70)
THRESH_TOPIC = _env_float("HR_TOPIC_SIM", 0.55)
THRESH_KEY = _env_float("HR_KEY_SIM", 0.60)
# 全文段落直检的混合分阈值（tests/eval_recall.py 标定：bge-m3 下 0.55 时
# 召回 100%、无关问题误命中 0%）；HR_CHUNK_SIM=1 可等效关闭该路径
THRESH_CHUNK = _env_float("HR_CHUNK_SIM", 0.55)
CHAT_BACKEND_URL = os.environ.get("HR_CHAT_BACKEND", EMBED_URL)
# 对话后端协议：1/0 强制 OpenAI/Ollama 协议；auto（默认）跟随嵌入后端探测结果
CHAT_BACKEND_OPENAI = os.environ.get("HR_CHAT_BACKEND_OPENAI", "auto")
# B3 修复：可选 Bearer 鉴权。设置 HR_API_TOKEN 后，除 /health 外的
# 接口要求 Authorization: Bearer <token>（LobeChat 侧把 API Key 填成
# 同名值即可）。默认为空 = 不鉴权（保持仅本机监听的默认形态）。
API_TOKEN = os.environ.get("HR_API_TOKEN", "")

config = HConfig(
    storage_root=STORAGE_ROOT,
    embedding_base_url=EMBED_URL,
    embedding_backend=EMBED_BACKEND,
    chat_backend_url=CHAT_BACKEND_URL,
    # M6 修复：API 服务主要跑中文对话，显式采用中文场景的相似度阈值，
    # 避免沿用英文默认阈值导致对话被误并入同一话题域、L3 形同虚设。
    topic_min_similarity=THRESH_TOPIC_MIN,
    topic_similarity_threshold=THRESH_TOPIC,
    key_sentence_similarity_threshold=THRESH_KEY,
    chunk_similarity_threshold=THRESH_CHUNK,
    context_char_budget=_env_int("HR_CONTEXT_CHAR_BUDGET", HConfig.context_char_budget),
    embedding_cache_size=_env_int("HR_EMBEDDING_CACHE_SIZE", 4096),
    retrieval_cache_size=_env_int("HR_RETRIEVAL_CACHE_SIZE", 256),
    fulltext_cache_size=_env_int("HR_FULLTEXT_CACHE_SIZE", 128),
)
# 嵌入模型预设覆盖（安装器注入；不设置则保持 bge-m3 默认，行为不变）
if EMBED_MODEL:
    config.embedding_model = EMBED_MODEL
if EMBED_DIM_RAW.isdigit():
    config.embedding_dim = int(EMBED_DIM_RAW)
if CHAT_MODEL:
    config.chat_model = CHAT_MODEL

logger.info(f"[启动] 存储目录: {STORAGE_ROOT}")
logger.info(f"[启动] Embedding 服务: {EMBED_URL} (backend={EMBED_BACKEND})")
logger.info(f"[启动] 对话后端: {CHAT_BACKEND_URL} (openai协议={CHAT_BACKEND_OPENAI})")

# 检索引擎（全局单例，惰性初始化）。
# v0.4 修复：import 本模块不再立即构建引擎——此前单元测试 import
# api_server 就会触发一次真实 Ollama 健康检查与存储目录创建；
# 引擎首次使用时才创建。测试/外部代码可直接对 api_server.hr 赋值
# 注入替身，get_hr() 会原样返回注入值。
hr: Optional[HierarchicalRetrieval] = None
_hr_lock = threading.Lock()


def get_hr() -> HierarchicalRetrieval:
    """获取检索引擎单例；允许测试预先注入 api_server.hr。"""
    global hr
    if hr is None:
        with _hr_lock:
            if hr is None:
                hr = HierarchicalRetrieval(config=config)
    return hr


# ── 可选 Bearer 鉴权（B3 修复）─────────────────────────
def verify_token(authorization: Optional[str] = Header(None)) -> None:
    """HR_API_TOKEN 已设置时，校验 Authorization: Bearer <token>。"""
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="未授权：缺少 Authorization: Bearer <token> 请求头",
        )
    if not secrets.compare_digest(authorization[len("Bearer "):].strip(), API_TOKEN):
        raise HTTPException(status_code=401, detail="未授权：API token 校验失败")


# ── 请求/响应模型 ──────────────────────────────────────

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="对话全文", example="今天讨论一下微服务架构方案...")
    metadata: Optional[dict] = Field(None, description="附加元数据", example={"source": "meeting", "speaker": "Alice"})


class IngestResponse(BaseModel):
    entry_id: str
    domain_id: str
    domain_name: str
    sentence_ids: List[str]


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询文本", example="微服务架构")
    top_k_sentences: Optional[int] = Field(None, ge=0, le=100, description="每个话题检索的关键句数")
    top_k_context: Optional[int] = Field(None, ge=0, le=100, description="每个关键句关联的全文段落数")


class RetrieveResponse(BaseModel):
    query: str
    detected_domain: Optional[dict] = None
    key_sentences: List[dict] = Field(default_factory=list)
    full_contexts: List[dict] = Field(default_factory=list)
    summary: str = ""


class HealthResponse(BaseModel):
    status: str
    ollama_url: str
    storage_root: str
    embedding_backend: str = ""


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, v: Any) -> Any:
        """兼容 OpenAI 多段内容格式 [{"type": "text", "text": ...}, ...]。

        LobeChat 等客户端带图片/附件时会发送数组形式的 content，
        原先 str 类型直接 422。这里只保留文本段（后端走纯文本）。
        """
        if v is None:
            return ""
        if isinstance(v, list):
            return "\n".join(
                part.get("text", "") for part in v
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return v


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    topic_workspace: Optional[TopicMode] = None
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ── 路由 ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查"""
    emb = get_hr().embedding
    return HealthResponse(
        status="ok",
        ollama_url=config.embedding_base_url,
        storage_root=STORAGE_ROOT,
        embedding_backend=getattr(emb, "backend", "unknown"),
    )


@app.get("/stats", dependencies=[Depends(verify_token)])
def stats():
    """缓存命中统计（嵌入向量 + 检索结果 + 全文记录）与库规模。"""
    return get_hr().status()


# ── OpenAI 兼容代理：上下文压缩的落点 ─────────────────────
#
# 为什么需要它：LobeChat 等客户端是"全量重放"模式，每轮请求都把完整
# 历史塞进 prompt 发给模型——只把分级检索挂在旁边（ingest 钩子）不改写
# 发出的 payload，历史该多长还是多长，本地小模型照样溢出。
# 把客户端的"模型服务商"指向本服务 /v1，历史在这里被压缩后再转发，
# 发出去的 prompt 从结构上就有界。

# M5.1：单条超长消息截断装配时追加的尾部标记（9 字符）
TRUNC_MARK = "（原文过长已截断）"


# ── 对话后端协议判定 ──────────────────────────────

def _chat_backend_is_openai() -> bool:
    """
    判定对话后端走 OpenAI 协议（/v1/chat/completions）还是 Ollama 协议（/api/chat）。

    HR_CHAT_BACKEND_OPENAI: 1/true → 强制 OpenAI；0/false → 强制 Ollama；
    auto（默认）→ 跟随嵌入后端探测结果（同一服务既出嵌入又出对话时，
    嵌入探测结果即可代表该服务的协议）。
    """
    flag = CHAT_BACKEND_OPENAI.strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    try:
        return get_hr().embedding.backend != "ollama"
    except Exception:
        return True


ROLE_LABELS = {"user": "用户", "assistant": "助手"}
MIN_TURN_CHARS = 8          # 一轮对话的正文少于该字数视为寒暄，不入库
FOLLOWUP_MAX_CHARS = 40
ACTION_SOURCE_RULE = re.compile(r"问动作(?:时)?(?:只|一律|统一|请|都)?(?:按|核对|查|看)(?:同章|该章|小说)?(原文|排演|校订)")
QUESTION_SEPARATOR = "\n\n本轮问题：\n"
CONTEXTUAL_QUESTION = re.compile(r"^(?:那[么、， ]*|然后|接着|除此之外|还有呢|他|她|它|这个|那个)")
PRONOUN_QUESTION = re.compile(r"^(?:那[么、， ]*)?(?:他|她|它|这件事|那件事)")
SHORT_FACT_QUESTION = re.compile(r"^[一-鿿]{2,8}(?:谁|哪里|多少|几点|多久|哪天)(?:来着)?[？?]?$")
NO_MEMORY_NOTE = "（更早的对话已被压缩，且没有检索到与当前问题相关的记录。如果问题涉及更早的内容，请直接说明不记得，不要猜测。）"


def _needs_question_context(question: str) -> bool:
    return len(question) <= FOLLOWUP_MAX_CHARS and bool(
        CONTEXTUAL_QUESTION.match(question)
        or question.rstrip("？?").endswith("呢")
        or SHORT_FACT_QUESTION.fullmatch(question)
    )


def _turn_texts(older: List[ChatMessage], continues: bool) -> List[str]:
    """把旧消息按轮分组并渲染成入库文本（每条消息一行，带"用户："/"助手："前缀）。

    一轮 = 一条用户消息 + 其后直到下一条用户消息之前的助手消息。
    continues=True 表示最后一轮的回复还留在"近期对话"里，该轮先不入库，
    等它完整滑出近期窗口再入——否则同一轮会以两种文本各入库一次。
    """
    turns: List[List[ChatMessage]] = []
    current: List[ChatMessage] = []
    for m in older:
        if m.role not in ROLE_LABELS or not m.content.strip():
            continue
        if m.role == "user" and current:
            turns.append(current)
            current = []
        current.append(m)
    if current and not continues:
        turns.append(current)
    texts = []
    for turn in turns:
        if sum(len(m.content.strip()) for m in turn) < MIN_TURN_CHARS:
            continue
        texts.append("\n".join(f"{ROLE_LABELS[m.role]}：{m.content.strip()}" for m in turn))
    return texts


def _compress_history(messages: List[ChatMessage]) -> List[ChatMessage]:
    """历史超阈值时：旧消息入库并检索记忆块，近期对话原样保留。

    输出有界：头部 system + 记忆块(≤context_char_budget) + 近 K 条消息
    (总量同样裁到 context_char_budget 以内，单条超长消息截断装配)，
    与历史长度无关（v0.3 边界说明：len(messages)≤keep 时不压缩，
    该旁路与 num_ctx 未下发的问题见修复说明-v0.3 遗留项）。
    """
    total = sum(len(m.content) for m in messages)
    keep = config.recent_turns_keep
    if total <= config.compress_threshold_chars or len(messages) <= keep:
        return list(messages)

    # 头部 system 提示词始终保留
    head: List[ChatMessage] = []
    body = list(messages)
    if body and body[0].role == "system":
        head = [body[0]]
        body = body[1:]

    older, recent = body[:-keep], body[-keep:]

    # 旧消息写入分级检索库（幂等：重复内容自动去重）。
    # B1 修复：补录窗口 = "未入库 ∩ 最旧优先"。
    # 旧实现取 older 段的 [-cap:]（最靠后的一批），窗口随对话增长向
    # 前滑动，最早的一批消息永远落在窗口外，与"留到下一轮继续补"
    # 的注释正好相反——60 条历史、cap=32 时，第 0~27 条将永久丢失。
    # 现在先按内容哈希剔除已入库消息，再从最旧的一批补起：任何
    # 历史消息最多 ceil(n/cap) 轮请求后必然入库。
    # 单次请求最多补录 N 条，防止超长历史的首个请求把对话卡死。
    # 入库单位是"一轮对话"（用户消息 + 紧随的助手回复），而不是单条消息：
    #   - "我对花生过敏。"这类短消息此前因不足 20 字被整条丢弃，实测 6 道
    #     短事实题三个对话模型全部答不出；和回复合成一轮后有了上下文，可以入库。
    #   - 文本自带"用户：/助手："标注，召回后模型能分清是谁说的。
    #   - 提问和回答留在同一条记录里，不会只召回问题而丢了答案。
    cap = config.max_ingest_per_request
    engine = get_hr()
    continues = bool(recent) and recent[0].role != "user"
    to_ingest = [
        t for t in _turn_texts(older, continues)
        if not engine.cloud_cache.has_content(t)
    ][:cap] if cap > 0 else []
    for text in to_ingest:
        try:
            engine.ingest(text, {"source": "proxy", "role": "turn"})
        except Exception as e:
            logger.warning(f"[代理] 旧对话入库失败（跳过该轮）: {e}")

    # 连续追问沿用最近仍能独立成立的提问，避免代词逐轮传递后丢掉主题。
    user_msgs = [m.content for m in messages if m.role == "user" and m.content.strip()]
    query = user_msgs[-1] if user_msgs else ""
    extra = []
    anchor = None
    action_rule = next((match[1] for message in (head[0].content if head else "", *reversed(user_msgs))
                        if (match := ACTION_SOURCE_RULE.search(message))), "")
    followup = TextProcessor.chapter_followup_context(user_msgs, original_actions=action_rule == "原文")
    resolved = followup.query
    update_query = TextProcessor.chapter_revision_question(query) if len(TextProcessor.split_sentences(query)) == 1 else ""
    normalized = TextProcessor.retrieval_query(query)
    needs_context = _needs_question_context(normalized)
    if resolved or update_query:
        query = resolved or update_query
        if resolved and recent[-1].role == "user":
            # The chapter offset has already been applied during retrieval. Send
            # the resulting standalone question so the model cannot apply it twice.
            _, action_request = TextProcessor.split_action_clause(user_msgs[-1].strip())
            question = resolved.rstrip("？?") + "？" + action_request
            if followup.action_question:
                question = "沿用的原文场景：" + resolved.rstrip("？?") + "。本轮只问：" + followup.action_question
            recent = [*recent[:-1], ChatMessage(role="user", content=question)]
    elif resolved is not None and len(user_msgs) >= 2 and needs_context:
        recent_questions = [TextProcessor.retrieval_query(text) for text in user_msgs[-7:-1]]
        previous = next((text for text in reversed(recent_questions)
                         if not _needs_question_context(text)),
                        recent_questions[-1])
        anchor = previous[:200]
        anchored = anchor + "\n" + normalized
        if PRONOUN_QUESTION.match(normalized):
            query = anchored
        else:
            extra.append(anchored)
    # 先为系统指令和最新消息留位，按实际剩余预算选择记忆片段。
    budget = config.context_char_budget
    memory_budget = budget - sum(len(m.content) for m in head) - len(recent[-1].content)
    if resolved is not None and needs_context and len(recent) >= 3 and recent[-3].role == "user" and recent[-2].role == "assistant":
        # Keep a short answer with the question that established its referent.
        previous_turn = len(recent[-3].content) + len(recent[-2].content)
        if previous_turn <= max(0, memory_budget // 4):
            memory_budget -= previous_turn
    memory = ""
    if resolved is None:
        clarification = "仅凭当前指代无法唯一确定话题。本轮先请用户确认章节或能区分场景的动作，确认前数值和动作留空。"
        choices = "；".join(f"第{ch}章{field.split('多少')[0]}" for ch, field in followup.candidates[:6])
        if choices and len(clarification) + len(choices) + 8 <= memory_budget:
            clarification += "可供确认：" + choices + "。"
        memory = clarification if len(clarification) <= memory_budget else ""
    elif query and memory_budget > len(engine.MEMORY_HEADER) + 3:
        revision_query = TextProcessor.normalize_chapter_aliases(TextProcessor.revision_query(query, anchor))
        revision = (user_msgs[-1] if update_query else
                    next((match for m in reversed(body[:-1]) if m.role == "user"
                          if (match := TextProcessor.matching_revision(m.content, revision_query))), ""))
        options = {}
        if TextProcessor.rehearsal_scope(revision_query):
            options["recent_user_texts"] = tuple(m.content for m in body[:-1] if m.role == "user")
        memory = engine.assemble_context(query, char_budget=memory_budget, extra_queries=extra,
                                         anchor_query=anchor, recent_revision=revision, **options)
    if not memory and len(NO_MEMORY_NOTE) <= memory_budget:
        # 历史已被裁掉而又没有召回任何记录：明确告诉模型，否则它会把
        # "没看到"当成"没发生"，甚至编一个答案（实测出现过）。
        memory = NO_MEMORY_NOTE

    # 近期对话也裁进预算：从最旧的开始丢。
    # M5.1 修复：旧逻辑对最新一条无条件保留（kept_recent 为空时不走
    # continue 分支），用户粘贴一条 50k 字的长文档就会把压缩旁路掉，
    # 输出照样 50k。现在最新一条超预算时截断装配并追加截断标记；
    # 极端情况下（记忆块占满预算，剩余空间连标记都放不下）仍保住
    # 最新一条的开头 64 字符——用户本轮的问题不能被整个丢掉，
    # 允许小幅超出预算是可接受的代价。
    kept_recent: List[ChatMessage] = []
    used = sum(len(m.content) for m in head) + len(memory)
    for m in reversed(recent):
        if used + len(m.content) > budget:
            if kept_recent:
                break
            remain = budget - used
            if remain > len(TRUNC_MARK):
                m = ChatMessage(
                    role=m.role,
                    content=m.content[: remain - len(TRUNC_MARK)] + TRUNC_MARK,
                )
            else:
                m = ChatMessage(role=m.role, content=m.content[:64] + TRUNC_MARK)
        kept_recent.insert(0, m)
        used += len(m.content)

    # A suffix starting mid-turn must not attach another user's question to this answer.
    if len(kept_recent) > 1 and kept_recent[0].role == "assistant" and (older or len(kept_recent) < len(recent)):
        kept_recent = kept_recent[1:]

    if resolved is None and kept_recent and kept_recent[-1].role == "user":
        # A prior numeric answer cannot distinguish the outstanding alternatives.
        kept_recent = kept_recent[-1:]

    # Once the original-book question is standalone, earlier numeric answers
    # and rehearsal edits are competing sources, including edits to this chapter.
    # Only filter the prompt copy; the actual conversation remains intact for
    # later switches back to the rehearsal edition.
    original_lookup = (TextProcessor.asks_for_book_source(query) and TextProcessor.is_distance_question(query)
                       and len(TextProcessor.chapter_ids(query)) == 1)
    if original_lookup and kept_recent and kept_recent[-1].role == "user":
        filtered = []
        previous_scope = False
        for m in kept_recent[:-1]:
            if m.role == "user":
                previous_scope = bool(re.search(r"章|排演|校订|原文|原书", m.content))
                if TextProcessor.chapter_revision_question(m.content):
                    continue
            if (m.role == "assistant" and TextProcessor.contains_measurement(m.content)
                    and (previous_scope or re.search(r"第.+?章|排演|校订|原文", m.content))):
                continue
            filtered.append(m)
        kept_recent = filtered + kept_recent[-1:]
    # For a verified single value, place the scoped quote next to the question.
    # Keep the caller's history untouched so the injected quote is never archived as user input.
    scoped_quote = ""
    if kept_recent and kept_recent[-1].role == "user" and TextProcessor.is_distance_question(query):
        revision_prefix = engine.MEMORY_HEADER + engine.REVISION_LABEL
        if memory.startswith(revision_prefix):
            scoped_quote = "本轮依据为用户最新校订：" + memory[len(revision_prefix):]
        elif memory.startswith(engine.BOOK_SOURCE_HEADER + "\n"):
            scoped_quote = "本轮依据为小说原文：" + memory.split("\n", 1)[1]
    if scoped_quote and len(scoped_quote) + len(QUESTION_SEPARATOR) <= len(memory):
        question_suffix = ""
        chapters = TextProcessor.chapter_ids(query)
        if len(chapters) == 1:
            chapter = chapters[0]
            # An earlier assistant's proposed renumbering is not evidence about
            # this chapter. Keep the actual user questions and other recent facts.
            filtered = []
            for m in kept_recent:
                if m.role == "assistant" and TextProcessor.competing_chapter_measurement(m.content, query, chapter):
                    if filtered and filtered[-1].role == "user" and filtered[-1].content in scoped_quote:
                        filtered.pop()
                    continue
                filtered.append(m)
            kept_recent = filtered
            label = f"仅适用于第{chapter}章。"
            # Carry the caller's requested attribution format through long runs
            # of similarly worded numeric answers.
            instructions = (head[0].content if head else "") + "\n" + user_msgs[-1]
            number_only = re.search(r"只(?:答|要|报|给|需)(?:出)?(?:数值|数字)|(?:不用|不必|不要|无需|省略).{0,5}动作",
                                    instructions)
            if (not number_only and
                    re.search(r"(?:给出|注明|标明|列出).{0,12}章[节号].{0,8}(?:版本|来源)", instructions)):
                source = "用户校订" if memory.startswith(revision_prefix) else "原文"
                label = f"作答注明第{chapter}章、{source}。"
            if (not number_only and memory.startswith(engine.BOOK_SOURCE_HEADER + "\n")
                    and re.search(r"(?:简述|描述|说明|交代|写出|概括|补充).{0,16}动作|谁做了什么", instructions)):
                question_suffix = "请依据引文填写：场景动作（谁做了什么）、测量数值（含单位）。"
                if label.startswith("作答注明"):
                    question_suffix = (f"请用这一行格式作答：第{chapter}章｜原文｜"
                                       "场景动作：<谁做了什么，摘出引文中的具体行为>｜数值：<测量结果和单位>。动作栏写测量状态以外发生的事，可从数值句的前后文取证。")
                if TextProcessor.scene_unit(query):
                    question_suffix = (
                        "核对哪一句的主体和事件与提问一致；没有对应句，只答原文依据不足。"
                        "有依据时，动作栏先摘出含数值的完整原句，再从紧接的上下文补出是谁、做了什么。"
                        "数值栏保留原句的单位、范围和假设语气。代词无法确认姓名时保留原称呼。"
                    )
                    if label.startswith("作答注明"):
                        question_suffix += f"格式：第{chapter}章｜原文｜场景动作：…｜数值：…。"
            if original_lookup and followup.action_question:
                question_suffix = "本轮核对动作，同时保留对话要求的章节、人物、版本字段和作答格式。优先用数值所在的原句说明具体动作，并保留句中区分场景的对象或起因，引文不足时说明。"
            if original_lookup and TextProcessor.scene_unit(query) and not question_suffix:
                question_suffix = "请核对引文是否描述所问人物和事件；无法对应时只答原文依据不足。"
            if (memory.startswith(revision_prefix) and not update_query and not number_only
                    and re.search(r"(?:问|核对|回答)原文(?:时|再|才|则).{0,12}动作", instructions)
                    and not re.search(r"动作|谁做了什么", user_msgs[-1])):
                question_suffix = f"本轮只核对第{chapter}章的校订数值，请仅给出章号、版本和数值。"
            if not number_only and "字段" in instructions and all(k in instructions for k in ("章节", "人物", "版本")):
                question_suffix += "已定位该场景，按要求填写章节、人物和版本字段。人物栏填写所问主体，同行者可在动作中交代。"
            if not number_only and any(re.search(r"第\s*[0-9零〇一二两三四五六七八九十百]+\s*[卷册]", m.content)
                                       for m in kept_recent[:-1]) and not re.search(r"[卷册]", scoped_quote):
                question_suffix += "出处只标本轮依据给出的章节与版本，引文未给出的卷册请省略。"
            # Share the remaining character budget between the question and
            # its citation. Preserve the evidence and original question first.
            available = budget - sum(len(m.content) for m in head + kept_recent) - len(scoped_quote) - len(QUESTION_SEPARATOR)
            if len(question_suffix) > available:
                question_suffix = ""
            if len(label) + len(question_suffix) + 1 <= available:
                if memory.startswith(engine.BOOK_SOURCE_HEADER + "\n"):
                    # The source header already identifies the chapter. Only
                    # append a response format when the caller requested one.
                    if label.startswith("作答注明"):
                        question_suffix = "\n" + label + question_suffix
                else:
                    scoped_quote += label
        kept_recent[-1] = ChatMessage(role="user", content=scoped_quote + QUESTION_SEPARATOR + kept_recent[-1].content + question_suffix)
        memory = ""
    out = head + ([ChatMessage(role="system", content=memory)] if memory else []) + kept_recent
    out_chars = sum(len(m.content) for m in out)
    logger.info(
        f"[代理] 压缩: {len(messages)} 条/{total} 字 → {len(out)} 条/{out_chars} 字"
        + (f"（记忆块 {len(memory)} 字）" if memory else "")
    )
    return out


def _official_openai_backend(url: str) -> bool:
    return bool(re.fullmatch(r"https://api\.openai\.com(?::443)?(?:/v1)?/?", url, flags=re.I))


def _gpt62_chat_model(model: str) -> bool:
    return bool(re.fullmatch(r"gpt-6\.2-(?:sol|astra)(?:-\d{4}-\d{2}-\d{2})?", model))


def _openai_target(resource: str) -> tuple[str, dict]:
    url = config.chat_backend_url
    if _official_openai_backend(url):
        key = os.environ.get("HR_OPENAI_API_KEY", "")
        options = {"headers": {"Authorization": "Bearer " + key}, "allow_redirects": False} if key else {}
        return "https://api.openai.com/v1/" + resource, options
    return url.rstrip("/") + "/v1/" + resource, {}


@app.get("/v1/models", dependencies=[Depends(verify_token)])
def list_models():
    """OpenAI 兼容模型列表（透传后端服务），供客户端探测"""
    names: list[str] = []
    try:
        if _chat_backend_is_openai():
            endpoint, options = _openai_target("models")
            resp = requests.get(endpoint, timeout=5, **options)
            resp.raise_for_status()
            names = [m.get("id", "") for m in resp.json().get("data", [])]
        else:
            resp = requests.get(
                f"{config.chat_backend_url.rstrip('/')}/api/tags", timeout=5
            )
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    if not names:
        names = [config.chat_model]
    return {
        "object": "list",
        "data": [{"id": n, "object": "model", "owned_by": "hr-proxy"} for n in names],
    }


def _supports_openai_cache_breakpoints(backend_url: str, model: str) -> bool:
    setting = os.environ.get("HR_OPENAI_CACHE_BREAKPOINTS", "auto").strip().lower()
    if setting != "auto":
        return setting in ("1", "true", "yes", "on")
    return _official_openai_backend(backend_url) and _gpt62_chat_model(model)


@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
def chat_completions(req: ChatCompletionRequest,
                     workspace_header: Annotated[Optional[str], Header(alias="X-Topic-Workspace")] = None):
    """
    OpenAI 兼容对话代理：压缩历史 → 转发本地后端（llama.cpp /v1 或 Ollama /api/chat）。

    LobeChat 接入：添加自定义 OpenAI 服务商，API 地址填
    http://localhost:8000/v1 ，模型名填本地后端的对话模型
    （llama.cpp 后端下模型名不校验，实际模型由 llama-server -m 启动参数决定）。
    """
    model = req.model or config.chat_model
    mode = req.topic_workspace
    if workspace_header is not None:
        if not workspace_header.strip() or len(workspace_header) > 256:
            raise HTTPException(422, "话题工作区标识为空或过长")
        if mode and workspace_header != mode.workspace_id:
            raise HTTPException(422, "请求头与请求体的工作区标识不一致")
        mode = mode or TopicMode(workspace_id=workspace_header)
    cache_scope = None
    if (
        mode and _gpt62_chat_model(model)
        and CHAT_BACKEND_OPENAI.strip().lower() not in ("0", "false", "no", "off")
        and (_official_openai_backend(config.chat_backend_url)
             or _supports_openai_cache_breakpoints(config.chat_backend_url, model))
    ):
        cache_scope = config.chat_backend_url
    branch = prepare_turn(
        mode, req.messages, model, config.storage_root, config.context_char_budget,
        temperature=req.temperature, max_tokens=req.max_tokens, cache_scope=cache_scope,
        chapter_cache=bool(cache_scope) and _supports_openai_cache_breakpoints(config.chat_backend_url, model),
    ) if mode else None
    messages = branch.messages if branch else [m.model_dump() for m in _compress_history(req.messages)]
    local = branch is not None and branch.local_answer is not None
    if local:
        backend_openai = True
    else:
        backend_openai = _chat_backend_is_openai()
    include_usage = req.stream and req.stream_options is not None and req.stream_options.include_usage
    measure_stream = include_usage or (
        req.stream and cache_scope and _official_openai_backend(config.chat_backend_url)
        and str(model).lower().split("-20")[0] == "gpt-6.2-sol"
    )
    upstream_options = {}
    # Both protocols accept the same role/content messages.
    if backend_openai:
        use_breakpoints = branch and not local and _supports_openai_cache_breakpoints(config.chat_backend_url, model)
        if use_breakpoints:
            messages = branch.openai_cache_messages()
        # OpenAI 协议（llama.cpp 等）：messages 透传，temperature/max_tokens 原样下发
        payload = {
            "model": model,
            "messages": messages,
            "stream": req.stream,
        }
        if use_breakpoints and (branch.one_time_document or branch.cache_evidence_prefix_only):
            payload["prompt_cache_options"] = {"mode": "explicit"}
        if use_breakpoints and branch.prompt_cache_key:
            payload["prompt_cache_key"] = branch.prompt_cache_key
        if measure_stream:
            payload["stream_options"] = {"include_usage": True}
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            official_model = _official_openai_backend(config.chat_backend_url) and _gpt62_chat_model(model)
            payload["max_completion_tokens" if official_model else "max_tokens"] = req.max_tokens
        if not local:
            backend, upstream_options = _openai_target("chat/completions")
    else:
        # Ollama 协议：options 包装 + NDJSON 流
        payload = {
            "model": model,
            "messages": messages,
            "stream": req.stream,
        }
        if req.temperature is not None:
            payload.setdefault("options", {})["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = req.max_tokens
        backend = config.chat_backend_url.rstrip("/") + "/api/chat"

    if not req.stream:
        finish = "stop"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if local:
            content = branch.local_answer
        else:
            try:
                resp = requests.post(backend, json=payload, timeout=300, **upstream_options)
                resp.raise_for_status()
                data = resp.json()
                if backend_openai:
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                    finish = choice.get("finish_reason") or finish
                    usage = data.get("usage") or usage
                else:
                    content = data["message"]["content"]
                    finish = data.get("done_reason") or finish
                    usage["prompt_tokens"] = data.get("prompt_eval_count", 0)
                    usage["completion_tokens"] = data.get("eval_count", 0)
                    usage["total_tokens"] = usage["completion_tokens"] + usage["prompt_tokens"]
                if not isinstance(content, str):
                    raise ValueError("后端未返回文本")
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
                raise HTTPException(status_code=502, detail=f"后端模型响应失败: {e}")
            if branch:
                content = branch.finish(content, usage)
        return {
            **({"topic_workspace": branch.metadata} if branch else {}),
            "id": f"chatcmpl-hr-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }],
            "usage": usage,
        }

    # 流式：后端 NDJSON（Ollama）或 SSE（OpenAI/llama.cpp）→ OpenAI SSE
    def gen():
        cid = f"chatcmpl-hr-{uuid.uuid4().hex[:12]}"

        def sse(delta: dict, finish_reason: Optional[str] = None, *, usage: Optional[dict] = None) -> str:
            body = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }
            if include_usage:
                body["usage"] = usage
                if usage is not None:
                    body["choices"] = []
            if finish_reason and branch:
                body["topic_workspace"] = branch.metadata
            return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

        def error_event(message: str, kind: str) -> str:
            err = {"error": {"message": message, "type": kind}}
            return f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

        if local:
            yield sse({"content": branch.local_answer})
            yield sse({}, "stop")
            if include_usage:
                yield sse({}, usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            yield "data: [DONE]\n\n"
            return
        pieces = []
        complete = False
        reported_usage = None
        finish_reason: Optional[str] = "stop"
        try:
            with requests.post(backend, json=payload, stream=True, timeout=300, **upstream_options) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    done = False
                    if line.startswith("data:"):
                        # OpenAI / llama.cpp SSE 行
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            complete = True
                            break
                        try:
                            chunk = json.loads(raw)
                        except ValueError:
                            if branch:
                                yield error_event("后端返回无效数据，本轮未归档", "invalid_stream")
                                finish_reason = None
                                break
                            continue
                        if measure_stream and isinstance(chunk.get("usage"), dict):
                            reported_usage = chunk["usage"]
                        choice = (chunk.get("choices") or [{}])[0]
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                            done = True
                    else:
                        # Ollama NDJSON 行
                        try:
                            chunk = json.loads(line)
                        except ValueError:
                            if branch and not backend_openai:
                                yield error_event("后端返回无效数据，本轮未归档", "invalid_stream")
                                finish_reason = None
                                break
                            continue
                        piece = (chunk.get("message") or {}).get("content") or ""
                        if chunk.get("done"):
                            if include_usage and "prompt_eval_count" in chunk and "eval_count" in chunk:
                                inputs, outputs = chunk["prompt_eval_count"], chunk["eval_count"]
                                reported_usage = {
                                    "prompt_tokens": inputs, "completion_tokens": outputs,
                                    "total_tokens": inputs + outputs,
                                }
                            if chunk.get("done_reason") == "length":
                                finish_reason = "length"
                            done = True
                    if chunk.get("error"):
                        # 后端流中途报错（如模型被卸载）同样以普通数据行返回，
                        # 原先被当作空片段静默吞掉，客户端只看到回答戛然而止。
                        error = chunk["error"]
                        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                        yield error_event(message, "backend_error")
                        finish_reason = None
                        break
                    if piece:
                        if branch:
                            pieces.append(piece)
                        yield sse({"content": piece})
                    if done:
                        complete = True
                        if not (measure_stream and backend_openai):
                            break
        except requests.RequestException as e:
            yield error_event(f"后端模型服务不可达: {e}", "proxy_error")
            finish_reason = None
        # OpenAI 流以带 finish_reason 的空 delta 收尾，部分客户端依赖它判定完成
        if branch and finish_reason:
            if complete:
                try:
                    streamed = "".join(pieces)
                    if branch.finish(streamed, reported_usage) != streamed:
                        raise HTTPException(409, "该请求已提交；请使用相同 request_id 重放已归档结果")
                except HTTPException as exc:
                    yield error_event(str(exc.detail), "workspace_commit_error")
                    finish_reason = None
            else:
                yield error_event("后端流提前结束，本轮未归档", "incomplete_stream")
                finish_reason = None
        if finish_reason:
            yield sse({}, finish_reason)
            if include_usage and reported_usage is not None:
                yield sse({}, usage=reported_usage)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/topic-workspaces/{workspace_id}", dependencies=[Depends(verify_token)])
def topic_directory(workspace_id: str):
    return workspace_call(workspace_store(config.storage_root).list_topics, workspace_id)


@app.post("/topic-workspaces/{workspace_id}/open", dependencies=[Depends(verify_token)])
def open_topic(workspace_id: str, location: TopicLocation):
    return workspace_call(workspace_store(config.storage_root).open_topic, workspace_id, **location.model_dump())


@app.post("/topic-workspaces/{workspace_id}/back", dependencies=[Depends(verify_token)])
def go_back(workspace_id: str):
    return workspace_call(workspace_store(config.storage_root).back, workspace_id)


class TopicSourceAttachment(BaseModel):
    topic: str = Field(min_length=1, max_length=256)
    entry_id: str = Field(pattern=r"^cc_[0-9a-f]{12}$")


@app.post("/topic-workspaces/{workspace_id}/attach", dependencies=[Depends(verify_token)])
def attach_topic_source(workspace_id: str, req: TopicSourceAttachment):
    """Attach a committed L1 source to a branch without changing the active conversation."""
    archive = get_hr().cloud_cache
    if not archive.get_ingest_result(req.entry_id):
        raise HTTPException(404, "未找到已完成入库的原文")
    source_text = archive.get_full_text(req.entry_id)
    if source_text is None:
        raise HTTPException(404, "未找到原文")
    store = workspace_store(config.storage_root)
    identity = json.dumps([req.topic, req.entry_id], ensure_ascii=False)
    snapshot = workspace_call(
        store.attach_source, workspace_id, req.topic, source_text,
        source=req.entry_id, request_id="source:" + uuid.uuid5(uuid.NAMESPACE_URL, identity).hex,
    )
    return {key: snapshot[key] for key in ("topic", "revision", "breadcrumb")}


@app.get("/topic-workspaces/{workspace_id}/search", dependencies=[Depends(verify_token)])
def search_topic(workspace_id: str, topic: str, query: str, limit: int = 5):
    return workspace_call(workspace_store(config.storage_root).search, workspace_id, topic, query, limit=limit)


@app.get("/topic-workspaces/{workspace_id}/source", dependencies=[Depends(verify_token)])
def expand_topic_source(workspace_id: str, topic: str, message_id: int):
    return workspace_call(workspace_store(config.storage_root).source_message, workspace_id, topic, message_id)


@lru_cache(maxsize=4)
def _receipt_store(root):
    return ToolReceiptStore(root)


@app.post("/tool-receipts", dependencies=[Depends(verify_token)])
def record_tool_receipt(receipt: ToolReceipt):
    """Archive an authoritative tool receipt. A replay of identical contents is idempotent."""
    try:
        return _receipt_store(config.storage_root).record(receipt)
    except ReceiptIdentityConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/tool-receipts/lookup", dependencies=[Depends(verify_token)])
def lookup_tool_receipt(scope: ReceiptScope):
    """Resolve the current action parameter within one task, resource and field."""
    return _receipt_store(config.storage_root).lookup(scope)


@app.post("/task-context/resolve", dependencies=[Depends(verify_token)])
def resolve_task_context(scope: ReceiptScope):
    """Resolve an archived selection and its current or explicitly pinned tool evidence."""
    return _receipt_store(config.storage_root).resolve_task_context(scope)


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_token)])
def ingest_dialogue(req: IngestRequest):
    """
    写入一段对话到分级检索系统

    流程: 全文存储 → 话题分配 → 关键句提取 → 向量索引
    """
    try:
        engine = get_hr()
        result = engine.ingest(text=req.text, metadata=req.metadata)
        try:
            engine.flush()
        except Exception as _e:
            logger.warning(f"[写入] flush 失败（下次写入重试）: {_e}")
        logger.info(f"[写入] entry_id={result['entry_id']} domain={result['domain_name']}")
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"[写入] 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(verify_token)])
def retrieve(req: RetrieveRequest):
    """
    分级检索主接口

    流程:
    1. 检测查询所属话题界域 (L3)
    2. 在界域内检索关键句 (L2)
    3. 关联完整上下文 (L1)
    4. 聚合返回结果
    """
    try:
        result = get_hr().retrieve(
            query=req.query,
            top_k_sentences=req.top_k_sentences,
            top_k_context=req.top_k_context,
        )
        logger.info(f"[检索] query={req.query[:30]}... 找到 {len(result['key_sentences'])} 个关键句")
        return RetrieveResponse(**result)
    except Exception as e:
        logger.error(f"[检索] 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 启动入口 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,  # 直接传对象：AutoClaw 内嵌 python 为 ._pth 布局，按字符串重导入会 ModuleNotFoundError
        host="127.0.0.1",   # 代理含对话内容，默认仅本机访问；如需局域网自行修改
        port=8000,
        reload=False,       # 常驻服务不应开 reload（会反复重启并重放初始化）
        log_level="info",
    )
