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
from pathlib import Path
from typing import Optional, List

# 确保能导入 hierarchical_retrieval
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from hierarchical_retrieval import HConfig, HierarchicalRetrieval

# ── 日志配置 ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI 应用 ──────────────────────────────────────
app = FastAPI(
    title="分级检索 API",
    description="基于本地嵌入模型（默认 bge-m3）的三级架构检索服务 (CloudCache + KeySentence + TopicDomain) + OpenAI 兼容上下文压缩代理",
    version="0.5.0",
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
OLLAMA_URL = os.environ.get("HR_OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("HR_CHAT_MODEL", "")  # /v1 代理的回退对话模型
# B3 修复：可选 Bearer 鉴权。设置 HR_API_TOKEN 后，除 /health 外的
# 接口要求 Authorization: Bearer <token>（LobeChat 侧把 API Key 填成
# 同名值即可）。默认为空 = 不鉴权（保持仅本机监听的默认形态）。
API_TOKEN = os.environ.get("HR_API_TOKEN", "")

config = HConfig(
    storage_root=STORAGE_ROOT,
    embedding_base_url=OLLAMA_URL,
    chat_backend_url=os.environ.get("HR_CHAT_BACKEND", OLLAMA_URL),
    # M6 修复：API 服务主要跑中文对话，显式采用中文场景的相似度阈值，
    # 避免沿用英文默认阈值导致对话被误并入同一话题域、L3 形同虚设。
    topic_min_similarity=0.70,
    topic_similarity_threshold=0.55,
    key_sentence_similarity_threshold=0.60,
)
if CHAT_MODEL:
    config.chat_model = CHAT_MODEL

logger.info(f"[启动] 存储目录: {STORAGE_ROOT}")
logger.info(f"[启动] Embedding 服务: {OLLAMA_URL}")

# 检索引擎（全局单例，惰性初始化）。
# v0.4 修复：import 本模块不再立即构建引擎——此前单元测试 import
# api_server 就会触发一次真实 Ollama 健康检查与存储目录创建；
# 引擎首次使用时才创建。测试/外部代码可直接对 api_server.hr 赋值
# 注入替身，get_hr() 会原样返回注入值。
hr: Optional[HierarchicalRetrieval] = None


def get_hr() -> HierarchicalRetrieval:
    """获取检索引擎单例；允许测试预先注入 api_server.hr。"""
    global hr
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
    if authorization[len("Bearer "):].strip() != API_TOKEN:
        raise HTTPException(status_code=401, detail="未授权：API token 校验失败")


# ── 请求/响应模型 ──────────────────────────────────────

class IngestRequest(BaseModel):
    text: str = Field(..., description="对话全文", example="今天讨论一下微服务架构方案...")
    metadata: Optional[dict] = Field(None, description="附加元数据", example={"source": "meeting", "speaker": "Alice"})


class IngestResponse(BaseModel):
    entry_id: str
    domain_id: str
    domain_name: str
    sentence_ids: List[str]


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="查询文本", example="微服务架构")
    top_k_sentences: Optional[int] = Field(None, description="每个话题检索的关键句数")
    top_k_context: Optional[int] = Field(None, description="每个关键句关联的全文段落数")


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


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ── 路由 ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok",
        ollama_url=config.embedding_base_url,
        storage_root=STORAGE_ROOT,
    )


# ── OpenAI 兼容代理：上下文压缩的落点 ─────────────────────
#
# 为什么需要它：LobeChat 等客户端是"全量重放"模式，每轮请求都把完整
# 历史塞进 prompt 发给模型——只把分级检索挂在旁边（ingest 钩子）不改写
# 发出的 payload，历史该多长还是多长，本地小模型照样溢出。
# 把客户端的"模型服务商"指向本服务 /v1，历史在这里被压缩后再转发，
# 发出去的 prompt 从结构上就有界。

# M5.1：单条超长消息截断装配时追加的尾部标记（9 字符）
TRUNC_MARK = "（原文过长已截断）"


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
    cap = config.max_ingest_per_request
    engine = get_hr()
    to_ingest = [
        m for m in older
        if m.role in ("user", "assistant")
        and len(m.content.strip()) >= 20
        and not engine.cloud_cache.has_content(m.content)
    ][:cap] if cap > 0 else []
    for m in to_ingest:
        try:
            engine.ingest(m.content, {"source": "proxy", "role": m.role})
        except Exception as e:
            logger.warning(f"[代理] 旧消息入库失败（跳过该条）: {e}")

    # 用最新的用户问题做检索，装配记忆块
    query = next((m.content for m in reversed(messages) if m.role == "user"), "")
    memory = engine.assemble_context(query) if query else ""

    # 近期对话也裁进预算：从最旧的开始丢。
    # M5.1 修复：旧逻辑对最新一条无条件保留（kept_recent 为空时不走
    # continue 分支），用户粘贴一条 50k 字的长文档就会把压缩旁路掉，
    # 输出照样 50k。现在最新一条超预算时截断装配并追加截断标记；
    # 极端情况下（记忆块占满预算，剩余空间连标记都放不下）仍保住
    # 最新一条的开头 64 字符——用户本轮的问题不能被整个丢掉，
    # 允许小幅超出预算是可接受的代价。
    budget = config.context_char_budget
    kept_recent: List[ChatMessage] = []
    used = sum(len(m.content) for m in head) + len(memory)
    for m in reversed(recent):
        if used + len(m.content) > budget:
            if kept_recent:
                continue
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

    out = head + ([ChatMessage(role="system", content=memory)] if memory else []) + kept_recent
    out_chars = sum(len(m.content) for m in out)
    logger.info(
        f"[代理] 压缩: {len(messages)} 条/{total} 字 → {len(out)} 条/{out_chars} 字"
        + (f"（记忆块 {len(memory)} 字）" if memory else "")
    )
    return out


@app.get("/v1/models", dependencies=[Depends(verify_token)])
def list_models():
    """OpenAI 兼容模型列表（透传后端 Ollama），供客户端探测"""
    try:
        resp = requests.get(
            f"{config.chat_backend_url.rstrip('/')}/api/tags", timeout=5
        )
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        names = [config.chat_model]
    return {
        "object": "list",
        "data": [{"id": n, "object": "model", "owned_by": "hr-proxy"} for n in names],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI 兼容对话代理：压缩历史 → 转发本地后端（默认 Ollama /api/chat）。

    LobeChat 接入：添加自定义 OpenAI 服务商，API 地址填
    http://localhost:8000/v1 ，模型名填本地 Ollama 的对话模型。
    """
    model = req.model or config.chat_model
    messages = _compress_history(req.messages)
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": req.stream,
    }
    if req.temperature is not None:
        payload.setdefault("options", {})["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload.setdefault("options", {})["num_predict"] = req.max_tokens

    backend = config.chat_backend_url.rstrip("/") + "/api/chat"

    if not req.stream:
        try:
            resp = requests.post(backend, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"后端模型服务不可达: {e}")
        content = data.get("message", {}).get("content", "")
        return {
            "id": f"chatcmpl-hr-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # 流式：Ollama NDJSON → OpenAI SSE
    def gen():
        cid = f"chatcmpl-hr-{uuid.uuid4().hex[:12]}"
        try:
            with requests.post(backend, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    piece = chunk.get("message", {}).get("content", "")
                    if not piece:
                        continue
                    sse = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(sse, ensure_ascii=False)}\n\n"
        except requests.RequestException as e:
            err = {"error": {"message": f"后端模型服务不可达: {e}", "type": "proxy_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_token)])
def ingest_dialogue(req: IngestRequest):
    """
    写入一段对话到分级检索系统

    流程: 全文存储 → 话题分配 → 关键句提取 → 向量索引
    """
    try:
        result = get_hr().ingest(text=req.text, metadata=req.metadata)
        # N-perf: 增量落盘兜底——把脏标记的索引/元数据刷盘
        try:
            get_hr().cloud_cache.flush()
            get_hr().key_sentence_lib._vector_store.flush()
            for _dom in get_hr().topic_manager._domains.values():
                _vs = getattr(_dom, "vector_store", None)
                if _vs is not None and hasattr(_vs, "flush"):
                    _vs.flush()
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
