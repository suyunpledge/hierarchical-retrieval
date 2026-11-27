"""
Embedding 服务 —— 双后端嵌入向量客户端（llama.cpp / Ollama）

同一个客户端支持两类本地嵌入服务，后端自动探测（也可显式指定）：

  llama.cpp (llama-server)          Ollama
  ─────────────────────────        ─────────────────────────
  POST /v1/embeddings               POST /api/embeddings
  {model, input: str}               {model, prompt}
  → data[0].embedding               → embedding
  GET /v1/models → 200              GET /api/tags → 200

默认模型 bge-m3（1024 维）。实测同一 bge-m3 权重在两个后端产生的
向量余弦 > 0.9999，FAISS 索引可跨后端直接续用，无需重建。
"""

import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import requests

from ..config import HConfig
from ..utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class NomicEmbedding:
    """
    嵌入向量客户端（类名 NomicEmbedding 沿用历史命名；实际模型由配置决定）

    依赖: 一个本地嵌入服务——llama.cpp 的 llama-server（推荐，
    --embedding 启动）或 Ollama（ollama serve），默认按地址自动探测。
    """

    def __init__(self, config: Optional[HConfig] = None):
        self.config = config or HConfig()
        self._model = self.config.embedding_model
        self._base_url = self.config.embedding_base_url.rstrip("/")
        self._timeout = self.config.embedding_request_timeout
        self._dim = self.config.embedding_dim
        self._backend = self._resolve_backend()

    # ── 后端探测 ─────────────────────────────────────────────

    def _resolve_backend(self) -> str:
        """解析嵌入后端类型: 'ollama' | 'openai'（探测失败保持 openai 并告警）"""
        want = getattr(self.config, "embedding_backend", "auto").lower()
        if want in ("ollama", "openai"):
            logger.info("Embedding 后端（显式配置）: %s @ %s", want, self._base_url)
            return want
        # auto: Ollama 独有 /api/tags 200；llama-server 对其 404
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                logger.info("Embedding 后端（探测）: ollama @ %s", self._base_url)
                return "ollama"
        except requests.RequestException:
            pass
        try:
            r = requests.get(f"{self._base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                logger.info("Embedding 后端（探测）: openai(llama.cpp) @ %s", self._base_url)
                return "openai"
        except requests.RequestException:
            pass
        logger.warning(
            "无法探测 Embedding 服务 (%s)：/api/tags 与 /v1/models 均不可达，"
            "按 openai 协议重试。请确认 llama-server --embedding 或 ollama serve 已启动",
            self._base_url,
        )
        return "openai"

    def _check_available(self):
        """启动健康检查（兼容旧调用点；探测已在构造时完成）"""
        if self._backend == "ollama":
            try:
                resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = [m.get("name", "") for m in resp.json().get("models", []) if isinstance(m, dict)]
                    if not any(self._model in m for m in models):
                        logger.warning(
                            "模型 %s 未在 Ollama 中找到，请先运行: ollama pull %s",
                            self._model, self._model,
                        )
                    else:
                        logger.info("Ollama 服务可用，模型 %s 已就绪", self._model)
            except requests.RequestException:
                logger.warning(
                    "无法连接到 Ollama 服务 (%s)，请确保已启动: ollama serve",
                    self._base_url,
                )
        else:
            logger.info(
                "llama.cpp 嵌入服务: %s（模型由 llama-server 启动参数 -m 指定，此处不校验）",
                self._base_url,
            )

    # ── 嵌入请求 ─────────────────────────────────────────────

    def _coerce_vector(self, embedding, *, strict: bool) -> np.ndarray:
        """Validate and normalize a backend vector before it reaches FAISS."""
        try:
            vec = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError) as e:
            if strict:
                raise RuntimeError(f"Embedding 返回格式无效: {e}") from e
            return np.zeros(self._dim, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self._dim or not np.isfinite(vec).all():
            msg = f"Embedding 维度/数值无效: 期望 ({self._dim},)，实际 {vec.shape}"
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return np.zeros(self._dim, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-10:
            if strict:
                raise RuntimeError("Embedding 返回零向量")
            return np.zeros(self._dim, dtype=np.float32)
        return vec / norm

    def embed(self, text: str, strict: bool = False) -> np.ndarray:
        """
        将单条文本转为嵌入向量

        Args:
            text: 输入文本
            strict: True 时失败抛异常。写入路径必须开启——否则嵌入
                    服务掉线时零向量会被静默写进 FAISS 索引，永久污染数据；
                    查询路径可保持 False，由调用方用 is_valid() 判空兜底。

        Returns:
            shape (dim,) 的 float32 向量（已 L2 归一化）
        """
        if not text or not text.strip():
            if strict:
                raise ValueError("embed(): 输入文本为空")
            return np.zeros(self._dim, dtype=np.float32)

        text = TextProcessor.normalize_query(text)
        cache = self._cache_state()
        if cache["capacity"] <= 0:
            return self._embed_uncached(text, strict)

        with cache["condition"]:
            while True:
                vec = cache["items"].get(text)
                if vec is not None:
                    cache["items"].move_to_end(text)
                    cache["hits"] += 1
                    return vec
                if text not in cache["inflight"]:
                    cache["misses"] += 1
                    cache["inflight"].add(text)
                    generation = cache["generation"]
                    break
                cache["condition"].wait()

        try:
            vec = self._embed_uncached(text, strict)
            # 失败降级的零向量不缓存，服务恢复后仍可正常重试。
            if self.is_valid(vec):
                vec.setflags(write=False)
                with cache["condition"]:
                    if generation == cache["generation"]:
                        cache["items"][text] = vec
                        cache["items"].move_to_end(text)
                        while len(cache["items"]) > cache["capacity"]:
                            cache["items"].popitem(last=False)
            return vec
        finally:
            with cache["condition"]:
                cache["inflight"].discard(text)
                cache["condition"].notify_all()

    # ── 嵌入缓存 ─────────────────────────────────────────────
    #
    # 同一段文本在一次 ingest 里会被嵌入多遍（短消息时 chunk == 全文 ==
    # 话题分配文本，关键句也常与 chunk 相同），检索时关键句锚点、重复/
    # 重新生成的查询也会反复嵌入。进程内 LRU 让这些请求不再打到嵌入服务。

    def _cache_state(self) -> dict:
        """惰性创建缓存状态（测试替身子类可能不调用本类 __init__）。"""
        state = self.__dict__.get("_embed_cache")
        if state is None:
            lock = threading.Lock()
            state = self.__dict__.setdefault("_embed_cache", {
                "items": OrderedDict(),
                "lock": lock,
                "condition": threading.Condition(lock),
                "inflight": set(),
                "generation": 0,
                "capacity": max(0, int(getattr(self.config, "embedding_cache_size", 4096))),
                "hits": 0,
                "misses": 0,
            })
        return state

    def cache_stats(self) -> dict:
        """嵌入缓存统计：容量、当前条数、命中/未命中次数与命中率。"""
        cache = self._cache_state()
        with cache["lock"]:
            hits, misses = cache["hits"], cache["misses"]
            size = len(cache["items"])
        total = hits + misses
        return {
            "capacity": cache["capacity"],
            "size": size,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    def clear_cache(self) -> None:
        cache = self._cache_state()
        with cache["condition"]:
            cache["items"].clear()
            cache["generation"] += 1

    def _embed_uncached(self, text: str, strict: bool) -> np.ndarray:
        """直接请求嵌入服务（不经缓存）。"""
        try:
            if self._backend == "ollama":
                embedding = self._embed_ollama(text)
            else:
                embedding = self._embed_openai(text)
            if not embedding:
                if strict:
                    raise RuntimeError("嵌入服务返回空 embedding")
                logger.warning("嵌入服务返回空 embedding，使用零向量")
                return np.zeros(self._dim, dtype=np.float32)
            return self._coerce_vector(embedding, strict=strict)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Embedding 服务不可用: {e}") from e
            logger.error("Embedding 请求失败（返回零向量）: %s", e)
            return np.zeros(self._dim, dtype=np.float32)

    def _embed_ollama(self, text: str) -> list:
        resp = requests.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    def _embed_openai(self, text: str) -> list:
        # llama-server / vLLM / 任意 OpenAI 兼容 embeddings 端点。
        # 注意：llama-server 不校验 model 名（模型由 -m 启动参数决定），
        # 字段仅作占位；对严格校验的服务请把 model 填成真实模型名。
        resp = requests.post(
            f"{self._base_url}/v1/embeddings",
            json={"model": self._model, "input": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return data[0].get("embedding", []) if data else []

    def embed_batch(self, texts: list[str], strict: bool = False) -> list[np.ndarray]:
        """
        批量嵌入（线程池并发调用，同步逐条太慢：
        一次 ingest 涉及全部 chunk 与关键句的嵌入，是首响时延的大头）

        Args:
            texts: 文本列表
            strict: 同 embed()，写入路径应开启

        Returns:
            向量列表（与输入顺序一致）
        """
        if not texts:
            return []
        # 批内去重：同一文本只请求一次（并发线程同时 miss 同一文本时
        # 缓存挡不住，会重复请求）
        unique = list(dict.fromkeys(texts))
        workers = max(1, int(getattr(self.config, "embed_workers", 6)))
        with ThreadPoolExecutor(max_workers=min(workers, len(unique))) as ex:
            vectors = dict(zip(unique, ex.map(lambda t: self.embed(t, strict=strict), unique)))
        return [vectors[t] for t in texts]

    @staticmethod
    def is_valid(vec: np.ndarray) -> bool:
        """零向量（服务失败/空文本的降级产物）判否，查询路径据此兜底"""
        try:
            return bool(
                np.asarray(vec).ndim == 1
                and np.isfinite(vec).all()
                and float(np.linalg.norm(vec)) > 1e-6
            )
        except (TypeError, ValueError):
            return False

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return self._backend
