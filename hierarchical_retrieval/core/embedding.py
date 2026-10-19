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
from typing import Optional

import numpy as np
import requests

from ..config import HConfig

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
                    models = [m["name"] for m in resp.json().get("models", [])]
                    if not any(self._model in m for m in models):
                        logger.warning(
                            "模型 %s 未在 Ollama 中找到，请先运行: ollama pull %s",
                            self._model, self._model,
                        )
                    else:
                        logger.info("Ollama 服务可用，模型 %s 已就绪", self._model)
            except requests.ConnectionError:
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
            vec = np.array(embedding, dtype=np.float32)
            # L2 归一化，使内积 = 余弦相似度，取值 [-1, 1]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            return vec
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
        workers = getattr(self.config, "embed_workers", 6)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, len(texts))) as ex:
            return list(ex.map(lambda t: self.embed(t, strict=strict), texts))

    @staticmethod
    def is_valid(vec: np.ndarray) -> bool:
        """零向量（服务失败/空文本的降级产物）判否，查询路径据此兜底"""
        return float(np.linalg.norm(vec)) > 1e-6

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return self._backend
