"""
Embedding 服务 —— 对接 nomic-embed-text 嵌入向量模型

通过 Ollama API 调用 nomic-embed-text:latest 生成文本嵌入向量。
"""

import logging
from typing import Optional

import numpy as np
import requests

from ..config import HConfig

logger = logging.getLogger(__name__)


class NomicEmbedding:
    """
    nomic-embed-text 嵌入向量客户端

    依赖: 本地需安装 Ollama 并已拉取 nomic-embed-text 模型
    """

    def __init__(self, config: Optional[HConfig] = None):
        self.config = config or HConfig()
        self._model = self.config.embedding_model
        self._base_url = self.config.embedding_base_url.rstrip("/")
        self._timeout = self.config.embedding_request_timeout
        self._dim = self.config.embedding_dim

        # 健康检查
        self._check_available()

    def _check_available(self):
        """检查 Ollama 服务是否可用"""
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
            else:
                logger.warning(
                    "Ollama 服务返回异常状态码 %s，请确保服务已启动", resp.status_code
                )
        except requests.ConnectionError:
            logger.warning(
                "无法连接到 Ollama 服务 (%s)，请确保已启动: ollama serve",
                self._base_url,
            )

    def embed(self, text: str, strict: bool = False) -> np.ndarray:
        """
        将单条文本转为嵌入向量

        Args:
            text: 输入文本
            strict: True 时失败抛异常。写入路径必须开启——否则 Ollama
                    掉线时零向量会被静默写进 FAISS 索引，永久污染数据；
                    查询路径可保持 False，由调用方用 is_valid() 判空兜底。

        Returns:
            shape (dim,) 的 float32 向量
        """
        if not text or not text.strip():
            if strict:
                raise ValueError("embed(): 输入文本为空")
            return np.zeros(self._dim, dtype=np.float32)

        payload = {
            "model": self._model,
            "prompt": text,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
            if not embedding:
                if strict:
                    raise RuntimeError("Ollama 返回空 embedding")
                logger.warning("Ollama 返回空 embedding，使用零向量")
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

    def embed_batch(self, texts: list[str], strict: bool = False) -> list[np.ndarray]:
        """
        批量嵌入（线程池并发调用 Ollama，同步逐条太慢：
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