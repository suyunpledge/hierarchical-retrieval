"""
关键句库模块 —— 第二级（二级信息库）

将对话内容提炼为精简的关键句，在独立向量空间中存储。
每个关键句关联到全文缓存中的条目位置。
支持根据关键句快速定位详细对话内容。
"""

import logging
import time
import uuid
from typing import Optional

import numpy as np

from ..config import HConfig
from ..storage.local_storage import LocalStorage
from ..storage.vector_store import VectorStore
from .embedding import NomicEmbedding
from ..utils.text_processor import TextProcessor
from ..utils.summarizer import KeySentenceExtractor

logger = logging.getLogger(__name__)


class KeySentenceLibrary:
    """
    关键句库

    层级: 第二级
    职责:
      - 从对话文本中提取关键句
      - 将关键句向量化并存储
      - 按向量相似度检索相关关键句
      - 通过关键句关联到全文缓存条目
    """

    def __init__(
        self,
        embedding: NomicEmbedding,
        config: Optional[HConfig] = None,
    ):
        self.config = config or HConfig()
        self.embedding = embedding
        self.extractor = KeySentenceExtractor(config)

        base_dir = self.config.storage_root
        self._text_store = LocalStorage(
            root_dir=f"{base_dir}/{self.config.key_sentence_dir}",
            use_json=True,
        )
        self._vector_store = VectorStore(
            root_dir=f"{base_dir}/{self.config.vector_index_dir}/key_sentences",
            dimension=self.config.embedding_dim,
        )

    # ── 写入 ────────────────────────────────────────────────────

    def ingest(self, text: str, source_entry_id: str) -> list[dict]:
        """
        从对话文本中提取关键句，存入关键句库。

        Args:
            text: 对话原文
            source_entry_id: 对应的全文缓存条目 ID

        Returns:
            [{"sentence_id", "text", "vector"}, ...]（带向量，供界域索引复用）
        """
        # 按块提取：配置项 max_key_sentences_per_chunk 的本意是"每块 N 句"。
        # 此前对全文只调用一次 extract，导致无论多长的对话最多只有 3 句
        # 进入 L2，长对话的关键句覆盖率约 2%，召回率严重不足。
        max_len = self.config.key_sentence_max_length
        seen_sents: set[str] = set()
        key_sentences: list[str] = []
        for chunk in TextProcessor.split_chunks(text):
            for sent in self.extractor.extract(chunk):
                if len(sent) > max_len:
                    sent = sent[: max_len - 1] + "…"
                if sent not in seen_sents:
                    seen_sents.add(sent)
                    key_sentences.append(sent)

        sentence_ids = []
        timestamp = time.time()

        # 批量嵌入（写入路径 strict：失败抛错，不让零向量进索引）
        vectors = self.embedding.embed_batch(key_sentences, strict=True)

        items = []
        for sent, vec in zip(key_sentences, vectors):
            sent_id = f"ks_{uuid.uuid4().hex[:12]}"
            # 存储文本
            record = {
                "sentence_id": sent_id,
                "text": sent,
                "source_entry_id": source_entry_id,
                "timestamp": timestamp,
            }
            self._text_store.save(sent_id, record)
            sentence_ids.append(sent_id)
            items.append((
                sent_id,
                vec,
                {
                    "sentence_id": sent_id,
                    "text": sent,
                    "source_entry_id": source_entry_id,
                    "timestamp": timestamp,
                    "type": "key_sentence",
                },
            ))

        # 批量入索引（一次写入）
        self._vector_store.add_many(items)

        logger.info(
            "关键句库: 从 %s 提取 %d 条关键句",
            source_entry_id, len(key_sentences),
        )
        # 返回带向量的记录，供上层（话题界域索引）复用，避免重复嵌入
        return [
            {"sentence_id": sid, "text": sent, "vector": vec}
            for sid, sent, vec in zip(sentence_ids, key_sentences, vectors)
        ]

    # ── 检索 ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        按查询语义检索相关关键句。

        Args:
            query: 查询文本
            top_k: 返回的关键句数量

        Returns:
            [{"sentence_id", "text", "source_entry_id", "score", "timestamp"}, ...]
            嵌入服务不可用（查询向量为零向量）时返回空列表（M3 修复）。
        """
        q_vec = self.embedding.embed(query)
        if not NomicEmbedding.is_valid(q_vec):
            logger.error("查询嵌入失败（Embedding 服务不可用），本次检索返回空")
            return []
        results = self._vector_store.search(q_vec, top_k=top_k)

        enriched = []
        for r in results:
            meta = r["metadata"]
            enriched.append({
                "sentence_id": meta.get("sentence_id", ""),
                "text": meta.get("text", ""),
                "source_entry_id": meta.get("source_entry_id", ""),
                "score": r["score"],
                "timestamp": meta.get("timestamp", 0),
            })
        return enriched

    # ── 管理 ────────────────────────────────────────────────────

    def get_sentence(self, sentence_id: str) -> Optional[dict]:
        """按 ID 获取关键句详情"""
        return self._text_store.load(sentence_id)

    def list_sentences(self, source_entry_id: Optional[str] = None) -> list[str]:
        """列出关键句 ID，可选按 source 过滤"""
        all_ids = self._text_store.list_keys(prefix="ks_")
        if source_entry_id is None:
            return all_ids
        filtered = []
        for sid in all_ids:
            record = self._text_store.load(sid)
            if record and record.get("source_entry_id") == source_entry_id:
                filtered.append(sid)
        return filtered

    def count(self) -> int:
        return self._text_store.count(prefix="ks_")