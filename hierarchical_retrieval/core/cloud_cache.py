"""
全文缓存模块 —— 第一级（基础信息库）

将所有对话全文存入本地磁盘，一字不漏，作为完整的信息基础。
支持逐条写入、向量索引检索、按时间/ID 读取。

（类名 CloudCache 沿用历史命名；数据始终存放在本地磁盘，
从未涉及任何"云端"存储。）
"""

import copy
import hashlib
import json
import logging
import time
import uuid
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from ..config import HConfig
from ..storage.local_storage import LocalStorage
from ..storage.vector_store import VectorStore
from .embedding import NomicEmbedding
from ..utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class CloudCache:
    """
    全文缓存（CloudCache）—— 全文存储与向量检索

    层级: 第一级
    职责: 将所有对话原文保存到本地磁盘，支持全文检索和语义检索。
    """

    def __init__(
        self,
        embedding: NomicEmbedding,
        config: Optional[HConfig] = None,
    ):
        self.config = config or HConfig()
        self.embedding = embedding

        # 存储后端
        base_dir = self.config.storage_root
        self._text_store = LocalStorage(
            root_dir=f"{base_dir}/{self.config.cloud_cache_dir}",
            use_json=True,
        )
        self._vector_store = VectorStore(
            root_dir=f"{base_dir}/{self.config.vector_index_dir}/cloud_cache",
            dimension=self.config.embedding_dim,
        )

        # 索引: entry_id → {chunk_index → chunk_text}
        self._chunk_map: dict[str, dict[int, str]] = {}

        # 内容去重索引: sha256(text) → entry_id（代理模式下同一历史会被
        # 反复送入 ingest，没有去重会让库膨胀且关键句重复）
        self._content_index_path = self._text_store.root / "content_index.json"
        self._content_index: dict[str, str] = {}
        self._content_lock = threading.RLock()
        if self._content_index_path.exists():
            try:
                self._content_index = json.loads(
                    self._content_index_path.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.warning("加载内容去重索引失败（忽略）: %s", e)

        self._record_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._record_cache_lock = threading.RLock()
        self._record_cache_capacity = max(0, int(self.config.fulltext_cache_size))
        self._record_hits = 0
        self._record_misses = 0

    def _load_record(self, entry_id: str) -> Optional[dict]:
        # 同步磁盘读取和回填，合并并发 miss；避免旧读取覆盖刚写入的记录。
        with self._record_cache_lock:
            return self._load_record_locked(entry_id)

    def _load_record_locked(self, entry_id: str) -> Optional[dict]:
        if self._record_cache_capacity > 0:
            with self._record_cache_lock:
                record = self._record_cache.get(entry_id)
                if record is not None:
                    self._record_cache.move_to_end(entry_id)
                    self._record_hits += 1
                    return copy.deepcopy(record)
                self._record_misses += 1
        record = self._text_store.load(entry_id)
        if record is not None and self._record_cache_capacity > 0:
            with self._record_cache_lock:
                self._record_cache[entry_id] = copy.deepcopy(record)
                self._record_cache.move_to_end(entry_id)
                while len(self._record_cache) > self._record_cache_capacity:
                    self._record_cache.popitem(last=False)
        return record

    def _save_record(self, entry_id: str, record: dict) -> None:
        with self._record_cache_lock:
            self._save_record_locked(entry_id, record)

    def _save_record_locked(self, entry_id: str, record: dict) -> None:
        self._text_store.save(entry_id, record)
        if self._record_cache_capacity > 0:
            with self._record_cache_lock:
                self._record_cache[entry_id] = copy.deepcopy(record)
                self._record_cache.move_to_end(entry_id)
                while len(self._record_cache) > self._record_cache_capacity:
                    self._record_cache.popitem(last=False)

    def _discard_record(self, entry_id: str) -> None:
        with self._record_cache_lock:
            self._record_cache.pop(entry_id, None)

    def _delete_record(self, entry_id: str) -> None:
        with self._record_cache_lock:
            self._text_store.delete(entry_id)
            self._discard_record(entry_id)

    def clear_cache(self) -> None:
        with self._record_cache_lock:
            self._record_cache.clear()

    def cache_stats(self) -> dict:
        with self._record_cache_lock:
            hits, misses = self._record_hits, self._record_misses
            size = len(self._record_cache)
        total = hits + misses
        return {
            "capacity": self._record_cache_capacity,
            "size": size,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    # ── 写入 ────────────────────────────────────────────────────

    def store(self, text: str, metadata: Optional[dict] = None, dedupe: bool = True) -> str:
        """
        将全文存入缓存，自动切块并构建向量索引。

        Args:
            text: 对话全文
            metadata: 附加元数据（如时间戳、来源等）
            dedupe: 内容级去重，重复文本直接复用已有条目

        Returns:
            entry_id: 本条记录的全局唯一 ID（重复内容返回原条目 ID）
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("CloudCache.store(): 文本不能为空")

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Hold the dedupe lock across the complete write so concurrent
        # requests with identical content cannot both append vectors.
        with self._content_lock:
            if dedupe and content_hash in self._content_index:
                existing = self._content_index[content_hash]
                logger.info("CloudCache 内容重复，复用条目 %s", existing)
                return existing
            entry_id = f"cc_{uuid.uuid4().hex[:12]}"
            timestamp = time.time()
            chunks = TextProcessor.split_chunks(text)
            if not chunks:
                raise ValueError("CloudCache.store(): 文本无法切分")
            vectors = self.embedding.embed_batch(chunks, strict=True)
            meta = {
                "entry_id": entry_id,
                "timestamp": timestamp,
                "total_chunks": len(chunks),
                "total_length": len(text),
                **(metadata or {}),
            }
            self._save_record(entry_id, {"meta": meta, "full_text": text})
            self._chunk_map[entry_id] = {}
            try:
                self._vector_store.add_many([
                    (
                        f"{entry_id}_chunk_{idx}",
                        vec,
                        {
                            "entry_id": entry_id,
                            "chunk_index": idx,
                            "chunk_text": chunk,
                            "timestamp": timestamp,
                        },
                    )
                    for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
                ])
            except Exception:
                self._chunk_map.pop(entry_id, None)
                self._delete_record(entry_id)
                raise
            for idx, chunk in enumerate(chunks):
                self._chunk_map[entry_id][idx] = chunk

            # Register the hash only after all L1 writes succeeded.
            self._content_index[content_hash] = entry_id
            tmp = self._content_index_path.with_suffix(".json.tmp")
            try:
                tmp.write_text(
                    json.dumps(self._content_index, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp.replace(self._content_index_path)
            except Exception as e:
                logger.warning("写入内容去重索引失败: %s", e)
            logger.info("CloudCache 写入: %s (%d 块, %d 字符)", entry_id, len(chunks), len(text))
            return entry_id

    # ── 检索 ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        语义检索相关全文段落（全库范围）。

        Args:
            query: 查询文本
            top_k: 返回的段落数

        Returns:
            [{"entry_id", "chunk_index", "chunk_text", "score", "timestamp", "full_text"?}, ...]
            嵌入服务不可用（查询向量为零向量）时返回空列表，避免
            FAISS 按插入序返回伪命中（M3 修复）。
        """
        q_vec = self.embedding.embed(query)
        if not NomicEmbedding.is_valid(q_vec):
            logger.error("查询嵌入失败（Embedding 服务不可用），本次检索返回空")
            return []
        results = self._vector_store.search(q_vec, top_k=top_k)
        return self._enrich(results)

    def search_in_entry(self, query: str, entry_id: str, top_k: int = 3) -> list[dict]:
        """
        在指定全文条目内检索最相关段落。

        这是"关键句 → 全文关联"的落点：传入关键句的 source_entry_id，
        返回的一定是该条目自己的段落，而不是全库里随便哪篇相似的东西。
        条目内无命中时回退为原文开头片段，保证关联不落空。
        嵌入服务不可用（零向量）时返回空列表（M3 修复）。
        """
        q_vec = self.embedding.embed(query)
        if not NomicEmbedding.is_valid(q_vec):
            logger.error("查询嵌入失败（Embedding 服务不可用），本次检索返回空")
            return []
        return self.search_in_entry_by_vector(q_vec, entry_id, top_k)

    def search_in_entry_by_vector(
        self, q_vec: np.ndarray, entry_id: str, top_k: int = 3
    ) -> list[dict]:
        """复用用户查询向量，使各召回路径的分数具有相同含义。"""
        if top_k <= 0 or not NomicEmbedding.is_valid(q_vec):
            return []
        # N-perf 修复: 改用正排索引（entry_id → faiss_ids）做条目内定向检索，
        # 复杂度 O(该条目向量数)，不再全库暴力扫描（此前 top_k=max(count, top_k)
        # 的全库扫描在库增长后会成为主检索路径的性能瓶颈）。
        hit = self._vector_store.search_by_entry(q_vec, entry_id, top_k=top_k)
        enriched = self._enrich(hit)
        if enriched:
            return enriched

        # 回退：条目向量不在索引中（历史坏数据/外部写入），直接给原文开头
        full = self.get_full_text(entry_id)
        if full:
            meta = self.get_metadata(entry_id) or {}
            return [{
                "entry_id": entry_id,
                "chunk_index": -1,
                "chunk_text": full[:512],
                "score": 0.0,
                "timestamp": meta.get("timestamp", 0),
                "full_text": full,
            }]
        return []

    def candidate_chunks(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        """全库段落候选（不读全文文件，供调用方先过滤再 attach_full_text）。"""
        if top_k <= 0 or not NomicEmbedding.is_valid(q_vec):
            return []
        return [
            {
                "entry_id": r["metadata"].get("entry_id", ""),
                "chunk_index": r["metadata"].get("chunk_index", -1),
                "chunk_text": r["metadata"].get("chunk_text", ""),
                "score": r["score"],
                "timestamp": r["metadata"].get("timestamp", 0),
            }
            for r in self._vector_store.search(q_vec, top_k=top_k)
        ]

    def attach_full_text(self, items: list[dict]) -> list[dict]:
        """为 candidate_chunks 的结果补上 full_text（同一条目只读一次盘）。"""
        full_texts: dict[str, Optional[str]] = {}
        for item in items:
            eid = item.get("entry_id", "")
            if eid not in full_texts:
                full_texts[eid] = self.get_full_text(eid) if eid else None
            item["full_text"] = full_texts[eid]
        return items

    def adjacent_chunks(self, hit: dict) -> list[dict]:
        """按已命中的段落定位同一条目的前后两段，兼容已有向量索引。"""
        entry_id = hit.get("entry_id")
        index = hit.get("chunk_index")
        if not entry_id or not isinstance(index, int) or index < 0:
            return []
        neighbors = []
        for offset in (-1, 1):
            other = index + offset
            meta = self._vector_store.get_metadata(f"{entry_id}_chunk_{other}")
            if meta and meta.get("chunk_text"):
                neighbors.append({**hit, "chunk_index": other, "chunk_text": meta["chunk_text"]})
        return neighbors

    def entry_chunk_vectors(self, entry_id: str) -> list[tuple[np.ndarray, str]]:
        return [(vector, meta["chunk_text"])
                for _, vector, meta in self._vector_store.vectors_by_entry(entry_id)]

    def iter_committed_text(self, *, with_metadata=False):
        """Read committed records for the local chapter and correction lookup."""
        for entry_id in self.list_entries():
            try:
                record = self._text_store.load(entry_id)
            except (OSError, ValueError):
                logger.warning("跳过无法读取的全文条目: %s", entry_id)
                continue
            if isinstance(record, dict) and record.get("ingest_result") and isinstance(record.get("full_text"), str):
                meta = record.get("meta") or {}
                yield entry_id, record["full_text"], meta if with_metadata else meta.get("timestamp", 0)

    def _enrich(self, results: list[dict]) -> list[dict]:
        """把 VectorStore 原始结果装配成带全文的检索结果"""
        enriched = []
        full_texts: dict[str, Optional[str]] = {}
        for r in results:
            meta = r["metadata"]
            chunk_text = meta.get("chunk_text", "")
            entry_id = meta.get("entry_id", "")

            if entry_id not in full_texts:
                full_texts[entry_id] = None
                if entry_id:
                    record = self._load_record(entry_id)
                    if record:
                        full_texts[entry_id] = record.get("full_text", "")
            full_text = full_texts[entry_id]

            enriched.append({
                "entry_id": entry_id,
                "chunk_index": meta.get("chunk_index", -1),
                "chunk_text": chunk_text,
                "score": r["score"],
                "timestamp": meta.get("timestamp", 0),
                "full_text": full_text,
            })
        return enriched

    # ── 读取 ────────────────────────────────────────────────────

    def has_content(self, text: str) -> bool:
        """
        该文本是否已入库（内容哈希查询，O(1)，不读条目文件）。

        供 /v1 代理的补录窗口做"未入库"判定（B1 修复）：
        补录前先剔除已入库内容，避免重复内容挤占补录名额。
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._content_lock:
            return content_hash in self._content_index

    def get_full_text(self, entry_id: str) -> Optional[str]:
        """按 entry_id 获取完整全文"""
        record = self._load_record(entry_id)
        if record:
            return record.get("full_text")
        return None

    def get_ingest_result(self, entry_id: str) -> Optional[dict]:
        """读取该条目上次完整入库的结果（内容去重时直接复用）"""
        record = self._load_record(entry_id)
        if record:
            return record.get("ingest_result")
        return None

    def save_ingest_result(self, entry_id: str, result: dict) -> None:
        """三级写入全部成功后记录结果，配合内容去重保证 ingest 幂等"""
        record = self._load_record(entry_id)
        if record:
            record["ingest_result"] = result
            self._save_record(entry_id, record)

    def delete_entry(self, entry_id: str) -> bool:
        """Remove an entry and its indexed chunks during a failed ingest."""
        record = self._load_record(entry_id)
        if not record:
            return False
        total_chunks = int((record.get("meta") or {}).get("total_chunks", 0))
        self._vector_store.delete_many(
            f"{entry_id}_chunk_{idx}" for idx in range(total_chunks)
        )
        self._chunk_map.pop(entry_id, None)
        self._delete_record(entry_id)
        with self._content_lock:
            hashes = [h for h, eid in self._content_index.items() if eid == entry_id]
            for content_hash in hashes:
                self._content_index.pop(content_hash, None)
            if hashes:
                tmp = self._content_index_path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(self._content_index, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp.replace(self._content_index_path)
        return True

    def get_metadata(self, entry_id: str) -> Optional[dict]:
        """按 entry_id 获取元数据"""
        record = self._load_record(entry_id)
        if record:
            return record.get("meta")
        return None

    def list_entries(self) -> list[str]:
        """列出所有缓存条目 ID"""
        return self._text_store.list_keys(prefix="cc_")

    def count(self) -> int:
        """缓存条目总数"""
        return self._text_store.count(prefix="cc_")

    def flush(self) -> None:
        """将全文向量索引的待持久化变更写入磁盘。"""
        self._vector_store.flush()
