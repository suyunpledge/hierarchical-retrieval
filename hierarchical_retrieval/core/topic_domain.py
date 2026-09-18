"""
话题界域模块 —— 三级架构中的界域划分层

将对话信息划分为不同的话题界域，每个界域拥有独立的
关键句库索引，从而实现"按话题分配记忆"。
"""

import json
import logging
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import HConfig
from ..storage.local_storage import LocalStorage
from ..storage.vector_store import VectorStore
from .embedding import NomicEmbedding
from ..utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class TopicDomain:
    """单个话题界域"""

    def __init__(self, domain_id: str, name: str, description: str = "", embedding: Optional[np.ndarray] = None):
        self.domain_id = domain_id
        self.name = name
        self.description = description
        self.embedding = embedding           # 该话题的语义向量代表
        self.created_at: float = time.time()
        self.entry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "description": self.description,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at,
            "entry_count": self.entry_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopicDomain":
        emb = np.array(d.get("embedding", []), dtype=np.float32) if d.get("embedding") else None
        domain = cls(
            domain_id=d["domain_id"],
            name=d.get("name", ""),
            description=d.get("description", ""),
            embedding=emb,
        )
        domain.created_at = d.get("created_at", 0.0)
        domain.entry_count = d.get("entry_count", 0)
        return domain


class TopicDomainManager:
    """
    话题界域管理器

    层级: 第三级（顶层）
    职责:
      - 自动检测/分配话题界域
      - 每个界域拥有独立的关键句向量空间
      - 根据查询确定当前话题界域，缩小检索范围
    """

    def __init__(
        self,
        embedding: NomicEmbedding,
        config: Optional[HConfig] = None,
    ):
        self.config = config or HConfig()
        self.embedding = embedding

        base_dir = self.config.storage_root
        self._domain_dir = Path(f"{base_dir}/{self.config.topic_domain_dir}")
        self._domain_dir.mkdir(parents=True, exist_ok=True)

        # 界域注册表
        self._domains: dict[str, TopicDomain] = {}
        self._load_registry()

        # 关键句向量存储（按界域隔离）
        self._vector_stores: dict[str, VectorStore] = {}
        self._lock = threading.RLock()

    # ── 界域注册表持久化 ──────────────────────────────────

    def _registry_path(self) -> Path:
        return self._domain_dir / "domain_registry.json"

    def _load_registry(self):
        path = self._registry_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for d in data:
                    domain = TopicDomain.from_dict(d)
                    self._domains[domain.domain_id] = domain
                logger.info("加载 %d 个话题界域", len(self._domains))
            except Exception as e:
                logger.warning("加载话题界域注册表失败: %s", e)

    def _save_registry(self):
        data = [d.to_dict() for d in self._domains.values()]
        # 原子写（TRAE v0.2 补丁）: 先写临时文件再替换，
        # 避免进程崩溃时写一半损坏注册表、丢失全部话题域
        path = self._registry_path()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    # ── 界域级向量存储 ────────────────────────────────────

    def _get_vector_store(self, domain_id: str) -> VectorStore:
        if domain_id not in self._vector_stores:
            base_dir = self.config.storage_root
            vs = VectorStore(
                root_dir=f"{base_dir}/{self.config.vector_index_dir}/domains/{domain_id}",
                dimension=self.config.embedding_dim,
            )
            self._vector_stores[domain_id] = vs
        return self._vector_stores[domain_id]

    # ── 界域分配 ──────────────────────────────────────────

    def assign_domain(self, text: str, metadata: Optional[dict] = None) -> TopicDomain:
        """
        将文本归入已有话题或创建新话题。

        Args:
            text: 对话文本
            metadata: 额外信息

        Returns:
            匹配的 TopicDomain 对象
        """
        # 写入路径 strict：嵌入失败抛错。否则零向量会成为新话题的质心，
        # 该话题永远无法被 detect_domain 匹配（相似度恒为 0），且每条
        # 后续文本都会因匹配失败而新建话题，造成话题域无限增殖。
        text_vec = self.embedding.embed(text, strict=True)

        # 界域注册表的匹配、更新、新建必须在同一临界区内，避免并发请求
        # 同时创建重复话题或互相覆盖 entry_count/质心。
        with self._lock:
            return self._assign_domain_locked(text, text_vec)

    def _assign_domain_locked(self, text: str, text_vec: np.ndarray) -> TopicDomain:
        """assign_domain 的锁内实现。"""
        # 寻找最相似已有话题
        best_domain = None
        best_score = -1.0

        for domain in self._domains.values():
            if domain.embedding is not None:
                sim = float(np.dot(text_vec, domain.embedding))
                if sim > best_score:
                    best_score = sim
                    best_domain = domain

        # 如果相似度超过阈值，归入已有话题
        if best_domain and best_score >= self.config.topic_min_similarity:
            # 平滑更新话题向量
            alpha = 0.15  # 新信息权重
            best_domain.embedding = (
                (1 - alpha) * best_domain.embedding + alpha * text_vec
            )
            best_domain.embedding = best_domain.embedding / (
                np.linalg.norm(best_domain.embedding) + 1e-10
            )
            best_domain.entry_count += 1
            self._save_registry()
            logger.info(
                "归入话题 [%s] %s (相似度 %.4f)",
                best_domain.domain_id, best_domain.name, best_score,
            )
            return best_domain

        # 话题域已达上限：强制归入当前最相似域（topic_max_domains 此前
        # 从未生效，界域数会无界增长，反噬检测精度与存储）
        if len(self._domains) >= self.config.topic_max_domains:
            if best_domain is not None:
                logger.warning(
                    "话题域已达上限(%d)，相似度 %.4f 低于阈值 %.2f，强制归入 [%s]",
                    self.config.topic_max_domains, best_score,
                    self.config.topic_min_similarity, best_domain.domain_id,
                )
                best_domain.entry_count += 1
                self._save_registry()
                return best_domain

        # 创建新话题
        domain_id = f"td_{uuid.uuid4().hex[:8]}"
        name = self._generate_domain_name(text)
        new_domain = TopicDomain(
            domain_id=domain_id,
            name=name,
            description=text[:100],
            embedding=text_vec / (np.linalg.norm(text_vec) + 1e-10),
        )
        new_domain.entry_count = 1
        self._domains[domain_id] = new_domain
        self._save_registry()

        logger.info("创建新话题 [%s] %s", domain_id, name)
        return new_domain

    # ── 话题检测 ──────────────────────────────────────────

    def detect_domain(self, query: str) -> Optional[TopicDomain]:
        """
        检测查询属于哪个话题界域。

        Returns:
            匹配的 TopicDomain，若无匹配则返回 None
        """
        if not self._domains:
            return None

        q_vec = self.embedding.embed(query)
        best_domain = None
        best_score = -1.0

        for domain in self._domains.values():
            if domain.embedding is not None:
                sim = float(np.dot(q_vec, domain.embedding))
                if sim > best_score:
                    best_score = sim
                    best_domain = domain

        if best_domain and best_score >= self.config.topic_similarity_threshold:
            logger.info(
                "话题检测: [%s] %s (相似度 %.4f)",
                best_domain.domain_id, best_domain.name, best_score,
            )
            return best_domain

        return None

    # ── 界域内关键句检索 ──────────────────────────────────

    def search_in_domain(
        self, query: str, domain: TopicDomain, top_k: int = 5
    ) -> list[dict]:
        """
        在指定话题界域内检索关键句。

        嵌入服务不可用（查询向量为零向量）时返回空列表（M3 修复）。
        """
        vs = self._get_vector_store(domain.domain_id)
        q_vec = self.embedding.embed(query)
        if not NomicEmbedding.is_valid(q_vec):
            logger.error("查询嵌入失败（Embedding 服务不可用），本次检索返回空")
            return []
        return vs.search(q_vec, top_k=top_k)

    def add_key_sentence_to_domain(
        self, domain_id: str, sentence_id: str, text: str, metadata: Optional[dict] = None
    ):
        """将关键句加入指定话题界域的向量索引"""
        vs = self._get_vector_store(domain_id)
        vec = self.embedding.embed(text, strict=True)
        vs.add(
            external_id=sentence_id,
            vector=vec,
            metadata={
                "sentence_id": sentence_id,
                "text": text,
                **(metadata or {}),
            },
        )

    def add_key_sentences_batch(
        self, domain_id: str, records: list[dict], extra_meta: Optional[dict] = None
    ):
        """
        批量将关键句加入界域索引，直接复用关键句库已算好的向量。

        此前每条关键句在界域索引里要重新嵌入一次——同样的文本在
        一次 ingest 里被嵌了两遍，是写入耗时的大头之一。
        """
        if not records:
            return
        vs = self._get_vector_store(domain_id)
        vs.add_many([
            (
                r["sentence_id"],
                r["vector"],
                {
                    "sentence_id": r["sentence_id"],
                    "text": r["text"],
                    **(extra_meta or {}),
                },
            )
            for r in records
        ])

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _generate_domain_name(text: str) -> str:
        """从文本片段生成话题名"""
        # 取前 20 个字符作为初始话题名
        clean = text.strip().replace("\n", " ")[:20]
        if len(clean) < 3:
            return f"话题-{uuid.uuid4().hex[:4]}"
        return f"{clean}..."

    def list_domains(self) -> list[TopicDomain]:
        return list(self._domains.values())

    def get_domain(self, domain_id: str) -> Optional[TopicDomain]:
        return self._domains.get(domain_id)

    def count(self) -> int:
        return len(self._domains)
