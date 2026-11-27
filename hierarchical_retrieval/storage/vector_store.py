"""
向量存储模块 —— 基于 FAISS 的本地向量索引
"""

import json
import os
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .base import StorageBase

logger = logging.getLogger(__name__)


class VectorStore(StorageBase):
    """
    轻量向量存储

    使用 FAISS (IndexFlatIP) 作为 ANN 索引后端，
    同时维护 id → metadata 的映射。
    """

    def __init__(self, root_dir: str, dimension: int = 768):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self._index: Optional["faiss.Index"] = None
        self._id_map: dict[str, int] = {}       # external_id → faiss_id
        self._reverse_map: dict[int, str] = {}   # faiss_id → external_id
        self._metadata: dict[str, dict] = {}     # external_id → metadata
        self._next_id: int = 0
        self._lock = threading.RLock()
        # 正排索引: entry_id → [faiss_id...]（条目内检索用，避免全库扫描）
        self._entry_to_ids: dict[str, list[int]] = {}
        # 增量落盘脏标记（N-perf 修复: 写操作不再每次全量重写索引）
        self._dirty = False

        self._load_index()

    # ── 延迟导入 FAISS（避免没有 GPU 时报错） ──────────────────
    @property
    def index(self):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            # 从磁盘恢复已有数据
            self._load_index()
            if self._index is None:
                self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    # ── 持久化 ──────────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.root / "vector_index.faiss"

    def _meta_path(self) -> Path:
        return self.root / "metadata.json"

    def _save_index(self):
        import faiss
        index_path = self._index_path()
        meta_path = self._meta_path()
        index_tmp = index_path.with_name(f".{index_path.name}.{uuid.uuid4().hex}.tmp")
        meta_tmp = meta_path.with_name(f".{meta_path.name}.{uuid.uuid4().hex}.tmp")
        faiss.write_index(self.index, str(index_tmp))
        meta = {
            "id_map": self._id_map,
            "reverse_map": {str(k): v for k, v in self._reverse_map.items()},
            "metadata": self._metadata,
            "next_id": self._next_id,
            "dimension": self.dimension,
            "entry_to_ids": {k: v for k, v in self._entry_to_ids.items()},
        }
        meta_tmp.write_text(
            json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8"
        )
        os.replace(index_tmp, index_path)
        os.replace(meta_tmp, meta_path)

    def _load_index(self):
        idx_path = self._index_path()
        meta_path = self._meta_path()
        if idx_path.exists() != meta_path.exists():
            raise RuntimeError("向量索引文件不完整，拒绝使用空索引覆盖已有数据")
        if idx_path.exists() and meta_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(idx_path))
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stored_dimension = int(meta.get("dimension", self.dimension))
                if stored_dimension != self.dimension or self._index.d != stored_dimension:
                    raise ValueError(
                        f"向量索引维度不匹配: 索引={self._index.d}, 元数据={stored_dimension}, 配置={self.dimension}"
                    )
                self._id_map = meta["id_map"]
                self._reverse_map = {int(k): v for k, v in meta["reverse_map"].items()}
                self._metadata = meta["metadata"]
                self._next_id = meta["next_id"]
                eti = meta.get("entry_to_ids")
                if isinstance(eti, dict):
                    self._entry_to_ids = {k: list(v) for k, v in eti.items()}
                else:
                    # 旧版文件无正排索引: 从 metadata 惰性重建
                    for ext, md in self._metadata.items():
                        eid = (md or {}).get("entry_id")
                        if not eid:
                            continue
                        fid = self._id_map.get(ext, -1)
                        if fid >= 0:
                            self._entry_to_ids.setdefault(eid, []).append(fid)
                self._dirty = False
            except Exception as e:
                self._index = None
                raise RuntimeError(f"加载向量索引失败，拒绝使用空索引覆盖已有数据: {e}") from e

    # ── 核心操作 ────────────────────────────────────────────────

    def add(self, external_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> int:
        """添加向量，返回 faiss_id。零向量将被拒绝（返回 -1），防止污染索引"""
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]
        if vec.shape != (1, self.dimension) or not np.isfinite(vec).all():
            raise ValueError(
                f"向量维度/数值无效: 期望 ({self.dimension},)，实际 {vec.shape}"
            )
        # 零向量防护（TRAE v0.2 补丁）: 嵌入服务失败时会产生零向量，
        # 写入会污染索引（与任何查询内积恒为 0，且占据 top_k 名额）
        if float(np.linalg.norm(vec)) < 1e-10:
            logger.warning(
                "拒绝添加零向量 (id=%s): 嵌入服务可能异常，跳过以防索引污染",
                external_id,
            )
            return -1
        with self._lock:
            if external_id in self._id_map:
                return self._id_map[external_id]
            faiss_id = self._next_id
            self.index.add(vec)
            self._id_map[external_id] = faiss_id
            self._reverse_map[faiss_id] = external_id
            self._metadata[external_id] = metadata or {}
            self._next_id += 1
            entry_id = (metadata or {}).get("entry_id")
            if entry_id:
                self._entry_to_ids.setdefault(entry_id, []).append(faiss_id)
            self._dirty = True
            self._save_index()
            self._dirty = False
        return faiss_id

    def add_many(self, items: list[tuple[str, np.ndarray, Optional[dict]]]) -> None:
        """批量添加：一次 FAISS 写入和一次持久化。"""
        if not items:
            return
        prepared = []
        seen_ids = set()
        for external_id, vector, metadata in items:
            vec = np.asarray(vector, dtype=np.float32)
            if external_id in seen_ids:
                continue
            if vec.ndim != 1 or vec.shape[0] != self.dimension or not np.isfinite(vec).all():
                raise ValueError(
                    f"向量维度/数值无效: id={external_id}, 期望 ({self.dimension},)，实际 {vec.shape}"
                )
            if float(np.linalg.norm(vec)) < 1e-10:
                logger.warning(
                    "拒绝添加零向量 (id=%s): 嵌入服务可能异常，跳过以防索引污染",
                    external_id,
                )
                continue
            seen_ids.add(external_id)
            prepared.append((external_id, vec, metadata))
        if not prepared:
            return
        with self._lock:
            items = [item for item in prepared if item[0] not in self._id_map]
            if not items:
                return
            vecs = np.vstack([v for _i, v, _m in items]).astype(np.float32, copy=False)
            self.index.add(vecs)
            for external_id, _v, metadata in items:
                faiss_id = self._next_id
                self._next_id += 1
                self._id_map[external_id] = faiss_id
                self._reverse_map[faiss_id] = external_id
                self._metadata[external_id] = metadata or {}
                entry_id = (metadata or {}).get("entry_id")
                if entry_id:
                    self._entry_to_ids.setdefault(entry_id, []).append(faiss_id)
            self._dirty = True
            self._save_index()
            self._dirty = False

    def flush(self) -> None:
        """脏标记落盘（原子写）。崩溃时最多丢失最近一批未 flush 的更新。"""
        with self._lock:
            if self._dirty:
                self._save_index()
                self._dirty = False

    def vectors_by_entry(self, entry_id: str) -> list[tuple[str, np.ndarray, dict]]:
        """Read the vectors and metadata for one entry under the index lock."""
        with self._lock:
            return [
                (ext_id, self.index.reconstruct(faiss_id), self._metadata[ext_id])
                for faiss_id in self._entry_to_ids.get(entry_id, [])
                if (ext_id := self._reverse_map.get(faiss_id)) is not None
            ]

    def search_by_entry(self, query_vector: np.ndarray, entry_id: str, top_k: int = 3) -> list[dict]:
        """
        条目内检索（正排索引版）: 直接取该 entry_id 的全部向量做内积排序，
        复杂度 O(该条目向量数)，不再全库暴力扫描（N-perf 修复）。
        条目无向量时返回空列表，由调用方走回退逻辑。
        """
        if top_k <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self.dimension or not np.isfinite(q).all():
            return []
        q = q.reshape(1, -1)
        rows = self.vectors_by_entry(entry_id)
        if not rows:
            return []
        vecs = np.vstack([vector for _, vector, _ in rows])
        scores = (vecs @ q.T).flatten()
        order = np.argsort(-scores)[:top_k]
        out = []
        for oi in order:
            ext, _, meta = rows[int(oi)]
            out.append({
                "id": ext,
                "score": float(scores[int(oi)]),
                "metadata": meta,
            })
        return out

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        检索最相似的 top_k 条记录。

        返回: [{"id": str, "score": float, "metadata": dict}, ...]
        """
        if top_k <= 0 or self.index.ntotal == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        if q.shape != (1, self.dimension) or not np.isfinite(q).all():
            return []
        with self._lock:
            # delete() 只做逻辑删除，被删向量仍留在 FAISS 中并可能占据
            # top_k 名额，导致实际返回远少于 top_k。按墓碑数量多取一些
            # 再过滤，保证有效结果数不被回滚/删除操作吞掉。
            ntotal = self.index.ntotal
            tombstones = max(0, ntotal - len(self._id_map))
            fetch_k = min(ntotal, top_k + tombstones)
            scores, indices = self.index.search(q, fetch_k)
            results = []
            for score, faiss_id in zip(scores[0], indices[0]):
                if faiss_id == -1:
                    continue
                ext_id = self._reverse_map.get(int(faiss_id), "")
                if not ext_id:
                    continue
                results.append({
                    "id": ext_id,
                    "score": float(score),
                    "metadata": self._metadata.get(ext_id, {}),
                })
                if len(results) >= top_k:
                    break
        return results

    def get_metadata(self, external_id: str) -> Optional[dict]:
        return self._metadata.get(external_id)

    def update_metadata(self, external_id: str, metadata: dict) -> bool:
        with self._lock:
            if external_id not in self._metadata:
                return False
            self._metadata[external_id].update(metadata)
            self._dirty = True
        # N-review2: 低频写操作兜底落盘（锁外执行）
        try:
            self.flush()
        except Exception as _e:
            logger.warning(f"[VectorStore] update_metadata flush 失败: {_e}")
        return True

    def delete(self, key: str) -> bool:
        """删除（当前 FAISS 不支持按 id 删除，仅标记清除）"""
        return self.delete_many([key]) > 0

    def delete_many(self, keys: Iterable[str]) -> int:
        """批量逻辑删除，只落盘一次。返回实际删除的条数。

        回滚路径一次要删几十个向量，逐条 delete() 会触发同样次数的
        全量索引重写（O(n²) I/O）。
        """
        removed = 0
        with self._lock:
            for key in keys:
                if key not in self._id_map:
                    continue
                faiss_id = self._id_map.pop(key)
                self._reverse_map.pop(faiss_id, None)
                meta = self._metadata.pop(key, None) or {}
                eid = meta.get("entry_id")
                ids = self._entry_to_ids.get(eid) if eid else None
                if ids is not None and faiss_id in ids:
                    ids.remove(faiss_id)
                    if not ids:
                        self._entry_to_ids.pop(eid, None)
                removed += 1
            if removed:
                self._dirty = True
        # N-review2: 低频写操作兜底落盘（锁外执行，避免锁内 I/O）
        if removed:
            try:
                self.flush()
            except Exception as e:
                logger.warning("[VectorStore] delete flush 失败: %s", e)
        return removed

    # ── StorageBase 接口 ────────────────────────────────────────

    def save(self, key: str, value: Any) -> None:
        """value 应为 {"vector": list, "metadata": dict}"""
        if isinstance(value, dict) and "vector" in value:
            self.add(key, np.array(value["vector"]), value.get("metadata"))

    def load(self, key: str) -> Optional[Any]:
        if key in self._metadata:
            return {"id": key, "metadata": self._metadata[key]}
        return None

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._id_map if k.startswith(prefix))

    def count(self, prefix: str = "") -> int:
        return sum(1 for k in self._id_map if k.startswith(prefix))
