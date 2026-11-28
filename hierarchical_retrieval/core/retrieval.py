"""
分级检索流水线 —— 三级架构的核心调度引擎

完整流程:
  1. 话题检测 → 确定当前查询所属的话题界域（第三级）
  2. 界域内关键句检索 → 在该话题的关键句库中查找最相关句（第二级）
  3. 关键句→全文关联 → 通过关键句关联到全文缓存的详细上下文（第一级）
  4. 结果聚合 → 按话题/相关性排序返回

该机制使"上下文长度"聚焦在指定话题界域对应的关键句库中发挥，
在信息较杂的对话中实现大容量信息记忆的精准召回。
"""

import copy
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..config import HConfig
from .embedding import NomicEmbedding
from .cloud_cache import CloudCache
from .key_sentence import KeySentenceLibrary
from .topic_domain import TopicDomainManager, TopicDomain
from ..utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class CachedRetrieval:
    result: dict
    query_vector: np.ndarray
    direct_floor: float
    candidate_floor: float
    hybrid_floor: float


class HierarchicalRetrieval:
    """
    分级检索控制器

    三级架构:
      L3 ─ TopicDomainManager  (话题界域)
      L2 ─ KeySentenceLibrary   (关键句库)
      L1 ─ CloudCache           (全文缓存，本地磁盘)
    """

    def __init__(
        self,
        embedding: Optional[NomicEmbedding] = None,
        config: Optional[HConfig] = None,
    ):
        self.config = config or HConfig()
        self.embedding = embedding or NomicEmbedding(self.config)
        self._ingest_lock = threading.RLock()

        # 三级存储
        self.cloud_cache = CloudCache(self.embedding, self.config)
        self.key_sentence_lib = KeySentenceLibrary(self.embedding, self.config)
        self.topic_manager = TopicDomainManager(self.embedding, self.config)

        # 成功追加后检查新增向量是否会影响已有结果。generation 阻止并发检索回填旧快照。
        self._result_cache: "OrderedDict[tuple, CachedRetrieval]" = OrderedDict()
        self._result_cache_lock = threading.Lock()
        self._result_cache_condition = threading.Condition(self._result_cache_lock)
        self._result_inflight: set[tuple] = set()
        self._result_cache_capacity = max(0, int(getattr(self.config, "retrieval_cache_size", 256)))
        self._book_quotes = OrderedDict()
        self._revision_evidence = OrderedDict()
        self._revision_hits = self._revision_misses = 0
        self._book_inflight = set()
        self._book_hits = self._book_misses = 0
        self._generation = 0
        self._result_hits = 0
        self._result_misses = 0
        self._write_began = False
        self._chapter_passages: Optional[dict[str, list[tuple[str, int, int]]]] = None
        self._user_revisions: list[tuple[float, str, str]] = []

    # ── 检索结果缓存 ───────────────────────────────────────

    def _invalidate_retrieval_cache(self) -> None:
        with self._result_cache_lock:
            self._generation += 1
            self._result_cache.clear()
            self._book_quotes.clear()
            self._revision_evidence.clear()

    def _revalidate_after_append(self, entry_id: str, sentences: list[dict]) -> None:
        chunks = self.cloud_cache.entry_chunk_vectors(entry_id)
        cfg = self.config
        signature = self._retrieval_signature()
        epsilon = 1e-5
        with self._result_cache_lock:
            for key, cached in list(self._result_cache.items()):
                query, sentence_count, context_count, cached_signature = key
                vector = cached.query_vector
                old_domain = cached.result["detected_domain"]
                domain = self.topic_manager.detect_domain_by_vector(vector)
                changes_domain = ((domain.domain_id if domain else None)
                                  != (old_domain["domain_id"] if old_domain else None))
                old_sentences = cached.result["key_sentences"]
                sentence_floor = (old_sentences[-1]["score"] if len(old_sentences) == sentence_count
                                  else cfg.key_sentence_similarity_threshold)
                if cached_signature != signature or changes_domain or any(
                    float(sentence["vector"] @ vector) >= sentence_floor - epsilon
                    for sentence in sentences
                ):
                    del self._result_cache[key]
                    continue
                if context_count and cfg.direct_chunk_search:
                    for chunk_vector, text in chunks:
                        score = float(chunk_vector @ vector)
                        # An append cannot enter a full dense candidate window below
                        # its old cutoff; the cutoff can only increase with more rows.
                        if score < cached.candidate_floor - epsilon:
                            continue
                        # A strong raw score can displace an old hit before reranking.
                        if score >= cached.direct_floor - epsilon or (
                            score + cfg.chunk_lexical_weight * TextProcessor.lexical_overlap(query, text)
                            >= cached.hybrid_floor - epsilon
                        ):
                            del self._result_cache[key]
                            break
                if key in self._result_cache and domain and old_domain["entry_count"] != domain.entry_count:
                    # 替换快照，保留已经取到旧快照的并发读者。
                    result = {**cached.result, "detected_domain": {**old_domain, "entry_count": domain.entry_count}}
                    result["summary"] = self._build_summary(result)
                    self._result_cache[key] = replace(cached, result=result)

    def _retrieval_signature(self) -> tuple:
        """调参或更换嵌入实例后，旧结果不应继续命中。"""
        return (
            id(self.embedding), self.embedding.model_name, self.embedding.dimension,
            self.config.topic_similarity_threshold,
            self.config.key_sentence_similarity_threshold,
            self.config.direct_chunk_search, self.config.chunk_candidates,
            self.config.chunk_similarity_threshold, self.config.chunk_lexical_weight,
        )

    def cache_stats(self) -> dict:
        """嵌入、检索结果与全文记录的缓存命中统计。"""
        with self._result_cache_lock:
            hits, misses = self._result_hits, self._result_misses
            size = len(self._result_cache)
            chapter_stats = {"hits": self._book_hits, "misses": self._book_misses, "size": len(self._book_quotes)}
            revision_stats = {"hits": self._revision_hits, "misses": self._revision_misses, "size": len(self._revision_evidence)}
        total = hits + misses
        emb_stats = getattr(self.embedding, "cache_stats", None)
        fulltext_stats = getattr(self.cloud_cache, "cache_stats", None)
        return {
            "embedding": emb_stats() if callable(emb_stats) else None,
            "chapter_evidence": chapter_stats,
            "revision_evidence": revision_stats,
            "retrieval": {
                "capacity": self._result_cache_capacity,
                "size": size,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 4) if total else 0.0,
            },
            "full_text": fulltext_stats() if callable(fulltext_stats) else None,
        }

    def _catalogue_entry(self, entry_id: str, text: str, meta: dict) -> None:
        revision = TextProcessor.stored_user_revision(text)
        if revision:
            self._user_revisions.append((meta.get("timestamp", 0), entry_id, revision))
        if not text.startswith(("用户：", "助手：")):
            for chapter, start, end in TextProcessor.chapter_spans(text, meta.get("start_chapter")):
                self._chapter_passages.setdefault(chapter, []).append((entry_id, start, end))

    def _context_catalogue(
        self, query: str, include_chapter: bool,
    ) -> tuple[list[tuple[float, str, str]], list[tuple[str, int, int]]]:
        with self._ingest_lock:
            if self._chapter_passages is None:
                self._chapter_passages = {}
                try:
                    for entry_id, text, meta in self.cloud_cache.iter_committed_text(with_metadata=True):
                        self._catalogue_entry(entry_id, text, meta)
                except Exception:
                    self._chapter_passages = None
                    self._user_revisions.clear()
                    raise
            chapters = TextProcessor.chapter_ids(query) if include_chapter else []
            passages = [span for chapter in dict.fromkeys(chapters)
                        for span in self._chapter_passages.get(chapter, ())]
            return list(self._user_revisions), passages

    # ── 写入流水线 ─────────────────────────────────────────

    def ingest(self, text: str, metadata: Optional[dict] = None) -> dict:
        """
        将一段对话写入三级存储体系。

        Args:
            text: 对话全文
            metadata: 附加元数据

        Returns:
            {"entry_id", "domain_id", "domain_name", "sentence_ids"}
        """
        with self._ingest_lock:
            try:
                return self._ingest_locked(text, metadata)
            except Exception:
                if self._write_began:
                    self._invalidate_retrieval_cache()
                raise
            finally:
                with self._result_cache_lock:
                    if self._write_began:
                        self._generation += 1
                    self._write_began = False

    def _ingest_locked(self, text: str, metadata: Optional[dict] = None) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ingest(): 文本不能为空")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("ingest(): metadata 必须是对象")
        content_existed = self.cloud_cache.has_content(text)
        if content_existed:
            # 内容已完整入库时直接复用上次结果：不产生写入，检索缓存不失效
            # （store 对已登记内容只返回原 ID，不写盘）
            existing_id = self.cloud_cache.store(text, metadata)
            prev = self.cloud_cache.get_ingest_result(existing_id)
            if prev:
                logger.info("ingest 内容重复，跳过重复入库: %s", existing_id)
                return prev
        with self._result_cache_lock:
            self._write_began = True
            self._generation += 1
        # L1: 存入全文缓存。内容级去重：同一文本重复 ingest 直接
        # 复用上次结果——代理模式下每轮都会把历史送进来，没有幂等性
        # 库会无限膨胀。
        entry_id = self.cloud_cache.store(text, metadata)
        prev = self.cloud_cache.get_ingest_result(entry_id)
        if prev:
            logger.info("ingest 内容重复，跳过重复入库: %s", entry_id)
            return prev

        domain = None
        sentence_ids: list[str] = []
        domains_before = {d.domain_id for d in self.topic_manager.list_domains()}
        try:
            # L3: 分配话题界域
            domain = self.topic_manager.assign_domain(text, metadata)

            # L2: 提取关键句并存入关键句库（返回带向量的记录）
            ks_records = self.key_sentence_lib.ingest(text, entry_id)
            sentence_ids = [r["sentence_id"] for r in ks_records]

            # L2→L3 关联: 将关键句批量加入话题界域的向量索引（复用向量）
            self.topic_manager.add_key_sentences_batch(
                domain_id=domain.domain_id,
                records=ks_records,
                extra_meta={
                    "source_entry_id": entry_id,
                    "type": "key_sentence",
                },
            )
        except Exception:
            if domain is not None:
                try:
                    self.topic_manager.remove_key_sentences(domain.domain_id, sentence_ids)
                    self.topic_manager.rollback_assignment(
                        domain.domain_id, domain.domain_id not in domains_before
                    )
                except Exception as cleanup_error:
                    logger.exception("回滚话题索引失败: %s", cleanup_error)
            try:
                self.key_sentence_lib.delete_source(entry_id)
            except Exception as cleanup_error:
                logger.exception("回滚关键句失败: %s", cleanup_error)
            if not content_existed:
                try:
                    self.cloud_cache.delete_entry(entry_id)
                except Exception as cleanup_error:
                    logger.exception("回滚全文缓存失败: %s", cleanup_error)
            raise

        result = {
            "entry_id": entry_id,
            "domain_id": domain.domain_id,
            "domain_name": domain.name,
            "sentence_ids": sentence_ids,
        }
        # 三级全部成功后落档，配合内容去重实现 ingest 幂等
        self.cloud_cache.save_ingest_result(entry_id, result)
        if self._chapter_passages is not None:
            meta = self.cloud_cache.get_metadata(entry_id) or {}
            self._catalogue_entry(entry_id, text, meta)
        if content_existed:
            self._invalidate_retrieval_cache()
        elif self._result_cache:
            try:
                self._revalidate_after_append(entry_id, ks_records)
            except Exception:
                # 入库已经提交，验证异常时丢弃快照并保留写入成功的语义。
                logger.exception("无法验证追加写入后的检索缓存")
                self._invalidate_retrieval_cache()
        return result

    # ── 检索流水线 ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k_sentences: Optional[int] = None,
        top_k_context: Optional[int] = None,
    ) -> dict:
        """
        分级检索主入口

        Args:
            query: 查询文本
            top_k_sentences: 每个话题检索的关键句数
            top_k_context:  每个关键句关联的全文段落数

        Returns:
            {
                "query": str,
                "detected_domain": dict or None,
                "key_sentences": [...],
                "full_contexts": [...],
                "summary": str,
            }
        """
        if not isinstance(query, str) or not query.strip():
            return {
                "query": query,
                "detected_domain": None,
                "key_sentences": [],
                "full_contexts": [],
                "summary": "查询文本为空",
            }
        tks = self.config.top_k_key_sentences if top_k_sentences is None else top_k_sentences
        tkc = self.config.top_k_full_context if top_k_context is None else top_k_context
        if not isinstance(tks, int) or tks < 0:
            raise ValueError("top_k_sentences 必须是非负整数")
        if not isinstance(tkc, int) or tkc < 0:
            raise ValueError("top_k_context 必须是非负整数")
        if tks == 0:
            return {
                "query": query,
                "detected_domain": None,
                "key_sentences": [],
                "full_contexts": [],
                "summary": "未请求返回结果",
            }
        key = (TextProcessor.retrieval_query(query), tks, tkc, self._retrieval_signature())
        use_cache = self._result_cache_capacity > 0
        cached = None
        if use_cache:
            with self._result_cache_condition:
                while True:
                    generation = self._generation
                    cached = self._result_cache.get(key) if not self._write_began else None
                    if cached is not None:
                        self._result_cache.move_to_end(key)
                        self._result_hits += 1
                        break
                    if key not in self._result_inflight:
                        self._result_misses += 1
                        self._result_inflight.add(key)
                        break
                    self._result_cache_condition.wait()
        else:
            with self._result_cache_lock:
                generation = self._generation
        if cached is not None:
            out = copy.deepcopy(cached.result)
            out["query"] = query
            return out

        try:
            result = self._retrieve_uncached(key[0], tks, tkc)
            dependencies = result.pop("_cache_dependencies", None)
            result["query"] = query
            if use_cache and dependencies is not None:
                snapshot = CachedRetrieval(copy.deepcopy(result), *dependencies)
                with self._result_cache_condition:
                    if (self._generation == generation and not self._write_began
                            and key[3] == self._retrieval_signature()):
                        self._result_cache[key] = snapshot
                        self._result_cache.move_to_end(key)
                        while len(self._result_cache) > self._result_cache_capacity:
                            self._result_cache.popitem(last=False)
            return result
        finally:
            if use_cache:
                with self._result_cache_condition:
                    self._result_inflight.discard(key)
                    self._result_cache_condition.notify_all()

    def _retrieve_uncached(self, query: str, tks: int, tkc: int) -> dict:
        """retrieve 的实际检索流程（不经结果缓存）。

        结果带 "_cache_dependencies" 表示可缓存；嵌入服务不可用等降级结果
        不带该标记，避免服务恢复后仍返回缓存的空结果。
        """
        thr = self.config.key_sentence_similarity_threshold

        # 嵌入服务可用性前置检查：查询向量是零向量时（Ollama 掉线），
        # IndexFlatIP 会按插入顺序返回任意记录伪装成命中，必须拦下
        query_vector = self.embedding.embed(query)
        if not NomicEmbedding.is_valid(query_vector):
            logger.error("查询嵌入失败（Embedding 服务不可用），本次检索返回空")
            result = {
                "query": query, "detected_domain": None,
                "key_sentences": [], "full_contexts": [],
                "summary": "Embedding 服务不可用，本次检索返回空结果",
            }
            return result

        result = {
            "query": query,
            "detected_domain": None,
            "key_sentences": [],
            "full_contexts": [],
            "summary": "",
        }

        # 全局候选始终参与排序：话题质心只能做提示，不能将更相关的
        # 跨域答案挡在门外。合并后统一阈值、去重、降序截断。
        domain = self.topic_manager.detect_domain_by_vector(query_vector)
        candidates = self.key_sentence_lib.search_by_vector(query_vector, top_k=tks)
        if domain is not None:
            result["detected_domain"] = {
                "domain_id": domain.domain_id,
                "name": domain.name,
                "description": domain.description,
                "entry_count": domain.entry_count,
            }
            candidates.extend(self._flatten_results(
                self.topic_manager.search_in_domain_by_vector(query_vector, domain, top_k=tks)
            ))
        by_sentence = {}
        for candidate in candidates:
            if candidate.get("score", 0.0) < thr:
                continue
            identity = candidate.get("sentence_id") or (
                candidate.get("source_entry_id"), candidate.get("text")
            )
            previous = by_sentence.get(identity)
            if previous is None or candidate["score"] > previous["score"]:
                by_sentence[identity] = candidate
        result["key_sentences"] = sorted(
            by_sentence.values(), key=lambda k: k["score"], reverse=True
        )[:tks]

        # ── Step 3: 关键句→全文关联 (L1) ──────────────────
        # 修复：此前 source_entry_id 收集进 seen_entries 后从未被使用，
        # 实际执行的是"拿关键句文本去全库重新向量搜索"，返回的全文
        # 可能来自任何一篇无关对话，分级隔离在最后一步被放大回全库。
        # 现在直接在关键句所属的全文条目内检索最相关段落，这才是
        # 设计文档里"根据关键句在缓存中查找详细对话"的本意。
        seen_entries = set()
        contexts = []
        for ks in result["key_sentences"] if tkc > 0 else []:
            source_id = ks.get("source_entry_id", "")
            if not source_id or source_id in seen_entries:
                continue
            seen_entries.add(source_id)
            contexts.extend(
                self.cloud_cache.search_in_entry_by_vector(
                    query_vector, source_id, top_k=tkc
                )
            )
        for ctx in contexts:
            ctx["via"] = "key_sentence"

        # ── Step 4: 全文段落直检（第二条召回路径）──────────
        # 关键句每块最多 3 句，约一半句子不会进入关键句库；问到这些句子里
        # 的事实时，上面的路径必然落空。实测（tests/eval_recall.py）全文段落
        # 向量检索能把目标对话排在第一，只是分数分布与关键句不同，故单独
        # 设阈值，并用关键词重合度校正专名/数字等被稠密向量稀释的命中。
        direct = []
        candidate_floor = float("-inf")
        hybrid_floor = self.config.chunk_similarity_threshold
        if tkc > 0 and getattr(self.config, "direct_chunk_search", False):
            window_size = max(tkc, self.config.chunk_candidates)
            window = self.cloud_cache.candidate_chunks(query_vector, window_size)
            if len(window) == window_size:
                candidate_floor = min(ctx["score"] for ctx in window)
            direct = self._direct_chunk_hits(query, query_vector, tkc, candidates=window)
            if len(direct) == tkc:
                # Hybrid scores are rounded to four decimals before sorting.
                hybrid_floor = max(hybrid_floor, direct[-1]["hybrid_score"] - 0.00005)
            contexts.extend(direct)

        # 去重并排序
        by_chunk = {}
        for ctx in contexts:
            # 两条路径统一使用 query→chunk 的余弦和同一字面校正。
            lexical = TextProcessor.lexical_overlap(query, ctx.get("chunk_text", ""))
            ctx["lexical"] = round(lexical, 4)
            ctx["hybrid_score"] = ctx["score"] + self.config.chunk_lexical_weight * lexical
            chunk_id = (ctx.get("entry_id"), ctx.get("chunk_index"))
            previous = by_chunk.get(chunk_id)
            if previous is None:
                ctx["sources"] = [ctx["via"]]
                by_chunk[chunk_id] = ctx
            else:
                if ctx["via"] not in previous["sources"]:
                    previous["sources"].append(ctx["via"])

        unique_contexts = sorted(
            by_chunk.values(), key=lambda c: c["hybrid_score"], reverse=True
        )

        result["full_contexts"] = unique_contexts[:tkc * tks]

        # ── 生成摘要信息 ──────────────────────────────────
        result["summary"] = self._build_summary(result)
        direct_floor = min((hit["score"] for hit in direct), default=float("inf"))
        result["_cache_dependencies"] = (query_vector, direct_floor, candidate_floor, hybrid_floor)
        return result

    def _direct_chunk_hits(self, query: str, query_vector, limit: int, *, candidates=None) -> list[dict]:
        """全库段落粗召回 → 混合分（余弦 + 权重×关键词重合度）过阈值 → 取前 limit 条。"""
        cfg = self.config
        if candidates is None:
            candidates = self.cloud_cache.candidate_chunks(query_vector, max(limit, cfg.chunk_candidates))
        kept = []
        for ctx in candidates:
            lexical = TextProcessor.lexical_overlap(query, ctx.get("chunk_text", ""))
            hybrid = ctx["score"] + cfg.chunk_lexical_weight * lexical
            if hybrid >= cfg.chunk_similarity_threshold:
                ctx["lexical"] = round(lexical, 4)
                ctx["hybrid_score"] = round(hybrid, 4)
                ctx["via"] = "direct"
                kept.append(ctx)
        kept.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return self.cloud_cache.attach_full_text(kept[:limit])

    # ── 上下文装配（代理模式核心原语） ─────────────────────

    MEMORY_HEADER = "以下是更早对话中的相关记录（按时间从早到晚，冲突时以靠后的为准；记录里没有的不要猜测）："
    REVISION_LABEL = "\n用户明确更正（优先于旧记录和助手曾给出的回答）："
    DIRECT_EVIDENCE_LABEL = "\n与问题直接对应的原文："
    BOOK_SOURCE_HEADER = "本轮询问小说原文。请按以下原文回答，近期对话中的排演改动只适用于排演版本。"

    def _rank_scene_quotes(self, field, unit_match, candidates):
        # Unit and physical chapter already constrain the search. Compare the
        # short numerical sentence separately from the surrounding narrative.
        topic = field[:unit_match.start()].rstrip("是")
        terms = TextProcessor.query_terms(TextProcessor.scene_terms_text(topic)) - {"步数", "时长", "时间"}
        sentences, windows = zip(*dict.fromkeys(candidates))
        vector = self.embedding.embed(topic)
        vectors = self.embedding.embed_batch([*sentences, *windows])
        valid = NomicEmbedding.is_valid(vector) and all(NomicEmbedding.is_valid(v) for v in vectors)
        ranked = []
        for sent, window, focal_vec, window_vec in zip(sentences, windows, vectors, vectors[len(sentences):]):
            focal_text = TextProcessor.scene_terms_text(sent)
            window_text = TextProcessor.scene_terms_text(window)
            direct = sum(t in focal_text for t in terms)
            nearby = sum(t in window_text for t in terms)
            similarity = max(float(vector @ focal_vec), float(vector @ window_vec))
            score = similarity + (2 * direct + nearby) / max(1, len(terms))
            if direct or nearby or valid:
                ranked.append((score, direct, sent, window))
        ranked.sort(key=lambda row: row[0], reverse=True)
        # With little lexical support, let the reader compare a bounded set
        # of actual source occurrences. The answer still requires a matching event.
        quotes = []
        for score, direct, sent, window in ranked[:3]:
            if quotes and ranked[0][1] and ranked[0][0] - score > self.config.chunk_lexical_weight:
                break
            quotes.append(f"数值所在句：{sent}\n场景上下文：{window}")
        return tuple(quotes), valid

    def _book_measurement_quotes(self, chapter, query, passages):
        field = TextProcessor.book_measurement_field(query)
        # A book quote depends only on the physical source spans for this chapter,
        # never on the newest chat revision. A fresh append yields a distinct key.
        spans = tuple(span for span in passages if self.cloud_cache.get_metadata(span[0]) is not None)
        key = (chapter, field, spans, self._retrieval_signature())
        capacity = self._result_cache_capacity
        if not capacity:
            return self._find_book_quotes(chapter, field, spans)[0]
        with self._result_cache_condition:
            while True:
                if key in self._book_quotes:
                    self._book_hits += 1
                    self._book_quotes.move_to_end(key)
                    return self._book_quotes[key]
                if key not in self._book_inflight:
                    generation = self._generation
                    self._book_inflight.add(key)
                    self._book_misses += 1
                    break
                self._result_cache_condition.wait()
        try:
            quotes, cacheable = self._find_book_quotes(chapter, field, spans)
            with self._result_cache_lock:
                if quotes and cacheable and generation == self._generation and not self._write_began:
                    self._book_quotes[key] = quotes
                    self._book_quotes.move_to_end(key)
                    while len(self._book_quotes) > capacity:
                        self._book_quotes.popitem(last=False)
            return quotes
        finally:
            with self._result_cache_condition:
                self._book_inflight.remove(key)
                self._result_cache_condition.notify_all()

    def _chapter_sections(self, spans):
        # Rebuild continuous chapter runs only when ingest supplied verified
        # document coordinates. Independent notes and legacy chunks stay separate.
        located = []
        for entry_id, start, end in spans:
            meta = self.cloud_cache.get_metadata(entry_id) or {}
            section = (self.cloud_cache.get_full_text(entry_id) or "")[start:end]
            document, offset = meta.get("document"), meta.get("offset")
            if not (isinstance(document, str) and document and type(offset) is int and offset >= 0):
                yield section
                continue
            located.append((document, offset + start, section))
        pieces = []
        previous = None
        for document, offset, section in sorted(located):
            if pieces and previous != (document, offset):
                yield "".join(pieces)
                pieces = []
            pieces.append(section)
            previous = (document, offset + len(section))
        if pieces:
            yield "".join(pieces)

    def _find_book_quotes(self, chapter, field, spans):
        scene_unit = TextProcessor.scene_unit(field)
        candidates = []
        best_score, best_quote = 0.0, ""
        for section in self._chapter_sections(spans):
            if scene_unit:
                candidates.extend(TextProcessor.scene_quantity_quotes(section, scene_unit[1]))
            else:
                score, quote = TextProcessor.distance_evidence(section, section, f"第{chapter}章{field}", chapter_section=True)
                if score >= 0.20 and score > best_score:
                    best_score, best_quote = score, quote
        quotes = (best_quote,) if best_quote else ()
        cacheable = True
        if candidates:
            quotes, cacheable = self._rank_scene_quotes(field, scene_unit, candidates)
        return quotes, cacheable

    def _rehearsal_evidence(self, scope, revisions, recent, passages):
        chapter, field, unit, topic = scope
        # The dependency key changes for edits to this chapter, including new competing fields.
        committed = sorted(((ts, i, text) for ts, i, text in revisions
                            if TextProcessor.chapter_ids(text) == [chapter] and self.cloud_cache.get_metadata(i) is not None),
                           key=lambda row: row[0])
        pending = tuple(text for text in recent if len(text) <= 240 and TextProcessor.chapter_ids(text) == [chapter]
                        and TextProcessor.chapter_revision_question(text))
        # Only expanded scene descriptions depend on physical book passages.
        spans = tuple(s for s in passages if self.cloud_cache.get_metadata(s[0]) is not None)
        key = (scope, tuple(i for _, i, _ in committed), pending, spans)
        capacity = self._result_cache_capacity
        if capacity:
            with self._result_cache_lock:
                if key in self._revision_evidence:
                    self._revision_hits += 1
                    self._revision_evidence.move_to_end(key)
                    return self._revision_evidence[key]
                self._revision_misses += 1
        latest = {}
        for note in [*(text for _, _, text in committed), *pending]:
            label, sep, _ = note.partition("：")
            if sep and not TextProcessor._REHEARSAL.fullmatch(label.strip()):
                continue
            question = TextProcessor.chapter_revision_question(note)
            if not question:
                continue
            stored = TextProcessor.rehearsal_scope("按排演笔记，" + question)
            if not stored:
                continue
            _, stored_field, stored_unit, stored_topic = stored
            latest[question.split("是多少", 1)[0]] = (stored_field, stored_unit, stored_topic, note)
        sections = tuple(self._chapter_sections(spans)) if spans else ()
        matches = [TextProcessor.split_sentences(note)[0] for stored_field, stored_unit, stored_topic, note in latest.values()
                   if (not field or field == stored_field) and (not unit or unit == stored_unit)
                   and (TextProcessor.revision_topic_matches(topic, stored_topic) or
                        (sections and TextProcessor.witnessed_revision_scene(topic, stored_topic, sections)))]
        result = (self.MEMORY_HEADER + self.REVISION_LABEL + matches[0] if len(matches) == 1 else
                  f"第{chapter}章排演校订依据不足，请核对具体项目及对应记录。")
        if capacity and len(matches) == 1:
            with self._result_cache_lock:
                self._revision_evidence[key] = result
                self._revision_evidence.move_to_end(key)
                while len(self._revision_evidence) > capacity:
                    self._revision_evidence.popitem(last=False)
        return result

    def assemble_context(
        self,
        query: str,
        char_budget: Optional[int] = None,
        extra_queries: Optional[list[str]] = None,
        anchor_query: Optional[str] = None,
        recent_revision: str = "",
        recent_user_texts: tuple[str, ...] = (),
    ) -> str:
        """
        检索并把结果装配成一段可直接放进 prompt 的紧凑记忆块。

        - 按相关性在字符预算内挑选原文段落与关键句（预算不够时丢弃最不相关的），
          无论库里有几百条对话，出去的这块永远有界。
        - 关键句若已包含在入选的原文段落里则不再重复列出（此前同一句话
          以"关键句"和"原文片段"各出现一次，白占预算）。
        - 入选条目按时间从早到晚排列，并在抬头说明"冲突时以靠后的为准"，
          让模型能处理"先定 20、后来改成 50"这类被更新过的事实。
        - 不再输出"当前话题：<话题名>"：话题名取自该话题第一条消息的前
          20 个字，本身可能是一条过时的事实（如旧房租），会误导模型。
        - extra_queries：附加检索式（如拼上上一条用户消息的追问），结果合并。
        - anchor_query：沿用话题时复用上一问题的少量原文，补足同场景细节。
        """
        budget = self.config.context_char_budget if char_budget is None else char_budget
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError("char_budget 必须是正整数")

        chunks: dict[tuple, dict] = {}
        sentences: dict[str, dict] = {}
        revision = recent_revision
        revision_time = float("inf") if revision else -1.0
        query = TextProcessor.normalize_chapter_aliases(query)
        revision_query = TextProcessor.revision_query(query, anchor_query)
        focus_distance = TextProcessor.is_distance_question(query)
        scope = TextProcessor.rehearsal_scope(revision_query)
        if scope:
            revisions, passages = self._context_catalogue(revision_query, "之前" in scope[3] and "后把" in scope[3])
            pending = recent_user_texts or ((recent_revision,) if recent_revision else ())
            evidence = self._rehearsal_evidence(scope, revisions, pending, passages)
            return evidence if len(evidence) <= budget else ""
        book_source = TextProcessor.asks_for_book_source(revision_query)
        if book_source:
            revision, revision_time = "", float("inf")
        if focus_distance and revision:
            note = self.MEMORY_HEADER + self.REVISION_LABEL + revision
            if len(note) <= budget:
                return note
        distance_quotes = []
        revisions, passages = self._context_catalogue(revision_query, focus_distance)
        for timestamp, entry_id, text in revisions:
            if timestamp > revision_time:
                matched = TextProcessor.matching_revision(text, revision_query)
                if matched and self.cloud_cache.get_metadata(entry_id) is not None:
                    revision, revision_time = matched, timestamp
        if focus_distance and revision:
            note = self.MEMORY_HEADER + self.REVISION_LABEL + revision
            if len(note) <= budget:
                return note
        chapters = TextProcessor.chapter_ids(revision_query)
        scene_unit = TextProcessor.scene_unit(revision_query)
        if focus_distance and len(chapters) == 1 and (book_source or (scene_unit and passages)):
            quotes = self._book_measurement_quotes(chapters[0], revision_query, passages)
            header = f"{self.BOOK_SOURCE_HEADER}\n第{chapters[0]}章原文："
            remaining = budget - len(header)
            selected = []
            for quote in quotes:
                cost = len(quote) + (1 if selected else 0)
                if cost > remaining:
                    # Drop surrounding narration before dropping an intact focal sentence.
                    focal, separator, _ = quote.partition("\n场景上下文：")
                    cost = len(focal) + (1 if selected else 0)
                    if not separator or cost > remaining:
                        break
                    quote = focal
                selected.append(quote)
                remaining -= cost
            if selected:
                return header + "\n".join(selected)
            # A measurement from an unlabelled fragment cannot establish what
            # a particular chapter says, even when its dense score is high.
            missing = f"未能定位第{chapters[0]}章与本题对应的原文数值，请说明依据不足。"
            return missing if len(missing) <= budget else ""
        if focus_distance and not revision and not scene_unit:
            for entry_id, start, end in passages:
                section = (self.cloud_cache.get_full_text(entry_id) or "")[start:end]
                coverage, quote = TextProcessor.distance_evidence(section, section, revision_query, chapter_section=True)
                # 章节已精确匹配，允许场景描述存在措辞差异。
                if coverage >= 0.20:
                    distance_quotes.append((coverage + 0.30, quote, entry_id))
        probes = [(q, False) for q in (query, *(extra_queries or []))]
        if anchor_query:
            probes.append((anchor_query, True))
        for q, is_anchor in probes:
            if not isinstance(q, str) or not q.strip():
                continue
            res = self.retrieve(q)
            hits = res.get("full_contexts", [])
            if is_anchor and hits:
                # 相邻原文可补到上一问的命中片段之前或之后，沿用话题分数。
                hits = [hits[0], *self.cloud_cache.adjacent_chunks(hits[0])]
            for ctx in hits:
                full = ctx.get("full_text") or ""
                if book_source and full.startswith(("用户：", "助手：")):
                    continue
                timestamp = ctx.get("timestamp") or 0
                if timestamp > revision_time:
                    first_line = TextProcessor.stored_user_revision(full)
                    matched = TextProcessor.matching_revision(first_line, revision_query)
                    if matched:
                        revision, revision_time = matched, timestamp
                # 条目只有一个段落时用全文（保留"用户：/助手："的换行）
                body = full if full and len(full) <= 512 else (ctx.get("chunk_text") or full[:512])
                if not body:
                    continue
                score = ctx.get("hybrid_score", ctx.get("score", 0.0))
                if focus_distance and not scene_unit:
                    coverage, quote = TextProcessor.distance_evidence(body, full, revision_query)
                    if coverage >= 0.30:
                        distance_quotes.append((coverage + 0.25 * score, quote, ctx.get("entry_id")))
                if is_anchor:
                    score -= self.config.chunk_lexical_weight / 2
                key = (ctx.get("entry_id"), ctx.get("chunk_index"), body if not ctx.get("entry_id") else "")
                if key not in chunks or score > chunks[key]["score"]:
                    chunks[key] = {"kind": "chunk", "score": score, "text": body,
                                   "ts": ctx.get("timestamp") or 0, "query": q,
                                   "chunk_index": ctx.get("chunk_index") or 0}
            for ks in ([] if is_anchor else res.get("key_sentences", [])):
                text = ks.get("text", "")
                if not text:
                    continue
                if book_source:
                    source = self.cloud_cache.get_full_text(ks["source_entry_id"]) if ks.get("source_entry_id") else ""
                    if not source or source.startswith(("用户：", "助手：")):
                        continue
                score = ks.get("score", 0.0)
                if text not in sentences or score > sentences[text]["score"]:
                    ts = ks.get("timestamp") or 0
                    if not ts and ks.get("source_entry_id"):
                        ts = (self.cloud_cache.get_metadata(ks["source_entry_id"]) or {}).get("timestamp", 0)
                    sentences[text] = {"kind": "sentence", "score": score, "text": text, "ts": ts}

        candidates = sorted(
            [*chunks.values(), *sentences.values()], key=lambda c: c["score"], reverse=True
        )
        sentence_scores = {text: s["score"] for text, s in sentences.items()}
        revision_note = self.REVISION_LABEL + revision if revision else ""
        if len(self.MEMORY_HEADER) + len(revision_note) > budget:
            revision_note = ""
        if focus_distance and revision_note:
            # 标量问题已有用户明确覆盖的当前值时，保留更正原句。
            return self.MEMORY_HEADER + revision_note
        direct_note = ""
        if distance_quotes:
            _, quote, entry_id = max(distance_quotes, key=lambda item: item[0])
            direct_note = self.DIRECT_EVIDENCE_LABEL + quote
            if len(self.MEMORY_HEADER) + len(direct_note) <= budget:
                # 用同一场景的段落补充语境。避免其他章节的相似数字淹没直接引文。
                chunks = {key: chunk for key, chunk in chunks.items() if key[0] == entry_id}
                candidates = sorted(chunks.values(), key=lambda item: item["score"], reverse=True)
            else:
                direct_note = ""
        used = len(self.MEMORY_HEADER) + len(revision_note) + len(direct_note)
        kept: list[dict] = []
        for item in candidates:
            core = item["text"].rstrip("…")
            if item["kind"] == "sentence" and any(
                k["kind"] == "chunk" and core in k["text"] for k in kept
            ):
                continue
            cost = len(item["text"]) + 3          # "- " 前缀与换行
            if item["kind"] == "chunk":
                # 新入选的段落已包含的关键句让出预算
                covered = [k for k in kept if k["kind"] == "sentence" and k["text"].rstrip("…") in item["text"]]
                refund = sum(len(k["text"]) + 3 for k in covered)
                if used - refund + cost > budget:
                    excerpt = TextProcessor.evidence_excerpt(
                        item["text"], item["query"], budget - used - 3, sentence_scores
                    )
                    if not excerpt or any(excerpt in k["text"] for k in kept):
                        continue
                    item = {**item, "text": excerpt}
                    cost = len(excerpt) + 3
                    covered = [k for k in covered if k["text"].rstrip("…") in excerpt]
                    refund = sum(len(k["text"]) + 3 for k in covered)
                kept = [k for k in kept if k not in covered]
                used -= refund
            elif used + cost > budget:
                continue
            kept.append(item)
            used += cost
        if not kept and not revision_note and not direct_note:
            return ""
        kept.sort(key=lambda k: (k["ts"], k.get("chunk_index", 0)))
        evidence = "".join(f"\n- {k['text']}" for k in kept)
        return self.MEMORY_HEADER + direct_note + evidence + revision_note

    # ── 批量写入 ──────────────────────────────────────────

    def ingest_batch(self, conversations: list[dict]) -> list[dict]:
        """
        批量写入多条对话。

        Args:
            conversations: [{"text": str, "metadata": dict}, ...]

        Returns:
            [{"entry_id", "domain_id", ...}, ...]
        """
        return [self.ingest(c["text"], c.get("metadata")) for c in conversations]

    def flush(self) -> None:
        """将三级向量索引中的待持久化变更统一刷盘。"""
        self.cloud_cache.flush()
        self.key_sentence_lib.flush()
        self.topic_manager.flush()

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _flatten_results(results: list[dict]) -> list[dict]:
        """
        将 VectorStore 返回的 {"id","score","metadata"} 展平为
        {"sentence_id","text","source_entry_id","score",...} 顶层结构，
        统一关键句检索结果的访问字段。
        """
        flattened = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            item = {
                "sentence_id": meta.get("sentence_id", r.get("id", "")),
                "text": meta.get("text", ""),
                "source_entry_id": meta.get("source_entry_id", ""),
                "score": r.get("score", 0.0),
                "timestamp": meta.get("timestamp", 0),
            }
            flattened.append(item)
        return flattened

    @staticmethod
    def _build_summary(result: dict) -> str:
        """生成检索结果摘要"""
        parts = []
        domain = result.get("detected_domain")
        if domain:
            parts.append(f"话题界域: [{domain['name']}] (共 {domain['entry_count']} 条记录)")
        else:
            parts.append("话题界域: 未匹配（全局检索）")

        ks = result.get("key_sentences", [])
        parts.append(f"关键句命中: {len(ks)} 条")

        ctx = result.get("full_contexts", [])
        parts.append(f"关联全文段落: {len(ctx)} 段")

        # 列出关键句内容
        if ks:
            parts.append("--- 关键句摘要 ---")
            for i, k in enumerate(ks[:5]):
                text = k.get("text", "")[:80]
                score = k.get("score", 0)
                parts.append(f"  [{i + 1}] (得分 {score:.4f}) {text}...")

        return "\n".join(parts)

    # ── 状态查询 ──────────────────────────────────────────

    def status(self) -> dict:
        """返回系统状态"""
        return {
            "embedding_model": self.embedding.model_name,
            "cloud_cache_entries": self.cloud_cache.count(),
            "key_sentence_count": self.key_sentence_lib.count(),
            "topic_domains": self.topic_manager.count(),
            "cache": self.cache_stats(),
            "domains": [
                {
                    "id": d.domain_id,
                    "name": d.name,
                    "entries": d.entry_count,
                }
                for d in self.topic_manager.list_domains()
            ],
        }
