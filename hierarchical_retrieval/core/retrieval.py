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

import logging
from typing import Optional

from ..config import HConfig
from .embedding import NomicEmbedding
from .cloud_cache import CloudCache
from .key_sentence import KeySentenceLibrary
from .topic_domain import TopicDomainManager, TopicDomain

logger = logging.getLogger(__name__)


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

        # 三级存储
        self.cloud_cache = CloudCache(self.embedding, self.config)
        self.key_sentence_lib = KeySentenceLibrary(self.embedding, self.config)
        self.topic_manager = TopicDomainManager(self.embedding, self.config)

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
        # L1: 存入全文缓存。内容级去重：同一文本重复 ingest 直接
        # 复用上次结果——代理模式下每轮都会把历史送进来，没有幂等性
        # 库会无限膨胀。
        entry_id = self.cloud_cache.store(text, metadata)
        prev = self.cloud_cache.get_ingest_result(entry_id)
        if prev:
            logger.info("ingest 内容重复，跳过重复入库: %s", entry_id)
            return prev

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

        result = {
            "entry_id": entry_id,
            "domain_id": domain.domain_id,
            "domain_name": domain.name,
            "sentence_ids": sentence_ids,
        }
        # 三级全部成功后落档，配合内容去重实现 ingest 幂等
        self.cloud_cache.save_ingest_result(entry_id, result)
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
        tks = top_k_sentences or self.config.top_k_key_sentences
        tkc = top_k_context or self.config.top_k_full_context
        thr = self.config.key_sentence_similarity_threshold

        # 嵌入服务可用性前置检查：查询向量是零向量时（Ollama 掉线），
        # IndexFlatIP 会按插入顺序返回任意记录伪装成命中，必须拦下
        if not NomicEmbedding.is_valid(self.embedding.embed(query)):
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

        # ── Step 1: 话题检测 (L3) ──────────────────────────
        domain = self.topic_manager.detect_domain(query)
        if domain is None:
            # 无匹配话题，降级为全局关键句检索
            logger.info("无匹配话题界域，降级为全局检索")
            result["key_sentences"] = self.key_sentence_lib.search(query, top_k=tks)
        else:
            result["detected_domain"] = {
                "domain_id": domain.domain_id,
                "name": domain.name,
                "description": domain.description,
                "entry_count": domain.entry_count,
            }

            # ── Step 2: 界域内关键句检索 (L2) ──────────────
            # M7 修复：先按分数阈值过滤，再判断是否补全局。
            # 旧顺序是"数量不足才补全局 → 最后统一过滤"：界域返回
            # 恰好 tks 条低分噪声时，既不触发补充、又被过滤清空，
            # 全局库里的高分答案被白白错过——明明聊过却检索不到。
            domain_ks = self._flatten_results(
                self.topic_manager.search_in_domain(query, domain, top_k=tks)
            )
            domain_ks = [k for k in domain_ks if k.get("score", 0.0) >= thr]

            # 界域内（过滤后的）结果不足，补充全局关键句检索（同样过滤）。
            # 注意：key_sentence_lib.search 返回的已是展平格式
            # （sentence_id/text/source_entry_id/score），不能再过
            # _flatten_results——v0.2 在这里多套了一层展平，把字段
            # 全部清空（text=""、sentence_id=""），补充的关键句进了
            # 结果列表却无法关联全文，等于白补（M7 实施时由回归测试暴露）。
            if len(domain_ks) < tks:
                global_ks = self.key_sentence_lib.search(query, top_k=tks)
                global_ks = [k for k in global_ks if k.get("score", 0.0) >= thr]
                # 去重补充（修复：sentence_id 缺失时不再把 "" / None 加进
                # 集合，避免后续无 ID 结果被误判为重复而丢弃）
                existing_ids = {k["sentence_id"] for k in domain_ks if k.get("sentence_id")}
                for ks in global_ks:
                    if ks.get("sentence_id") not in existing_ids:
                        domain_ks.append(ks)
                        if ks.get("sentence_id"):
                            existing_ids.add(ks["sentence_id"])
                    if len(domain_ks) >= tks:
                        break
            result["key_sentences"] = domain_ks

        # ── 分数门槛（统一兜底）─────────────────────────
        # key_sentence_similarity_threshold 此前定义了但从未使用：FAISS
        # 永远返回 top_k 个"最相似"，哪怕全是 0.1 分的噪声。低于阈值
        # 一律丢弃——宁可少返回，不返回假的。
        # 界域/全局补充路径已在入口处按阈值过滤（M7，先过滤后补全局）；
        # 此处统一兜底，覆盖"无匹配话题 → 全局检索"分支，并防御未来
        # 新增的检索入口漏配过滤。
        result["key_sentences"] = [
            k for k in result["key_sentences"] if k.get("score", 0.0) >= thr
        ]

        # ── Step 3: 关键句→全文关联 (L1) ──────────────────
        # 修复：此前 source_entry_id 收集进 seen_entries 后从未被使用，
        # 实际执行的是"拿关键句文本去全库重新向量搜索"，返回的全文
        # 可能来自任何一篇无关对话，分级隔离在最后一步被放大回全库。
        # 现在直接在关键句所属的全文条目内检索最相关段落，这才是
        # 设计文档里"根据关键句在缓存中查找详细对话"的本意。
        seen_entries = set()
        contexts = []
        for ks in result["key_sentences"]:
            source_id = ks.get("source_entry_id", "")
            if not source_id or source_id in seen_entries:
                continue
            seen_entries.add(source_id)
            contexts.extend(
                self.cloud_cache.search_in_entry(
                    ks.get("text", query), source_id, top_k=tkc
                )
            )

        # 去重并排序
        seen_chunks = set()
        unique_contexts = []
        for ctx in contexts:
            chunk_id = f"{ctx.get('entry_id')}_{ctx.get('chunk_index')}"
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_contexts.append(ctx)

        result["full_contexts"] = unique_contexts[:tkc * tks]

        # ── 生成摘要信息 ──────────────────────────────────
        result["summary"] = self._build_summary(result)

        return result

    # ── 上下文装配（代理模式核心原语） ─────────────────────

    def assemble_context(
        self,
        query: str,
        char_budget: Optional[int] = None,
    ) -> str:
        """
        检索并把结果装配成一段可直接放进 prompt 的紧凑记忆块。

        优先级：话题界域名 > 关键句（按分数降序） > 原文段落（按分数降序），
        超出字符预算时从最不重要的条目开始丢弃。这是"上下文扩展"的
        出口：无论库里有几百条对话，出去的这块永远有界。
        """
        budget = char_budget or self.config.context_char_budget
        res = self.retrieve(query)

        entries: list[tuple[int, float, str]] = []  # (优先级, 分数, 文本)
        if res.get("detected_domain"):
            d = res["detected_domain"]
            entries.append((0, 1.0, f"当前话题：{d['name']}"))
        for ks in res.get("key_sentences", []):
            entries.append((1, ks.get("score", 0.0), f"- {ks.get('text', '')}"))
        for ctx in res.get("full_contexts", []):
            body = ctx.get("chunk_text") or (ctx.get("full_text") or "")[:512]
            entries.append((2, ctx.get("score", 0.0), f"· 原文片段：{body}"))

        entries.sort(key=lambda x: (x[0], -x[1]))
        header = "以下是与当前问题相关的历史记忆（分级检索自动召回）："
        used = len(header)
        kept = []
        for _pri, _score, text in entries:
            if used + len(text) + 1 > budget:
                continue
            kept.append(text)
            used += len(text) + 1
        if not kept:
            return ""
        return header + "\n" + "\n".join(kept)

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
            "domains": [
                {
                    "id": d.domain_id,
                    "name": d.name,
                    "entries": d.entry_count,
                }
                for d in self.topic_manager.list_domains()
            ],
        }