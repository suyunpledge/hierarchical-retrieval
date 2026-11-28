# -*- coding: utf-8 -*-
"""
v0.2/v0.3 修复验证测试（不依赖 pytest / Ollama，直接 python 运行）

覆盖:
  1. 全文关联修复   —— full_contexts 必须来自关键句所属条目，而非全库
  2. 零向量污染修复 —— 嵌入服务挂掉时 ingest 抛错、retrieve 返回空
  3. 分数门槛生效   —— 低于阈值的关键句被丢弃
  4. 分块提取修复   —— 长对话关键句数 > 3
  5. 话题域上限生效 —— 达到 max 后不再新建
  6. ingest 幂等    —— 重复内容不膨胀库
  7. 装配预算       —— 记忆块长度受 char_budget 约束
  8. 历史压缩有界   —— 代理压缩输出与历史长度无关（需 fastapi）
  9. B1 补录窗口    —— 未入库∩最旧优先，多轮请求后历史全部入库（需 fastapi）
 10. M5.1 截断装配  —— 单条超长消息不再旁路压缩预算（需 fastapi）
 11. M7 先过滤后补 —— 界域低分噪声也触发全局补充，不漏高分答案
"""

import hashlib
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


def _find_package_root() -> str:
    """向上查找包含 hierarchical_retrieval 包的目录（兼容扁平/嵌套两种布局）"""
    for anc in Path(__file__).resolve().parents:
        if (anc / "hierarchical_retrieval" / "__init__.py").exists():
            return str(anc)
    raise RuntimeError("未找到 hierarchical_retrieval 包根目录")


sys.path.insert(0, _find_package_root())

import numpy as np

from hierarchical_retrieval import HConfig, HierarchicalRetrieval
from hierarchical_retrieval.core.embedding import NomicEmbedding


# ── 假嵌入：确定性、离线 ──────────────────────────────────────

class FakeEmbedding(NomicEmbedding):
    """同一文本永远得到同一向量，不同文本近似正交。不联网。"""

    def __init__(self, config=None):
        self.config = config or HConfig()
        self._model = self.config.embedding_model
        self._dim = self.config.embedding_dim

    def embed(self, text, strict=False):
        if not text or not text.strip():
            if strict:
                raise ValueError("empty text")
            return np.zeros(self._dim, dtype=np.float32)
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        # 标准正态：正负混合，无关文本余弦≈0。
        # （若用 rand() 会得到全正向量，任何两向量余弦都≈0.75，
        #  ，会把测试变成假阳性/假阴性的温床）
        vec = rng.standard_normal(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)


class DeadEmbedding(NomicEmbedding):
    """模拟 Ollama 掉线：strict 抛错，非 strict 返回零向量。"""

    def __init__(self, config=None):
        self.config = config or HConfig()
        self._model = self.config.embedding_model
        self._dim = self.config.embedding_dim

    def embed(self, text, strict=False):
        if strict:
            raise RuntimeError("Embedding 服务不可用: 模拟掉线")
        return np.zeros(self._dim, dtype=np.float32)


def make_hr(embedding_cls=FakeEmbedding, **cfg) -> HierarchicalRetrieval:
    root = tempfile.mkdtemp(prefix="hr_test_")
    config = HConfig(storage_root=root, **cfg)
    return HierarchicalRetrieval(embedding=embedding_cls(config), config=config)


CONV_A = (
    "今天讨论微服务架构方案。我们决定把用户、订单、支付三个模块拆分部署。"
    "数据库选型确定为 PostgreSQL 主库加 Redis 缓存。"
    "API Gateway 用 Kong，下周开始搭框架。"
    "最终结论是采用微服务方案，责任人是 Alice。"
)
CONV_B = (
    "关于季度预算，服务器成本每月五万元。"
    "人员成本大约三十万，需要重新规划。"
    "建议削减云服务预算百分之十五。"
    "结论是总预算控制在一百二十万以内。"
)


# ── 用例 ─────────────────────────────────────────────────────

def test_association():
    """修复 #1：full_contexts 必须来自关键句所属的条目"""
    hr = make_hr()
    ra = hr.ingest(CONV_A, {"src": "A"})
    rb = hr.ingest(CONV_B, {"src": "B"})
    assert ra["entry_id"] != rb["entry_id"]

    # 用 A 的一条关键句原文做查询（假嵌入下与自身完全匹配，得分 1.0）
    sid = ra["sentence_ids"][0]
    sent = hr.key_sentence_lib.get_sentence(sid)
    assert sent and sent["source_entry_id"] == ra["entry_id"]

    result = hr.retrieve(sent["text"])
    assert result["key_sentences"], "应至少命中查询的那条关键句"
    assert result["full_contexts"], "命中关键句后应关联出全文"
    bad = [c for c in result["full_contexts"] if c["entry_id"] != ra["entry_id"]]
    assert not bad, f"关联泄漏：返回了别的条目的全文 {bad}"
    # 旧实现在此处必然失败：全局搜索会混入 CONV_B 的块


def test_strict_ingest_blocks_zero_vector():
    """修复 #2：嵌入服务挂掉时 ingest 抛错且不污染索引"""
    hr = make_hr(DeadEmbedding)
    try:
        hr.ingest(CONV_A)
        raise AssertionError("DeadEmbedding 下 ingest 应当抛错")
    except RuntimeError:
        pass
    assert hr.cloud_cache.count() == 0, "失败写入不应留下全文条目"
    assert hr.key_sentence_lib.count() == 0, "失败写入不应留下关键句"
    assert hr.topic_manager.count() == 0, "失败写入不应留下话题域"


def test_dead_embedding_query_returns_empty():
    """修复 #2：查询嵌入失败时返回空结果而非按插入序的垃圾"""
    hr = make_hr(FakeEmbedding)
    hr.ingest(CONV_A)
    hr.ingest(CONV_B)
    # 换成坏嵌入再查询
    hr.embedding = DeadEmbedding(hr.config)
    hr.cloud_cache.embedding = hr.embedding
    hr.key_sentence_lib.embedding = hr.embedding
    hr.topic_manager.embedding = hr.embedding
    result = hr.retrieve("架构方案是什么")
    assert result["key_sentences"] == [], "嵌入不可用时不应返回任何'命中'"
    assert "不可用" in result["summary"]


def test_noise_filtered_by_threshold():
    """修复 #3：与库内内容无关的查询得分≈0，应被阈值过滤"""
    hr = make_hr()
    hr.ingest(CONV_A)
    hr.ingest(CONV_B)
    result = hr.retrieve("量子纠缠与波函数坍缩的哲学含义")
    assert result["key_sentences"] == [], "噪声查询不应返回伪命中"


def test_per_chunk_extraction():
    """修复 #6：长对话提取的关键句应多于 3 条（原实现全文只提 3 句）"""
    hr = make_hr()
    long_text = "".join(
        f"第{i}个要点是关于模块{ i % 7 }的说明，涉及参数配置与阈值设定。" for i in range(40)
    )
    r = hr.ingest(long_text)
    assert len(r["sentence_ids"]) > 3, f"长对话应提取 >3 条关键句，实际 {len(r['sentence_ids'])}"


def test_domain_cap():
    """修复 #4：话题域达到上限后强制归入，不再新建"""
    hr = make_hr(topic_max_domains=2)
    for i in range(5):
        hr.ingest(f"话题{i}的独立内容：编号{i}的方案讨论与结论。" * 3)
    assert hr.topic_manager.count() <= 2, "话题域数量不应超过上限"


def test_ingest_idempotent():
    """新增：重复内容 ingest 幂等，库不膨胀"""
    hr = make_hr()
    r1 = hr.ingest(CONV_A)
    n_ks = hr.key_sentence_lib.count()
    r2 = hr.ingest(CONV_A)
    assert r2["entry_id"] == r1["entry_id"], "重复内容应复用条目"
    assert hr.key_sentence_lib.count() == n_ks, "重复 ingest 不应新增关键句"
    assert hr.cloud_cache.count() == 1


def test_assemble_context_budget():
    """新增：装配出的记忆块长度受预算约束"""
    hr = make_hr()
    r = hr.ingest(CONV_A)
    hr.ingest(CONV_B)
    # 假嵌入只有精确文本才能高分命中，用关键句原文做查询
    sent = hr.key_sentence_lib.get_sentence(r["sentence_ids"][0])["text"]
    small = hr.assemble_context(sent, char_budget=200)
    if small:
        assert len(small) <= 200, f"记忆块应 ≤200 字符，实际 {len(small)}"
    big = hr.assemble_context(sent, char_budget=5000)
    assert big, "命中查询应装配出记忆块"
    assert len(big) >= len(small)


def test_compress_history_bounded():
    """新增（需 fastapi）：代理压缩输出与历史长度无关"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # api_server.py 所在目录
        import api_server
    except ImportError as e:
        print(f"    [跳过] 无法导入 api_server: {e}")
        return

    tmp = Path(tempfile.mkdtemp(prefix="hr_proxy_"))
    old_cwd = Path.cwd()
    import os
    os.chdir(tmp)
    os.environ["HR_STORAGE_ROOT"] = str(tmp / "storage")
    try:
        # 重新绑定测试引擎与测试配置（避免碰真实存储与真实 Ollama）
        cfg = HConfig(
            storage_root=str(tmp / "storage"),
            recent_turns_keep=6,
            compress_threshold_chars=3000,
            context_char_budget=2000,
        )
        api_server.config = cfg
        api_server.hr = make_hr(FakeEmbedding,
                                recent_turns_keep=6,
                                compress_threshold_chars=3000,
                                context_char_budget=2000)

        from api_server import ChatMessage, _compress_history
        msgs = [ChatMessage(role="system", content="你是一个助手。")]
        for i in range(40):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(ChatMessage(role=role, content=f"第{i}轮对话，" + "内容填充" * 40))

        total_in = sum(len(m.content) for m in msgs)
        out = _compress_history(msgs)
        total_out = sum(len(m.content) for m in out)
        assert len(out) <= 8, f"压缩后条数应 ≤8（system+记忆+近6轮），实际 {len(out)}"
        assert total_out <= 2 * 2000 + 400, f"压缩后字符应受预算约束，实际 {total_out}"
        assert out[0].role == "system", "头部 system 提示词应保留"
        print(f"    压缩: {len(msgs)} 条/{total_in} 字 → {len(out)} 条/{total_out} 字")
    finally:
        os.chdir(old_cwd)


def test_backfill_oldest_first_and_complete():
    """B1 修复：补录窗口 = 未入库 ∩ 最旧优先，多轮请求后历史全部入库。

    旧实现取 older 段 [-cap:]（最靠后的一批），窗口随对话增长前移，
    60 条历史、cap=32 时最早的消息永久丢失（与"留到下一轮继续补"
    的注释相反）。
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import api_server
    except ImportError as e:
        print(f"    [跳过] 无法导入 api_server: {e}")
        return

    tmp = Path(tempfile.mkdtemp(prefix="hr_proxy_"))
    old_cwd = Path.cwd()
    import os
    os.chdir(tmp)
    os.environ["HR_STORAGE_ROOT"] = str(tmp / "storage")
    try:
        cfg = HConfig(
            storage_root=str(tmp / "storage"),
            recent_turns_keep=6,
            compress_threshold_chars=3000,
            context_char_budget=2000,
            max_ingest_per_request=16,
        )
        api_server.config = cfg
        api_server.hr = make_hr(
            FakeEmbedding,
            recent_turns_keep=6,
            compress_threshold_chars=3000,
            context_char_budget=2000,
            max_ingest_per_request=16,
        )
        hr = api_server.hr
        from api_server import ChatMessage, _compress_history

        # 1 条 system + 59 条对话（每条 56 字，总长 > 3000 触发压缩）；
        # keep=6 → older 53 条 = 26 轮完整对话 + 1 条回复还在近期窗口里的
        # 用户消息（该轮暂不入库）；入库单位是"轮"，cap=16
        msgs = [ChatMessage(role="system", content="你是一个助手。")]
        for i in range(59):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(ChatMessage(role=role, content=f"第{i}轮对话内容，" + "细节填充" * 12))
        older = msgs[1:-6]
        assert len(older) == 53
        assert sum(len(m.content) for m in msgs) > 3000  # 确保触发压缩

        from api_server import _turn_texts
        turns = _turn_texts(older, continues=True)
        assert len(turns) == 26
        assert turns[0] == f"用户：{older[0].content}\n助手：{older[1].content}"

        _compress_history(msgs)
        n1 = hr.cloud_cache.count()
        assert n1 == 16, f"首轮应补录最旧的 16 轮，实际 {n1}"
        # 最旧的一轮必须首轮入库（旧实现会永久丢掉它）
        assert hr.cloud_cache.has_content(turns[0]), "最旧一轮首轮必须入库"
        # 窗口外的最新一轮还不该入库
        assert not hr.cloud_cache.has_content(turns[-1])

        _compress_history(msgs)
        n2 = hr.cloud_cache.count()
        assert n2 == 26, f"两轮后完整的旧对话应全部入库（已入库不占窗口名额），实际 {n2}"
        assert hr.cloud_cache.has_content(turns[-1]), "第二轮应补完剩余 10 轮"

        _compress_history(msgs)
        assert hr.cloud_cache.count() == 26, "重复请求不应再入库"
        # 回复还留在近期窗口的那一轮，等对话继续、它完整滑出后才入库
        msgs += [ChatMessage(role="assistant", content="收到。" * 10),
                 ChatMessage(role="user", content="继续。" * 10)]
        _compress_history(msgs)
        assert hr.cloud_cache.count() == 27
        print(f"    补录: 26 轮旧对话 → 两轮请求后全部入库 ({n1} → {n2})")
    finally:
        os.chdir(old_cwd)


def test_single_oversized_message_truncated():
    """M5.1 修复：单条超长消息截断装配，输出不再被一条巨无霸旁路。

    旧逻辑对最新一条消息无条件保留：用户粘贴 5 万字文档时，
    压缩后的输出仍带完整 5 万字，"结构性有界"失效。
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import api_server
    except ImportError as e:
        print(f"    [跳过] 无法导入 api_server: {e}")
        return

    tmp = Path(tempfile.mkdtemp(prefix="hr_proxy_"))
    old_cwd = Path.cwd()
    import os
    os.chdir(tmp)
    os.environ["HR_STORAGE_ROOT"] = str(tmp / "storage")
    try:
        cfg = HConfig(
            storage_root=str(tmp / "storage"),
            recent_turns_keep=6,
            compress_threshold_chars=3000,
            context_char_budget=2000,
        )
        api_server.config = cfg
        api_server.hr = make_hr(
            FakeEmbedding,
            recent_turns_keep=6,
            compress_threshold_chars=3000,
            context_char_budget=2000,
        )
        from api_server import ChatMessage, TRUNC_MARK, _compress_history

        # 8 条消息：1 system + 6 常规 + 1 条 5 万字的最新提问
        msgs = [ChatMessage(role="system", content="你是一个助手。")]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(ChatMessage(role=role, content=f"第{i}轮常规对话，" + "细节填充" * 8))
        msgs.append(
            ChatMessage(role="user", content="请分析这篇超长文档：" + "填充内容" * 12500)
        )
        assert sum(len(m.content) for m in msgs) > 3000  # 触发压缩
        assert len(msgs) > 6  # 越过 len≤keep 旁路

        out = _compress_history(msgs)
        total_out = sum(len(m.content) for m in out)
        # 预算 2000 + 头部/标记的少量宽放；旧实现此处是 ~5 万
        assert total_out <= 2200, f"单条超长消息应被截断，输出仍有界，实际 {total_out}"
        assert any(TRUNC_MARK in m.content for m in out), "截断后应带截断标记"
        assert any(
            m.content.startswith("请分析这篇超长文档：") for m in out
        ), "截断应保留消息开头"
        print(f"    压缩: 5 万字单条消息 → 输出总长 {total_out} 字")
    finally:
        os.chdir(old_cwd)


def test_m7_domain_noise_triggers_global_supplement():
    """M7 修复：界域内"数量够但质量不够"（低分噪声）也触发全局补充。

    旧顺序（先按数量补全局 → 最后统一过滤）：界域返回恰好 tks 条
    低于阈值的噪声时既不补充、又被过滤清空——明明聊过却检索不到。
    """
    hr = make_hr()
    ra = hr.ingest(CONV_A)
    rb = hr.ingest(CONV_B)
    tks = hr.config.top_k_key_sentences
    # 全局库里的"高分答案"：B 的第一条关键句（假嵌入下与自身余弦=1）
    gold = hr.key_sentence_lib.get_sentence(rb["sentence_ids"][0])["text"]
    domain_a = hr.topic_manager.get_domain(ra["domain_id"])
    assert domain_a is not None

    # 场景注入：话题检测误命中 A 域（模拟跨域/误并），
    # 界域内只返回恰好 tks 条低于阈值的关键句噪声
    hr.topic_manager.detect_domain_by_vector = lambda query_vector: domain_a
    hr.topic_manager.search_in_domain_by_vector = lambda query_vector, domain, top_k=5: [
        {
            "id": f"ks_noise_{i}",
            "score": 0.10,
            "metadata": {
                "sentence_id": f"ks_noise_{i}",
                "text": f"界域内低分噪声句{i}",
                "source_entry_id": ra["entry_id"],
                "timestamp": 0,
            },
        }
        for i in range(tks)
    ]

    result = hr.retrieve(gold)
    texts = [k.get("text", "") for k in result["key_sentences"]]
    assert gold in texts, "界域噪声应触发全局补充，找回高分关键句"
    assert all(
        k.get("score", 0.0) >= hr.config.key_sentence_similarity_threshold
        for k in result["key_sentences"]
    ), "过滤后不应残留低于阈值的结果"
    assert all("噪声" not in t for t in texts), "界域噪声不应出现在最终结果里"
    print(f"    界域噪声({tks} 条) → 全局补充找回高分关键句 ✓")


def test_failed_ingest_rolls_back_all_layers():
    """写入流水线失败时不应留下孤儿全文、关键句或话题域。"""
    hr = make_hr()
    original = hr.topic_manager.add_key_sentences_batch

    def fail(*args, **kwargs):
        raise RuntimeError("模拟 L3 索引失败")

    hr.topic_manager.add_key_sentences_batch = fail
    try:
        try:
            hr.ingest(CONV_A)
            raise AssertionError("模拟失败应向上传播")
        except RuntimeError:
            pass
    finally:
        hr.topic_manager.add_key_sentences_batch = original
    assert hr.cloud_cache.count() == 0
    assert hr.key_sentence_lib.count() == 0
    assert hr.topic_manager.count() == 0


def test_vector_store_rejects_duplicate_ids_and_bad_dimensions():
    """重复外部 ID 不得留下幽灵向量，维度错误应尽早失败。"""
    from hierarchical_retrieval.storage.vector_store import VectorStore

    root = tempfile.mkdtemp(prefix="hr_vs_")
    vs = VectorStore(root, dimension=4)
    vec = np.array([1, 0, 0, 0], dtype=np.float32)
    vs.add_many([("same", vec, {"value": 1})])
    vs.add_many([("same", vec, {"value": 2})])
    assert vs.count() == 1
    assert len(vs.search(vec, top_k=10)) == 1
    try:
        vs.add("bad", np.ones(3, dtype=np.float32))
        raise AssertionError("维度错误应抛出 ValueError")
    except ValueError:
        pass


def test_retrieve_reuses_query_embedding():
    """L3/L2 及全局补充必须共享一次查询嵌入。"""
    class CountingEmbedding(FakeEmbedding):
        def __init__(self, config=None):
            super().__init__(config)
            self.calls = 0

        def embed(self, text, strict=False):
            self.calls += 1
            return super().embed(text, strict=strict)

    hr = make_hr(CountingEmbedding)
    result = hr.ingest(CONV_A)
    sentence = hr.key_sentence_lib.get_sentence(result["sentence_ids"][0])["text"]
    hr.embedding.calls = 0
    hr.retrieve(sentence, top_k_context=0)
    assert hr.embedding.calls == 1, f"查询嵌入应只调用 1 次，实际 {hr.embedding.calls}"


def test_api_singleton_is_thread_safe():
    """并发首次访问只能创建一个检索引擎。"""
    from concurrent.futures import ThreadPoolExecutor
    import time
    import api_server

    old_hr = api_server.hr
    old_factory = api_server.HierarchicalRetrieval
    created = []
    try:
        def factory(config):
            created.append(config)
            time.sleep(0.02)
            return object()

        api_server.hr = None
        api_server.HierarchicalRetrieval = factory
        with ThreadPoolExecutor(max_workers=12) as pool:
            engines = list(pool.map(lambda _i: api_server.get_hr(), range(24)))
        assert len(created) == 1
        assert len({id(engine) for engine in engines}) == 1
    finally:
        api_server.hr = old_hr
        api_server.HierarchicalRetrieval = old_factory


def test_vector_store_concurrent_duplicate_add():
    """并发批量添加同一外部 ID 时只能产生一个向量。"""
    from concurrent.futures import ThreadPoolExecutor
    from hierarchical_retrieval.storage.vector_store import VectorStore

    root = tempfile.mkdtemp(prefix="hr_test_")
    vs = VectorStore(root, dimension=4)
    vec = np.array([1, 0, 0, 0], dtype=np.float32)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _i: vs.add_many([("same", vec, {"value": 1})]), range(16)))
    assert vs.count() == 1
    assert vs.index.ntotal == 1
    assert len(vs.search(vec, top_k=10)) == 1


def test_vector_store_faiss_failure_leaves_no_mappings():
    """FAISS 添加失败时不得提前提交 ID 和元数据映射。"""
    from hierarchical_retrieval.storage.vector_store import VectorStore

    class FailingIndex:
        ntotal = 0

        def add(self, _vectors):
            raise RuntimeError("模拟 FAISS 写入失败")

    vs = VectorStore(tempfile.mkdtemp(prefix="hr_test_"), dimension=4)
    vs._index = FailingIndex()
    try:
        vs.add_many([("bad", np.ones(4, dtype=np.float32), {"value": 1})])
        raise AssertionError("FAISS 失败应向上传播")
    except RuntimeError:
        pass
    assert vs.count() == 0
    assert not vs._reverse_map and not vs._metadata and vs._next_id == 0


def test_vector_store_rejects_corrupt_existing_index():
    """已有索引损坏时必须失败关闭，不能退回空索引。"""
    from hierarchical_retrieval.storage.vector_store import VectorStore

    root = Path(tempfile.mkdtemp(prefix="hr_test_"))
    (root / "vector_index.faiss").write_text("broken", encoding="utf-8")
    (root / "metadata.json").write_text("{}", encoding="utf-8")
    try:
        VectorStore(str(root), dimension=4)
        raise AssertionError("损坏索引应抛出 RuntimeError")
    except RuntimeError as e:
        assert "拒绝使用空索引" in str(e)


def test_vector_store_single_add_is_durable():
    """单条 StorageBase 写入无需外部 flush 也应可重载。"""
    from hierarchical_retrieval.storage.vector_store import VectorStore

    root = tempfile.mkdtemp(prefix="hr_test_")
    vec = np.array([1, 0, 0, 0], dtype=np.float32)
    VectorStore(root, dimension=4).add("one", vec, {"value": 1})
    reloaded = VectorStore(root, dimension=4)
    assert reloaded.count() == 1
    assert reloaded.get_metadata("one") == {"value": 1}


def test_split_chunks_enforces_limit_for_unpunctuated_text():
    """无标点长文也必须按上限切块，且不丢字符。"""
    from hierarchical_retrieval.utils.text_processor import TextProcessor

    text = "abcdefghij" * 200
    chunks = TextProcessor.split_chunks(text, max_chars=128, overlap=16)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 128 for chunk in chunks)
    assert "".join(chunks) == text


def test_zero_key_sentence_limit_and_api_flush():
    """零关键句配置应生效，API 写入后应调用统一刷盘。"""
    from hierarchical_retrieval.utils.summarizer import KeySentenceExtractor
    import api_server

    assert KeySentenceExtractor(HConfig()).extract("第一句。第二句。", max_sentences=0) == []
    old_hr = api_server.hr
    try:
        class Engine:
            flushed = False

            def ingest(self, text, metadata=None):
                return {
                    "entry_id": "cc_test",
                    "domain_id": "td_test",
                    "domain_name": "测试",
                    "sentence_ids": [],
                }

            def flush(self):
                self.flushed = True

        engine = Engine()
        api_server.hr = engine
        response = api_server.ingest_dialogue(api_server.IngestRequest(text="路由测试"))
        assert response.entry_id == "cc_test" and engine.flushed
    finally:
        api_server.hr = old_hr


# ── v0.6 移植回归用例 ────────────────────────────────────────

def test_vector_search_skips_tombstones():
    """逻辑删除的向量不应吞掉 top_k 名额；delete_many 一次删多条。"""
    from hierarchical_retrieval.storage.vector_store import VectorStore

    vs = VectorStore(tempfile.mkdtemp(prefix="hr_test_"), dimension=4)
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    items = []
    for i in range(6):
        v = np.array([1.0, 0.1 * i, 0, 0], dtype=np.float32)
        items.append((f"v{i}", v / np.linalg.norm(v), {}))
    vs.add_many(items)
    assert vs.delete_many(["v0", "v1", "v2", "missing"]) == 3
    hits = vs.search(q, top_k=2)
    assert [h["id"] for h in hits] == ["v3", "v4"], hits


def test_registry_dimension_mismatch_tolerated():
    """换嵌入模型后旧话题质心维度不符，不应让 ingest/检索抛错。"""
    from hierarchical_retrieval.core.topic_domain import TopicDomainManager

    hr = make_hr(embedding_dim=768)
    hr.ingest(CONV_A)
    cfg = HConfig(storage_root=hr.config.storage_root, embedding_dim=1024)
    tm = TopicDomainManager(FakeEmbedding(cfg), cfg)
    assert tm.count() == 1
    assert tm.detect_domain("微服务") is None
    tm.assign_domain(CONV_B)  # 旧实现在 np.dot 处抛 ValueError
    assert tm.count() == 2


def test_chat_message_accepts_content_parts():
    """OpenAI 数组形式的 content（LobeChat 带附件时）不再 422。"""
    from api_server import ChatMessage

    m = ChatMessage(role="user", content=[
        {"type": "text", "text": "你好"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
        {"type": "text", "text": "看图"},
    ])
    assert m.content == "你好\n看图"
    assert ChatMessage(role="assistant", content=None).content == ""


def test_stream_proxy_errors_and_finish_reason():
    """流式代理：两种后端协议都发 finish_reason 收尾块，中途错误被透传。"""
    import asyncio
    import json as _json
    import api_server

    class FakeResp:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            return iter(self.lines)

    def run(openai_backend, lines):
        api_server.requests.post = lambda *a, **k: FakeResp(lines)
        api_server._chat_backend_is_openai = lambda: openai_backend
        req = api_server.ChatCompletionRequest(
            model="m", stream=True,
            messages=[api_server.ChatMessage(role="user", content="hi")],
        )
        resp = api_server.chat_completions(req)

        async def collect():
            return [c if isinstance(c, str) else c.decode() async for c in resp.body_iterator]

        events = [e for e in "".join(asyncio.run(collect())).split("\n\n") if e]
        assert events[-1] == "data: [DONE]"
        return [_json.loads(e[len("data: "):]) for e in events[:-1]]

    old_post, old_probe = api_server.requests.post, api_server._chat_backend_is_openai
    try:
        ollama_ok = run(False, [
            _json.dumps({"message": {"content": "你"}}),
            _json.dumps({"message": {"content": "好"}, "done": True, "done_reason": "length"}),
        ])
        assert [e["choices"][0]["delta"].get("content") for e in ollama_ok[:2]] == ["你", "好"]
        assert ollama_ok[-1]["choices"][0]["finish_reason"] == "length"

        openai_ok = run(True, [
            "data: " + _json.dumps({"choices": [{"delta": {"content": "嗨"}, "finish_reason": None}]}),
            "data: " + _json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ])
        assert openai_ok[-1]["choices"][0]["finish_reason"] == "stop"

        ollama_err = run(False, [
            _json.dumps({"message": {"content": "你"}}),
            _json.dumps({"error": "model unloaded"}),
        ])
        assert ollama_err[-1] == {"error": {"message": "model unloaded", "type": "backend_error"}}

        openai_err = run(True, ["data: " + _json.dumps({"error": {"message": "boom"}})])
        assert openai_err[-1]["error"]["message"] == "boom"
    finally:
        api_server.requests.post, api_server._chat_backend_is_openai = old_post, old_probe


# ── 缓存命中用例 ─────────────────────────────────────────────

class CountingEmbedding(NomicEmbedding):
    """走真实 NomicEmbedding.embed()（含缓存），只替换网络请求并计数。"""

    def __init__(self, config=None, fail=False):
        self.config = config or HConfig()
        self._model = self.config.embedding_model
        self._dim = self.config.embedding_dim
        self._backend = "openai"
        self.requests = []
        self.fail = fail

    def _embed_uncached(self, text, strict):
        self.requests.append(text)
        if self.fail:
            if strict:
                raise RuntimeError("模拟掉线")
            return np.zeros(self._dim, dtype=np.float32)
        return FakeEmbedding(self.config).embed(text)


def make_counting_hr(**cfg):
    root = tempfile.mkdtemp(prefix="hr_test_")
    config = HConfig(storage_root=root, **cfg)
    emb = CountingEmbedding(config)
    return HierarchicalRetrieval(embedding=emb, config=config), emb


def test_embedding_cache_hits_and_safety():
    """同文本只请求一次；失败不缓存；缓存向量只读；LRU 淘汰；0 = 关闭。"""
    emb = CountingEmbedding(HConfig(embedding_cache_size=2))
    a1, a2 = emb.embed("甲"), emb.embed("甲")
    assert emb.requests == ["甲"] and a1 is a2
    try:
        a1[0] = 9.0
        raise AssertionError("缓存向量应为只读")
    except ValueError:
        pass
    emb.embed("乙"), emb.embed("丙")          # 容量 2：最久未用的"甲"被淘汰
    emb.embed("甲")
    assert emb.requests.count("甲") == 2
    s = emb.cache_stats()
    assert s["size"] == 2 and s["hits"] == 1 and s["misses"] == 4

    dead = CountingEmbedding(HConfig(), fail=True)
    dead.embed("丁"), dead.embed("丁")
    assert dead.requests == ["丁", "丁"], "失败降级的零向量不能被缓存"

    off = CountingEmbedding(HConfig(embedding_cache_size=0))
    off.embed("戊"), off.embed("戊")
    assert len(off.requests) == 2

    batch = CountingEmbedding(HConfig(embedding_cache_size=0))
    out = batch.embed_batch(["x", "y", "x", "x"])
    assert sorted(batch.requests) == ["x", "y"], "批内重复文本只请求一次"
    assert np.array_equal(out[0], out[2])


def test_ingest_reuses_embeddings_of_same_text():
    """短消息 ingest：chunk / 话题分配 / 关键句是同一文本，只请求一次嵌入。"""
    hr, emb = make_counting_hr()
    hr.ingest("数据库选型确定为 PostgreSQL。")
    assert len(emb.requests) == len(set(emb.requests)), emb.requests
    assert emb.requests == ["数据库选型确定为 PostgreSQL。"]


def test_retrieval_cache_hit_invalidate_and_isolation():
    """无关写入保留缓存，相关写入失效，调用方修改返回值与缓存隔离。"""
    hr, emb = make_counting_hr()
    ra = hr.ingest(CONV_A)
    query = hr.key_sentence_lib.get_sentence(ra["sentence_ids"][0])["text"]

    first = hr.retrieve(query)
    n = len(emb.requests)
    second = hr.retrieve("  " + query + "\n")   # 首尾空白归一后命中
    assert len(emb.requests) == n, "命中检索缓存时不应再请求嵌入"
    assert second["query"] == "  " + query + "\n"
    second["query"] = query
    assert second == first
    second["key_sentences"].clear()
    assert hr.retrieve(query)["key_sentences"], "调用方修改返回值不应污染缓存"
    assert hr.cache_stats()["retrieval"]["hits"] == 2

    hr.ingest(CONV_B)
    hr.ingest(CONV_B)                          # 重复内容：无写入，不失效
    gen = hr._generation
    assert hr.retrieve(query) == first
    assert hr.cache_stats()["retrieval"]["misses"] == 1, "无关追加后应复用结果"
    hr.ingest(CONV_B)
    assert hr._generation == gen, "重复 ingest 不应使缓存失效"
    update = hr.ingest(query)
    refreshed = hr.retrieve(query)
    assert hr.cache_stats()["retrieval"]["misses"] == 2, "相关追加后应重新检索"
    assert update["entry_id"] in {item["source_entry_id"] for item in refreshed["key_sentences"]}


def test_retrieval_cache_skips_degraded_results():
    """嵌入服务不可用时的空结果不进缓存，服务恢复后能正常检索。"""
    hr, emb = make_counting_hr()
    ra = hr.ingest(CONV_A)
    query = hr.key_sentence_lib.get_sentence(ra["sentence_ids"][0])["text"]
    hr.embedding.clear_cache()
    emb.fail = True
    assert hr.retrieve(query)["key_sentences"] == []
    emb.fail = False
    assert hr.retrieve(query)["key_sentences"], "服务恢复后不应返回缓存的空结果"


def test_concurrent_write_does_not_cache_stale_result():
    """检索期间发生写入时，该次结果不回填缓存。"""
    hr, emb = make_counting_hr()
    hr.ingest(CONV_A)
    original = hr._retrieve_uncached

    def racing(query, tks, tkc):
        out = original(query, tks, tkc)
        hr.ingest(CONV_B)                       # 模拟检索中途完成一次写入
        return out

    hr._retrieve_uncached = racing
    hr.retrieve("预算")
    hr._retrieve_uncached = original
    assert hr.cache_stats()["retrieval"]["size"] == 0


# ── 全文段落直检（召回）用例 ─────────────────────────────────

class CharGramEmbedding(FakeEmbedding):
    """字二元组哈希向量：字面越重合余弦越高，可离线模拟"句子 ⊂ 段落"的相似。"""

    def embed(self, text, strict=False):
        if not text or not text.strip():
            if strict:
                raise ValueError("empty text")
            return np.zeros(self._dim, dtype=np.float32)
        vec = np.zeros(self._dim, dtype=np.float32)
        for i in range(len(text) - 1):
            h = int(hashlib.md5(text[i:i + 2].encode("utf-8")).hexdigest()[:8], 16)
            vec[h % self._dim] += 1.0
        return vec / (np.linalg.norm(vec) + 1e-10)


FACT_CONV = (
    "今天和房东谈了续租。"
    "楼下的停车位以后要单独收费每月一百五十元。"
    "热水器下个月换新的。"
    "总的来说这次沟通还算顺利。"
)


def test_lexical_overlap_terms():
    """疑问/功能词被剔除；ASCII 术语整体作为检索词。"""
    from hierarchical_retrieval.utils.text_processor import TextProcessor as TP

    assert TP.query_terms("什么时候") == set()
    assert "bge-m3" in TP.query_terms("bge-m3 的向量维度是多少")
    # 检索词 楼下/下车/车位/位收/收费，原文命中 楼下/车位/收费 → 3/5
    assert abs(TP.lexical_overlap("楼下车位收费吗", FACT_CONV) - 0.6) < 1e-9
    assert TP.lexical_overlap("如何给植物施肥", FACT_CONV) == 0.0


def test_direct_chunk_search_recovers_non_key_sentence_fact():
    """事实不在关键句里时，旧路径检索不到，全文段落直检能找回。"""
    cfg = dict(max_key_sentences_per_chunk=1, key_sentence_similarity_threshold=0.9,
               chunk_similarity_threshold=0.5)
    hr = make_hr(CharGramEmbedding, **cfg)
    entry = hr.ingest(FACT_CONV)["entry_id"]
    key_texts = [hr.key_sentence_lib.get_sentence(s)["text"] for s in hr.key_sentence_lib.list_sentences()]
    assert all("停车位" not in t for t in key_texts), "前提：目标事实不是关键句"
    query = "楼下的停车位以后要单独收费吗"

    hr.config.direct_chunk_search = False
    hr._invalidate_retrieval_cache()
    assert not hr.retrieve(query)["full_contexts"], "旧路径下应检索不到"

    hr.config.direct_chunk_search = True
    hr._invalidate_retrieval_cache()
    ctxs = hr.retrieve(query)["full_contexts"]
    assert [c["entry_id"] for c in ctxs] == [entry]
    assert ctxs[0]["via"] == "direct" and ctxs[0]["full_text"] == FACT_CONV
    assert ctxs[0]["hybrid_score"] >= 0.5

    assert hr.retrieve("量子计算机的工作原理")["full_contexts"] == [], "无关问题不应命中"


def test_direct_chunk_hits_respect_threshold_and_limit():
    """混合分低于阈值的段落被丢弃，返回条数不超过 top_k_context。"""
    hr = make_hr(CharGramEmbedding, chunk_similarity_threshold=0.99)
    hr.ingest(FACT_CONV)
    assert hr._direct_chunk_hits("停车位收费", hr.embedding.embed("停车位收费"), 3) == []
    hr.config.chunk_similarity_threshold = 0.0
    for i in range(5):
        hr.ingest(f"第{i}段：停车位收费相关的补充说明。")
    hits = hr._direct_chunk_hits("停车位收费", hr.embedding.embed("停车位收费"), 2)
    assert len(hits) == 2 and hits[0]["hybrid_score"] >= hits[1]["hybrid_score"]


def test_embedding_cache_coalesces_concurrent_misses():
    """并发相同文本只产生一次嵌入请求，等待者复用结果。"""
    from concurrent.futures import ThreadPoolExecutor
    import time

    emb = CountingEmbedding(HConfig(embedding_cache_size=8))
    original = emb._embed_uncached

    def slow_embed(text, strict):
        time.sleep(0.05)
        return original(text, strict)

    emb._embed_uncached = slow_embed
    with ThreadPoolExecutor(max_workers=8) as pool:
        vectors = list(pool.map(lambda _i: emb.embed("并发缓存"), range(16)))
    assert emb.requests == ["并发缓存"]
    assert all(vector is vectors[0] for vector in vectors)
    stats = emb.cache_stats()
    assert stats["misses"] == 1 and stats["hits"] == 15


def test_retrieval_cache_coalesces_concurrent_misses():
    """并发相同查询只执行一次完整检索，其余请求命中同一结果。"""
    from concurrent.futures import ThreadPoolExecutor
    import time

    hr, _emb = make_counting_hr()
    ra = hr.ingest(CONV_A)
    query = hr.key_sentence_lib.get_sentence(ra["sentence_ids"][0])["text"]
    original = hr._retrieve_uncached
    calls = []

    def slow_retrieve(text, tks, tkc):
        calls.append(text)
        time.sleep(0.05)
        return original(text, tks, tkc)

    hr._retrieve_uncached = slow_retrieve
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: hr.retrieve(query), range(12)))
    assert calls == [query]
    assert all(result == results[0] for result in results)
    stats = hr.cache_stats()["retrieval"]
    assert stats["misses"] == 1 and stats["hits"] == 11


def test_fulltext_cache_hits_update_and_isolation():
    """全文记录只读一次盘，写回/删除即时生效，返回对象与缓存隔离。"""
    hr, _emb = make_counting_hr(fulltext_cache_size=2)
    entry_id = hr.ingest(CONV_A)["entry_id"]
    cache = hr.cloud_cache
    cache.clear_cache()
    baseline = cache.cache_stats()
    original_load = cache._text_store.load
    disk_reads = []

    def counting_load(key):
        disk_reads.append(key)
        return original_load(key)

    cache._text_store.load = counting_load
    assert cache.get_full_text(entry_id) == CONV_A
    assert cache.get_metadata(entry_id)["total_length"] == len(CONV_A)
    first = cache.get_ingest_result(entry_id)
    assert disk_reads == [entry_id]
    stats = cache.cache_stats()
    assert stats["misses"] - baseline["misses"] == 1
    assert stats["hits"] - baseline["hits"] == 2
    assert stats["size"] == 1

    first["domain_name"] = "调用方误改"
    cached = cache.get_ingest_result(entry_id)
    assert cached["domain_name"] != "调用方误改"
    cached["domain_name"] = "写回更新"
    cache.save_ingest_result(entry_id, cached)
    assert cache.get_ingest_result(entry_id)["domain_name"] == "写回更新"
    assert disk_reads == [entry_id]

    assert cache.delete_entry(entry_id)
    assert cache.get_full_text(entry_id) is None
    assert entry_id not in cache._record_cache


TESTS = [
    test_association,
    test_strict_ingest_blocks_zero_vector,
    test_dead_embedding_query_returns_empty,
    test_noise_filtered_by_threshold,
    test_per_chunk_extraction,
    test_domain_cap,
    test_ingest_idempotent,
    test_assemble_context_budget,
    test_compress_history_bounded,
    test_backfill_oldest_first_and_complete,
    test_single_oversized_message_truncated,
    test_m7_domain_noise_triggers_global_supplement,
    test_failed_ingest_rolls_back_all_layers,
    test_vector_store_rejects_duplicate_ids_and_bad_dimensions,
    test_retrieve_reuses_query_embedding,
    test_api_singleton_is_thread_safe,
    test_vector_store_concurrent_duplicate_add,
    test_vector_store_faiss_failure_leaves_no_mappings,
    test_vector_store_rejects_corrupt_existing_index,
    test_vector_store_single_add_is_durable,
    test_split_chunks_enforces_limit_for_unpunctuated_text,
    test_zero_key_sentence_limit_and_api_flush,
    test_vector_search_skips_tombstones,
    test_registry_dimension_mismatch_tolerated,
    test_chat_message_accepts_content_parts,
    test_stream_proxy_errors_and_finish_reason,
    test_embedding_cache_hits_and_safety,
    test_ingest_reuses_embeddings_of_same_text,
    test_retrieval_cache_hit_invalidate_and_isolation,
    test_retrieval_cache_skips_degraded_results,
    test_concurrent_write_does_not_cache_stale_result,
    test_lexical_overlap_terms,
    test_direct_chunk_search_recovers_non_key_sentence_fact,
    test_direct_chunk_hits_respect_threshold_and_limit,
    test_embedding_cache_coalesces_concurrent_misses,
    test_retrieval_cache_coalesces_concurrent_misses,
    test_fulltext_cache_hits_update_and_isolation,
]


def main():
    passed = failed = 0
    for t in TESTS:
        name = t.__name__
        try:
            t()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception:
            print(f"  [FAIL] {name}")
            traceback.print_exc()
            failed += 1
        finally:
            # 清理测试存储
            for p in Path(tempfile.gettempdir()).glob("hr_test_*"):
                shutil.rmtree(p, ignore_errors=True)
            for p in Path(tempfile.gettempdir()).glob("hr_proxy_*"):
                shutil.rmtree(p, ignore_errors=True)
    print(f"\n结果: {passed} 通过, {failed} 失败")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
