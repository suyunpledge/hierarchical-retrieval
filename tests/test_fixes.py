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
            max_ingest_per_request=32,
        )
        api_server.config = cfg
        api_server.hr = make_hr(
            FakeEmbedding,
            recent_turns_keep=6,
            compress_threshold_chars=3000,
            context_char_budget=2000,
            max_ingest_per_request=32,
        )
        hr = api_server.hr
        from api_server import ChatMessage, _compress_history

        # 1 条 system + 59 条对话（每条 56 字，总长 > 3000 触发压缩）；
        # keep=6 → older 53 条，cap=32
        msgs = [ChatMessage(role="system", content="你是一个助手。")]
        for i in range(59):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(ChatMessage(role=role, content=f"第{i}轮对话内容，" + "细节填充" * 12))
        older = msgs[1:-6]
        assert len(older) == 53
        assert sum(len(m.content) for m in msgs) > 3000  # 确保触发压缩

        _compress_history(msgs)
        n1 = hr.cloud_cache.count()
        assert n1 == 32, f"首轮应补录最旧的 32 条，实际 {n1}"
        # 最旧的一条必须首轮入库（旧实现会永久丢掉它）
        assert hr.cloud_cache.has_content(older[0].content), "最旧消息首轮必须入库"
        # 窗口外的最新一条 older 还不该入库
        assert not hr.cloud_cache.has_content(older[-1].content)

        _compress_history(msgs)
        n2 = hr.cloud_cache.count()
        assert n2 == 53, f"两轮后 older 应全部入库（已入库不占窗口名额），实际 {n2}"
        assert hr.cloud_cache.has_content(older[-1].content), "第二轮应补完剩余 21 条"
        print(f"    补录: 53 条 older → 两轮请求后全部入库 ({n1} → {n2})")
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
    hr.topic_manager.detect_domain = lambda query: domain_a
    hr.topic_manager.search_in_domain = lambda query, domain, top_k=5: [
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