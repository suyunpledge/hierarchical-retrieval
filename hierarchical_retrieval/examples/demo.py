"""
分级检索开发工具包 —— 完整使用示例

本示例演示:
  1. 初始化分级检索系统
  2. 写入多话题对话数据
  3. 针对不同话题进行检索
  4. 查看系统状态

运行前请确保:
  - Ollama 已启动 (ollama serve)
  - nomic-embed-text 模型已拉取 (ollama pull nomic-embed-text)
"""

import logging
import sys
import os
import shutil

# 将项目根目录（hierarchical_retrieval 包的父目录）加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hierarchical_retrieval import HConfig, HierarchicalRetrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    print("=" * 60)
    print("  分级检索 (Hierarchical Retrieval) 开发工具包")
    print("  三级架构: 全文缓存 → 关键句库 → 话题界域")
    print("=" * 60)

    # ── 1. 配置 ────────────────────────────────────────────
    # 清理旧存储（TRAE v0.2 补丁）: 保证 demo 可重复运行，
    # 否则数据累积导致话题漂移、检索结果越来越怪
    demo_storage = "./demo_storage"
    if os.path.exists(demo_storage):
        shutil.rmtree(demo_storage)
        print(f"\n[清理] 已移除旧存储目录: {demo_storage}")

    config = HConfig(
        storage_root=demo_storage,  # 存储根目录
        top_k_key_sentences=3,
        top_k_full_context=2,
        # nomic-embed-text 对中文余弦相似度普遍偏高，
        # 写入归并阈值需调高，否则不同话题会被误并；
        # 查询匹配阈值可略低，保证召回。
        topic_min_similarity=0.70,
        topic_similarity_threshold=0.55,
    )
    print(f"\n[配置] 存储目录: {config.storage_root}")
    print(f"[配置] Embedding 模型: {config.embedding_model}")

    # ── 2. 初始化 ──────────────────────────────────────────
    print("\n[初始化] 正在连接 Ollama / nomic-embed-text...")
    hr = HierarchicalRetrieval(config=config)
    print("[初始化] 完成")

    # ── 3. 写入多话题对话数据 ──────────────────────────────
    print("\n" + "-" * 40)
    print("  [写入阶段] 注入多话题对话数据")
    print("-" * 40)

    conversations = [
        {
            "text": (
                "今天我们来讨论一下项目的技术架构方案。"
                "我建议采用微服务架构，将用户模块、订单模块和支付模块拆分。"
                "每个模块独立部署，通过 API Gateway 进行路由。"
                "数据库方面，我认为使用 PostgreSQL 作为主库，Redis 做缓存层比较合适。"
                "最终我们决定采用这个方案，下周开始搭建基础框架。"
            ),
            "metadata": {"source": "arch_meeting", "speaker": "Alice"},
        },
        {
            "text": (
                "关于预算问题，Q3 的研发预算需要重新规划。"
                "目前服务器成本每月约 5 万，人员成本约 30 万。"
                "建议将云服务预算削减 15%，通过优化实例规格来节省成本。"
                "另外，需要申请 10 万的 GPU 算力预算用于 AI 模型训练。"
                "结论是：总预算控制在 120 万以内，下周五前提交审批。"
            ),
            "metadata": {"source": "budget_meeting", "speaker": "Bob"},
        },
        {
            "text": (
                "产品上线计划初步定在 11 月 15 日。"
                "第一阶段：10 月 1 日前完成核心功能开发。"
                "第二阶段：10 月 15 日至 11 月 1 日进行内部测试。"
                "第三阶段：11 月 1 日至 11 月 14 日进行灰度发布。"
                "关键里程碑：10 月 20 日需要完成安全审计。"
                "建议安排专人跟进每个里程碑的进度。"
            ),
            "metadata": {"source": "planning_meeting", "speaker": "Charlie"},
        },
        {
            "text": (
                "刚才讨论的技术架构方案，我补充一下细节。"
                "关于用户模块，我们需要支持 OAuth2.0 和 SSO。"
                "订单模块需要处理高并发场景，建议使用消息队列削峰填谷。"
                "支付模块必须通过 PCI-DSS 合规认证，这个很重要。"
                "API Gateway 选用 Kong 或者 APISIX，还需要进一步评估。"
            ),
            "metadata": {"source": "arch_meeting_2", "speaker": "David"},
        },
    ]

    for i, conv in enumerate(conversations):
        result = hr.ingest(conv["text"], conv["metadata"])
        print(f"  [{i + 1}] 写入完成 → 话题: {result['domain_name']}")

    print(f"\n  [写入结果] 共 {len(conversations)} 条对话")
    print(f"  [写入结果] 系统状态: {hr.status()}")

    # ── 4. 检索测试 ────────────────────────────────────────
    print("\n" + "-" * 40)
    print("  [检索阶段] 测试不同话题的检索能力")
    print("-" * 40)

    queries = [
        "我们的技术架构用了什么方案？数据库选型是什么？",
        "Q3 的预算怎么安排的？GPU 算力预算多少？",
        "上线时间怎么安排的？安全审计什么时候？",
        "API Gateway 选型考虑了什么？",
    ]

    for q in queries:
        print(f"\n  >>> 查询: {q}")
        result = hr.retrieve(q)
        print(f"  <<< 话题: {result['detected_domain']['name'] if result['detected_domain'] else '无'}")
        print(f"  <<< 关键句: {len(result['key_sentences'])} 条命中")
        for i, ks in enumerate(result["key_sentences"][:3]):
            print(f"       [{i + 1}] {ks.get('text', '')[:60]}...")
        print(f"  <<< 全文段落: {len(result['full_contexts'])} 段")

    # ── 5. 系统状态 ────────────────────────────────────────
    print("\n" + "-" * 40)
    print("  [系统状态]")
    print("-" * 40)
    status = hr.status()
    print(f"  Embedding 模型: {status['embedding_model']}")
    print(f"  全文缓存条目:    {status['cloud_cache_entries']}")
    print(f"  关键句库数量:    {status['key_sentence_count']}")
    print(f"  话题界域数量:    {status['topic_domains']}")
    for d in status["domains"]:
        print(f"    ├─ [{d['id']}] {d['name']} ({d['entries']} 条)")

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()