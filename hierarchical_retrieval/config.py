"""
全局配置模块
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HConfig:
    """分级检索全局配置"""

    # ── Embedding 服务 ──────────────────────────────────────────
    embedding_model: str = "bge-m3:latest"
    embedding_dim: int = 1024          # bge-m3 输出维度（中文/多语）
    embedding_backend: str = "auto"   # auto | ollama | openai（llama.cpp llama-server）
    embedding_base_url: str = "http://localhost:11434"
    embedding_request_timeout: int = 60

    # ── 存储路径 ────────────────────────────────────────────────
    storage_root: str = "./hierarchical_storage"
    cloud_cache_dir: str = "cloud_cache"          # 全文缓存（本地磁盘；目录名沿用历史命名）
    key_sentence_dir: str = "key_sentence_lib"    # 关键句库
    topic_domain_dir: str = "topic_domains"       # 话题界域
    vector_index_dir: str = "vector_index"        # 向量索引

    # ── 检索参数 ────────────────────────────────────────────────
    top_k_key_sentences: int = 5       # 每个话题检索的关键句数量
    top_k_full_context: int = 3        # 每个关键句关联的全文段落数
    topic_similarity_threshold: float = 0.55  # 话题匹配阈值（bge-m3 分布）
    key_sentence_similarity_threshold: float = 0.55  # 关键句匹配阈值（bge-m3 分布）
    # 全文段落直检（第二条召回路径）：关键句只覆盖约一半句子，未被选为
    # 关键句的事实只能靠直接检索全文段落找回。混合分 = 余弦 + 权重×关键词重合度。
    direct_chunk_search: bool = True
    chunk_similarity_threshold: float = 0.55   # 混合分阈值（bge-m3 实测标定）
    chunk_lexical_weight: float = 0.15         # 关键词重合度权重
    chunk_candidates: int = 20                 # 向量粗召回候选数

    # ── 关键句提取 ──────────────────────────────────────────────
    max_key_sentences_per_chunk: int = 3
    key_sentence_max_length: int = 128  # 单句最大字符数

    # ── 话题界域 ────────────────────────────────────────────────
    topic_min_similarity: float = 0.62  # 归入已有话题的最小相似度（bge-m3 实测选值）
    topic_max_domains: int = 100        # 最大话题域数量

    # ── 上下文装配与压缩（/v1 代理模式） ─────────────────
    context_char_budget: int = 3000     # 记忆块字符预算（超预算从低分项开始丢弃）
    recent_turns_keep: int = 6          # 近期对话原样保留的消息条数
    compress_threshold_chars: int = 3000  # 历史总字符超过该值才触发压缩

    # ── 对话后端（/v1 代理转发目标，默认本地 Ollama） ─────
    chat_backend_url: str = "http://localhost:11434"
    chat_model: str = ""  # /v1 代理的回退对话模型，通过 HR_CHAT_MODEL 环境变量或代码设置

    # ── 性能与代理保护 ─────────────────────────────────────
    embed_workers: int = 6               # 批量嵌入的并发线程数（Ollama 可并发）
    max_ingest_per_request: int = 32     # 代理单次请求最多补录的旧消息条数

    # ── 进程内缓存（0 = 关闭） ─────────────────────────────
    embedding_cache_size: int = 4096     # 嵌入向量 LRU 条数（1024 维约 4KB/条）
    retrieval_cache_size: int = 256      # 检索结果 LRU 条数（追加时校验受影响的结果）
    fulltext_cache_size: int = 128       # 全文记录 LRU 条数（写入/删除时同步更新）

    # ── 日志 ────────────────────────────────────────────────────
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Fail fast on values later consumed by FAISS or thread pools."""
        positive_ints = {
            "embedding_dim": self.embedding_dim,
            "topic_max_domains": self.topic_max_domains,
            "context_char_budget": self.context_char_budget,
            "recent_turns_keep": self.recent_turns_keep,
            "compress_threshold_chars": self.compress_threshold_chars,
            "embed_workers": self.embed_workers,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        non_negative_ints = {
            "top_k_key_sentences": self.top_k_key_sentences,
            "top_k_full_context": self.top_k_full_context,
            "max_key_sentences_per_chunk": self.max_key_sentences_per_chunk,
            "key_sentence_max_length": self.key_sentence_max_length,
            "max_ingest_per_request": self.max_ingest_per_request,
            "chunk_candidates": self.chunk_candidates,
            "embedding_cache_size": self.embedding_cache_size,
            "retrieval_cache_size": self.retrieval_cache_size,
            "fulltext_cache_size": self.fulltext_cache_size,
        }
        for name, value in non_negative_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "topic_similarity_threshold",
            "key_sentence_similarity_threshold",
            "topic_min_similarity",
            "chunk_similarity_threshold",
            "chunk_lexical_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def from_dict(cls, d: dict) -> "HConfig":
        """从字典创建配置，仅覆盖已提供的字段"""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
