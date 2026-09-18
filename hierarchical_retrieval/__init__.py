"""
hierarchical_retrieval — 分级检索开发工具包

基于"全文缓存 + 关键句库 + 话题界域"三级架构，
兼容 Ollama 本地嵌入模型（默认 bge-m3），
在 prompt 体积有界的前提下扩展可检索的历史记忆容量。

快速开始:
    from hierarchical_retrieval import HConfig, HierarchicalRetrieval

    hr = HierarchicalRetrieval(config=HConfig(storage_root="./my_store"))
    hr.ingest("对话全文...")
    result = hr.retrieve("查询问题")
    print(result["summary"])
"""

__version__ = "0.5.0"
__author__ = "hierarchical-retrieval"

from .config import HConfig
from .core.embedding import NomicEmbedding
from .core.cloud_cache import CloudCache
from .core.key_sentence import KeySentenceLibrary
from .core.topic_domain import TopicDomain, TopicDomainManager
from .core.retrieval import HierarchicalRetrieval
from .storage.base import StorageBase
from .storage.local_storage import LocalStorage
from .storage.vector_store import VectorStore
from .utils.text_processor import TextProcessor
from .utils.summarizer import KeySentenceExtractor

__all__ = [
    # 配置
    "HConfig",
    # 核心引擎
    "HierarchicalRetrieval",
    "NomicEmbedding",
    "CloudCache",
    "KeySentenceLibrary",
    "TopicDomain",
    "TopicDomainManager",
    # 存储
    "StorageBase",
    "LocalStorage",
    "VectorStore",
    # 工具
    "TextProcessor",
    "KeySentenceExtractor",
]