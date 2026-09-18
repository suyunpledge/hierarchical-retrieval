# hierarchical-retrieval 项目整合文档

> 整合时间：2026-08-14 (Asia/Shanghai)
> 源目录：`Desktop/hierarchical_retrieval/`
> 共 21 个源文件（含 17 个 Python 源文件 + 4 个配置/启动文件），本文件将其全部整合为单一文档。

---

## 目录

- [项目结构总览](#项目结构总览)
- [README.md](#readmemd)
- [pyproject.toml](#pyprojecttoml)
- [requirements_api.txt](#requirements_apitxt)
- [.gitignore](#gitignore)
- [LICENSE](#license)
- [主包源码](#主包源码)
  - [\_\_init\_\_.py](#__init__py)
  - [config.py](#configpy)
  - [core/\_\_init\_\_.py](#core__init__py)
  - [core/embedding.py](#coreembeddingpy)
  - [core/cloud_cache.py](#corecloud_cachepy)
  - [core/key_sentence.py](#corekey_sentencepy)
  - [core/topic_domain.py](#coretopic_domainpy)
  - [core/retrieval.py](#coreretrievalpy)
  - [storage/\_\_init\_\_.py](#storage__init__py)
  - [storage/base.py](#storagebasepy)
  - [storage/local_storage.py](#storagelocal_storagepy)
  - [storage/vector_store.py](#storagevector_storepy)
  - [utils/\_\_init\_\_.py](#utils__init__py)
  - [utils/text_processor.py](#utilstext_processorpy)
  - [utils/summarizer.py](#utilssummarizerpy)
  - [examples/\_\_init\_\_.py](#examples__init__py)
  - [examples/demo.py](#examplesdemopy)
- [API 服务](#api-服务)
  - [api_server.py](#api_serverpy)
  - [start_api.bat](#start_apibat)
  - [start_hr_api_hidden.vbs](#start_hr_api_hiddenvbs)

---

## 项目结构总览

```
hierarchical_retrieval/
├── pyproject.toml                  # 打包配置（PEP 517/518）
├── README.md                       # 项目说明文档
├── LICENSE                         # MIT 许可证
├── .gitignore                      # Git 忽略规则
├── requirements_api.txt            # API 服务额外依赖
├── api_server.py                   # FastAPI HTTP 接口
├── start_api.bat                   # Windows 启动脚本
├── start_hr_api_hidden.vbs         # Windows 静默启动脚本
├── py.typed                        # PEP 561 类型标记
├── hierarchical-retrieval-整合文档.md  # ← 本文件
│
├── hierarchical_retrieval/         # 主包
│   ├── __init__.py                 # 公共 API 导出
│   ├── config.py                   # HConfig 全局配置
│   ├── py.typed
│   │
│   ├── core/                       # 核心引擎
│   │   ├── __init__.py
│   │   ├── embedding.py            # NomicEmbedding — Ollama 嵌入客户端
│   │   ├── cloud_cache.py          # CloudCache — L1 全文缓存
│   │   ├── key_sentence.py         # KeySentenceLibrary — L2 关键句库
│   │   ├── topic_domain.py         # TopicDomainManager — L3 话题界域
│   │   └── retrieval.py            # HierarchicalRetrieval — 三级流水线调度
│   │
│   ├── storage/                    # 存储层
│   │   ├── __init__.py
│   │   ├── base.py                 # StorageBase 抽象接口
│   │   ├── local_storage.py        # LocalStorage 本地键值存储
│   │   └── vector_store.py         # VectorStore FAISS 索引
│   │
│   ├── utils/                      # 工具
│   │   ├── __init__.py
│   │   ├── text_processor.py       # 文本分割/清洗
│   │   └── summarizer.py           # 关键句启发式提取
│   │
│   └── examples/
│       ├── __init__.py
│       └── demo.py                 # 完整使用示例
│
├── hierarchical_api_storage/       # API 运行时数据（自动生成）
└── .idea/                          # PyCharm IDE 配置（非项目本体）
```

---

## README.md

```markdown
# hierarchical-retrieval

> 分级检索开发工具包 — 云端缓存 + 关键句库 + 话题界域三级架构，兼容 `nomic-embed-text` 嵌入向量模型，实现大模型上下文记忆的几何倍数扩展。

## 背景与动机

当前大模型在长上下文记忆方面面临：全文通记导致遗忘、Token 浪费、细节丢失与召回困难、话题干扰等问题。

**分级检索**通过将信息分层存储、按话题隔离检索，将"上下文长度"聚焦在当前话题对应的关键句库中发挥，从而在不改变模型本身的前提下，显著扩展有效记忆容量。

## 核心思想

### 1. 云端缓存（基础信息库）
将所有对话信息一字不漏地缓存，作为基础信息库。

### 2. 关键句库（二级信息库）
AI 在对话中将内容提炼为精简的关键句，在独立空间中存储。检索时先命中关键句，再根据关键句在云端缓存中查找详细对话。

### 3. 话题界域
AI 将对话内容划分为不同的话题界域，按界域分配关键句库。检索时先确定当前查询所处的话题界域，再精准关联。

## 架构总览

```
查询 → L3 话题界域检测 → L2 界域内关键句检索 → L1 关键句关联全文 → 聚合结果
```

## 快速开始

```python
from hierarchical_retrieval import HConfig, HierarchicalRetrieval

hr = HierarchicalRetrieval(config=HConfig(storage_root="./my_store"))
hr.ingest("对话全文...", metadata={"source": "meeting"})
result = hr.retrieve("查询问题")
print(result["summary"])
```

## 安装

```bash
pip install -e .
# 前置：ollama serve + ollama pull nomic-embed-text
```

## License

MIT
```

---

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hierarchical-retrieval"
version = "0.1.0"
description = "分级检索开发工具包 — 云端缓存 + 关键句库 + 话题界域三级架构，兼容 nomic-embed-text"
readme = "README.md"
requires-python = ">=3.8"
license = { text = "MIT" }
authors = [
    { name = "hierarchical-retrieval" },
]
keywords = [
    "retrieval", "embedding", "nomic-embed-text", "memory",
    "rag", "topic-domain", "llm", "context-management",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.21.0",
    "requests>=2.28.0",
    "faiss-cpu>=1.7.4",
]

[project.optional-dependencies]
gpu = ["faiss-gpu>=1.7.4"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "build>=1.0", "twine>=4.0"]

[project.urls]
"Homepage" = "https://github.com/hierarchical-retrieval/hierarchical-retrieval"
"Issues" = "https://github.com/hierarchical-retrieval/hierarchical-retrieval/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["hierarchical_retrieval*"]
exclude = ["demo_storage*", "tests*"]

[tool.setuptools.package-data]
hierarchical_retrieval = ["py.typed"]
```

---

## requirements_api.txt

```text
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
httpx>=0.24.0
```

---

## .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
env/
ENV/
.venv/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# 测试与覆盖率
.pytest_cache/
.coverage
htmlcov/
.tox/

# 分级检索运行时数据
demo_storage/
hierarchical_storage/
*.faiss

# 日志
*.log
```

---

## LICENSE

```text
MIT License

Copyright (c) 2026 hierarchical-retrieval

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 主包源码

> 以下按依赖顺序排列所有 Python 源文件。

---

### \_\_init\_\_.py

> `hierarchical_retrieval/__init__.py` — 公共 API 导出

```python
"""
hierarchical_retrieval — 分级检索开发工具包

基于"云端缓存 + 关键句库 + 话题界域"三级架构，
兼容 nomic-embed-text 嵌入向量模型，
实现上下文记忆的几何倍数扩展。

快速开始:
    from hierarchical_retrieval import HConfig, HierarchicalRetrieval

    hr = HierarchicalRetrieval(config=HConfig(storage_root="./my_store"))
    hr.ingest("对话全文...")
    result = hr.retrieve("查询问题")
    print(result["summary"])
"""

__version__ = "0.1.0"
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
```

---

### config.py

> `hierarchical_retrieval/config.py` — 全局配置

```python
"""
全局配置模块
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HConfig:
    """分级检索全局配置"""

    # ── Embedding 服务 ──────────────────────────────────────────
    embedding_model: str = "nomic-embed-text:latest"
    embedding_dim: int = 768           # nomic-embed-text 输出维度
    embedding_base_url: str = "http://localhost:11434"
    embedding_request_timeout: int = 60

    # ── 存储路径 ────────────────────────────────────────────────
    storage_root: str = "./hierarchical_storage"
    cloud_cache_dir: str = "cloud_cache"          # 云端缓存（全文）
    key_sentence_dir: str = "key_sentence_lib"    # 关键句库
    topic_domain_dir: str = "topic_domains"       # 话题界域
    vector_index_dir: str = "vector_index"        # 向量索引

    # ── 检索参数 ────────────────────────────────────────────────
    top_k_key_sentences: int = 5       # 每个话题检索的关键句数量
    top_k_full_context: int = 3        # 每个关键句关联的全文段落数
    topic_similarity_threshold: float = 0.45  # 话题匹配阈值
    key_sentence_similarity_threshold: float = 0.50  # 关键句匹配阈值

    # ── 关键句提取 ──────────────────────────────────────────────
    max_key_sentences_per_chunk: int = 3
    key_sentence_max_length: int = 128  # 单句最大字符数

    # ── 话题界域 ────────────────────────────────────────────────
    topic_min_similarity: float = 0.40  # 归入已有话题的最小相似度
    topic_max_domains: int = 100        # 最大话题域数量

    # ── 日志 ────────────────────────────────────────────────────
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, d: dict) -> "HConfig":
        """从字典创建配置，仅覆盖已提供的字段"""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
```

---

### core/\_\_init\_\_.py

> `hierarchical_retrieval/core/__init__.py`

```python
from .embedding import NomicEmbedding
from .cloud_cache import CloudCache
from .key_sentence import KeySentenceLibrary
from .topic_domain import TopicDomainManager
from .retrieval import HierarchicalRetrieval

__all__ = [
    "NomicEmbedding",
    "CloudCache",
    "KeySentenceLibrary",
    "TopicDomainManager",
    "HierarchicalRetrieval",
]
```

---

### core/embedding.py

> `hierarchical_retrieval/core/embedding.py` — L0 嵌入服务，对接 Ollama nomic-embed-text

```python
"""
Embedding 服务 —— 对接 nomic-embed-text 嵌入向量模型

通过 Ollama API 调用 nomic-embed-text:latest 生成文本嵌入向量。
"""

import logging
from typing import Optional

import numpy as np
import requests

from ..config import HConfig

logger = logging.getLogger(__name__)


class NomicEmbedding:
    """
    nomic-embed-text 嵌入向量客户端

    依赖: 本地需安装 Ollama 并已拉取 nomic-embed-text 模型
    """

    def __init__(self, config: Optional[HConfig] = None):
        self.config = config or HConfig()
        self._model = self.config.embedding_model
        self._base_url = self.config.embedding_base_url.rstrip("/")
        self._timeout = self.config.embedding_request_timeout
        self._dim = self.config.embedding_dim

        # 健康检查
        self._check_available()

    def _check_available(self):
        """检查 Ollama 服务是否可用"""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if not any(self._model in m for m in models):
                    logger.warning(
                        "模型 %s 未在 Ollama 中找到，请先运行: ollama pull %s",
                        self._model, self._model,
                    )
                else:
                    logger.info("Ollama 服务可用，模型 %s 已就绪", self._model)
            else:
                logger.warning(
                    "Ollama 服务返回异常状态码 %s，请确保服务已启动", resp.status_code
                )
        except requests.ConnectionError:
            logger.warning(
                "无法连接到 Ollama 服务 (%s)，请确保已启动: ollama serve",
                self._base_url,
            )

    def embed(self, text: str) -> np.ndarray:
        """
        将单条文本转为嵌入向量

        Args:
            text: 输入文本

        Returns:
            shape (dim,) 的 float32 向量（L2 归一化）
        """
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)

        payload = {
            "model": self._model,
            "prompt": text,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
            if not embedding:
                logger.warning("Ollama 返回空 embedding，使用零向量")
                return np.zeros(self._dim, dtype=np.float32)
            vec = np.array(embedding, dtype=np.float32)
            # L2 归一化，使内积 = 余弦相似度，取值 [-1, 1]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            return vec
        except Exception as e:\n            logger.error("Embedding 请求失败: %s", e)
            return np.zeros(self._dim, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量嵌入"""
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model
```

---

### core/cloud_cache.py

> `hierarchical_retrieval/core/cloud_cache.py` — L1 全文缓存

```python
"""
云端缓存模块 —— 第一级（基础信息库）

存储所有对话全文，一字不漏，作为完整的信息基础。
支持逐条写入、向量索引检索、按时间/ID 读取。
"""

import logging
import time
import uuid
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
    云端缓存 —— 全文存储与向量检索

    层级: 第一级
    职责: 保存所有对话原文，支持全文检索和语义检索。
    """

    def __init__(
        self,
        embedding: NomicEmbedding,
        config: Optional[HConfig] = None,
    ):
        self.config = config or HConfig()
        self.embedding = embedding

        base_dir = self.config.storage_root
        self._text_store = LocalStorage(
            root_dir=f"{base_dir}/{self.config.cloud_cache_dir}",
            use_json=True,
        )
        self._vector_store = VectorStore(
            root_dir=f"{base_dir}/{self.config.vector_index_dir}/cloud_cache",
            dimension=self.config.embedding_dim,
        )
        self._chunk_map: dict[str, dict[int, str]] = {}

    def store(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        将全文存入缓存，自动切块并构建向量索引。

        Returns:
            entry_id: 本条记录的全局唯一 ID
        """
        entry_id = f"cc_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()

        chunks = TextProcessor.split_chunks(text)
        meta = {
            "entry_id": entry_id,
            "timestamp": timestamp,
            "total_chunks": len(chunks),
            "total_length": len(text),
            **(metadata or {}),
        }

        self._text_store.save(entry_id, {"meta": meta, "full_text": text})

        self._chunk_map[entry_id] = {}
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{entry_id}_chunk_{idx}"
            vec = self.embedding.embed(chunk)
            self._vector_store.add(
                external_id=chunk_id,
                vector=vec,
                metadata={
                    "entry_id": entry_id,
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "timestamp": timestamp,
                },
            )
            self._chunk_map[entry_id][idx] = chunk

        logger.info("CloudCache 写入: %s (%d 块, %d 字符)", entry_id, len(chunks), len(text))
        return entry_id

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索相关全文段落"""
        q_vec = self.embedding.embed(query)
        results = self._vector_store.search(q_vec, top_k=top_k)

        enriched = []
        for r in results:
            meta = r["metadata"]
            chunk_text = meta.get("chunk_text", "")
            entry_id = meta.get("entry_id", "")

            full_text = None
            if entry_id:
                record = self._text_store.load(entry_id)
                if record:
                    full_text = record.get("full_text", "")

            enriched.append({
                "entry_id": entry_id,
                "chunk_index": meta.get("chunk_index", -1),
                "chunk_text": chunk_text,
                "score": r["score"],
                "timestamp": meta.get("timestamp", 0),
                "full_text": full_text,
            })
        return enriched

    def get_full_text(self, entry_id: str) -> Optional[str]:
        """按 entry_id 获取完整全文"""
        record = self._text_store.load(entry_id)
        if record:
            return record.get("full_text")
        return None

    def get_metadata(self, entry_id: str) -> Optional[dict]:
        """按 entry_id 获取元数据"""
        record = self._text_store.load(entry_id)
        if record:
            return record.get("meta")
        return None

    def list_entries(self) -> list[str]:
        """列出所有缓存条目 ID"""
        return self._text_store.list_keys(prefix="cc_")

    def count(self) -> int:
        """缓存条目总数"""
        return self._text_store.count(prefix="cc_")
```

---

### core/key_sentence.py

> `hierarchical_retrieval/core/key_sentence.py` — L2 关键句库

```python
"""
关键句库模块 —— 第二级（二级信息库）

将对话内容提炼为精简的关键句，在独立向量空间中存储。
每个关键句关联到云端缓存中的全文位置。
"""

import logging
import time
import uuid
from typing import Optional

import numpy as np

from ..config import HConfig
from ..storage.local_storage import LocalStorage
from ..storage.vector_store import VectorStore
from .embedding import NomicEmbedding
from ..utils.text_processor import TextProcessor
from ..utils.summarizer import KeySentenceExtractor

logger = logging.getLogger(__name__)


class KeySentenceLibrary:
    """
    关键句库

    层级: 第二级
    职责:
      - 从对话文本中提取关键句
      - 将关键句向量化并存储
      - 按向量相似度检索相关关键句
      - 通过关键句关联到云端缓存全文
    """

    def __init__(
        self,
        embedding: NomicEmbedding,
        config: Optional[HConfig] = None,
    ):
        self.config = config or HConfig()
        self.embedding = embedding
        self.extractor = KeySentenceExtractor(config)

        base_dir = self.config.storage_root
        self._text_store = LocalStorage(
            root_dir=f"{base_dir}/{self.config.key_sentence_dir}",
            use_json=True,
        )
        self._vector_store = VectorStore(
            root_dir=f"{base_dir}/{self.config.vector_index_dir}/key_sentences",
            dimension=self.config.embedding_dim,
        )

    def ingest(self, text: str, source_entry_id: str) -> list[str]:
        """
        从对话文本中提取关键句，存入关键句库。

        Returns:
            提取的关键句 ID 列表
        """
        key_sentences = self.extractor.extract(text)
        sentence_ids = []

        for sent in key_sentences:
            sent_id = f"ks_{uuid.uuid4().hex[:12]}"
            timestamp = time.time()

            record = {
                "sentence_id": sent_id,
                "text": sent,
                "source_entry_id": source_entry_id,
                "timestamp": timestamp,
            }
            self._text_store.save(sent_id, record)

            vec = self.embedding.embed(sent)
            self._vector_store.add(
                external_id=sent_id,
                vector=vec,
                metadata={
                    "sentence_id": sent_id,
                    "text": sent,
                    "source_entry_id": source_entry_id,
                    "timestamp": timestamp,
                    "type": "key_sentence",
                },
            )
            sentence_ids.append(sent_id)

        logger.info(
            "关键句库: 从 %s 提取 %d 条关键句",
            source_entry_id, len(key_sentences),
        )
        return sentence_ids

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """按查询语义检索相关关键句"""
        q_vec = self.embedding.embed(query)
        results = self._vector_store.search(q_vec, top_k=top_k)

        enriched = []
        for r in results:
            meta = r["metadata"]
            enriched.append({
                "sentence_id": meta.get("sentence_id", ""),
                "text": meta.get("text", ""),
                "source_entry_id": meta.get("source_entry_id", ""),
                "score": r["score"],
                "timestamp": meta.get("timestamp", 0),
            })
        return enriched

    def get_sentence(self, sentence_id: str) -> Optional[dict]:
        """按 ID 获取关键句详情"""
        return self._text_store.load(sentence_id)

    def list_sentences(self, source_entry_id: Optional[str] = None) -> list[str]:
        """列出关键句 ID，可选按 source 过滤"""
        all_ids = self._text_store.list_keys(prefix="ks_")
        if source_entry_id is None:
            return all_ids
        filtered = []
        for sid in all_ids:
            record = self._text_store.load(sid)
            if record and record.get("source_entry_id") == source_entry_id:
                filtered.append(sid)
        return filtered

    def count(self) -> int:
        return self._text_store.count(prefix="ks_")
```

---

### core/topic_domain.py

> `hierarchical_retrieval/core/topic_domain.py` — L3 话题界域

```python
"""
话题界域模块 —— 三级架构中的界域划分层

将对话信息划分为不同的话题界域，每个界域拥有独立的
关键句库索引，从而实现"按话题分配记忆"。
"""

import json
import logging
import time
import uuid
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

    def __init__(self, domain_id: str, name: str, description: str = "",
                 embedding: Optional[np.ndarray] = None):
        self.domain_id = domain_id
        self.name = name
        self.description = description
        self.embedding = embedding
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

        self._domains: dict[str, TopicDomain] = {}
        self._load_registry()
        self._vector_stores: dict[str, VectorStore] = {}

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
            except Exception as e:\n                logger.warning("加载话题界域注册表失败: %s", e)

    def _save_registry(self):
        data = [d.to_dict() for d in self._domains.values()]
        self._registry_path().write_text(
            json.dumps(data, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )

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

        Returns:
            匹配的 TopicDomain 对象
        """
        text_vec = self.embedding.embed(text)

        best_domain = None
        best_score = -1.0

        for domain in self._domains.values():
            if domain.embedding is not None:
                sim = float(np.dot(text_vec, domain.embedding))
                if sim > best_score:
                    best_score = sim
                    best_domain = domain

        if best_domain and best_score >= self.config.topic_min_similarity:
            # 平滑更新话题向量
            alpha = 0.15
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
        """在指定话题界域内检索关键句"""
        vs = self._get_vector_store(domain.domain_id)
        q_vec = self.embedding.embed(query)
        return vs.search(q_vec, top_k=top_k)

    def add_key_sentence_to_domain(
        self, domain_id: str, sentence_id: str, text: str, metadata: Optional[dict] = None
    ):
        """将关键句加入指定话题界域的向量索引"""
        vs = self._get_vector_store(domain_id)
        vec = self.embedding.embed(text)
        vs.add(
            external_id=sentence_id,
            vector=vec,
            metadata={
                "sentence_id": sentence_id,
                "text": text,
                **(metadata or {}),
            },
        )

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _generate_domain_name(text: str) -> str:
        """从文本片段生成话题名"""
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
```

---

### core/retrieval.py

> `hierarchical_retrieval/core/retrieval.py` — 三级流水线调度引擎

```python
"""
分级检索流水线 —— 三级架构的核心调度引擎

完整流程:
  1. 话题检测 → 确定当前查询所属的话题界域（第三级）
  2. 界域内关键句检索 → 在该话题的关键句库中查找最相关句（第二级）
  3. 关键句→全文关联 → 通过关键句关联到云端缓存的详细上下文（第一级）
  4. 结果聚合 → 按话题/相关性排序返回
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
      L1 ─ CloudCache           (云端缓存/全文)
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

        Returns:
            {"entry_id", "domain_id", "domain_name", "sentence_ids"}
        """
        # L1: 存入云端缓存（全文）
        entry_id = self.cloud_cache.store(text, metadata)

        # L3: 分配话题界域
        domain = self.topic_manager.assign_domain(text, metadata)

        # L2: 提取关键句并存入关键句库
        sentence_ids = self.key_sentence_lib.ingest(text, entry_id)

        # L2→L3 关联: 将关键句也加入话题界域的向量索引
        for sid in sentence_ids:
            record = self.key_sentence_lib.get_sentence(sid)
            if record:
                self.topic_manager.add_key_sentence_to_domain(
                    domain_id=domain.domain_id,
                    sentence_id=sid,
                    text=record["text"],
                    metadata={
                        "source_entry_id": entry_id,
                        "type": "key_sentence",
                    },
                )

        return {
            "entry_id": entry_id,
            "domain_id": domain.domain_id,
            "domain_name": domain.name,
            "sentence_ids": sentence_ids,
        }

    # ── 检索流水线 ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k_sentences: Optional[int] = None,
        top_k_context: Optional[int] = None,
    ) -> dict:
        """
        分级检索主入口

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
            domain_ks = self._flatten_results(
                self.topic_manager.search_in_domain(query, domain, top_k=tks)
            )
            result["key_sentences"] = domain_ks

            # 不足则补充全局检索
            if len(domain_ks) < tks:
                global_ks = self._flatten_results(
                    self.key_sentence_lib.search(query, top_k=tks)
                )
                existing_ids = {k["sentence_id"] for k in domain_ks if k.get("sentence_id")}
                for ks in global_ks:
                    if ks.get("sentence_id") not in existing_ids:
                        domain_ks.append(ks)
                        existing_ids.add(ks.get("sentence_id", ""))
                    if len(domain_ks) >= tks:
                        break

        # ── Step 3: 关键句→全文关联 (L1) ──────────────────
        seen_entries = set()
        contexts = []
        for ks in result["key_sentences"]:
            source_id = ks.get("source_entry_id", "")
            if source_id and source_id not in seen_entries:
                seen_entries.add(source_id)
                ctx = self.cloud_cache.search(ks.get("text", query), top_k=tkc)
                contexts.extend(ctx)

        seen_chunks = set()
        unique_contexts = []
        for ctx in contexts:
            chunk_id = f"{ctx.get('entry_id')}_{ctx.get('chunk_index')}"
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_contexts.append(ctx)

        result["full_contexts"] = unique_contexts[:tkc * tks]

        # ── 生成摘要 ──────────────────────────────────────
        result["summary"] = self._build_summary(result)

        return result

    # ── 批量写入 ──────────────────────────────────────────

    def ingest_batch(self, conversations: list[dict]) -> list[dict]:
        """批量写入多条对话。"""
        return [self.ingest(c["text"], c.get("metadata")) for c in conversations]

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _flatten_results(results: list[dict]) -> list[dict]:
        """将 VectorStore 返回展平为统一的关键句结果结构。"""
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
```

---

### storage/\_\_init\_\_.py

> `hierarchical_retrieval/storage/__init__.py`

```python
from .base import StorageBase
from .local_storage import LocalStorage
from .vector_store import VectorStore

__all__ = ["StorageBase", "LocalStorage", "VectorStore"]
```

---

### storage/base.py

> `hierarchical_retrieval/storage/base.py` — 存储层抽象接口

```python
"""
存储层抽象接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class StorageBase(ABC):
    """所有存储后端的统一抽象"""

    @abstractmethod
    def save(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        ...

    @abstractmethod
    def count(self, prefix: str = "") -> int:
        ...
```

---

### storage/local_storage.py

> `hierarchical_retrieval/storage/local_storage.py` — 本地文件键值存储

```python
"""
本地文件存储实现
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

from .base import StorageBase


class LocalStorage(StorageBase):
    """基于本地文件系统的键值存储"""

    def __init__(self, root_dir: str, use_json: bool = True):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.use_json = use_json

    def _resolve_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        ext = ".json" if self.use_json else ".pkl"
        return self.root / f"{safe}{ext}"

    def save(self, key: str, value: Any) -> None:
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_json:
            path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")
        else:
            path.write_bytes(pickle.dumps(value))

    def load(self, key: str) -> Optional[Any]:
        path = self._resolve_path(key)
        if not path.exists():
            return None
        if self.use_json:
            return json.loads(path.read_text(encoding="utf-8"))
        return pickle.loads(path.read_bytes())

    def delete(self, key: str) -> bool:
        path = self._resolve_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        ext = ".json" if self.use_json else ".pkl"
        keys = []
        for f in self.root.iterdir():
            if f.suffix == ext:
                key = f.stem
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def count(self, prefix: str = "") -> int:
        return len(self.list_keys(prefix))
```

---

### storage/vector_store.py

> `hierarchical_retrieval/storage/vector_store.py` — FAISS 向量索引

```python
"""
向量存储模块 —— 基于 FAISS 的本地向量索引
"""

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .base import StorageBase


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

        self._load_index()

    @property
    def index(self):
        """延迟导入 FAISS"""
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            self._load_index()
            if self._index is None:
                self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def _index_path(self) -> Path:
        return self.root / "vector_index.faiss"

    def _meta_path(self) -> Path:
        return self.root / "metadata.json"

    def _save_index(self):
        import faiss
        faiss.write_index(self.index, str(self._index_path()))
        meta = {
            "id_map": self._id_map,
            "reverse_map": {str(k): v for k, v in self._reverse_map.items()},
            "metadata": self._metadata,
            "next_id": self._next_id,
            "dimension": self.dimension,
        }
        self._meta_path().write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")

    def _load_index(self):
        idx_path = self._index_path()
        meta_path = self._meta_path()
        if idx_path.exists() and meta_path.exists():
            import faiss
            try:
                self._index = faiss.read_index(str(idx_path))
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._id_map = meta["id_map"]
                self._reverse_map = {int(k): v for k, v in meta["reverse_map"].items()}
                self._metadata = meta["metadata"]
                self._next_id = meta["next_id"]
                self.dimension = meta.get("dimension", 768)
            except Exception:
                self._index = None

    def add(self, external_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> int:
        """添加向量，返回 faiss_id"""
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]
        faiss_id = self._next_id
        self.index.add(vec)
        self._id_map[external_id] = faiss_id
        self._reverse_map[faiss_id] = external_id
        self._metadata[external_id] = metadata or {}
        self._next_id += 1
        self._save_index()
        return faiss_id

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        检索最相似的 top_k 条记录。

        返回: [{"id": str, "score": float, "metadata": dict}, ...]
        """
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        scores, indices = self.index.search(q, top_k)
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
        return results

    def get_metadata(self, external_id: str) -> Optional[dict]:
        return self._metadata.get(external_id)

    def update_metadata(self, external_id: str, metadata: dict) -> bool:
        if external_id in self._metadata:
            self._metadata[external_id].update(metadata)
            self._save_index()
            return True
        return False

    def delete(self, key: str) -> bool:
        """删除（FAISS 不支持按 id 删除，仅清除映射）"""
        if key in self._id_map:
            faiss_id = self._id_map.pop(key)
            self._reverse_map.pop(faiss_id, None)
            self._metadata.pop(key, None)
            self._save_index()
            return True
        return False

    # ── StorageBase 接口 ────────────────────────────────────────

    def save(self, key: str, value: any) -> None:
        if isinstance(value, dict) and "vector" in value:
            self.add(key, np.array(value["vector"]), value.get("metadata"))

    def load(self, key: str) -> Optional[any]:
        if key in self._metadata:
            return {"id": key, "metadata": self._metadata[key]}
        return None

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._id_map if k.startswith(prefix))

    def count(self, prefix: str = "") -> int:
        return sum(1 for k in self._id_map if k.startswith(prefix))
```

---

### utils/\_\_init\_\_.py

> `hierarchical_retrieval/utils/__init__.py`

```python
from .text_processor import TextProcessor
from .summarizer import KeySentenceExtractor

__all__ = ["TextProcessor", "KeySentenceExtractor"]
```

---

### utils/text_processor.py

> `hierarchical_retrieval/utils/text_processor.py` — 文本分割/清洗

```python
"""
文本处理工具
"""

import re


class TextProcessor:
    """文本分割、清洗工具"""

    SENTENCE_DELIMITERS = re.compile(r"(?<=[。！？.!?])")

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """按句末标点分割句子"""
        raw = TextProcessor.SENTENCE_DELIMITERS.split(text)
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences

    @staticmethod
    def split_chunks(text: str, max_chars: int = 512, overlap: int = 32) -> list[str]:
        """将文本切分为重叠块，保证语义连续性。"""
        sentences = TextProcessor.split_sentences(text)
        chunks: list[str] = []
        buffer: list[str] = []
        buf_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if buf_len + sent_len > max_chars and buffer:
                chunks.append("".join(buffer))
                overlap_text = "".join(buffer)
                overlap_start = max(0, len(overlap_text) - overlap)
                overlap_sents = TextProcessor.split_sentences(overlap_text[overlap_start:])
                buffer = overlap_sents if overlap_sents else []
                buf_len = sum(len(s) for s in buffer)
            buffer.append(sent)
            buf_len += sent_len

        if buffer:
            chunks.append("".join(buffer))
        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        """清理多余空白"""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
```

---

### utils/summarizer.py

> `hierarchical_retrieval/utils/summarizer.py` — 启发式关键句提取

```python
"""
关键句提取器 —— 从对话文本中提取代表性关键句

支持两种模式:
  1. 基于位置/统计的启发式提取（轻量，无需 LLM）
  2. 预留 LLM 接口（可对接本地模型进行语义提取）
"""

import re
from typing import Optional

from ..config import HConfig
from .text_processor import TextProcessor


class KeySentenceExtractor:
    """
    从文本块中提取关键句。

    启发式策略:
      - 包含"总结/关键/重点/重要的是/核心/问题/方案/结论/决定/最终"等词的句子优先
      - 较长句子（含有较多信息量）优先
      - 首尾句（话题引入/总结）优先
    """

    SIGNAL_WORDS = {
        "总结": 3, "综上所述": 3, "关键": 3, "重点": 3,
        "核心": 3, "重要的是": 3, "关键在于": 3,
        "问题": 2, "方案": 2, "结论": 2, "决定": 2,
        "最终": 2, "建议": 2, "需要": 2, "必须": 2,
        "第一": 1, "第二": 1, "首先": 1, "其次": 1, "最后": 1,
        "因此": 1, "所以": 1, "但是": 1, "然而": 1,
        "summary": 3, "key": 3, "important": 3, "critical": 3,
        "conclusion": 2, "result": 2, "proposal": 2,
        "first": 1, "second": 1, "finally": 1,
    }

    def __init__(self, config: Optional[HConfig] = None):
        self.config = config or HConfig()

    def extract(self, text: str, max_sentences: Optional[int] = None) -> list[str]:
        """
        从文本中提取关键句。

        Returns:
            关键句列表（按重要性降序）
        """
        max_s = max_sentences or self.config.max_key_sentences_per_chunk
        sentences = TextProcessor.split_sentences(text)
        if len(sentences) <= max_s:\n            return sentences\n\n        scored = []\n        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences))
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:max_s]]

    def _score_sentence(self, sentence: str, idx: int, total: int) -> float:
        """为单句打分"""
        score = 0.0
        s_lower = sentence.lower()

        for word, weight in self.SIGNAL_WORDS.items():
            if word in sentence or word in s_lower:
                score += weight

        length = len(sentence)
        if 20 <= length <= self.config.key_sentence_max_length:
            score += 1.0
        elif length < 10:
            score -= 0.5

        if idx == 0:
            score += 1.5
        elif idx == total - 1:
            score += 1.0

        if sentence.strip().endswith(("?", "？")):
            score -= 0.5

        return score
```

---

### examples/\_\_init\_\_.py

> `hierarchical_retrieval/examples/__init__.py`

```python
```

---

### examples/demo.py

> `hierarchical_retrieval/examples/demo.py` — 完整使用示例

```python
"""
分级检索开发工具包 —— 完整使用示例

运行前请确保:
  - Ollama 已启动 (ollama serve)
  - nomic-embed-text 模型已拉取 (ollama pull nomic-embed-text)
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hierarchical_retrieval import HConfig, HierarchicalRetrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    print("=" * 60)
    print("  分级检索 (Hierarchical Retrieval) 开发工具包")
    print("  三级架构: 云端缓存 → 关键句库 → 话题界域")
    print("=" * 60)

    # ── 1. 配置 ────────────────────────────────────────────
    config = HConfig(
        storage_root="./demo_storage",
        top_k_key_sentences=3,
        top_k_full_context=2,
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
    print(f"  云端缓存条目:    {status['cloud_cache_entries']}")
    print(f"  关键句库数量:    {status['key_sentence_count']}")
    print(f"  话题界域数量:    {status['topic_domains']}")
    for d in status["domains"]:
        print(f"    ├─ [{d['id']}] {d['name']} ({d['entries']} 条)")

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## API 服务

> 以下文件位于项目根目录（非 `hierarchical_retrieval/` 包内），用于将检索引擎暴露为 HTTP 服务。

---

### api_server.py

> `api_server.py` — FastAPI HTTP 接口

```python
"""
分级检索 API 服务 (FastAPI)
将 HierarchicalRetrieval 暴露为 HTTP 接口
"""
import sys
import os
import logging
from pathlib import Path
from typing import Optional, List

# 确保能导入 hierarchical_retrieval
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from hierarchical_retrieval import HConfig, HierarchicalRetrieval

# ── 日志配置 ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI 应用 ──────────────────────────────────────
app = FastAPI(
    title="分级检索 API",
    description="基于 nomic-embed-text 的三级架构检索服务 (CloudCache + KeySentence + TopicDomain)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局配置 ──────────────────────────────────────────
STORAGE_ROOT = os.environ.get("HR_STORAGE_ROOT", "./hierarchical_api_storage")
OLLAMA_URL = os.environ.get("HR_OLLAMA_URL", "http://localhost:11434")

config = HConfig(
    storage_root=STORAGE_ROOT,
    embedding_base_url=OLLAMA_URL,
)

logger.info(f"[启动] 存储目录: {STORAGE_ROOT}")
logger.info(f"[启动] Embedding 服务: {OLLAMA_URL}")

# 初始化检索引擎（全局单例）
hr = HierarchicalRetrieval(config=config)


# ── 请求/响应模型 ──────────────────────────────────────

class IngestRequest(BaseModel):
    text: str = Field(..., description="对话全文", example="今天讨论一下微服务架构方案...")
    metadata: Optional[dict] = Field(None, description="附加元数据")


class IngestResponse(BaseModel):
    entry_id: str
    domain_id: str
    domain_name: str
    sentence_ids: List[str]


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    top_k_sentences: Optional[int] = Field(None)
    top_k_context: Optional[int] = Field(None)


class RetrieveResponse(BaseModel):
    query: str
    detected_domain: Optional[dict] = None
    key_sentences: List[dict] = Field(default_factory=list)
    full_contexts: List[dict] = Field(default_factory=list)
    summary: str = ""


class HealthResponse(BaseModel):
    status: str
    ollama_url: str
    storage_root: str


# ── 路由 ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok",
        ollama_url=config.embedding_base_url,
        storage_root=STORAGE_ROOT,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_dialogue(req: IngestRequest):
    """
    写入一段对话到分级检索系统

    流程: 全文存储 → 话题分配 → 关键句提取 → 向量索引
    """
    try:
        result = hr.ingest(text=req.text, metadata=req.metadata)
        logger.info(f"[写入] entry_id={result['entry_id']} domain={result['domain_name']}")
        return IngestResponse(**result)
    except Exception as e:\n        logger.error(f"[写入] 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """
    分级检索主接口

    流程:
    1. 检测查询所属话题界域 (L3)
    2. 在界域内检索关键句 (L2)
    3. 关联完整上下文 (L1)
    4. 聚合返回结果
    """
    try:
        result = hr.retrieve(
            query=req.query,
            top_k_sentences=req.top_k_sentences,
            top_k_context=req.top_k_context,
        )
        logger.info(f"[检索] query={req.query[:30]}... 找到 {len(result['key_sentences'])} 个关键句")
        return RetrieveResponse(**result)
    except Exception as e:\n        logger.error(f"[检索] 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 启动入口 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
```

---

### start_api.bat

> `start_api.bat` — Windows 启动脚本

```batch
@echo off
echo ========================================
echo   分级检索 API 服务启动器
echo ========================================
echo.

REM 检查 Ollama 是否运行
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [警告] Ollama 未运行，请先执行: ollama serve
    echo [提示] 并确保已拉取 nomic-embed-text: ollama pull nomic-embed-text
    echo.
    pause
    exit /b 1
)

echo [OK] Ollama 服务正常
echo.

REM 启动 API 服务
echo [启动] 正在启动 API 服务 http://localhost:8000 ...
echo [文档] Swagger UI: http://localhost:8000/docs
echo [文档] ReDoc: http://localhost:8000/redoc
echo.

python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

pause
```

---

### start_hr_api_hidden.vbs

> `start_hr_api_hidden.vbs` — Windows 静默启动脚本（自动拉起 Ollama）

```vbscript
' HR API + Ollama autostart (hidden)
Set ws = CreateObject("WScript.Shell")

Function IsRunning(procName)
  Dim svc
  IsRunning = False
  Set svc = GetObject("winmgmts:\\.\root\cimv2")
  If svc.ExecQuery("SELECT Name FROM Win32_Process WHERE Name='" & procName & "'").Count > 0 Then IsRunning = True
End Function

If Not IsRunning("ollama.exe") Then
  ws.Run """%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"" serve", 0, False
  WScript.Sleep 8000
End If

ws.Run "cmd.exe /c cd /d ""%USERPROFILE%\Desktop\hierarchical_retrieval"" && ""C:\Program Files\AutoClaw\resources\python\pythonw.exe"" -m uvicorn api_server:app --host 0.0.0.0 --port 8000", 0, False
```

---

## 附：配置参数速查

| 参数 | 默认值 | 中文推荐 | 说明 |
|------|--------|----------|------|
| `embedding_model` | `nomic-embed-text:latest` | 同默认 | Ollama 嵌入模型 |
| `embedding_dim` | `768` | 同默认 | 向量维度 |
| `embedding_base_url` | `http://localhost:11434` | 同默认 | Ollama 地址 |
| `storage_root` | `./hierarchical_storage` | — | 存储根目录 |
| `top_k_key_sentences` | `5` | `3` | 检索关键句数 |
| `top_k_full_context` | `3` | `2` | 关联全文段落数 |
| `topic_similarity_threshold` | `0.45` | `0.55` | 查询匹配话题阈值 |
| `topic_min_similarity` | `0.40` | `0.70` | 写入归并话题阈值 |
| `topic_max_domains` | `100` | 同默认 | 最大话题域数 |
| `max_key_sentences_per_chunk` | `3` | 同默认 | 每块关键句上限 |
| `key_sentence_max_length` | `128` | 同默认 | 关键句最大字符数 |

> **中文提示**：nomic-embed-text 对中文余弦相似度偏高，建议 `topic_min_similarity=0.70`、`topic_similarity_threshold=0.55`。

---

## 附：API 端点速查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/ingest` | 写入对话（全文→话题→关键句→索引） |
| `POST` | `/retrieve` | 分级检索（话题检测→关键句→全文关联） |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc 文档 |

---

*End of document.*
