# hierarchical-retrieval

> 分级检索开发工具包 — 全文缓存 + 关键句库 + 话题界域三级架构，基于 Ollama 本地嵌入（默认 `bge-m3`，1024 维），在 prompt 体积有界的前提下扩展可检索的历史记忆容量。

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)]()

---

## 目录

- [背景与动机](#背景与动机)
- [核心思想：分级检索](#核心思想分级检索)
- [架构总览](#架构总览)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置参数](#配置参数)
- [API 参考](#api-参考)
- [检索流程详解](#检索流程详解)
- [实际效果](#实际效果)
- [项目结构](#项目结构)
- [进阶用法](#进阶用法)
- [安全注意事项](#安全注意事项)
- [常见问题](#常见问题)
- [License](#license)

---

## 背景与动机

当前大模型（如 DeepSeek）在长上下文记忆方面已有显著进展，1M 上下文是一大优势，但仍面临若干根本性难题：

| 问题 | 表现 |
|------|------|
| **全文通记导致遗忘** | 超出上下文限制后信息丢失 |
| **Token 浪费** | 简单分析与重复问答消耗大量记忆 Token |
| **细节丢失与召回困难** | 长文本记忆存在信息衰减 |
| **话题干扰** | 无关话题插入导致召回混乱 |

**分级检索**通过将信息分层存储、按话题隔离检索，将"上下文长度"聚焦在当前话题对应的关键句库中发挥，从而在不改变模型本身的前提下，扩展可检索的历史记忆容量。

> 量化口径（v0.3 起）：本方案对记忆的收益取决于**检索命中率**，需按实际语料评测后给出数字，文档不再引用未经推导与实测的倍数估算（v0.2 曾宣称"1M 可当 100M"，无依据，已删除）。可验证的硬指标是 `/v1` 对话代理的 prompt 体积**结构性有界**——头部提示 + 记忆块（≤ `context_char_budget`）+ 近 K 条消息（总量同样裁进预算，单条超长消息截断装配），与历史长度无关。

---

## 核心思想：分级检索

分级检索由三部分组成：

### 1. 全文缓存（基础信息库，本地磁盘）
将**所有对话信息一字不漏地缓存到本地磁盘**（`CloudCache`，类名沿用历史命名），作为基础信息库。在检索管线中它是**详情层**：由命中的关键句按 `source_entry_id` 关联回原文段落；也可以通过 `cache.search()` 直接做全库语义检索。自动检索管线中的"L1 兜底召回"已列入路线图（见修复说明-v0.3 的遗留项）。

### 2. 关键句库（二级信息库）
AI 在对话中将内容**提炼为精简的关键句**，在独立空间中存储。检索时先命中关键句，再根据关键句在全文缓存中查找详细对话，避免全文扫描。

### 3. 话题界域
AI 将对话内容**划分为不同的话题界域**，按界域分配关键句库。检索时先确定当前查询所处的话题界域，再通过该界域的关键句库与全文缓存进行精准关联。

这样，"上下文长度"只在**指定话题界域对应的关键句库**中发挥作用，无关信息被天然隔离，从而实现大容量信息记忆。

---

## 架构总览

┌─────────────────────────────────────────────────────────────┐
│                     查询 (Query)                             │
└──────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  L3  话题界域 (TopicDomainManager)                           │
│  ── 检测查询所属话题，缩小检索范围                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │ 话题域 A    │ │ 话题域 B    │ │ 话题域 C    │  ...         │
│  └─────┬──────┘ └────────────┘ └────────────┘              │
│        │ 命中域                                                │
└────────┼────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  L2  关键句库 (KeySentenceLibrary)                           │
│  ── 在命中域内检索最相关关键句                                  │
│  ┌──────────────────────────────────────────┐               │
│  │ "采用微服务架构..."  (score 0.74)         │               │
│  │ "数据库使用 PostgreSQL..."  (score 0.68)  │               │
│  │ "API Gateway 路由..."  (score 0.61)       │               │
│  └──────────────────┬───────────────────────┘               │
│                     │ 关键句关联 source_entry_id             │
└─────────────────────┼───────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  L1  全文缓存 (CloudCache，本地磁盘)                          │
│  ── 通过关键句关联到全文段落                                   │
│  ┌──────────────────────────────────────────┐               │
│  │ entry_id: cc_xxx  chunk: 0  全文段落...   │               │
│  │ entry_id: cc_yyy  chunk: 1  全文段落...   │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  聚合结果：{ 话题域, 关键句列表, 全文段落, 摘要 }               │
└─────────────────────────────────────────────────────────────┘

---

## 特性

- **三级分级检索** — 全文缓存 → 关键句库 → 话题界域，逐层收敛检索范围
- **bge-m3 集成** — 通过 Ollama API 调用，向量自动 L2 归一化（内积 = 余弦相似度）
- **FAISS 向量索引** — 基于 `IndexFlatIP`，支持持久化到磁盘，进程重启自动恢复
- **话题自动聚类** — 新内容自动归入已有话题或创建新话题，话题向量平滑更新
- **关键句启发式提取** — 基于信号词权重、位置、长度的轻量提取，无需额外 LLM 调用
- **降级检索** — 无匹配话题时自动降级为全局关键句检索，尽量弥补召回缺口
- **全量持久化** — 所有数据写入本地磁盘，零外部服务依赖（除 Ollama 外）
- **可配置阈值** — 话题归并/匹配阈值、top-k 等参数全部可调

---

## 安装

### 前置环境

| 步骤 | 说明 | 命令 |
|------|------|------|
| 1. Python | 需要 3.9+ | `python --version` |
| 2. 拉取模型 | 下载 bge-m3 嵌入模型（默认） | `ollama pull bge-m3` |
| 3. 启动 Ollama | 默认监听 `http://localhost:11434` | `ollama serve` |

### 安装本包

| 方式 | 说明 | 命令 |
|------|------|------|
| 开发模式 | 源码可编辑安装（推荐开发） | `pip install -e .` |
| 普通安装 | 构建后安装到 site-packages | `pip install .` |
| GPU 加速 | 用 `faiss-gpu` 替代 `faiss-cpu` | `pip install -e ".[gpu]"` |
| API 服务 | 附带 FastAPI / uvicorn / pydantic | `pip install -e ".[api]"` |
| 开发依赖 | 附带 pytest / build / twine | `pip install -e ".[dev]"` |

### 运行时依赖

| 依赖 | 用途 |
|------|------|
| `numpy` | 向量运算 |
| `requests` | 调用 Ollama API |
| `faiss-cpu` | 向量近似最近邻索引 |

---

## 快速开始

from hierarchical_retrieval import HConfig, HierarchicalRetrieval
1. 初始化（自动连接 Ollama）
hr = HierarchicalRetrieval(config=HConfig(storage_root="./my_store"))
2. 写入对话（自动: 全文缓存 + 关键句提取 + 话题分配）
hr.ingest(
"今天讨论技术架构。建议采用微服务，数据库用 PostgreSQL。"
"最终决定下周开始搭建框架。",
metadata={"source": "arch_meeting"},
)
3. 检索
result = hr.retrieve("技术架构用了什么方案？")
print(result["summary"])
print(result["key_sentences"])    # 关键句列表
print(result["full_contexts"])    # 关联的全文段落

### 运行示例

python hierarchical_retrieval/examples/demo.py

---

## 配置参数

`HConfig` 全部参数，按功能分组。第三列为中文场景的推荐值（默认值适合英文，中文需调高话题阈值）。

**Embedding 服务**

| 参数 | 默认值 | 中文推荐 | 说明 |
|------|--------|----------|------|
| `embedding_model` | `bge-m3:latest` | 同默认 | Ollama 嵌入模型名 |
| `embedding_dim` | `1024` | 同默认 | 嵌入向量维度 |
| `embedding_base_url` | `http://localhost:11434` | 同默认 | Ollama 服务地址 |
| `embedding_request_timeout` | `60` | 同默认 | 请求超时（秒） |

**存储路径**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `storage_root` | `./hierarchical_storage` | 存储根目录 |
| `cloud_cache_dir` | `cloud_cache` | 全文缓存子目录 |
| `key_sentence_dir` | `key_sentence_lib` | 关键句库子目录 |
| `topic_domain_dir` | `topic_domains` | 话题界域子目录 |
| `vector_index_dir` | `vector_index` | 向量索引子目录 |

**检索参数**

| 参数 | 默认值 | 中文推荐 | 说明 |
|------|--------|----------|------|
| `top_k_key_sentences` | `5` | `3` | 每次检索返回的关键句数 |
| `top_k_full_context` | `3` | `2` | 每个关键句关联的全文段落数 |
| `key_sentence_similarity_threshold` | `0.55` | 同默认 | 关键句匹配相似度阈值（bge-m3 分布实测选值） |

**话题界域**

| 参数 | 默认值 | 中文推荐 | 说明 |
|------|--------|----------|------|
| `topic_similarity_threshold` | `0.55` | 同默认 | 查询匹配话题的相似度阈值（bge-m3 分布实测选值） |
| `topic_min_similarity` | `0.62` | 同默认 | 写入时归入已有话题的最小相似度（bge-m3 分布实测选值） |
| `topic_max_domains` | `100` | 同默认 | 最大话题域数量 |

**关键句提取**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_key_sentences_per_chunk` | `3` | 每块提取的最大关键句数 |
| `key_sentence_max_length` | `128` | 单个关键句最大字符数 |

**上下文装配与代理（`/v1` 对话代理使用）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `context_char_budget` | `3000` | 记忆块/近期对话的字符预算（超预算从低分项丢弃，单条超长消息截断装配） |
| `recent_turns_keep` | `6` | 近期原样保留的**消息条数**（约 3 轮对话） |
| `compress_threshold_chars` | `3000` | 历史总字符超过该值才触发代理压缩 |
| `chat_backend_url` | `http://localhost:11434` | 代理转发目标（本地 Ollama） |
| `chat_model` | `glm4:9b` | 代理回退对话模型 |

**性能与代理保护**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_workers` | `6` | 批量嵌入的并发线程数（Ollama 可并发） |
| `max_ingest_per_request` | `32` | 代理单次请求最多补录的旧消息条数（未入库 ∩ 最旧优先） |

**其他**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_level` | `INFO` | 日志级别 |

> **中文调参提示**：默认嵌入为 bge-m3（1024 维），默认阈值（0.55 / 0.55 / 0.62）按其中文分布实测选定，通常可直接使用；若你的语料出现话题误并或漏召，可再微调这两个阈值。可直接复制：

config = HConfig(
storage_root="./my_store",
topic_min_similarity=0.62,       # 写入归并阈值（bge-m3 实测选值）
topic_similarity_threshold=0.55, # 查询匹配阈值（bge-m3 实测选值）
top_k_key_sentences=3,
top_k_full_context=2,
)

---

## API 参考

### `HierarchicalRetrieval` — 主入口

hr = HierarchicalRetrieval(embedding=None, config=None)

| 方法 | 说明 |
|------|------|
| `ingest(text, metadata=None) -> dict` | 写入一段对话到三级存储，返回 `{entry_id, domain_id, domain_name, sentence_ids}` |
| `ingest_batch(conversations) -> list[dict]` | 批量写入 |
| `retrieve(query, top_k_sentences=None, top_k_context=None) -> dict` | 分级检索，返回 `{query, detected_domain, key_sentences, full_contexts, summary}` |
| `status() -> dict` | 返回系统状态（各层级数量、话题列表） |

### `NomicEmbedding` — 嵌入服务

emb = NomicEmbedding(config=None)
emb.embed("文本")        # -> np.ndarray (1024,)
emb.embed_batch(["t1"])  # -> list[np.ndarray]

### `CloudCache` — L1 全文缓存

cache = CloudCache(embedding, config=None)
cache.store(text, metadata=None)   # -> entry_id
cache.search(query, top_k=3)       # -> list[dict]
cache.has_content(text)             # -> bool（内容哈希去重查询）
cache.get_full_text(entry_id)      # -> str | None
cache.count()                       # -> int

### `KeySentenceLibrary` — L2 关键句库

ks_lib = KeySentenceLibrary(embedding, config=None)
ks_lib.ingest(text, source_entry_id)  # -> list[dict]（含 sentence_id / text / vector，供界域索引复用）
ks_lib.search(query, top_k=5)          # -> list[dict]
ks_lib.get_sentence(sentence_id)       # -> dict | None

### `TopicDomainManager` — L3 话题界域

tm = TopicDomainManager(embedding, config=None)
tm.assign_domain(text)              # -> TopicDomain（归入或新建）
tm.detect_domain(query)             # -> TopicDomain | None
tm.search_in_domain(query, domain)  # -> list[dict]
tm.list_domains()                   # -> list[TopicDomain]

### 存储与工具类

- `LocalStorage` — 基于 JSON/Pickle 的本地键值存储
- `VectorStore` — FAISS 向量索引封装，支持持久化
- `TextProcessor` — 句子分割、切块、清洗
- `KeySentenceExtractor` — 启发式关键句提取

---

## 检索流程详解

输入: query
│
├─ Step 1: 话题检测 (L3)
│    将 query 向量化，与所有话题域的代表向量计算余弦相似度
│    ├─ 命中（≥ topic_similarity_threshold）→ 进入该域
│    └─ 未命中 → 降级为全局关键句检索
│
├─ Step 2: 界域内关键句检索 (L2)
│    在命中域的独立向量空间中检索 top_k_key_sentences 条关键句
│    先按 key_sentence_similarity_threshold 过滤低分噪声；
│    过滤后不足 top_k 条时补充全局关键句检索（同样过滤）并去重
│
├─ Step 3: 关键句 → 全文关联 (L1)
│    对每条关键句的 source_entry_id，在全文缓存中检索全文段落
│    去重后截取 top_k_full_context * top_k_key_sentences 段
│
└─ Step 4: 聚合
生成 summary 文本，返回完整结果字典

---

## 实际效果

以 4 段不同主题的对话（技术架构 / 预算 / 上线计划 / 架构补充）为例：

**写入结果** — 自动形成 3 个话题界域：
- 技术架构域（2 条，架构方案 + 架构补充自动归并）
- 预算域（1 条）
- 上线计划域（1 条）

**检索结果**：

| 查询 | 命中话题 | Top1 关键句 |
|------|----------|-------------|
| 技术架构方案/数据库选型 | 技术架构域 | "刚才讨论的技术架构方案，我补充一下细节" |
| Q3 预算/GPU 算力 | 预算域 | "关于预算问题，Q3 的研发预算需要重新规划" |
| 上线时间/安全审计 | 上线计划域 | "建议安排专人跟进每个里程碑的进度" |
| API Gateway 选型 | 技术架构域 | "API Gateway 选用 Kong 或者 APISIX..." |

每个查询都正确路由到对应话题界域，关键句精准命中，全文段落成功关联。

---

## 项目结构

hierarchical-retrieval/
├── pyproject.toml              # 打包配置（PEP 517/518）
├── README.md                   # 本文档
├── LICENSE                     # MIT
├── .gitignore
│
└── hierarchical_retrieval/     # 主包
├── __init__.py             # 公共 API 导出
├── config.py               # HConfig 全局配置
├── py.typed                # PEP 561 类型标记
│
├── core/                   # 核心引擎
│   ├── embedding.py        # NomicEmbedding — Ollama 嵌入客户端
│   ├── cloud_cache.py      # CloudCache — L1 全文缓存
│   ├── key_sentence.py     # KeySentenceLibrary — L2 关键句库
│   ├── topic_domain.py     # TopicDomainManager — L3 话题界域
│   └── retrieval.py        # HierarchicalRetrieval — 三级流水线
│
├── storage/                # 存储层
│   ├── base.py             # StorageBase 抽象接口
│   ├── local_storage.py    # LocalStorage 本地键值存储
│   └── vector_store.py     # VectorStore FAISS 索引
│
├── utils/                  # 工具
│   ├── text_processor.py   # 文本分割/清洗
│   └── summarizer.py       # 关键句启发式提取
│
└── examples/
└── demo.py             # 完整使用示例

---

## 进阶用法

### 自定义嵌入服务

from hierarchical_retrieval import HConfig
使用其他 Ollama 嵌入模型
config = HConfig(
embedding_model="mxbai-embed-large:latest",
embedding_dim=1024,
)

### 直接操作各层级

from hierarchical_retrieval import HConfig, NomicEmbedding, CloudCache
emb = NomicEmbedding(HConfig())
cache = CloudCache(emb)
手动存全文
entry_id = cache.store("长对话全文...", metadata={"date": "2026-08-06"})
按需检索
results = cache.search("相关问题", top_k=5)

### 批量写入

conversations = [
{"text": "对话1...", "metadata": {"src": "a"}},
{"text": "对话2...", "metadata": {"src": "b"}},
]
hr.ingest_batch(conversations)

### 查看系统状态

status = hr.status()
{
"embedding_model": "bge-m3:latest",
"cloud_cache_entries": 4,
"key_sentence_count": 12,
"topic_domains": 3,
"domains": [{"id": "td_xxx", "name": "...", "entries": 2}, ...]
}

---

## 安全注意事项

`/v1` 对话代理会接收并缓存完整对话内容，默认部署形态按"仅本机使用"设计（v0.3 起对齐实现）：

- **默认仅本机监听**：三个启动方式（`start_api.bat`、`start_hr_api_hidden.vbs`、计划任务 `hr-api-task.xml`）的 `--host` 均已默认 `127.0.0.1`，不再绑定 `0.0.0.0`；
- **可选 Bearer 鉴权**：设置环境变量 `HR_API_TOKEN=<你的密钥>` 后，除 `/health` 外的接口均要求 `Authorization: Bearer <你的密钥>` 请求头。LobeChat 接入时把"API Key"填成同一个值即可；不设置则不鉴权（仅本机监听时风险可控）；
- **如需局域网/远程访问**：把启动参数 `--host` 改为 `0.0.0.0`，**并务必同时设置 `HR_API_TOKEN`**——代理转发含全部对话明文，不应无鉴权暴露到非本机网络；
- **跨域**：CORS 保持 `allow_origins=["*"]` 但不带凭证（v0.3 删去了违反规范的 `allow_credentials` 组合）；
- **数据落盘位置**：全部数据（全文、关键句、向量索引）写在本地磁盘 `storage_root` 目录下，不上传任何远端服务。

---

## 常见问题

**Q: 启动时报 "无法连接到 Ollama 服务"？**

A: 确保已运行 `ollama serve`，默认地址 `http://localhost:11434` 可达。若 Ollama 在其他地址，通过 `HConfig(embedding_base_url="http://your-host:11434")` 配置。

**Q: 所有对话被归入同一话题？**

A: 通常是相似度阈值过低。默认阈值（`topic_min_similarity` = 0.62）按 bge-m3 中文分布选定；若仍出现误并，可继续调高该值。

**Q: 检索结果关键句为空？**

A: 检查是否写入了数据（`hr.status()`），并确认查询与已存内容语义相关。无匹配话题时会自动降级为全局检索。

**Q: 如何重置所有数据？**

A: 删除 `storage_root` 目录（默认 `./hierarchical_storage` 或 demo 的 `./demo_storage`）。

**Q: 支持 GPU 加速吗？**

A: 支持。安装 `faiss-gpu` 替代 `faiss-cpu`：`pip install -e ".[gpu]"`。

---

## License

[MIT](LICENSE)