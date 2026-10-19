# hierarchical-retrieval

> 分级检索开发工具包 — 全文缓存 + 关键句库 + 话题界域三级架构，本地嵌入（默认 `bge-m3`，1024 维）支持 **llama.cpp** 与 Ollama 双后端，在 prompt 体积有界的前提下扩展可检索的历史记忆容量。

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)]()

---

## 目录

- [30 秒理解](#30-秒理解)
- [背景与动机](#背景与动机)
- [架构总览](#架构总览)
- [安装与部署（llama.cpp 路线）](#安装与部署llamacpp-路线)
- [Ollama 路线（可选）](#ollama-路线可选)
- [配置参数](#配置参数)
- [API 参考](#api-参考)
- [检索流程详解](#检索流程详解)
- [实际效果](#实际效果)
- [项目结构](#项目结构)
- [进阶用法](#进阶用法)
- [安全注意事项](#安全注意事项)
- [常见问题](#常见问题)
- [Changelog](#changelog)
- [License](#license)

---

## 30 秒理解

你要做的一共三件事（Windows 为例，Linux/macOS 同理）：

```
① 起嵌入服务   llama-server -m bge-m3.gguf --embedding --pooling cls --port 8101
② 装 HR 并启动  pip install -e ".[api]" → uvicorn api_server:app --port 8000
③ 验证        curl http://127.0.0.1:8000/health → {"status":"ok",...}
```

不想读长文档的话，跟着 [快速落地（5 分钟）](#快速落地5-分钟) 走一遍即可。

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

## 架构总览

```
                        查询 (Query)
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│  L3  话题界域 (TopicDomainManager)                          │
│  ── 检测查询所属话题，缩小检索范围                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ 话题域 A    │  │ 话题域 B    │  │ 话题域 C    │  ...      │
│  └─────┬──────┘  └────────────┘  └────────────┘           │
│        │ 命中域                                              │
└────────┼───────────────────────────────────────────────────┘
         ▼
┌───────────────────────────────────────────────────────────┐
│  L2  关键句库 (KeySentenceLibrary)                          │
│  ── 在命中域内检索最相关关键句                                 │
│  ┌──────────────────────────────────────────┐              │
│  │ "采用微服务架构..."  (score 0.74)          │              │
│  │ "数据库使用 PostgreSQL..."  (score 0.68)   │              │
│  │ "API Gateway 路由..."  (score 0.61)        │              │
│  └──────────────────┬───────────────────────┘              │
│                     │ 关键句关联 source_entry_id             │
└─────────────────────┼──────────────────────────────────────┘
         ▼
┌───────────────────────────────────────────────────────────┐
│  L1  全文缓存 (CloudCache，本地磁盘)                          │
│  ── 通过关键句关联到全文段落                                   │
│  ┌──────────────────────────────────────────┐              │
│  │ entry_id: cc_xxx  chunk: 0  全文段落...    │              │
│  │ entry_id: cc_yyy  chunk: 1  全文段落...    │              │
│  └──────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│  聚合结果：{ 话题域, 关键句列表, 全文段落, 摘要 }                │
└───────────────────────────────────────────────────────────┘
```

三层存储全部落在本地磁盘（JSON + FAISS），嵌入向量化由本地服务提供：**llama.cpp 的 llama-server**（推荐，进程轻、无守护进程、显存占用小）或 **Ollama**（老路线，继续兼容）。

---

## 安装与部署（llama.cpp 路线）

推荐路线：**llama.cpp + bge-m3 GGUF**。无守护进程、无模型仓库管理、单 exe 即服务；嵌入与对话可分别起进程，互不抢显存时可共存。

### 快速落地（5 分钟）

以下命令 Windows / Linux / macOS 通用（Windows 用 `pwsh` 或 PowerShell 均可）。

**第 1 步 · 拿到两个可执行文件与模型**

llama.cpp 官方发布页：<https://github.com/ggml-org/llama.cpp/releases> — 按平台下载预编译包（CUDA 版 `llama-*-bin-win-cuda12.4-x64.zip` 通用性最好；纯 CPU 用 `llama-*-bin-win-x64.zip`），解压即用，无需编译。

嵌入模型（bge-m3 GGUF，任选一个源）：

```bash
# 方式 A：huggingface（直链下载，约 1.2GB）
https://huggingface.co/bbvch-ai/bge-m3-GGUF/resolve/main/bge-m3-q8_0.gguf
# 国内镜像（hf-mirror）：
https://hf-mirror.com/bbvch-ai/bge-m3-GGUF/resolve/main/bge-m3-q8_0.gguf

# 方式 B：llama-server 直接拉（首次联网，之后有本地缓存）
llama-server --hf-repo bbvch-ai/bge-m3-GGUF --hf-file bge-m3-q8_0.gguf --embedding
```

> 量化档位说明：嵌入模型对精度敏感度低于对话模型，但为保险起见推荐 `q8_0` 或 F16；实测 q8_0 与原版余弦 > 0.9998。F16 约 2.3GB，q8_0 约 1.2GB，q4 档不推荐（检索质量会明显下降）。

**第 2 步 · 起嵌入服务（一条命令）**

```bash
llama-server -m /path/to/bge-m3-q8_0.gguf \
  --embedding --pooling cls \
  --host 127.0.0.1 --port 8101 \
  -c 8192 -ub 8192 --no-webui
```

参数逐条解释：

| 参数 | 作用 |
|------|------|
| `--embedding` | 开启嵌入模式（llama-server 出 `/v1/embeddings` 端点） |
| `--pooling cls` | bge 系模型的标准池化方式（与 Ollama/原版对齐） |
| `-c 8192` | 上下文窗口，决定单条文本最大长度（按块切分前） |
| `-ub 8192` | 物理批大小，设成与 `-c` 一致避免长文本报错 |
| `--no-webui` | 不起内置网页（省一点资源，不影响 API） |

验证：

```bash
curl http://127.0.0.1:8101/v1/models
# 应返回 {"models":[...]} 且 meta.n_embd = 1024
```

**第 3 步 · 装 HR 并启动 API 服务**

```bash
git clone https://github.com/suyunpledge/hierarchical-retrieval.git
cd hierarchical-retrieval
pip install -e ".[api]"

# 指向 llama.cpp 嵌入服务（关键是 HR_EMBED_URL 指到 8101）
set HR_EMBED_URL=http://127.0.0.1:8101
set HR_CHAT_MODEL=qwen3.5-9b            # 任意名字，llama.cpp 不校验

uvicorn api_server:app --host 127.0.0.1 --port 8000
```

**第 4 步 · 验证**

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","ollama_url":"http://127.0.0.1:8101","storage_root":"./hierarchical_api_storage","embedding_backend":"openai"}

# 写入 + 检索冒烟
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" ^
  -d "{\"text\": \"今天讨论技术架构。建议采用微服务，数据库用 PostgreSQL。\", \"metadata\": {\"source\": \"demo\"}}"

curl -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" ^
  -d "{\"query\": \"技术架构用了什么方案？\"}"
# key_sentences 应命中 "建议采用微服务..." 那句
```

### 对话模型（可选）

`/v1/chat/completions` 代理需要一个对话后端。llama.cpp 路线下另起一个 llama-server 加载对话模型即可：

```bash
llama-server -m /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -c 8192 --reasoning off --no-webui

# HR 侧把对话后端指过去（与嵌入服务分进程）
set HR_CHAT_BACKEND=http://127.0.0.1:8080
set HR_CHAT_MODEL=qwen3.5-9b
```

> 8GB 显存参考：bge-m3(F16) 独占约 1.6GB，Qwen3.5-9B(Q4) 约 6GB——两者同时常驻会挤满 8GB 卡，建议**嵌入服务常驻 + 对话服务按需起**，或对话模型换小一档（4B 级 Q4 约 3GB，可共存）。

### 常驻与开机自启（Windows）

把第 2 步命令存成 `start-embed-server.bat`（记得给路径），再建计划任务：

```bat
:: start-embed-server.bat
@echo off
C:\llama.cpp\llama-server.exe -m D:\models\bge-m3.gguf --embedding --pooling cls ^
  --host 127.0.0.1 --port 8101 -c 8192 -ub 8192 --no-webui
```

```powershell
# 管理员 PowerShell：登录自启（隐藏窗口，不弹终端）
schtasks /Create /TN "HREmbed-AutoStart" /SC ONLOGON /RL LIMITED `
  /TR "C:\Windows\System32\conhost.exe --headless C:\llama.cpp\start-embed-server.bat" /F
```

> 计划任务跑控制台程序必须套 `conhost.exe --headless`，否则 Win11 默认终端会闪窗。

---

## Ollama 路线（可选）

已部署 Ollama 的环境继续可用，代码自动探测后端（`HR_EMBED_BACKEND=auto`），无需改任何配置：

```bash
ollama pull bge-m3
ollama serve   # 默认 http://localhost:11434
```

HR 侧不设 `HR_EMBED_URL` 即默认指向 11434，自动识别为 Ollama 协议。

**两条路线的等价性**：同一 bge-m3 权重在 llama.cpp 与 Ollama 下产生的向量余弦 > 0.9999（实测 0.999991），FAISS 索引跨后端直接续用，**换后端不需要重建存储**。

**为什么要换 llama.cpp**：

| 维度 | llama.cpp llama-server | Ollama |
|------|----------------------|--------|
| 进程形态 | 单 exe，无守护进程 | 常驻服务 + 模型仓库管理 |
| 显存占用 | 只加载指定的那一个模型 | 驻留最近使用的模型 |
| 模型获取 | GGUF 文件即用（本地文件/直链/hf-repo） | 必须先 pull 进仓库（ blobs 目录） |
| 嵌入质量 | bge-m3 GGUF q8_0 ≈ 原版（cos>0.9998） | 同源模型，等价 |
| API | OpenAI 兼容（/v1/embeddings、/v1/chat/completions） | Ollama 协议（/api/embeddings、/api/chat） |
| 版本节奏 | 跟随上游，更新快 | 发行版滞后 |

---

## 配置参数

`HConfig` 全部参数按功能分组。中文场景推荐值已作为默认值（0.55 / 0.55 / 0.62 三档阈值按 bge-m3 中文分布实测选定）。

**Embedding 服务**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_backend` | `auto` | `auto` 自动探测；`openai` 强制 llama.cpp/OpenAI 协议；`ollama` 强制 Ollama 协议 |
| `embedding_model` | `bge-m3:latest` | 嵌入模型名（llama.cpp 下不校验，实际由 `-m` 启动参数决定） |
| `embedding_dim` | `1024` | 嵌入向量维度（bge-m3 输出维度，换模型时需同步改） |
| `embedding_base_url` | `http://localhost:11434` | 嵌入服务地址（llama.cpp 路线指到 llama-server 端口） |
| `embedding_request_timeout` | `60` | 请求超时（秒） |

**存储路径**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `storage_root` | `./hierarchical_storage` | 存储根目录 |
| `cloud_cache_dir` | `cloud_cache` | 全文缓存子目录 |
| `key_sentence_dir` | `key_sentence_lib` | 关键句库子目录 |
| `topic_domain_dir` | `topic_domains` | 话题界域子目录 |
| `vector_index_dir` | `vector_index` | 向量索引子目录 |

**检索参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k_key_sentences` | `5` | 每次检索返回的关键句数 |
| `top_k_full_context` | `3` | 每个关键句关联的全文段落数 |
| `key_sentence_similarity_threshold` | `0.55` | 关键句匹配相似度阈值 |
| `topic_similarity_threshold` | `0.55` | 查询匹配话题的相似度阈值 |
| `topic_min_similarity` | `0.62` | 写入时归入已有话题的最小相似度 |
| `topic_max_domains` | `100` | 最大话题域数量 |

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
| `chat_backend_url` | 同嵌入服务 | 代理转发目标（llama.cpp 或 Ollama 的对话服务） |
| `chat_model` | `glm4:9b` | 代理回退对话模型（llama.cpp 下任意名字均可） |

**性能与代理保护**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_workers` | `6` | 批量嵌入的并发线程数 |
| `max_ingest_per_request` | `32` | 代理单次请求最多补录的旧消息条数（未入库 ∩ 最旧优先） |

**环境变量（api_server 读取）**

| 变量 | 默认 | 说明 |
|------|------|------|
| `HR_STORAGE_ROOT` | `./hierarchical_api_storage` | 存储根目录 |
| `HR_EMBED_URL` | `HR_OLLAMA_URL` 或 `http://localhost:11434` | 嵌入服务地址 |
| `HR_EMBED_BACKEND` | `auto` | `auto` / `openai` / `ollama` |
| `HR_CHAT_BACKEND` | 同嵌入服务 | 对话后端地址（可与嵌入分开部署） |
| `HR_CHAT_BACKEND_OPENAI` | `auto` | 对话后端协议：`1`/`0` 强制，`auto` 跟随嵌入探测 |
| `HR_CHAT_MODEL` | `glm4:9b` | 代理回退对话模型 |
| `HR_API_TOKEN` | 空 | Bearer 鉴权（空 = 不鉴权，仅本机监听时风险可控） |

---

## API 参考

### `HierarchicalRetrieval` — 主入口

```python
hr = HierarchicalRetrieval(embedding=None, config=None)
```

| 方法 | 说明 |
|------|------|
| `ingest(text, metadata=None) -> dict` | 写入一段对话到三级存储，返回 `{entry_id, domain_id, domain_name, sentence_ids}` |
| `ingest_batch(conversations) -> list[dict]` | 批量写入 |
| `retrieve(query, top_k_sentences=None, top_k_context=None) -> dict` | 分级检索，返回 `{query, detected_domain, key_sentences, full_contexts, summary}` |
| `status() -> dict` | 返回系统状态（各层级数量、话题列表） |

### `NomicEmbedding` — 嵌入服务客户端

```python
emb = NomicEmbedding(config=None)
emb.embed("文本")        # -> np.ndarray (1024,)，已 L2 归一化
emb.embed_batch(["t1"])  # -> list[np.ndarray]
emb.backend              # -> "ollama" | "openai"（实际探测结果）
```

后端探测逻辑（`embedding_backend="auto"` 时）：

```
GET /api/tags   → 200 ? 是 → Ollama 协议
GET /v1/models  → 200 ? 是 → OpenAI 协议（llama.cpp llama-server 等）
两者均不可达    → 按 OpenAI 协议重试（请求时给出明确告警日志）
```

### `CloudCache` — L1 全文缓存

```python
cache = CloudCache(embedding, config=None)
cache.store(text, metadata=None)   # -> entry_id
cache.search(query, top_k=3)       # -> list[dict]
cache.has_content(text)            # -> bool（内容哈希去重查询）
cache.get_full_text(entry_id)      # -> str | None
cache.count()                      # -> int
```

### `KeySentenceLibrary` — L2 关键句库

```python
ks_lib = KeySentenceLibrary(embedding, config=None)
ks_lib.ingest(text, source_entry_id)  # -> list[dict]（含 sentence_id / text / vector，供界域索引复用）
ks_lib.search(query, top_k=5)          # -> list[dict]
ks_lib.get_sentence(sentence_id)       # -> dict | None
```

### `TopicDomainManager` — L3 话题界域

```python
tm = TopicDomainManager(embedding, config=None)
tm.assign_domain(text)              # -> TopicDomain（归入或新建）
tm.detect_domain(query)             # -> TopicDomain | None
tm.search_in_domain(query, domain)  # -> list[dict]
tm.list_domains()                   # -> list[TopicDomain]
```

### 存储与工具类

- `LocalStorage` — 基于 JSON/Pickle 的本地键值存储
- `VectorStore` — FAISS 向量索引封装，支持持久化
- `TextProcessor` — 句子分割、切块、清洗
- `KeySentenceExtractor` — 启发式关键句提取

---

## 检索流程详解

```
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
```

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

```
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
    │   ├── embedding.py        # NomicEmbedding — 双后端嵌入客户端（llama.cpp/Ollama）
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
```

---

## 进阶用法

### 自定义嵌入服务

```python
from hierarchical_retrieval import HConfig

# 换其他 bge 系模型（维度不同时必须同步改）
config = HConfig(
    embedding_model="bge-large-zh",
    embedding_dim=1024,
)

# 显式指定后端（跳过探测）
config = HConfig(
    embedding_base_url="http://127.0.0.1:8101",
    embedding_backend="openai",
)
```

### 直接操作各层级

```python
from hierarchical_retrieval import HConfig, NomicEmbedding, CloudCache

emb = NomicEmbedding(HConfig(embedding_base_url="http://127.0.0.1:8101"))
cache = CloudCache(emb)

# 手动存全文
entry_id = cache.store("长对话全文...", metadata={"date": "2026-08-06"})

# 按需检索
results = cache.search("相关问题", top_k=5)
```

### 批量写入

```python
conversations = [
    {"text": "对话1...", "metadata": {"src": "a"}},
    {"text": "对话2...", "metadata": {"src": "b"}},
]
hr.ingest_batch(conversations)
```

### 查看系统状态

```python
status = hr.status()
# {
#     "embedding_model": "bge-m3:latest",
#     "cloud_cache_entries": 4,
#     "key_sentence_count": 12,
#     "topic_domains": 3,
#     "domains": [{"id": "td_xxx", "name": "...", "entries": 2}, ...]
# }
```

### 从 Ollama 迁移到 llama.cpp

存量存储无需重建（向量等价，余弦 > 0.9999）：

1. 起 llama-server（`--embedding --pooling cls`，加载同一 bge-m3 GGUF）；
2. `HR_EMBED_URL` 指到新端口（如 8101）；
3. 重启 api_server。`/health` 的 `embedding_backend` 应变为 `openai`。
4. Ollama 可卸载，模型 blobs 可删（`~/.ollama/models` 目录，先确认无其他模型依赖）。

---

## 安全注意事项

`/v1` 对话代理会接收并缓存完整对话内容，默认部署形态按"仅本机使用"设计（v0.3 起对齐实现）：

- **默认仅本机监听**：api_server 与 llama-server 均默认 `127.0.0.1`；
- **可选 Bearer 鉴权**：设置环境变量 `HR_API_TOKEN=<密钥>` 后，除 `/health` 外的接口均要求 `Authorization: Bearer <密钥>` 请求头。LobeChat 接入时把"API Key"填成同一个值即可；不设置则不鉴权（仅本机监听时风险可控）；
- **如需局域网/远程访问**：把 `--host` 改为 `0.0.0.0`，**并务必同时设置 `HR_API_TOKEN`**——代理转发含全部对话明文，不应无鉴权暴露到非本机网络；
- **跨域**：CORS 保持 `allow_origins=["*"]` 但不带凭证；
- **数据落盘位置**：全部数据（全文、关键句、向量索引）写在本地磁盘 `storage_root` 目录下，不上传任何远端服务。

---

## 常见问题

**Q: 启动时报 "无法探测 Embedding 服务"？**

A: llama.cpp 路线确认 llama-server 已带 `--embedding` 启动且 `/v1/models` 可达（curl 一下）；Ollama 路线确认 `ollama serve` 在跑。`HR_EMBED_URL` 端口写错是最常见原因。

**Q: llama.cpp 下 embedding_backend 一直显示 openai，我明明想用 Ollama？**

A: 这就是正确的：`openai` 指 OpenAI 兼容协议（llama.cpp），`ollama` 指 Ollama 协议。auto 探测按 `/api/tags` 是否 200 判定。想强制可设 `HR_EMBED_BACKEND=ollama`。

**Q: 所有对话被归入同一话题？**

A: 通常是相似度阈值过低。默认阈值（`topic_min_similarity` = 0.62）按 bge-m3 中文分布选定；若仍出现误并，可继续调高该值。

**Q: 检索结果关键句为空？**

A: 检查是否写入了数据（`hr.status()`），并确认查询与已存内容语义相关。无匹配话题时会自动降级为全局检索。

**Q: 如何重置所有数据？**

A: 删除 `storage_root` 目录（默认 `./hierarchical_storage` 或 api_server 的 `./hierarchical_api_storage`）。

**Q: 支持 GPU 加速吗？**

A: 支持。faiss 侧装 `faiss-gpu` 替代 `faiss-cpu`；推理侧 llama.cpp 直接下 CUDA 预编译包即可（Ollama 同理自动用 GPU）。

**Q: 显存不够同时跑嵌入 + 对话模型怎么办？**

A: 嵌入服务常驻（bge-m3 约 1.6GB 显存），对话模型按需起；或对话模型换小一档（4B Q4 约 3GB）；或对话后端指到另一台机器/CPU 推理（`HR_CHAT_BACKEND` 独立配置）。

---

## Changelog

- **0.6.0（2026-10-19）** — llama.cpp 后端支持：`NomicEmbedding` 双后端自动探测（llama.cpp /v1/embeddings + Ollama /api/embeddings）；api_server 对话代理支持 OpenAI 协议转发（llama.cpp /v1/chat/completions，含流式 SSE）；新增 `HR_EMBED_URL` / `HR_EMBED_BACKEND` / `HR_CHAT_BACKEND_OPENAI` 环境变量；`/health` 暴露 `embedding_backend`。实测 Ollama↔llama.cpp 向量余弦 0.999991，存量存储无缝迁移。
- **0.5.0** — bge-m3 迁移：嵌入模型 nomic-embed-text(768d) → bge-m3(1024d)，阈值按中文分布重定（0.55/0.55/0.62），索引重建。
- **0.4.0** — 性能与代理修复（B1 补录窗口 / M5.1 截断装配 / M7 先过滤后补 / N-perf 增量落盘）。
- **0.3.0** — 安全对齐（默认 127.0.0.1 / Bearer 鉴权 / CORS 规范化）。
- **0.2.0** — 首个公开版本：三级架构 + Ollama 嵌入 + OpenAI 兼容代理。

---

## License

[MIT](LICENSE)
