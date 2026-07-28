# voyage_trace

> 中文 | [English](README.md)

一个元智能体（meta-agent）系统，用于收集其他智能体的执行追踪数据，并生成治理与优化方案。基于 [deepagents](https://github.com/langchain-ai/deepagents) 扩展机制构建。

## 特性

- **后端无关的追踪协议** — 统一的 `CanonicalTrace` / `TraceSpan` 规范模式，将任意可观测性后端的追踪数据标准化。
- **六种源适配器** — 内置支持 LangSmith、Langfuse、OpenTelemetry GenAI、A2A、MCP 以及原始/半结构化数据。
- **Markdown 执行图** — 从追踪数据生成 Git 可 diff、GitHub 可原生渲染的执行图（YAML 前置元数据 + Mermaid `flowchart TD` + 节点统计表）。
- **确定性回放与假设模拟** — 使用记录的 I/O 作为磁带回放追踪；在应用修改（模型替换、循环上限、节点删除）前投影其效果。
- **可插拔工作区存储** — `WorkspaceStorage` 抽象基类，提供真实内存后端和生产级 Postgres 后端（`psycopg` v3 + 异步连接池）。
- **分区记忆系统** — 四类记忆分区（情景、语义、程序、工作），按目标智能体 + 治理轮次隔离，支持动态插拔与跨轮次召回。
- **deepagents 原生** — 仅通过 deepagents 公共扩展点（`BackendProtocol`、中间件、工具、子智能体）进行扩展，不 vendor 或 fork 内部代码。

## 安装

```bash
pip install voyage-trace
```

开发环境安装：

```bash
pip install -e ".[test]"
```

### 环境要求

- Python >= 3.11
- `deepagents >= 0.6.0`
- `psycopg[binary] >= 3.2`（Postgres 存储所需）
- `pyyaml >= 6.0`

## 快速开始

### 1. 从任意后端适配追踪数据

```python
from voyage_trace.adapters import adapt

# LangSmith、Langfuse、OTel、A2A、MCP 或原始数据 — 自动检测
trace = adapt(raw_payload, source_protocol="langsmith")

# 或让适配器从 payload 形状自动推断协议
trace = adapt(raw_payload)
```

### 2. 构建执行图

```python
from voyage_trace.execution_graph import build_execution_graph, render_markdown

graph = build_execution_graph(trace)
markdown_doc = render_markdown(graph)
print(markdown_doc)  # 在 GitHub 上原生渲染
```

### 3. 回放或模拟

```python
from voyage_trace.simulator import replay, simulate, Modification

# 使用记录的 I/O 进行确定性回放
result = replay(trace)
print(f"回放了 {len(result.steps)} 步, OK={result.ok}")

# 假设分析：投影替换为更便宜模型的效果
modified = simulate(trace, [
    Modification(target_node_id="chat-step", kind="swap_model",
                 params={"cost_multiplier": 0.3, "token_multiplier": 0.8})
])
```

### 4. 使用分区记忆

```python
from voyage_trace.memory import PartitionedMemory
from voyage_trace.storage import InMemoryStorage

storage = InMemoryStorage()
pm = PartitionedMemory(storage)

async with pm.use("agent-A", "round-1"):
    await pm.episodic().remember(
        pm.current(), "f1",
        {"trace_id": "t1", "agent_id": "agent-A",
         "failure_signature": "loop:web_search", "outcome": "capped"}
    )

# 在后续轮次中跨轮次召回
hits = await pm.recall_cross_round("agent-A", "loop:web_search")
```

### 5. Postgres 存储

```python
from voyage_trace.storage import PostgresStorage

storage = PostgresStorage(
    "host=127.0.0.1 port=5432 dbname=voyage user=voyage password=voyage"
)
# 首次使用时自动创建表结构
```

## 文档

| 文档 | 描述 |
|------|------|
| [架构设计](docs/architecture_zh-CN.md) | 系统设计、模块分解与数据流 |
| [追踪协议](docs/protocol_zh-CN.md) | 规范模式、`dotted_order`、不变量与适配器映射规则 |
| [使用指南](docs/usage_zh-CN.md) | 适配器、执行图、模拟、记忆与存储的详细示例 |

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    源后端                                │
│  LangSmith · Langfuse · OTel · A2A · MCP · Raw          │
└──────────────────────┬──────────────────────────────────┘
                       │ adapt()
                       ▼
┌─────────────────────────────────────────────────────────┐
│              CanonicalTrace (协议)                       │
│  TraceSpan[] · dotted_order · 不变量强制校验             │
└──────┬──────────────┬───────────────┬───────────────────┘
       │              │               │
       ▼              ▼               ▼
┌────────────┐ ┌────────────┐ ┌────────────────┐
│ 执行图     │ │ 模拟器     │ │  记忆          │
│ (Markdown) │ │ 回放/      │ │  情景          │
│            │ │ 模拟       │ │  语义          │
│            │ │            │ │  程序          │
│            │ │            │ │  工作          │
└─────┬──────┘ └─────┬──────┘ └───────┬────────┘
      │              │                │
      ▼              ▼                ▼
┌─────────────────────────────────────────────────────────┐
│              WorkspaceStorage (抽象基类)                 │
│        InMemoryStorage  ·  PostgresStorage               │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
src/voyage_trace/
├── types.py              # 核心领域类型（TraceSpan, CanonicalTrace, 枚举）
├── protocol.py           # JSON 序列化, dotted_order, 不变量强制校验
├── adapters/             # 源协议适配器（LangSmith, Langfuse, OTel, A2A, MCP, raw）
├── execution_graph.py    # Markdown 执行图（构建, 聚合, 渲染, 解析）
├── simulator.py          # 确定性回放 + 假设模拟
├── storage/              # 工作区存储（抽象基类, 内存, Postgres, BackendProtocol 桥接）
└── memory/               # 四类分区记忆 + 管理器
```

## 许可证

MIT
