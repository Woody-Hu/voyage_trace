---
session_id: 0005
title: 轻量重构（去重 / 死代码清理 / 文档同步）+ trace 信号丢失 bug 修复 + DeepSeek e2e 复验
date: 2026-08-06
status: implemented
research-kind: closed-loop-verification
files_touched:
  # 轻量重构：抽取共享 helper，消除跨模块重复
  - src/voyage_trace/_internal.py                  # 新增：utcnow / new_id / dt_to_str / dt_from_str
  - src/voyage_trace/types.py                      # 改：移除本地 _utcnow，从 _internal 导入
  - src/voyage_trace/protocol.py                   # 改：移除本地 _dt_to_str/_str_to_dt，从 _internal 导入
  - src/voyage_trace/analysis.py                   # 改：移除本地 _utcnow/_new_id/_dt_to_str/_dt_from_str
  - src/voyage_trace/verification.py               # 改：移除本地 helpers + 不可达 else 分支
  - src/voyage_trace/execution_graph.py            # 改：移除死正则 _NODE_DEF_RE，修过时 docstring
  - src/voyage_trace/automl.py                     # 改：移除死常量 TARGETS，traces 只聚合一次
  - src/voyage_trace/storage/in_memory.py          # 改：用 _internal.utcnow
  - src/voyage_trace/storage/postgres.py           # 改：移除死 datetime import
  - src/voyage_trace/adapters/otel.py              # 改：用共享 _otel_status_code
  - src/voyage_trace/adapters/mcp.py               # 改：同上
  - src/voyage_trace/memory/base.py                # 改：recall/search/forget 上提为默认实现 + _query_records
  - src/voyage_trace/memory/episodic.py            # 改：移除重复 recall/search/forget
  - src/voyage_trace/memory/procedural.py          # 改：同上
  - src/voyage_trace/memory/working.py             # 改：同上
  - src/voyage_trace/memory/semantic.py            # 改：search 用 _query_records
  - src/voyage_trace/memory/manager.py             # 改：移除未用的 self.partitions
  - src/voyage_trace/integrations/langfuse_export.py  # 改：移除死函数 export_observation_now/parse_langfuse_datetime
  - src/voyage_trace/integrations/acs.py           # 改：lf_client → cs_client（命名清晰）+ 统一 AdapterError
  # trace 信号丢失 bug 修复（e2e 暴露）
  - sample_agents/tracing.py                       # 改：observer 写 metadata["name"] + 用 _model_name 读 model_name/model
  # 测试
  - tests/test_sample_agents.py                    # 改：+1 回归测试（distinct per-tool nodes）
  # 文档同步
  - docs/protocol.md                              # 改：SourceProtocol 表补 DEEPEVAL/ACS
  - docs/protocol_zh-CN.md                         # 改：同上
  - docs/usage.md                                  # 改：同上
  - docs/usage_zh-CN.md                            # 改：同上
  - README.md                                      # 改：Six→Eight adapters + 项目结构树
  - README_zh-CN.md                                # 改：同上
  - docs/architecture.md                           # 改：同步
  - docs/architecture_zh-CN.md                     # 改：同步
tests: 单元 376 passed（+1 回归）；e2e 5 passed（真实 DeepSeek）
mocks_used: false
---

# Session 0005 — 轻量重构 + trace 信号丢失 bug 修复 + DeepSeek e2e 复验

## 1. 任务

对 voyage_trace 系统做轻量级重构：清理无用代码、消除重复、在保证效果与性能的
前提下使代码尽可能优雅，并同步文档体系。然后用真实 DeepSeek API（deepseek-v4-flash）
跑一遍基于 trace 优化过的智能体，看实际效果。

## 2. 轻量重构：去重 + 死代码清理

### 2.1 抽取共享 helper（`_internal.py`）

五个模块各自重复定义了同一组 one-liner：`_utcnow`、`_new_id`、`_dt_to_str`、
`_dt_from_str`。集中到 [src/voyage_trace/_internal.py](../../src/voyage_trace/_internal.py)：

```python
def utcnow() -> datetime:           return datetime.now(timezone.utc)
def new_id(prefix: str) -> str:     return f"{prefix}-{uuid.uuid4().hex[:12]}"
def dt_to_str(dt) -> str | None:    return dt.isoformat() if dt else None
def dt_from_str(s) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
```

`types.py` / `protocol.py` / `analysis.py` / `verification.py` / `storage/in_memory.py`
全部改为从 `_internal` 导入，删掉各自的本地副本。该模块刻意只有标准库依赖、全部
下划线私有——公共 API 仍在 `types` / `protocol`。

### 2.2 死代码清理

| 文件 | 删除 | 理由 |
|---|---|---|
| `execution_graph.py` | `_NODE_DEF_RE` 正则 | 无任何引用 |
| `execution_graph.py` | `_detect_bottlenecks` docstring 中对 `voyage_trace.governance.findings` 的引用 | 该模块不存在 |
| `automl.py` | `TARGETS` 常量 | 无任何引用 |
| `storage/postgres.py` | `datetime` import | 未使用 |
| `integrations/langfuse_export.py` | `export_observation_now` / `parse_langfuse_datetime` | 无任何调用方 |
| `memory/manager.py` | `self.partitions` dict | 写入后从不读取 |
| `agents.py` | `Modification` import | 未使用（Modification 通过 simulator 间接消费） |

### 2.3 冗余计算消除

- `automl.run_automl`：原先把 traces 聚合成 graph 两次（一次取 matrix、一次取
  graph），改为聚合一次后复用。
- `memory/base.py`：把 `recall` / `search` / `forget` 上提为 `MemoryPartition`
  的默认实现 + `_query_records` helper；`episodic` / `procedural` / `working`
  三个子类删掉各自的重复实现，`semantic.search` 改用 `_query_records`。

### 2.4 命名 / 一致性

- `integrations/acs.py`：`lf_client` → `cs_client`（Content Safety 客户端，不是
  Langfuse）。
- `adapters/acs.py` / `adapters/deepeval.py`：统一抛 `AdapterError`（而非裸
  `ValueError`）。
- `adapters/otel.py` / `adapters/mcp.py`：status 处理改用共享 `_otel_status_code`。

## 3. 文档同步

把所有文档（中英）对齐到当前代码：

- `SourceProtocol` 表补 `DEEPEVAL` / `ACS`（之前 README 说 "Six adapters"，
  实际是八个 + 两个保留枚举）。
- 项目结构树补 `_internal.py` / `analysis.py` / `automl.py` / `agents.py` /
  `integrations/` / `storage/` / `memory/`。
- 涉及文件：`protocol.md` / `protocol_zh-CN.md` / `usage.md` / `usage_zh-CN.md` /
  `README.md` / `README_zh-CN.md` / `architecture.md` / `architecture_zh-CN.md`。

## 4. trace 信号丢失 bug（e2e 暴露 → 修复）

### 4.1 现象

跑 3 条真实 DeepSeek trace 后聚合执行图，发现**只有 2 个节点**：

```
chat:chat                          chat              30   ...  159219 tokens
execute_tool:execute_tool          execute_tool      50   ...  0 tokens
```

所有 chat span 坍缩成 `chat:chat`，所有 tool span 坍缩成
`execute_tool:execute_tool`。AutoML 只有 2 行特征、R²=0.0，治理管线无信号可学，
只能诚实返回 "no explanatory signal above the mean baseline"。

### 4.2 根因（两个互相关联的键不匹配）

`aggregate_execution_graph` 的分桶键 `_aggregate_key` 读：

```python
label = span.metadata.get("name") or span.metadata.get("tool_name") or span.operation_type.value
```

但 `TraceObserver`（[sample_agents/tracing.py](../../sample_agents/tracing.py)）写的是：

1. **tool span**：`metadata={"tool": name, ...}` —— 写 `tool`，aggregator 读
   `name` / `tool_name`，对不上 → 回退到 `operation_type.value` → 全坍缩成
   `execute_tool:execute_tool`。
2. **chat span**：`model_name = getattr(model, "name", "")` —— LangChain chat
   model 的标识在 `.model_name` / `.model`，不在 `.name`（`.name` 通常是类名或
   空）→ `model_name=""` → label 回退到 `chat` → 全坍缩成 `chat:chat`。

所有 adapter 与合成测试都用 `metadata["name"]`，**只有 TraceObserver 是异类**。

### 4.3 修复

在 `tracing.py` 里：

- tool span 的 metadata 加 `"name": name`（保留 `"tool"` 兼容现有测试）。
- chat span 的 metadata 加 `"name": model_name or "chat"`。
- 新增 `_model_name(model)` helper，按 `model_name` → `model` → `name` 顺序读，
  覆盖 LangChain 各版本的字段命名。

### 4.4 修复后效果（同一组 3 条真实 trace）

```
nodes: 6  (was 2 before the metadata['name'] fix)

  node_id                                type           calls   p50(s)   p99(s)   tokens   err%
  chat:deepseek-v4-flash                 chat              30    6.162   44.076   159219   0.0%
  execute_tool:task                      execute_tool       6   15.685   27.797        0   0.0%
  execute_tool:search                    execute_tool      30    0.001    0.001        0   0.0%
  execute_tool:summarise                 execute_tool       6    0.001    0.001        0   0.0%
  execute_tool:glob                      execute_tool       2    0.001    0.001        0   0.0%
  execute_tool:ls                        execute_tool       2    0.001    0.001        0   0.0%
```

- 节点数 2 → 6（3× 粒度）。
- AutoML 特征矩阵 2 行 → 6 行。
- 真实瓶颈可见：`execute_tool:task`（subagent 委派）p50=15.7s 是延迟热点；
  chat p99=44s 是 DeepSeek 长尾延迟。
- 真实 LLM 行为可见：模型偶尔回退到 deepagents 内置的 `glob`/`ls`（尽管 prompt
  明确禁止）—— 这是有用的治理信号。

### 4.5 治理管线的诚实表现

即便节点粒度恢复，3 条 trace 仍不足以让 AutoML 越过 mean-baseline 门：

| target | R² | top feature | 接受提案 | 说明 |
|---|---|---|---|---|
| `cost_usd` | 0.0 | calls | 0 | DeepSeek 只回 token 不回 $，cost 全 0 → 无信号（诚实） |
| `total_tokens` | 0.0435 | p99_duration | 0 | 有微弱信号但低于 mean baseline → 拒绝编造节省（诚实） |

系统在信号不足时**拒绝编造节省**，正是反作弊边界该有的行为。

## 5. 测试

### 5.1 单元测试

- 376 passed（+1 新回归测试 `test_observer_spans_aggregate_to_distinct_per_tool_nodes`）。
- 8 个 warning 全是 FLAML/AutoGluon 在退化数据上的数值警告，非失败。

### 5.2 e2e 测试（真实 DeepSeek API，deepseek-v4-flash）

```
tests/test_e2e_deepseek.py::TestResearchAgentE2E::test_runs_and_captures_real_trace PASSED
tests/test_e2e_deepseek.py::TestResearchAgentE2E::test_finalized_trace_is_canonical_and_aggregates PASSED
tests/test_e2e_deepseek.py::TestKBQAAgentE2E::test_answers_grounded_in_kb PASSED
tests/test_e2e_deepseek.py::TestCodeReviewAgentE2E::test_produces_nonempty_review PASSED
tests/test_e2e_deepseek.py::TestGovernancePipelineE2E::test_aggregate_multiple_real_traces_and_run_automl PASSED
5 passed in 560.36s
```

### 5.3 回归测试

`test_observer_spans_aggregate_to_distinct_per_tool_nodes` 锁定 §4 的修复：
scripted research agent 跑一轮后，聚合图必须有 ≥3 个 distinct tool 节点
（`execute_tool:search` / `:summarise` / `:task`），且不得出现坍缩签名
`execute_tool:execute_tool` 或 `chat:chat`。

## 6. 不作弊 / 不伪造审计

| 风险 | 防护 |
|---|---|
| 重构改行为 | 共享 helper 是纯 one-liner 抽取；376 单元测试一字不改全绿 |
| 编造节省 | AutoML mean-baseline 门 + leakage 守卫在 e2e 中同样生效；R²<baseline 时 0 accepted |
| 伪造 token | TraceObserver 从 `usage_metadata` 读，缺失记 0；e2e 断言总 token > 0 |
| key 进源码 | `test_no_api_key_in_source_files` 扫 `sample_agents/` 全部 `.py`；`grep -rE 'sk-[a-f0-9]{32}' /workspace` 零匹配 |
| e2e 用 mock | `tests/test_e2e_deepseek.py` 用真实 `ChatOpenAI` 指向 `api.deepseek.com` |

## 7. 后续可选

- 接 ≥5 条真实 trace 跑治理管线，让 AutoML 越过 mean-baseline 门、产出真实
  accepted proposal（3 条信号太弱是预期的，不是 bug）。
- 给 TraceObserver 加可选的 token→$ 定价表，让 `cost_usd` target 也有信号
  （当前 DeepSeek 只回 token 不回 $）。
- 把 `glob`/`ls` 回退信号反馈到 sample agent 的 prompt 调优（模型偶尔无视
  "do not use glob/ls" 指令）。
