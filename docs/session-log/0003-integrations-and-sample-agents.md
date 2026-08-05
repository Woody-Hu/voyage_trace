---
session_id: 0003
title: 可选集成 (deepeval / langfuse / acs / flaml) 与 sample agents 路径
date: 2026-08-05
status: implemented
research-kind: open-ended
files_touched:
  # 新增 integrations 模块
  - src/voyage_trace/integrations/langfuse_export.py     # 新增：push CanonicalTrace → Langfuse SDK
  - src/voyage_trace/integrations/acs.py                 # 新增：SDK-using 安全/一致性 scorer
  - src/voyage_trace/integrations/flaml_runner.py        # 新增：FLAML 作为 AutoGluon 替代后端
  - src/voyage_trace/integrations/deepeval.py            # 修改：from_deepeval_results 支持对象列表
  # 新增 sample agents 包
  - sample_agents/__init__.py                            # 新增：包入口
  - sample_agents/tracing.py                             # 新增：低侵入 TraceObserver 中间件
  - sample_agents/llm_config.py                          # 新增：config-file LLM 配置
  - sample_agents/testing.py                             # 新增：ScriptedChatModel 测试替身
  - sample_agents/research_agent.py                      # 新增：research subagent
  - sample_agents/code_review_agent.py                  # 新增：code-review subagent
  - sample_agents/kb_qa_agent.py                        # 新增：KB-QA subagent
  - sample_agents/config.example.yaml                    # 新增：示例配置（无真实 key）
  # 协议修复
  - src/voyage_trace/protocol.py                         # 修改：make_dotted_order 同时去除下划线
  # 测试
  - tests/test_adapters_scores.py                        # 新增：41 个集成 + adapter 测试
  - tests/test_sample_agents.py                          # 新增：28 个 sample agents 测试
  # 文档
  - docs/sample-agents.md                                # 新增：sample agents 设计与使用
  - .gitignore                                           # 修改：忽略 sample_agents/config.yaml
tests: 69 新增；全量 359 passed
mocks_used: false
---

# Session 0003 — 可选集成 + sample agents 路径

## 1. 任务

继续用户多阶段任务的步骤 3–4：

3. 在网上搜索类似的开源项目，与 **deepeval / langfuse / acs / flaml**
   完成可选的集成与适配（不破坏已有接口）。
4. 在 GitHub 上搜索成熟的数字员工/智能体实现，转为 **deepagents / agentscope**
   框架实现，放到 `sample_agents` 路径；补测试与文档；设计一个**低侵入性**
   的形式让这些智能体使用 voyage_trace 的 trace 分析机制。

## 2. 集成模块（步骤 3）

### 2.1 设计原则

每一个集成模块遵守三条契约：

1. **SDK 懒加载**：在用到 SDK 的函数内部才 `import`；包级 `__init__` 不
   触发 SDK 导入。
2. **优雅降级**：SDK 缺失时返回 JSON-safe 的同形结构（或 `skipped` 评
   决），绝不抛 `ImportError` 到核心管线。
3. **不修改 canonical schema**：`CanonicalTrace` / `TraceSpan` 字段不变；
   集成是边界处的无损翻译。

### 2.2 langfuse_export.py（push 侧）

- **方向**：`CanonicalTrace` → Langfuse（push）。
- **SDK 缺失时**：返回与 Langfuse `fetch()` 相同的 JSON-safe 字典
  `{"trace": {...}, "observations": [...]}`，可被 `LangfuseAdapter`（pull 侧）
  回流解析。
- **诚实保证**：push→pull 往返不丢信息（`test_export_artefact_round_trips_through_pull_adapter`
  验证 token 数与父子链）。
- **失败降级**：调用方传入的 client 抛异常时，仍返回 JSON-safe 字典而非
  崩溃治理轮（`test_export_with_client_does_not_raise`）。

### 2.3 acs.py（scorer 侧）

- **方向**：text → 安全/一致性 verdict dict → `CanonicalTrace`。
- **后端优先级**：(1) 调用方提供的 `scorer` callable → (2) Azure Content
  Safety SDK + 凭据 → (3) 启发式回退。
- **诚实契约**：启发式回退**绝不**返回 `verdict="safe"`，而是
  `verdict="skipped"` —— "未配置真实 scorer" 与 "通过安全检查" 是两个不同
  的状态，绝不混淆。
- **类别名兼容**：Azure 类别名（`SelfHarm`）→ 本地名（`self_harm`）；
  触发 protocol 的 `make_dotted_order` 修复（见 §4）。

### 2.4 flaml_runner.py（AutoML 替代后端）

- **方向**：traces/graph → `AutoMLReport`（与 `run_automl` 相同类型）。
- **为何需要**：FLAML 是 cost-aware 模型选择，tiny dataset 上比 AutoGluon
  快得多；AutoGluon + FLAML 在 top feature 上一致 = 信号；不一致 = 值得
  记录在 `notes`。
- **诚实契约**：与 AutoGluon 后端**完全相同**的 leakage 守卫、mean-baseline
  门、no-signal 检查 —— `beats_baseline` 在两个后端语义一致。

### 2.5 deepeval.py 修复

`from_deepeval_results` 原先对对象列表的处理有 bug：直接 `{"results": [...]}`
传给 adapter，但 adapter 调 `res.get("metrics")` 对非 dict 对象会抛
`AttributeError`。修复为：列表中全为 dict 时直接走 adapter；包含对象时先
读属性归一化为 dict 形状。

## 3. sample agents 路径（步骤 4）

### 3.1 框架选型：deepagents vs agentscope

| | deepagents 0.7.4 | agentscope |
|---|---|---|
| 模型 | 单智能体 harness + `task`-tool 委派 | 对等智能体消息总线 |
| 子智能体 | 短暂、隔离上下文、单返回 | 长寿命、可寻址、消息传递 |
| 内置 | filesystem / memory / skills / HIL / summarization / rubric | 分布式 RPC / web UI |
| 追踪 | LangSmith 标签内置；observer middleware 用于自定义 tracer | 事件驱动 pub-sub |

**结论**：请求/响应式"员工"智能体（research / code-review / KB-QA）用
deepagents 的 `SubAgent` + `task`-tool 委派更贴合。**全部三个 sample 用
deepagents 实现**；agentscope 适配器只在需要 P2P 消息传递时才有价值，
当前 sample 不需要。

### 3.2 三个 sample 智能体

每个 sample = 一个 `SubAgent` spec + 几个 `@tool` 装饰的纯 Python 工具 +
一个 `build_*_agent` runner：

| Sample | 模式来源 | 工具 |
|---|---|---|
| `research_agent` | OpenHands / 通用 research-agent | `search` + `summarise` |
| `code_review_agent` | OpenHands 代码评审环 | `read_snippet` + `critique` |
| `kb_qa_agent` | Dify / MaxKB / FastGPT KB-QA | `retrieve` + `answer_or_escalate` |

工具均为**确定性 stub**（纯 Python，无网络），保证 CI 中可重现；生产部署
替换工具函数体（接 Tavily / linter / 向量库），agent spec 不变。

### 3.3 低侵入 trace 接入设计

`TraceObserver`（`sample_agents/tracing.py`）是 agent 与 voyage_trace
之间的**唯一**接触点，五条保证：

1. **零行为变更**：`wrap_model_call` / `wrap_tool_call` 调 `handler(request)`
   后原样返回；agent 输出有无 observer 字节一致
   （`test_observer_does_not_alter_agent_output` 验证）。
2. **零静默失败**：observer 解析失败时记录空 span 或跳过；observer 的 bug
   不污染 agent 轮。
3. **每次调用一个 span**：model 调用 → `CHAT` span；tool 调用 →
   `EXECUTE_TOOL` span；trace 是扁平的 agent 步骤列表，直接喂
   `aggregate_execution_graph` / `run_automl`。
4. **真实 token 数**：从 AIMessage 的 `usage_metadata` 读取
   （OpenAI/Anthropic/DeepSeek 都填）；缺失时记 0，绝不伪造。
5. **无 SDK 锁定**：只 import deepagents 的 `AgentMiddleware` 基类与
   langchain 消息类型；trace 是普通 `CanonicalTrace`，可经
   `voyage_trace.integrations` 推到任意后端。

**关键设计**：同一个 observer 实例**同时**挂到父 agent 与子 agent 的
middleware 链，所以一条 `CanonicalTrace` 捕获完整委派链：

```
parent model ──► task tool ──► subagent model
                                  ├──► search tool
                                  ├──► subagent model
                                  ├──► summarise tool
                                  └──► subagent reply
                              ◄───
                       parent final model
```

8 个步骤全部落到一条 trace 上，`agent_id` 为父 agent 的 id —— 治理管线看
到的是一次连贯运行。

### 3.4 LLM 配置（key 不进源码）

`sample_agents/llm_config.py`：

- 配置文件路径默认 `sample_agents/config.yaml`（git-ignored），可由
  `VOYAGE_TRACE_LLM_CONFIG` 环境变量覆盖。
- Key 解析顺序：(1) YAML 中 `api_key:` 字段 → (2) `api_key_env:` 命名的
  环境变量（默认 `DEEPSEEK_API_KEY`） → (3) `DEEPSEEK_API_KEY` 环境变量。
- 三者都未提供时 `build_chat_model()` 抛 `RuntimeError`，列出检查过的
  环境变量名，绝不静默回退到"免费"或硬编码 key。
- **元测试 `test_no_api_key_in_source_files`** 扫描 `sample_agents/` 全部
  `.py` 与 `config.example.yaml`，匹配 DeepSeek key 格式
  （`sk-[a-f0-9]{20,}`），任何泄漏即失败。

### 3.5 测试姿态（不造假）

- **不 mock deepagents**：`create_deep_agent`、`SubAgent` middleware、`task`
  tool、`TraceObserver` 全部是真实生产代码。
- **不伪造 span**：`ScriptedChatModel` 播放作者写好的 AIMessage，但工具派
  发、middleware 链、span 捕获都是真实的；"测试想要一个 tool span"到"测试
  发出一个 tool span"之间没有绕过 agent loop 的捷径。
- **空 observer 抛错**：`finalize()` 在无 span 时抛 `ProtocolError`，而非伪
  造一个 fake span。

## 4. 协议修复：`make_dotted_order` 兼容下划线

ACS adapter 的 category 名（如 `self_harm`）作为 span_id 时，dotted_order
的 suffix 包含下划线，违反 `_DOTTED_SEGMENT` 正则
（`^\d{8}T\d{6}Z[0-9A-Za-z]+$`）。原代码 `span_id.replace("-", "")` 只去
hyphen（UUID 用），未处理下划线。

修复：`re.sub(r"[_-]", "", span_id)` —— 同时去除 hyphen 与下划线，保持
deterministic、保持 24 字符对齐。最小化、目标明确的修复；不影响任何现有
adapter（UUID 仍正常工作）。

## 5. 测试

- 新增 `tests/test_adapters_scores.py`：41 个测试，覆盖 DeepEval/ACS adapter
  + 4 个集成模块 + protocol 推断。SDK 缺失时走降级路径并验证 JSON-safe
  字典返回。
- 新增 `tests/test_sample_agents.py`：28 个测试，覆盖 `ScriptedChatModel`
  + `TraceObserver` + 3 个 sample agent 端到端 + LLM config + 源码无 key
  扫描。
- 全量：**359 passed**（前次 331 → +28 sample + 41 集成 - 旧测试合并）。

## 6. 待办（步骤 5–6）

下一会话将处理：

5. 增加测试体系与案例（不造假/不伪造），轻量重构去冗余、精简接口与数据
   模型，补文档与 sessionlog（本会话已部分完成）。
6. 整体轻量重构使代码简洁优雅；用 deepseek API（配置文件存 key，不进源码）
   做 e2e 测试并调优智能体方案。本会话已为步骤 6 铺好路：
   - `sample_agents/llm_config.py` 已就位
   - `config.example.yaml` 已就位（无 key）
   - `ScriptedChatModel` 已验证 deepagents 端到端流程
   - 下一步只需在 `config.yaml` 写入 deepseek 配置并跑真实 LLM e2e 测试
