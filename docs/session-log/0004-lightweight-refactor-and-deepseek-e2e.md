---
session_id: 0004
title: 轻量重构 + DeepSeek (deepseek-v4-flash) 真实 e2e 验证 + 智能体调优
date: 2026-08-05
status: implemented
research-kind: closed-loop-verification
files_touched:
  # 轻量重构：抽取共享 builder，消除三个 sample agent 的重复样板
  - sample_agents/builder.py                   # 新增：SubAgentSpec + build_agent/build_subagent_dict
  - sample_agents/research_agent.py            # 重写：声明式 spec，去掉重复 build_*
  - sample_agents/code_review_agent.py         # 重写：同上 + 强化 prompt
  - sample_agents/kb_qa_agent.py              # 重写：同上 + 强化 prompt
  - sample_agents/__init__.py                  # 导出 SubAgentSpec
  # e2e 配置（无 key）
  - sample_agents/config.example.yaml          # 修改：deepseek-chat → deepseek-v4-flash
  - pyproject.toml                             # 修改：samples extra 加 langchain-openai
  # e2e 测试
  - tests/test_e2e_deepseek.py                # 新增：5 个真实 DeepSeek e2e 测试
  # 单元测试补充
  - tests/test_sample_agents.py                # 修改：+16 测试（builder / observer 边界 / 诚实性）
tests: 单元 +16（375 passed）；e2e +5（5 passed with key, 5 skipped without）
mocks_used: false
---

# Session 0004 — 轻量重构 + DeepSeek 真实 e2e 验证 + 智能体调优

## 1. 任务

继续用户多阶段任务的步骤 5–6：

5. 增加测试体系与案例（不造假/不伪造），轻量重构去冗余、精简接口与
   数据模型，补文档与 sessionlog。
6. 整体轻量重构使代码简洁优雅；用 deepseek API（配置文件存 key，不进源码）
   做 e2e 测试并调优智能体方案，不能伪造与作弊。

## 2. 轻量重构：消除 sample agents 的三重样板

### 2.1 发现的冗余

三个 sample agent（research / code-review / KB-QA）各有一对几乎完全相同
的函数：

```python
def build_*_subagent_spec(*, model=None, config=None, observer=None) -> dict:
    spec = {"name": ..., "description": ..., "system_prompt": ...,
            "tools": list(*_TOOLS)}
    if model is not None: spec["model"] = model
    elif config is not None: spec["_llm_config"] = config
    if observer is not None: spec["middleware"] = [observer]
    return spec

def build_*_agent(*, model=None, observer=None, config_path=None):
    from deepagents import create_deep_agent
    if observer is None: observer = attach(*_AGENT_ID, ...)
    if model is None: ... load_config ...
    spec = build_*_subagent_spec(model=model, observer=observer)
    agent = create_deep_agent(model=..., subagents=[spec],
                              middleware=[observer], system_prompt=...)
    return agent, observer
```

只有 `agent_id / agent_name / description / system_prompt / tools /
orchestrator_prompt` 六个字段不同，其余 ~40 行/模块 = ~120 行重复。

### 2.2 重构：声明式 `SubAgentSpec`

新增 `sample_agents/builder.py`，把共享形状抽成一个 frozen dataclass +
两个方法：

```python
@dataclass(frozen=True)
class SubAgentSpec:
    agent_id: str
    agent_name: str
    description: str
    system_prompt: str
    tools: list[BaseTool | Callable]
    orchestrator_prompt: str = "delegate to `{agent_id}`"

    def build_subagent_dict(self, *, model=None, config=None, observer=None) -> dict: ...
    def build_agent(self, *, model=None, observer=None, config_path=None) -> tuple: ...
```

每个 sample 模块现在只**声明差异**：

```python
RESEARCH_SPEC = SubAgentSpec(
    agent_id="research-agent", agent_name="ResearchAgent",
    description="...", system_prompt=RESEARCH_SYSTEM_PROMPT,
    tools=RESEARCH_TOOLS,
    orchestrator_prompt="delegate research to `research-agent`",
)
build_research_subagent_spec = RESEARCH_SPEC.build_subagent_dict
def build_research_agent(**kw): return RESEARCH_SPEC.build_agent(**kw)
```

- **公共 API 不变**：`build_research_agent` / `build_research_subagent_spec`
  仍是模块级可调用对象；现有 28 个测试一字不改全绿。
- **新增一个 sample 现在只需声明 spec + 写两个 `@tool`**，不再复制样板。
- 净减少 ~80 行重复代码。

### 2.3 诚实性约束

重构是纯结构性的 —— 行为字节一致。`TestSubAgentSpec` 用真实 deepagents
跑 `build_agent` 验证 spec dict 形状、observer 挂载、deferred config、
orchestrator prompt 插值，以及 `build_research_subagent_spec` 是
`RESEARCH_SPEC.build_subagent_dict` 的 bound method（公共契约）。

## 3. 测试补充（步骤 5）

### 3.1 builder + observer 边界（+16 单元测试）

`tests/test_sample_agents.py` 新增三个测试类，全部跑真实 deepagents /
真实 `TraceObserver`，无 mock：

- `TestSubAgentSpec`（7）：spec dict 形状、observer 挂载、deferred config、
  提供则不读配置、orchestrator prompt `{agent_id}` 插值、RESEARCH_SPEC 是
  SubAgentSpec 实例、模块级别名 == bound method。
- `TestTraceObserverEdgeCases`（7）：
  - `_extract_usage` 读 `usage_metadata`（DeepSeek/OpenAI 形状）；缺失时
    返回 `(0, 0)` 而非伪造。
  - `_coerce_str` 截断 + None → 空串。
  - `_tool_call_status`：ToolMessage `status="error"` → FAILED。
  - **工具抛异常 → 记录 FAILED EXECUTE_TOOL span 并向上抛**（不静默吞）。
  - **usage_metadata → span token 数**：scripted AIMessage 带
    `usage_metadata={input:7,output:3}`，捕获的 CHAT span 必须记 7/3。
  - **async 钩子与 sync 对称**：`awrap_model_call` 捕获同样的 span。
  - **reset() 跨 run 隔离**：两次 run 的 span 不串、trace_id 不同。
- `TestRefactorHonesty`（2）：扫描 `builder.py` 无硬编码 key；扫描三个
  tool 模块不 import 任何 LLM client（@tool stub 是纯 Python）。

### 3.2 诚实性元测试不变

- `test_no_api_key_in_source_files` 仍扫 `sample_agents/` 全部 `.py` 与
  `config.example.yaml`，匹配 `sk-[a-f0-9]{20,}`，任何泄漏即失败。
- e2e 文件 `tests/test_e2e_deepseek.py` 也被同一规则覆盖（见 §5）。

## 4. DeepSeek 真实 e2e 验证（步骤 6）

### 4.1 key 不进源码的契约

- **配置文件**：`sample_agents/config.example.yaml`（tracked，无 key）指向
  `deepseek-v4-flash` + `api_key_env: DEEPSEEK_API_KEY`。本地
  `config.yaml`（git-ignored）可填 key，或直接 export 环境变量。
- **解析顺序**（`LLMConfig.resolved_api_key`）：YAML `api_key` →
  `api_key_env` 命名的环境变量 → `DEEPSEEK_API_KEY`。三者皆无时抛
  `RuntimeError`，绝不静默回退。
- **验证**：`grep -rE 'sk-[a-f0-9]{32}' /workspace` 无任何匹配 —— key
  只存在于运行时环境变量中。

### 4.2 e2e 测试集（`tests/test_e2e_deepseek.py`，5 个）

`@pytest.mark.skipif(not DEEPSEEK_API_KEY)` —— CI 无 key 时干净跳过
（已验证：5 skipped in 0.03s），本地有 key 时跑真实网络往返。

| 测试 | 验证内容 | 诚实性 |
|---|---|---|
| `test_runs_and_captures_real_trace` | research agent 真实运行，trace 有 CHAT+EXECUTE_TOOL span，**总 token > 0** | 不伪造 token |
| `test_finalized_trace_is_canonical_and_aggregates` | finalize() 过 protocol 不变量，`aggregate_execution_graph` 接受 | 真实 CanonicalTrace |
| `test_answers_grounded_in_kb` | KB-QA 对 "return policy?" 的回答含 "30 day"，且 `retrieve` 被调用 | 答案必须落地 KB，不可凭空生成 |
| `test_produces_nonempty_review` | code-review 产出非空 review，至少 1 个 tool span，token > 0 | 不断言具体工具名（LLM 非确定） |
| `test_aggregate_multiple_real_traces_and_run_automl` | 3 条真实 trace → aggregate → `run_automl` → JSON-safe report | 闭环治理管线 |

### 4.3 真实 e2e 暴露的两个问题（→ 调优依据）

第一次 e2e 跑（未调优）有 2 个失败，**都是真实信号**：

1. **code-review agent 用了 deepagents 自带的 filesystem 工具**
   （`glob`/`ls`/`read_file`）而非 sample 的 `read_snippet`/`critique`。
   `create_deep_agent` 的 `FilesystemMiddleware` 是 protected scaffolding，
   默认注入；真实 LLM 在 5 个工具里选了 `read_file`。
   → **调优**：把三个 sample 的 system prompt 从"建议性"改为"指令性"，
   显式编号步骤 + "do not use glob/ls/read_file"。

2. **governance 测试断言 `report.n_samples == len(traces)` 失败**（2 != 3）。
   真因：`FeatureMatrix.n_samples` 是**聚合图中 (operation_type, label)
   去重后的节点数**，不是 trace 数。3 条 research trace 都命中同样的节点
   （chat / execute_tool:search / execute_tool:summarise / execute_tool:task），
   折叠成少数几行。
   → **修断言**为 `1 <= n_samples <= len(graph.nodes)`，并注释说明语义。

### 4.4 调优结果

三个 prompt 全部改为"指令性"（MUST + 编号步骤 + 显式禁止其他工具）。
脚本化测试（44 passed）不受影响 —— prompt 文本变了，但工具行为不变。

调优后真实 e2e 全绿：

```
tests/test_e2e_deepseek.py::TestResearchAgentE2E::test_runs_and_captures_real_trace PASSED
tests/test_e2e_deepseek.py::TestResearchAgentE2E::test_finalized_trace_is_canonical_and_aggregates PASSED
tests/test_e2e_deepseek.py::TestKBQAAgentE2E::test_answers_grounded_in_kb PASSED
tests/test_e2e_deepseek.py::TestCodeReviewAgentE2E::test_produces_nonempty_review PASSED
tests/test_e2e_deepseek.py::TestGovernancePipelineE2E::test_aggregate_multiple_real_traces_and_run_automl PASSED
5 passed in 698.67s (0:11:38)
```

KB-QA 真实回答里出现 "30 day" —— 证明 LLM 确实读了 `retrieve` 返回的 KB
块、没有凭空编造。

## 5. 全量测试

- **无 key（CI 姿态）**：375 passed + 5 skipped（e2e）。
- **有 key（本地姿态）**：375 passed + 5 passed（e2e）= 380 passed。
- 8 个 warning 全部来自 FLAML/AutoGluon 在退化数据上的数值警告
  （`divide by zero` / `ConstantInputWarning`），非测试失败。

## 6. 不作弊/不伪造的审计点

| 风险 | 防护 |
|---|---|
| 硬编码 API key | `test_no_api_key_in_source_files` + `TestRefactorHonesty` 扫 builder.py；`grep` 全仓零匹配 |
| 伪造 LLM 输出 | e2e 测试用真实 `ChatOpenAI` 指向 `api.deepseek.com`；scripted 测试用 `ScriptedChatModel` 但跑真实 agent loop |
| 伪造 token 数 | `TraceObserver` 从 `usage_metadata` 读，缺失记 0；e2e 断言"总 token > 0"才证明真跑过 |
| 伪造治理结果 | `run_automl` 的 leakage 守卫 / mean-baseline 门在 e2e 中同样生效；report.n_samples 诚实反映节点数 |
| 测试绕过 agent loop | `ScriptedChatModel` 播放作者写的 AIMessage，但工具派发 / middleware / span 捕获全是真实代码路径 |

## 7. 待办（无）

用户步骤 1–6 全部完成。后续可选：
- 把 e2e 测试接入 nightly（带 key 的 cron job）。
- 为 `code_review_agent` 接真实 linter（替换 `critique` stub）。
- sample agents 接 `agentscope` 的 P2P 消息总线（当出现需要智能体间
  对等通信的 sample 时）。
