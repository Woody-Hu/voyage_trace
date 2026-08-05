# Sample agents — deepagents + voyage_trace tracing

This directory ships three reference "digital employee" agents built on
**deepagents** and wired to voyage_trace's tracing protocol via a
low-intrusion middleware. The samples are working code (runnable end-to-end
in tests against a scripted chat model) and serve as a template for adding
more agents that plug straight into the trace-driven governance pipeline.

## What's here

```
sample_agents/
├── __init__.py            # public exports
├── builder.py             # SubAgentSpec — shared declarative spec + build_agent
├── tracing.py             # TraceObserver — the low-intrusion trace bridge
├── llm_config.py          # config-file LLM setup (DeepSeek key from env, never source)
├── testing.py             # ScriptedChatModel — deterministic test model
├── research_agent.py      # research subagent (search + summarise)
├── code_review_agent.py   # code-review subagent (read + critique)
├── kb_qa_agent.py         # knowledge-base QA subagent (retrieve + answer/escalate)
└── config.example.yaml    # example LLM config (no real keys; copy to config.yaml)
```

## Why deepagents (and not agentscope)

Both `deepagents` and `agentscope` were considered. They cover different
philosophies:

| | deepagents | agentscope |
|---|---|---|
| Model | opinionated single-agent harness with `task`-tool delegation | peer-to-peer multi-agent message bus |
| Sub-agents | ephemeral, isolated-context, single-return | long-lived, addressable, message-passing |
| Built-ins | filesystem, memory, skills, HIL, summarization, rubric | distributed RPC, web UI |
| Tracing | LangSmith tags built in; observer middleware for custom tracers | event-driven pub-sub |

For request/response "employee" agents (research, code review, KB-QA),
deepagents' `SubAgent` + `task`-tool delegation pattern is a cleaner fit
than agentscope's peer-to-peer message bus. **All three samples are
implemented in deepagents alone.** An agentscope adapter is only worth
building if you specifically need peer-to-peer message passing between equal
long-lived agents — none of the three sample patterns require it.

## The low-intrusion trace contract

`TraceObserver` is the **only** point of contact between the agent and
voyage_trace. It satisfies five guarantees:

1. **No behaviour change.** `wrap_model_call` and `wrap_tool_call` forward
   the request to `handler` and return its result unchanged. Agent outputs
   are byte-identical with and without the observer — verified by
   `test_observer_does_not_alter_agent_output`.
2. **No silent failures.** Anything the observer cannot interpret is
   recorded as an empty span (or skipped). A bug in the observer never
   crashes the agent loop.
3. **One span per call.** Each model invocation → one `CHAT` span; each
   tool invocation → one `EXECUTE_TOOL` span. The trace is a flat list of
   agent steps that flows straight through `aggregate_execution_graph` /
   `run_automl`.
4. **Real token counts when available.** Reads `usage_metadata` from the
   AIMessage when the model fills it (OpenAI / Anthropic / DeepSeek do).
   When absent, records zeros rather than fabricating numbers.
5. **No SDK lock-in.** Imports only deepagents' public `AgentMiddleware`
   base and langchain message types. The trace is a plain `CanonicalTrace`
   pushable to any backend via `voyage_trace.integrations`.

### How the observer attaches

The same observer instance is attached to BOTH the parent agent and the
subagent's own middleware chain, so a single `CanonicalTrace` captures the
full delegation:

```
parent model call ──► `task` tool ──► subagent model call
                                          │
                                          ├──► `search` tool
                                          ├──► subagent model call
                                          ├──► `summarise` tool
                                          └──► subagent final reply
                                       ◄───
                              parent final model call
```

All eight steps land as spans on one trace, with `agent_id` set to the
parent's `agent_id` (so the governance pipeline sees one coherent run).

## Running a sample

```python
from sample_agents import build_research_agent

# Uses the LLM config (config.yaml or env var DEEPSEEK_API_KEY).
agent, observer = build_research_agent()
result = agent.invoke({"messages": [HumanMessage(content="research autogluon")]})
trace = observer.finalize()
# `trace` is a voyage_trace.CanonicalTrace — feed it to run_automl, the
# execution-graph builder, the simulator, or push it to Langfuse / DeepEval.
```

For tests without a live LLM, pass a `ScriptedChatModel`:

```python
from sample_agents import ScriptedChatModel, build_research_agent
from langchain_core.messages import AIMessage

model = ScriptedChatModel(script=[
    AIMessage(content="", tool_calls=[{"name": "task", "args": {...}, "id": "c1"}]),
    AIMessage(content="..."),
])
agent, observer = build_research_agent(model=model)
```

## LLM config — credentials never in source

The DeepSeek API key is resolved in this order (first hit wins):

1. `api_key:` field in `sample_agents/config.yaml` (git-ignored; rarely used).
2. Env var named by `api_key_env:` in the config (default `DEEPSEEK_API_KEY`).
3. `DEEPSEEK_API_KEY` env var directly.

If none resolve, `build_chat_model()` raises a `RuntimeError` listing the
env vars that were checked — never a silent fallback to a free/hardcoded
key. The meta-test `test_no_api_key_in_source_files` scans the package
source for the DeepSeek key format and fails if any leak in.

### Setup

```bash
# 1. Copy the example config and edit if you want per-agent overrides.
cp sample_agents/config.example.yaml sample_agents/config.yaml

# 2. Export your DeepSeek key (recommended over editing config.yaml).
export DEEPSEEK_API_KEY="sk-..."

# 3. Run any sample.
python -c "from sample_agents import build_research_agent; a, o = build_research_agent(); \
  print(a.invoke({'messages': [{'role': 'user', 'content': 'hi'}]}))"
```

## Adding a new agent

The shared `SubAgentSpec` (`sample_agents/builder.py`) makes adding a sample
a declarative exercise — no boilerplate to copy:

1. Pick a name and an `agent_id` constant.
2. Write the tools as `@tool`-decorated callables (LangChain's interface;
   deepagents has no tool base class of its own).
3. Declare a `SubAgentSpec` with the agent's id, name, description,
   system_prompt, tools, and orchestrator_prompt.
4. Expose `build_<name>_subagent_spec = SPEC.build_subagent_dict` and a
   thin `build_<name>_agent(**kw)` wrapper around `SPEC.build_agent(**kw)`.
5. Add a test that drives the agent with a `ScriptedChatModel` and asserts
   the observer captured the expected tool spans.

`build_agent` handles observer creation, deferred model construction from
the LLM config, mounting the spec as a `task`-tool subagent, and attaching
the same observer to both parent and subagent. New agents only declare
*what is different*.

### Prompt style (tuned against a real LLM)

The three samples' system prompts are **directive**, not advisory: numbered
MUST steps that name the exact tool + argument, plus an explicit "do not
use glob/ls/read_file" line. This was tuned against the real DeepSeek
(`deepseek-v4-flash`) e2e run — without it, deepagents' built-in
`FilesystemMiddleware` tools (always injected) compete with the sample's
custom tools, and the LLM occasionally picks `read_file` over the sample's
`read_snippet`. The directive prompt reliably steers it back. See
`docs/session-log/0004-*.md` §4.3 for the e2e findings.

## Honest test posture

- **No mocks of deepagents.** `create_deep_agent`, the `SubAgent`
  middleware, the `task` tool, and the `TraceObserver` are all the real
  production code.
- **No fabricated spans.** The `ScriptedChatModel` plays author-written
  AIMessages, but the tool dispatch, the middleware chain, and the span
  capture are all real. There is no path from "the test wants a tool span"
  to "the test emits a tool span" that bypasses the agent loop.
- **No hardcoded keys.** The meta-test `test_no_api_key_in_source_files`
  scans every `.py` and the example YAML for the DeepSeek key format.
- **Empty observers raise.** `finalize()` on an observer with no spans
  raises `ProtocolError` rather than fabricating a fake span.
- **Real-LLM e2e layer.** `tests/test_e2e_deepseek.py` runs the same
  agents against the real DeepSeek API (`deepseek-v4-flash`), asserting
  real token usage and KB-grounded answers. It is skipped unless
  `DEEPSEEK_API_KEY` is set, so CI without a key stays green; locally with
  a key it verifies the loop end-to-end. The deterministic tool-name
  behaviour is covered by the scripted tests; the e2e tests assert
  *plumbing* (real tokens, real trace, governance pipeline runs), not
  exact LLM wording.

## Reference patterns (origin)

The three samples re-express common open-source "digital employee" patterns
onto deepagents:

| Sample | Origin pattern | deepagents mapping |
|---|---|---|
| research_agent | OpenHands / generic research-agent | `SubAgent` with `search` + `summarise` tools; structured return via final AIMessage |
| code_review_agent | OpenHands code-review loop | `SubAgent` with `read_snippet` + `critique` tools; structured Markdown return |
| kb_qa_agent | Dify / MaxKB / FastGPT KB-QA | `SubAgent` with `retrieve` + `answer_or_escalate` tools; human-in-the-loop escalation |

Each ships with **deterministic stub tools** (pure-Python, no network) so
the agent loop is reproducible in CI. Production deployments swap the tool
bodies for real search APIs / linters / vector stores without touching the
agent spec.
