"""Code-review subagent — reviews a file for issues + suggests fixes.

Pattern origin: OpenHands-style code-review loop. Re-expressed as a
deepagents :class:`SubAgent` with two custom tools (``read_snippet`` and
``critique``). The agent reads a snippet, runs the critique tool, then
returns the structured review.

The tools are deterministic pure-Python stubs here — production deployments
swap the body of ``critique`` for a linter / AST check without changing the
agent spec.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .builder import SubAgentSpec
from .tracing import TraceObserver

CODE_REVIEW_AGENT_ID = "code-review-agent"
CODE_REVIEW_SYSTEM_PROMPT = """\
You are a code-review sub-agent. To review a file you MUST follow these
steps in order:

1. Call the `read_snippet` tool with the file path (the user's message
   contains the path to review).
2. Call the `critique` tool, passing the snippet returned by `read_snippet`
   as `snippet`.
3. Return the `critique` output verbatim as your final answer.

Do not use `glob`, `ls`, `read_file`, or any other tool — your only tools
are `read_snippet` and `critique`.
"""


@tool
def read_snippet(path: str) -> str:
    """Return the source text of ``path`` (canned sample for the demo).

    In production, swap the body for a real ``open(path).read()`` call; the
    tool signature (and the agent loop that drives it) does not change.
    """
    sample = (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def divide(a, b):\n"
        "    return a / b  # possible ZeroDivisionError\n"
    )
    return f"# {path}\n{sample}"


@tool
def critique(snippet: str) -> str:
    """Return a structured review of ``snippet``.

    Deterministic pure-Python check: looks for common smells (bare
    ``except``, ``return / 0`` etc.) and emits 3 sections (style / bugs /
    suggestions). Production deployments replace the body with a real
    linter / AST walker; the agent spec stays identical.
    """
    bugs: list[str] = []
    if "/ 0" in snippet or "/0" in snippet:
        bugs.append("possible ZeroDivisionError — guard the divisor")
    if "except:" in snippet:
        bugs.append("bare `except:` swallows all exceptions")
    if "import os" in snippet and "os.system" in snippet:
        bugs.append("os.system call — prefer subprocess with shell=False")

    style: list[str] = []
    if "\t" in snippet:
        style.append("uses tabs — project convention is spaces")
    if any(len(line) > 100 for line in snippet.splitlines()):
        style.append("line(s) > 100 chars — wrap them")

    suggestions: list[str] = []
    if "print(" in snippet:
        suggestions.append("replace print() with logging")
    if "TODO" in snippet or "FIXME" in snippet:
        suggestions.append("address TODO/FIXME markers before merge")

    style_lines = [f"- {s}" for s in style] or ["- (no issues)"]
    bug_lines = [f"- {b}" for b in bugs] or ["- (no issues)"]
    sug_lines = [f"- {s}" for s in suggestions] or ["- (no issues)"]
    parts = ["## Style", *style_lines,
             "## Bugs", *bug_lines,
             "## Suggestions", *sug_lines]
    return "\n".join(parts)


CODE_REVIEW_TOOLS = [read_snippet, critique]

CODE_REVIEW_SPEC = SubAgentSpec(
    agent_id=CODE_REVIEW_AGENT_ID,
    agent_name="CodeReviewAgent",
    description=(
        "Reviews a source file for style, bugs, and improvement "
        "suggestions. Returns a structured Markdown review."
    ),
    system_prompt=CODE_REVIEW_SYSTEM_PROMPT,
    tools=CODE_REVIEW_TOOLS,
    orchestrator_prompt=(
        "You are a code-review orchestrator. Delegate review requests "
        "to the `code-review-agent` subagent via the `task` tool."
    ),
)

build_code_review_subagent_spec = CODE_REVIEW_SPEC.build_subagent_dict


def build_code_review_agent(
    *,
    model: Any | None = None,
    observer: TraceObserver | None = None,
    config_path: str | None = None,
) -> tuple[Any, TraceObserver]:
    """Build a runnable code-review agent + its trace observer (see
    :meth:`SubAgentSpec.build_agent`)."""
    return CODE_REVIEW_SPEC.build_agent(
        model=model, observer=observer, config_path=config_path
    )
