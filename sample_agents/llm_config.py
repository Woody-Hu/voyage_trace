"""Config-file LLM setup for sample agents.

The DeepSeek API key (and any other credentials) are read from a YAML config
file *only*. The config file path defaults to ``sample_agents/config.yaml``
but can be overridden via the ``VOYAGE_TRACE_LLM_CONFIG`` env var. The config
file is in ``.gitignore`` so credentials never land in source.

Configuration shape (YAML)::

    default:
      provider: deepseek
      model: deepseek-chat
      base_url: https://api.deepseek.com
      api_key_env: DEEPSEEK_API_KEY   # name of env var holding the key
      temperature: 0.2
    research_agent:
      provider: deepseek
      model: deepseek-reasoner
      base_url: https://api.deepseek.com
      api_key_env: DEEPSEEK_API_KEY
      temperature: 0.0

Key resolution order (per ``LLMConfig.api_key``):

1. ``api_key`` field in the YAML (rarely used; if you must hardcode for a
   local dev box, put it here and git-ignore the file).
2. ``api_key_env`` in the YAML — the *name* of an env var holding the key
   (default ``DEEPSEEK_API_KEY``). The env var's *value* is returned.
3. ``DEEPSEEK_API_KEY`` env var directly (backwards-compatible default).

If none of these resolve, :meth:`build_chat_model` raises a clear
``RuntimeError`` listing the config path and the env vars that were checked —
never a silent fallback to a "free" or hardcoded key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATHS: tuple[str, ...] = (
    "sample_agents/config.yaml",
    "sample_agents/config.example.yaml",
    "config/llm.yaml",
)


@dataclass
class LLMConfig:
    """One model configuration (one entry in the YAML)."""

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.2
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_api_key(self, *, config_path: str = "<config>") -> str:
        """Resolve the API key, with a clear error if missing.

        Resolution order: ``api_key`` field → env var named by
        ``api_key_env`` → ``DEEPSEEK_API_KEY`` env var.

        Raises ``RuntimeError`` (not ``ImportError``) when no key is found,
        listing the env vars that were checked so the operator knows what
        to export.
        """
        if self.api_key:
            return self.api_key
        env_vars = [self.api_key_env, "DEEPSEEK_API_KEY"]
        for name in env_vars:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        raise RuntimeError(
            f"No API key resolved for {self.provider}/{self.model}. "
            f"Checked env vars {env_vars} and the `api_key` field in {config_path}. "
            "Set the env var (recommended) or add `api_key:` to the config file."
        )

    def build_chat_model(self, *, config_path: str = "<config>") -> Any:
        """Build a LangChain ``BaseChatModel`` from this config.

        Currently supports the ``deepseek`` provider (via
        ``langchain_openai.ChatOpenAI`` with a DeepSeek ``base_url``) and
        ``openai`` (same constructor, default OpenAI base URL). Other
        providers raise ``ValueError`` with a clear message rather than a
        silent import failure.
        """
        api_key = self.resolved_api_key(config_path=config_path)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain_openai is not installed; install it with "
                "`pip install langchain-openai` to use LLM-backed sample agents."
            ) from e

        if self.provider in ("deepseek", "openai"):
            return ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                **self.extra,
            )
        raise ValueError(
            f"Unknown provider {self.provider!r}. Supported: 'deepseek', 'openai'."
        )


@dataclass
class LLMConfigSet:
    """A loaded config file: ``default`` + named per-agent overrides."""

    default: LLMConfig = field(default_factory=LLMConfig)
    overrides: dict[str, LLMConfig] = field(default_factory=dict)
    path: str = "<config>"

    def for_agent(self, name: str) -> LLMConfig:
        """Return the config for ``name``, falling back to ``default``."""
        if name in self.overrides:
            return self.overrides[name]
        return self.default


def _find_config_path() -> str | None:
    """Resolve the config path: ``VOYAGE_TRACE_LLM_CONFIG`` env var first."""
    env_path = os.environ.get("VOYAGE_TRACE_LLM_CONFIG", "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    for candidate in DEFAULT_CONFIG_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


def load_config(path: str | None = None) -> LLMConfigSet:
    """Load the LLM config; return an empty :class:`LLMConfigSet` if absent.

    When no config file is found, the returned set has ``default`` set to a
    plain DeepSeek config (with no API key — :meth:`build_chat_model` will
    raise a clear error if you try to use it without an env var). This means
    importing the module never fails; only *using* a model does.
    """
    resolved = path or _find_config_path()
    if not resolved:
        return LLMConfigSet(path="<no config file>")
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is not installed; install it with `pip install pyyaml`."
        ) from e

    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    default = _parse_section(raw.get("default", {}))
    overrides: dict[str, LLMConfig] = {}
    for name, section in raw.items():
        if name == "default":
            continue
        if isinstance(section, dict):
            overrides[name] = _parse_section(section)
    return LLMConfigSet(default=default, overrides=overrides, path=resolved)


def _parse_section(section: dict[str, Any]) -> LLMConfig:
    extra = {k: v for k, v in section.items()
             if k not in {"provider", "model", "base_url", "api_key",
                          "api_key_env", "temperature"}}
    return LLMConfig(
        provider=section.get("provider", "deepseek"),
        model=section.get("model", "deepseek-chat"),
        base_url=section.get("base_url", "https://api.deepseek.com"),
        api_key=section.get("api_key"),
        api_key_env=section.get("api_key_env", "DEEPSEEK_API_KEY"),
        temperature=float(section.get("temperature", 0.2)),
        extra=extra,
    )
