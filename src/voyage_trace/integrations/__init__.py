"""Optional, SDK-backed integrations for voyage_trace.

This package holds the **bidirectional / SDK-using** integration helpers that
sit *beside* the pure-pull, SDK-free :mod:`voyage_trace.adapters`. Every
module here:

* imports its upstream SDK **lazily** inside the function that needs it;
* degrades **gracefully** (returns a JSON-safe artefact, a ``skipped``
  verdict, or falls back to an alternative) when the SDK is absent — so
  ``voyage_trace`` never hard-depends on deepeval / langfuse / the Azure
  Content Safety SDK / FLAML being installed;
* never edits the core :class:`~voyage_trace.types.CanonicalTrace` /
  :class:`~voyage_trace.types.TraceSpan` schema.

Available integrations:

* :mod:`voyage_trace.integrations.deepeval` — push a governance outcome
  into a DeepEval ``EvaluationDataset``; run / pull DeepEval metric results
  back as a :class:`CanonicalTrace` of score spans.
* :mod:`voyage_trace.integrations.langfuse_export` — push a
  :class:`CanonicalTrace` into Langfuse via the v3/v4 OTel-based SDK
  (spans + ``create_score``); no-op when the SDK is absent (the pull side
  already lives in :mod:`voyage_trace.adapters.langfuse`).
* :mod:`voyage_trace.integrations.acs` — generic safety / consistency
  scorer (Azure Content Safety when the SDK is present; ``skipped``
  otherwise). See the module docstring for the "ACS" naming caveat.
* :mod:`voyage_trace.integrations.flaml_runner` — FLAML as an alternative
  AutoML backend behind the same
  ``FeatureMatrix -> AutoMLReport`` interface as
  :func:`voyage_trace.automl.run_automl`.

Install the optional dependencies to enable the SDK paths::

    pip install "voyage-trace[integrations]"
"""

from __future__ import annotations

from .acs import acs_score
from .deepeval import from_deepeval_results, to_deepeval_dataset
from .flaml_runner import run_automl_flaml
from .langfuse_export import export_to_langfuse

__all__ = [
    "acs_score",
    "from_deepeval_results",
    "to_deepeval_dataset",
    "run_automl_flaml",
    "export_to_langfuse",
]
