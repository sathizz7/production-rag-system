from __future__ import annotations

import litellm
import structlog

from rag.config import Settings, apply_provider_env, get_settings

log = structlog.get_logger()


def configure_observability(settings: Settings | None = None) -> bool:
    """Enable the LiteLLM→Langfuse callback when Langfuse keys are configured.

    LiteLLM emits a trace per model call (cost, tokens, latency) to Langfuse via
    its built-in callback; we only register it and export the keys. Returns True
    if tracing was enabled, False if it is a no-op (no keys). Consult the current
    LiteLLM + Langfuse docs before changing callback wiring (spec §18).
    """
    settings = settings or get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    apply_provider_env(settings)  # exports LANGFUSE_* into os.environ for LiteLLM
    if "langfuse" not in litellm.success_callback:
        litellm.success_callback = [*litellm.success_callback, "langfuse"]
    if "langfuse" not in litellm.failure_callback:
        litellm.failure_callback = [*litellm.failure_callback, "langfuse"]
    log.info("observability_enabled", backend="langfuse", host=settings.langfuse_host)
    return True
