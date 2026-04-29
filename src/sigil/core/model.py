from __future__ import annotations

import os


_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_BEDROCK_MODEL = "us.amazon.nova-2-lite-v1:0"


def make_model():
    """Return a Strands model instance based on environment configuration.

    Provider resolution order:
    1. SIGIL_PROVIDER env var ("anthropic" or "bedrock") — explicit override
    2. ANTHROPIC_API_KEY present → AnthropicModel
    3. Fallback → BedrockModel (requires AWS credentials)

    Model ID is always overridable via SIGIL_MODEL_ID.
    """
    provider = os.environ.get("SIGIL_PROVIDER", "").lower()

    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "bedrock"

    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel
        model_id = os.environ.get("SIGIL_MODEL_ID", _DEFAULT_ANTHROPIC_MODEL)
        return AnthropicModel(
            model_id=model_id,
            max_tokens=12000,
            params={"temperature": 0.0},
        )

    from strands.models import BedrockModel
    model_id = os.environ.get("SIGIL_MODEL_ID", _DEFAULT_BEDROCK_MODEL)
    return BedrockModel(model_id=model_id, temperature=0.0, max_tokens=12000)
