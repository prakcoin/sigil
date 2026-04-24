from __future__ import annotations

import os

from strands import Agent
from strands.models import BedrockModel

from ..core.inventory import Inventory

_SYSTEM_PROMPT = """\
You are a technical writer analyzing text artifacts from an agentic AI system.
Given a set of system prompts and tool descriptions, draft a sigil.yaml specification
that captures the cross-cutting concerns you observe across all agents.

The sigil.yaml format is:

tone: |
  <prose describing the tone and voice conventions you observe>

vocabulary:
  - canonical: <preferred term>
    avoid:
      - <synonym to avoid>

required_constraints:
  - <constraint every agent should enforce, one per line>

Output ONLY valid YAML. No explanations, no markdown fences, no preamble.\
"""


def draft_spec(inventory: Inventory) -> str:
    """Run the spec drafter agent and return raw YAML string for human review."""
    model = BedrockModel(
        model_id=os.environ.get("SIGIL_MODEL_ID", "us.amazon.nova-2-lite-v1:0"),
        temperature=0.0,
        max_tokens=12000,
    )
    agent = Agent(system_prompt=_SYSTEM_PROMPT, model=model, callback_handler=None)

    prompt = (
        "Here are all text artifacts from the project. "
        "Analyze them and draft the sigil.yaml specification.\n\n"
        + inventory.to_prompt_text()
    )

    result = str(agent(prompt)).strip()
    # Strip markdown code fences if the model ignores the "no fences" instruction
    if result.startswith("```"):
        lines = result.splitlines()
        result = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return result.strip()
