from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from strands import Agent
from strands.models import BedrockModel

from ..core.inventory import Inventory
from ..core.models import Finding, FindingCategory, Severity, make_id
from ..core.spec import Spec

_SHARED_CONTEXT = """\
You are analyzing text artifacts from an agentic AI system.
Each artifact has an ID, type, agent name, and content.
Respond ONLY with a JSON array of findings. Each finding must have:
  - "category": one of tone|vocabulary|constraints|routing
  - "severity": one of info|warning|error
  - "description": clear explanation of the issue
  - "affected_artifact_ids": list of artifact IDs involved

Return [] if you find no issues. No explanation outside the JSON array.\
"""

_TONE_PROMPT = """\
{context}

Analyze these artifacts for TONE inconsistencies only.
The expected tone is:
{tone}

Look for: mixed formality levels, inconsistent voice (first-person vs third-person),
varying levels of apology or hedging, different error message styles.\
"""

_VOCAB_PROMPT = """\
{context}

Analyze these artifacts for VOCABULARY inconsistencies only.
The canonical vocabulary is:
{vocabulary}

Look for: use of non-canonical synonyms, inconsistent naming of the same concept
across artifacts, terms that should be standardized.\
"""

_CONSTRAINTS_PROMPT = """\
{context}

Analyze these artifacts for missing or inconsistent CONSTRAINTS only.
Required constraints every agent should enforce:
{constraints}

Look for: agents missing required fallback responses, inconsistent scope-limiting
language, constraints stated differently across agents.\
"""

_ROUTING_PROMPT = """\
{context}

Analyze these artifacts for ROUTING CONSISTENCY only.
Look for: orchestrator descriptions of a subagent that contradict that subagent's
own system prompt, capability claims in routing logic that don't match what the
subagent actually does, subagents referenced by name that have no matching artifact.\
"""


def _make_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.environ.get("SIGIL_MODEL_ID", "us.amazon.nova-lite-v1:0"),
        temperature=0.0,
    )


def _run_analyzer(prompt: str, category: FindingCategory) -> list[Finding]:
    agent = Agent(system_prompt=_SHARED_CONTEXT, model=_make_model(), callback_handler=None)
    raw = str(agent(prompt)).strip()

    # Strip markdown fences if the model adds them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    findings: list[Finding] = []
    for item in items:
        findings.append(Finding(
            id=make_id(category.value, str(uuid.uuid4())),
            category=FindingCategory(item.get("category", category.value)),
            severity=Severity(item.get("severity", "warning")),
            description=item["description"],
            affected_artifact_ids=item.get("affected_artifact_ids", []),
        ))
    return findings


def run_analysis(inventory: Inventory, spec: Spec) -> list[Finding]:
    """Run all four analyzers in parallel and return merged findings."""
    artifacts_text = inventory.to_prompt_text()

    vocab_text = "\n".join(
        f"  canonical: {e.canonical}, avoid: {e.avoid}"
        for e in spec.vocabulary
    ) or "  (no vocabulary spec defined)"

    constraints_text = "\n".join(
        f"  - {c}" for c in spec.required_constraints
    ) or "  (no constraints spec defined)"

    prompts = [
        (
            _TONE_PROMPT.format(context=artifacts_text, tone=spec.tone or "(no tone spec defined)"),
            FindingCategory.TONE,
        ),
        (
            _VOCAB_PROMPT.format(context=artifacts_text, vocabulary=vocab_text),
            FindingCategory.VOCABULARY,
        ),
        (
            _CONSTRAINTS_PROMPT.format(context=artifacts_text, constraints=constraints_text),
            FindingCategory.CONSTRAINTS,
        ),
        (
            _ROUTING_PROMPT.format(context=artifacts_text),
            FindingCategory.ROUTING,
        ),
    ]

    findings: list[Finding] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_analyzer, prompt, category): category
            for prompt, category in prompts
        }
        for future in as_completed(futures):
            findings.extend(future.result())

    return findings
