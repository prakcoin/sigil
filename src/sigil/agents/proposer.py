from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from strands import Agent
from strands.models import BedrockModel

from ..core.inventory import Inventory
from ..core.models import Finding, ProposedChange

_SYSTEM_PROMPT = """\
You are a technical editor improving text artifacts in an agentic AI system.
Given a finding and the affected artifacts, propose specific text changes.

Respond ONLY with a JSON array. Each element must have:
  - "artifact_id": the artifact ID to change
  - "original": the EXACT current content of that artifact (copy verbatim)
  - "proposed": your improved version
  - "reasoning": one sentence explaining the change

Return [] if no change is needed. No explanation outside the JSON array.\
"""


def _make_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.environ.get("SIGIL_MODEL_ID", "us.amazon.nova-lite-v1:0"),
        temperature=0.0,
    )


def _propose_for_finding(finding: Finding, inventory: Inventory) -> list[ProposedChange]:
    artifacts_text = "\n\n".join(
        f"[{a.id}] {a.type.value} (agent: {a.agent_name})\n{a.content}"
        for aid in finding.affected_artifact_ids
        if (a := inventory.get(aid)) is not None
    )
    if not artifacts_text:
        return []

    prompt = (
        f"Finding ({finding.severity.value} / {finding.category.value}):\n"
        f"{finding.description}\n\n"
        f"Affected artifacts:\n{artifacts_text}\n\n"
        "Propose specific text changes to fix this finding."
    )

    agent = Agent(system_prompt=_SYSTEM_PROMPT, model=_make_model(), callback_handler=None)
    raw = str(agent(prompt)).strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    changes: list[ProposedChange] = []
    for item in items:
        artifact = inventory.get(item.get("artifact_id", ""))
        if artifact is None:
            continue
        changes.append(ProposedChange(
            artifact_id=item["artifact_id"],
            original=item["original"],
            proposed=item["proposed"],
            reasoning=item.get("reasoning", ""),
        ))
    return changes


def generate_proposals(findings: list[Finding], inventory: Inventory) -> list[Finding]:
    """Attach proposed changes to each finding. Runs proposals in parallel."""
    with ThreadPoolExecutor(max_workers=min(len(findings), 8)) as executor:
        futures = {
            executor.submit(_propose_for_finding, f, inventory): i
            for i, f in enumerate(findings)
        }
        for future in as_completed(futures):
            i = futures[future]
            findings[i].proposed_changes = future.result()

    return findings
