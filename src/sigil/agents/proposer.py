from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel

from ..core.inventory import Inventory
from ..core.models import Finding, ProposedChange


class _ChangeItem(BaseModel):
    artifact_id: str = Field(description="The artifact ID to change")
    original: str = Field(description="The complete current content of the artifact, copied verbatim")
    proposed: str = Field(description="The complete revised content of the artifact")
    reasoning: str = Field(description="One sentence explaining the change")


class _ChangesResult(BaseModel):
    changes: list[_ChangeItem]


_SYSTEM_PROMPT = """\
You are a technical editor improving text artifacts in an agentic AI system.
Given a finding and the affected artifacts, propose specific text changes.

Edit thoughtfully so the result reads naturally. Never add new sentences,
bullet points, or lines — only rephrase or remove existing text. If an
artifact does not contain the specific problematic text, return it unchanged.

Return an empty changes list if no change is needed.\
"""


def _make_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.environ.get("SIGIL_MODEL_ID", "us.amazon.nova-2-lite-v1:0"),
        temperature=0.0,
        max_tokens=12000,
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
    result = agent(prompt, structured_output_model=_ChangesResult)
    items = result.structured_output.changes if result.structured_output else []

    changes: list[ProposedChange] = []
    for item in items:
        if inventory.get(item.artifact_id) is None:
            continue
        # Reject proposals that add new lines — a line count increase means
        # new bullets or sentences were invented rather than existing text edited.
        if item.proposed.strip().count('\n') > item.original.strip().count('\n'):
            continue
        changes.append(ProposedChange(
            artifact_id=item.artifact_id,
            original=item.original,
            proposed=item.proposed,
            reasoning=item.reasoning,
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
            try:
                findings[i].proposed_changes = future.result()
            except Exception:
                pass

    return findings
