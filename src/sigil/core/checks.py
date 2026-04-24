from __future__ import annotations

import re

from .models import Finding, FindingCategory, ProposedChange, Severity, make_id
from .spec import Spec
from .inventory import Inventory


def check_vocabulary(inventory: Inventory, spec: Spec) -> list[Finding]:
    """Deterministic vocabulary check: flag and propose fixes for any avoid-term found."""
    findings: list[Finding] = []

    for entry in spec.vocabulary:
        for avoid_term in entry.avoid:
            pattern = re.compile(rf"\b{re.escape(avoid_term)}\b", re.IGNORECASE)
            for artifact in inventory.all():
                if not pattern.search(artifact.content):
                    continue

                proposed_content = pattern.sub(entry.canonical, artifact.content)
                findings.append(Finding(
                    id=make_id("vocabulary", artifact.id, avoid_term),
                    category=FindingCategory.VOCABULARY,
                    severity=Severity.WARNING,
                    description=(
                        f'Found "{avoid_term}" in {artifact.agent_name} / '
                        f'{artifact.type.value} — use "{entry.canonical}" instead.'
                    ),
                    affected_artifact_ids=[artifact.id],
                    proposed_changes=[ProposedChange(
                        artifact_id=artifact.id,
                        original=artifact.content,
                        proposed=proposed_content,
                        reasoning=f'Replace "{avoid_term}" with canonical term "{entry.canonical}".',
                    )],
                ))

    return findings
