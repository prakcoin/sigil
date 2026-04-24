from __future__ import annotations

import re

from .models import Finding, FindingCategory, Severity, make_id
from .spec import Spec
from .inventory import Inventory


def check_vocabulary(inventory: Inventory, spec: Spec) -> list[Finding]:
    """Deterministic vocabulary check: flag any avoid-term found in artifact content."""
    findings: list[Finding] = []

    for entry in spec.vocabulary:
        for avoid_term in entry.avoid:
            pattern = re.compile(rf"\b{re.escape(avoid_term)}\b", re.IGNORECASE)
            for artifact in inventory.all():
                if pattern.search(artifact.content):
                    findings.append(Finding(
                        id=make_id("vocabulary", artifact.id, avoid_term),
                        category=FindingCategory.VOCABULARY,
                        severity=Severity.WARNING,
                        description=(
                            f'Found "{avoid_term}" in {artifact.agent_name} / '
                            f'{artifact.type.value} — use "{entry.canonical}" instead.'
                        ),
                        affected_artifact_ids=[artifact.id],
                    ))

    return findings
