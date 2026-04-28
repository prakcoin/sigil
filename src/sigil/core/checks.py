from __future__ import annotations

import re

from .models import Finding, FindingCategory, ProposedChange, Severity, make_id
from .spec import Spec
from .inventory import Inventory


def check_vocabulary(inventory: Inventory, spec: Spec) -> list[Finding]:
    """Deterministic vocabulary check: one finding per artifact with all violations coalesced."""
    findings: list[Finding] = []

    for artifact in inventory.all():
        violations: list[tuple[str, str]] = []  # (avoid_term, canonical)
        proposed_content = artifact.content

        for entry in spec.vocabulary:
            for avoid_term in entry.avoid:
                pattern = re.compile(rf"\b{re.escape(avoid_term)}\b", re.IGNORECASE)
                if pattern.search(proposed_content):
                    violations.append((avoid_term, entry.canonical))
                    proposed_content = pattern.sub(entry.canonical, proposed_content)

        if not violations:
            continue

        terms_summary = ", ".join(f'"{t}" → "{c}"' for t, c in violations)
        findings.append(Finding(
            id=make_id("vocabulary", artifact.id),
            category=FindingCategory.VOCABULARY,
            severity=Severity.WARNING,
            description=(
                f'{artifact.agent_name} / {artifact.type.value} uses avoided vocabulary: '
                f'{terms_summary}.'
            ),
            affected_artifact_ids=[artifact.id],
            proposed_changes=[ProposedChange(
                artifact_id=artifact.id,
                original=artifact.content,
                proposed=proposed_content,
                reasoning=f'Replace avoided terms with canonical equivalents: {terms_summary}.',
            )],
        ))

    return findings
