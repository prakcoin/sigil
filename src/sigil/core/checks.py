from __future__ import annotations

import re

from .models import Finding, FindingCategory, ProposedChange, Severity, make_id
from .spec import Spec
from .inventory import Inventory


# Pure grammatical function words — can never be meaningful vocabulary entries.
_FUNCTION_WORDS = frozenset({
    'a', 'an', 'the',
    'and', 'or', 'but', 'nor', 'so', 'yet',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'over', 'after', 'before', 'about', 'up', 'out', 'through',
    'between', 'during', 'without', 'within', 'across', 'per', 'via',
    'than', 'then', 'when', 'where', 'while', 'since', 'unless',
    'although', 'though', 'whether', 'however',
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your',
    'he', 'she', 'his', 'her', 'it', 'its', 'they', 'them', 'their',
    'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'which',
    'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'must', 'can',
    'not', 'no', 'also', 'just', 'very', 'both', 'either', 'neither',
    'each', 'every', 'all', 'any', 'few', 'more', 'most', 'some', 'such',
    'other', 'same',
})

_TOKEN_RE = re.compile(r'\b[a-zA-Z][a-zA-Z-]{2,}\b')


def extract_vocabulary_candidates(inventory: Inventory, top_n: int = 50) -> list[str]:
    """Return terms that appear across multiple artifacts but not so widely as to be generic.

    Terms are ranked by cross-artifact frequency. Too rare (< 2 artifacts) and too
    common (> 80% of artifacts) are both excluded, leaving domain-specific shared
    vocabulary as candidates for the spec drafter to define.
    """
    artifacts = inventory.all()
    total = len(artifacts)
    if total == 0:
        return []

    upper = max(2, int(total * 0.8))

    term_count: dict[str, int] = {}
    for artifact in artifacts:
        tokens = {t.lower() for t in _TOKEN_RE.findall(artifact.content)} - _FUNCTION_WORDS
        for token in tokens:
            term_count[token] = term_count.get(token, 0) + 1

    candidates = [t for t, n in term_count.items() if 2 <= n <= upper]
    candidates.sort(key=lambda t: term_count[t], reverse=True)
    return candidates[:top_n]


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
