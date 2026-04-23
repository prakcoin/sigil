from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class ArtifactType(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    TOOL_DESCRIPTION = "tool_description"
    HANDLER_PROMPT = "handler_prompt"


class ArtifactRole(str, Enum):
    OWNED_BY = "owned_by"    # artifact belongs to agent_name
    DESCRIBES = "describes"  # artifact describes referenced_agent (e.g. routing logic)


class FindingCategory(str, Enum):
    TONE = "tone"
    VOCABULARY = "vocabulary"
    CONSTRAINTS = "constraints"
    ROUTING = "routing"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Artifact:
    id: str
    type: ArtifactType
    role: ArtifactRole
    content: str
    file_path: str
    line_start: int
    line_end: int
    agent_name: str
    source_segment: str          # exact text in source file including quotes
    referenced_agent: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Artifact:
        d["type"] = ArtifactType(d["type"])
        d["role"] = ArtifactRole(d["role"])
        return cls(**d)

    def preview(self, max_chars: int = 80) -> str:
        text = self.content.replace("\n", " ").strip()
        return text[:max_chars] + ("..." if len(text) > max_chars else "")


@dataclass
class VocabularyEntry:
    canonical: str
    avoid: list[str] = field(default_factory=list)


@dataclass
class Spec:
    tone: str = ""
    vocabulary: list[VocabularyEntry] = field(default_factory=list)
    required_constraints: list[str] = field(default_factory=list)
    examples: dict[str, str] = field(default_factory=dict)

    def vocabulary_set(self) -> dict[str, list[str]]:
        """Return {canonical: [avoid, ...]} for fast lookup."""
        return {e.canonical: e.avoid for e in self.vocabulary}


@dataclass
class ProposedChange:
    artifact_id: str
    original: str
    proposed: str
    reasoning: str


@dataclass
class Finding:
    id: str
    category: FindingCategory
    severity: Severity
    description: str
    affected_artifact_ids: list[str]
    proposed_changes: list[ProposedChange] = field(default_factory=list)
    approved: Optional[bool] = None  # None=pending, True=approved, False=rejected

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Finding:
        d["category"] = FindingCategory(d["category"])
        d["severity"] = Severity(d["severity"])
        d["proposed_changes"] = [ProposedChange(**c) for c in d["proposed_changes"]]
        return cls(**d)


def make_id(*parts: str) -> str:
    return hashlib.md5(":".join(parts).encode()).hexdigest()[:8]
