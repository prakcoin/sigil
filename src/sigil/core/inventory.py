from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .models import Artifact, ArtifactType


class Inventory:
    def __init__(self, artifacts: list[Artifact]):
        self._artifacts = {a.id: a for a in artifacts}

    def all(self) -> list[Artifact]:
        return list(self._artifacts.values())

    def get(self, artifact_id: str) -> Optional[Artifact]:
        return self._artifacts.get(artifact_id)

    def by_agent(self) -> dict[str, list[Artifact]]:
        grouped: dict[str, list[Artifact]] = defaultdict(list)
        for a in self._artifacts.values():
            grouped[a.agent_name].append(a)
        return dict(grouped)

    def by_type(self, artifact_type: ArtifactType) -> list[Artifact]:
        return [a for a in self._artifacts.values() if a.type == artifact_type]

    def agents(self) -> list[str]:
        return sorted({a.agent_name for a in self._artifacts.values()})

    def __len__(self) -> int:
        return len(self._artifacts)

    def to_prompt_text(self) -> str:
        """Render inventory as a readable text block for agent prompts."""
        lines: list[str] = []
        for agent_name, artifacts in sorted(self.by_agent().items()):
            lines.append(f"## Agent: {agent_name}")
            for a in artifacts:
                lines.append(f"### {a.type.value} [{a.id}] ({a.file_path}:{a.line_start})")
                lines.append(a.content.strip())
                lines.append("")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([a.to_dict() for a in self._artifacts.values()], f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Inventory:
        with open(path) as f:
            data = json.load(f)
        return cls([Artifact.from_dict(d) for d in data])
