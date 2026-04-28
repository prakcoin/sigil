from __future__ import annotations

from pathlib import Path

import yaml

from .models import Spec, VocabularyEntry

SPEC_FILENAME = "sigil.yaml"


def spec_path(project_root: Path) -> Path:
    return project_root / SPEC_FILENAME


def spec_exists(project_root: Path) -> bool:
    return spec_path(project_root).exists()


def load_spec(project_root: Path) -> Spec:
    path = spec_path(project_root)
    if not path.exists():
        return Spec()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    vocab = [
        VocabularyEntry(
            canonical=entry["canonical"],
            avoid=entry.get("avoid", []),
            definition=entry.get("definition", ""),
        )
        for entry in data.get("vocabulary", [])
    ]

    return Spec(
        tone=data.get("tone", ""),
        vocabulary=vocab,
        required_constraints=data.get("required_constraints", []),
        examples=data.get("examples", {}),
    )


def write_spec(project_root: Path, content: str) -> None:
    spec_path(project_root).write_text(content)
