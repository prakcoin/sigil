from __future__ import annotations

from pathlib import Path

import yaml

from .models import Spec, SpecException, VocabularyEntry

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
        raw = f.read()
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"sigil.yaml is not valid YAML — fix the file and re-scan.\n{e}") from e

    vocab = [
        VocabularyEntry(
            canonical=entry["canonical"],
            avoid=entry.get("avoid", []),
            definition=entry.get("definition", ""),
        )
        for entry in data.get("vocabulary", [])
    ]

    exceptions = [
        SpecException(
            artifact_id=e["artifact_id"],
            category=e["category"],
            reason=e.get("reason", ""),
        )
        for e in data.get("exceptions", [])
    ]

    return Spec(
        tone=data.get("tone", ""),
        vocabulary=vocab,
        required_constraints=data.get("required_constraints", []),
        examples=data.get("examples", {}),
        exceptions=exceptions,
    )


def write_spec(project_root: Path, content: str) -> None:
    spec_path(project_root).write_text(content)


def _literal_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def add_exception(project_root: Path, artifact_id: str, category: str, reason: str = "") -> None:
    path = spec_path(project_root)
    if not path.exists():
        return

    with open(path) as f:
        data = yaml.safe_load(f.read()) or {}

    existing = data.get("exceptions", [])
    for e in existing:
        if e.get("artifact_id") == artifact_id and e.get("category") == category:
            return

    existing.append({"artifact_id": artifact_id, "category": category, "reason": reason})
    data["exceptions"] = existing

    dumper = yaml.Dumper
    dumper.add_representer(str, _literal_str_representer)
    path.write_text(yaml.dump(data, Dumper=dumper, default_flow_style=False, allow_unicode=True, sort_keys=False))
