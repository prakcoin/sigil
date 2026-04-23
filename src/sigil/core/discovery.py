from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import yaml

from .models import Artifact, ArtifactRole, ArtifactType, make_id

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".sigil"}

# (string_value, source_segment, line_start, line_end)
_StringVar = tuple[str, str, int, int]


def discover(project_root: Path) -> list[Artifact]:
    """Walk all Python files and SKILL.md files and return discovered artifacts."""
    artifacts: list[Artifact] = []
    for py_file in sorted(project_root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in py_file.parts):
            continue
        artifacts.extend(_discover_in_file(py_file, project_root))

    # Build skills-dir → agent-name mapping from AgentSkills(skills=...) call sites
    skills_map = _collect_skills_map(project_root)

    for skill_file in sorted(project_root.rglob("SKILL.md")):
        if any(part in _SKIP_DIRS for part in skill_file.parts):
            continue
        artifact = _discover_skill_file(skill_file, project_root, skills_map)
        if artifact:
            artifacts.append(artifact)
    return artifacts


def _collect_skills_map(project_root: Path) -> dict[str, str]:
    """Scan Python files for AgentSkills(skills=...) calls.

    Returns a mapping of skill directory path (relative to project_root)
    to the agent name that registers those skills.
    e.g. {"src/agents/skills/archive_skills": "archive_assistant"}
    """
    result: dict[str, str] = {}
    for py_file in sorted(project_root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in py_file.parts):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        string_vars = _collect_string_vars(tree, source)
        visitor = _SkillsMapVisitor(py_file, project_root, string_vars)
        visitor.visit(tree)
        result.update(visitor.skills_map)
    return result


class _SkillsMapVisitor(ast.NodeVisitor):
    """Finds AgentSkills(skills=...) calls and maps skill dirs to agent names."""

    def __init__(
        self,
        py_file: Path,
        project_root: Path,
        string_vars: dict[str, _StringVar],
    ):
        self.py_file = py_file
        self.project_root = project_root
        self.string_vars = string_vars
        self.skills_map: dict[str, str] = {}
        self._ctx: list[tuple[str, str]] = []

    @property
    def _agent_name(self) -> str:
        for kind, name in reversed(self._ctx):
            if kind == "tool":
                return name
        for kind, name in reversed(self._ctx):
            if kind == "class":
                return name
        return self.py_file.stem

    def visit_ClassDef(self, node: ast.ClassDef):
        self._ctx.append(("class", node.name))
        self.generic_visit(node)
        self._ctx.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        self._ctx.append(("tool" if _has_tool_decorator(node) else "function", node.name))
        self.generic_visit(node)
        self._ctx.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        if _call_name(node) == "AgentSkills":
            for kw in node.keywords:
                if kw.arg == "skills":
                    path_str = _extract_string(kw.value)
                    if not path_str and isinstance(kw.value, ast.Name):
                        var = self.string_vars.get(kw.value.id)
                        if var:
                            path_str = var[0]
                    if path_str:
                        resolved = (self.py_file.parent / path_str).resolve()
                        try:
                            rel = str(resolved.relative_to(self.project_root.resolve()))
                            self.skills_map[rel] = self._agent_name
                        except ValueError:
                            pass
        self.generic_visit(node)


def _discover_skill_file(
    skill_file: Path,
    project_root: Path,
    skills_map: dict[str, str],
) -> Optional[Artifact]:
    """Parse a SKILL.md file and return a skill artifact."""
    try:
        text = skill_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None

    name, body = _parse_skill_md(text)
    if not body.strip():
        return None

    rel_path = str(skill_file.relative_to(project_root))

    # Walk up the skill file's parents looking for a match in the skills_map
    agent_name = None
    for parent in skill_file.parents:
        try:
            rel_parent = str(parent.relative_to(project_root))
        except ValueError:
            break
        if rel_parent in skills_map:
            agent_name = skills_map[rel_parent]
            break
    if not agent_name:
        agent_name = _infer_skill_agent(skill_file, project_root)

    skill_name = name or skill_file.parent.name  # noqa: F841

    return Artifact(
        id=make_id(rel_path, "0", "skill"),
        type=ArtifactType.SKILL,
        role=ArtifactRole.OWNED_BY,
        content=body.strip(),
        file_path=rel_path,
        line_start=1,
        line_end=text.count("\n") + 1,
        agent_name=agent_name,
        source_segment=body.strip(),  # body only — no Python quotes
        referenced_agent=None,
    )


def _parse_skill_md(text: str) -> tuple[str, str]:
    """Return (name, body) from a SKILL.md file.

    Body is everything after the closing frontmatter fence.
    If there is no frontmatter, the entire text is the body.
    """
    if not text.startswith("---"):
        return "", text

    end = text.find("\n---", 3)
    if end == -1:
        return "", text

    try:
        frontmatter = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        frontmatter = {}

    name = frontmatter.get("name", "")
    body = text[end + 4:].lstrip("\n")
    return name, body


def _infer_skill_agent(skill_file: Path, project_root: Path) -> str:
    """Infer agent name from the skills directory structure.

    Looks for a 'skills' directory in the path and uses its immediate child
    as the agent group, stripping common suffixes (_skills, -skills).
    Falls back to the skill's parent directory name.

    e.g. src/agents/skills/archive_skills/look-analysis/SKILL.md → archive
    """
    parts = skill_file.relative_to(project_root).parts
    for i, part in enumerate(parts):
        if part == "skills" and i + 1 < len(parts):
            group = parts[i + 1]
            return group.removesuffix("_skills").removesuffix("-skills")
    return skill_file.parent.parent.name


def _discover_in_file(file_path: Path, project_root: Path) -> list[Artifact]:
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    rel_path = str(file_path.relative_to(project_root))
    assignment_map = _collect_agent_assignments(tree)
    string_vars = _collect_string_vars(tree, source)
    visitor = _ArtifactVisitor(rel_path, source, assignment_map, string_vars)
    visitor.visit(tree)
    return visitor.artifacts


def _collect_agent_assignments(tree: ast.Module) -> dict[int, str]:
    """Map line number of an Agent() call to the bare variable name it's assigned to.

    Handles `my_agent = Agent(...)`. Does not handle `self.x = Agent(...)` —
    those fall back to class-name context.
    """
    result: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _call_name(node.value) == "Agent" and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    result[node.value.lineno] = target.id
    return result


def _collect_string_vars(tree: ast.Module, source: str) -> dict[str, _StringVar]:
    """Map variable name to (value, source_segment, line_start, line_end).

    Handles `PROMPT = "..."` and `PROMPT = \"\"\"...\"\"\"` at any scope.
    When system_prompt=PROMPT, we resolve to this definition so line numbers
    and source_segment point at the actual string, not the reference.
    """
    result: dict[str, _StringVar] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                content = _extract_string(node.value)
                if content:
                    segment = ast.get_source_segment(source, node.value) or ""
                    result[target.id] = (
                        content,
                        segment,
                        node.value.lineno,
                        node.value.end_lineno,
                    )
    return result


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _has_tool_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "tool":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "tool":
            return True
    return False


def _extract_string(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _ArtifactVisitor(ast.NodeVisitor):
    def __init__(
        self,
        rel_path: str,
        source: str,
        assignment_map: dict[int, str],
        string_vars: dict[str, _StringVar],
    ):
        self.rel_path = rel_path
        self.source = source
        self.assignment_map = assignment_map
        self.string_vars = string_vars
        self.artifacts: list[Artifact] = []
        self._ctx: list[tuple[str, str]] = []  # ("tool"|"class"|"function", name)

    @property
    def _agent_name(self) -> str:
        for kind, name in reversed(self._ctx):
            if kind == "tool":
                return name
        for kind, name in reversed(self._ctx):
            if kind == "class":
                return name
        for kind, name in reversed(self._ctx):
            if kind == "function":
                return name
        return Path(self.rel_path).stem

    def visit_ClassDef(self, node: ast.ClassDef):
        self._ctx.append(("class", node.name))
        self.generic_visit(node)
        self._ctx.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        is_tool = _has_tool_decorator(node)
        self._ctx.append(("tool" if is_tool else "function", node.name))

        if is_tool:
            docstring = ast.get_docstring(node)
            if docstring and node.body:
                doc_node = node.body[0]
                segment = ast.get_source_segment(self.source, doc_node.value) or ""
                self.artifacts.append(Artifact(
                    id=make_id(self.rel_path, str(node.lineno), "tool_description"),
                    type=ArtifactType.TOOL_DESCRIPTION,
                    role=ArtifactRole.OWNED_BY,
                    content=docstring,
                    file_path=self.rel_path,
                    line_start=doc_node.lineno,
                    line_end=doc_node.end_lineno,
                    agent_name=node.name,
                    source_segment=segment,
                ))

        self.generic_visit(node)
        self._ctx.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _resolve_prompt(self, value_node: ast.expr) -> Optional[tuple[str, str, int, int]]:
        """Return (content, source_segment, line_start, line_end) or None if unresolvable."""
        content = _extract_string(value_node)
        if content:
            segment = ast.get_source_segment(self.source, value_node) or ""
            return content, segment, value_node.lineno, value_node.end_lineno
        if isinstance(value_node, ast.Name) and value_node.id in self.string_vars:
            return self.string_vars[value_node.id]
        return None

    def _resolve_agent_name(self, call_node: ast.Call) -> str:
        """Resolve the owning agent name for a call node."""
        for kind, name in reversed(self._ctx):
            if kind == "tool":
                return name
        for k in call_node.keywords:
            if k.arg == "name":
                name = _extract_string(k.value)
                if name:
                    return name
        return self.assignment_map.get(call_node.lineno) or self._agent_name

    def visit_Call(self, node: ast.Call):
        name = _call_name(node)
        if not name:
            self.generic_visit(node)
            return

        is_agent = name == "Agent"
        is_handler = not is_agent and name.endswith("Handler")

        if is_agent or is_handler:
            for kw in node.keywords:
                if kw.arg == "system_prompt":
                    resolved = self._resolve_prompt(kw.value)
                    if not resolved:
                        break
                    content, segment, line_start, line_end = resolved
                    if not content.strip():
                        break

                    artifact_type = (
                        ArtifactType.SYSTEM_PROMPT if is_agent
                        else ArtifactType.HANDLER_PROMPT
                    )
                    self.artifacts.append(Artifact(
                        id=make_id(self.rel_path, str(line_start), artifact_type.value),
                        type=artifact_type,
                        role=ArtifactRole.OWNED_BY,
                        content=content,
                        file_path=self.rel_path,
                        line_start=line_start,
                        line_end=line_end,
                        agent_name=self._resolve_agent_name(node),
                        source_segment=segment,
                    ))
                    break

        self.generic_visit(node)
