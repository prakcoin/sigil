from __future__ import annotations

import difflib
import textwrap
import threading
from pathlib import Path

from rich.markup import escape

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, RichLog, Tree
from textual.widgets.tree import TreeNode

from .core.discovery import discover
from .core.inventory import Inventory
from .core.models import Artifact, ArtifactType, Finding
from .core.spec import Spec, load_spec


_TYPE_LABEL = {
    ArtifactType.SYSTEM_PROMPT:    "system_prompt",
    ArtifactType.TOOL_DESCRIPTION: "docstring",
    ArtifactType.HANDLER_PROMPT:   "handler",
    ArtifactType.SKILL:            "skill",
}


def _word_diff_line(original: str, proposed: str) -> tuple[str, str]:
    """Inline word-level diff for a single line. Returns (before_markup, after_markup)."""
    orig_words = original.split()
    prop_words = proposed.split()
    matcher = difflib.SequenceMatcher(None, orig_words, prop_words, autojunk=False)
    before_parts: list[str] = []
    after_parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        o = escape(" ".join(orig_words[i1:i2]))
        p = escape(" ".join(prop_words[j1:j2]))
        if tag == "equal":
            before_parts.append(o)
            after_parts.append(p)
        elif tag == "replace":
            before_parts.append(f"[bold red]{o}[/bold red]")
            after_parts.append(f"[bold green]{p}[/bold green]")
        elif tag == "delete":
            before_parts.append(f"[bold red]{o}[/bold red]")
        elif tag == "insert":
            after_parts.append(f"[bold green]{p}[/bold green]")
    return " ".join(before_parts), " ".join(after_parts)


def _diff_markup(original: str, proposed: str) -> tuple[str, str]:
    """Line-level diff with inline word-level highlighting for changed lines.
    Returns (before_markup, after_markup) ready for RichLog.write()."""
    orig_lines = original.splitlines()
    prop_lines = proposed.splitlines()
    matcher = difflib.SequenceMatcher(None, orig_lines, prop_lines, autojunk=False)
    before_lines: list[str] = []
    after_lines: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in orig_lines[i1:i2]:
                before_lines.append(escape(line))
            for line in prop_lines[j1:j2]:
                after_lines.append(escape(line))
        elif tag == "replace":
            orig_chunk = orig_lines[i1:i2]
            prop_chunk = prop_lines[j1:j2]
            if len(orig_chunk) == len(prop_chunk):
                for old, new in zip(orig_chunk, prop_chunk):
                    b, a = _word_diff_line(old, new)
                    before_lines.append(b)
                    after_lines.append(a)
            else:
                for line in orig_chunk:
                    before_lines.append(f"[red]{escape(line)}[/red]")
                for line in prop_chunk:
                    after_lines.append(f"[green]{escape(line)}[/green]")
        elif tag == "delete":
            for line in orig_lines[i1:i2]:
                before_lines.append(f"[red]{escape(line)}[/red]")
        elif tag == "insert":
            for line in prop_lines[j1:j2]:
                after_lines.append(f"[green]{escape(line)}[/green]")
    return "\n".join(before_lines), "\n".join(after_lines)


class ArtifactViewer(RichLog):
    """Displays the content of the selected artifact."""

    DEFAULT_CSS = """
    ArtifactViewer {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
    }
    """

    def show(self, title: str, content: str) -> None:
        self.clear()
        self.write(f"[bold cyan]{title}[/bold cyan]\n\n{textwrap.dedent(content).strip()}")
        self.scroll_home(animate=False)


class FindingsPanel(RichLog):
    """Scrollable findings panel — contextual or global queue mode."""

    DEFAULT_CSS = """
    FindingsPanel {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    """


class SigilTUI(App):
    """Sigil watch — live artifact manager."""

    TITLE = "sigil watch"
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
        layout: horizontal;
    }
    #left {
        width: 30;
        border: solid $accent;
    }
    #right {
        width: 1fr;
        layout: vertical;
    }
    #status {
        height: 1;
        background: $boost;
        padding: 0 2;
        color: $text-muted;
    }
    Tree {
        height: 1fr;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Re-scan"),
        Binding("f", "toggle_panel", "Toggle findings view"),
        Binding("y", "approve", "Approve"),
        Binding("n", "reject", "Reject"),
        Binding("u", "undo_decision", "Undo decision"),
        Binding("a", "apply", "Apply approved"),
        Binding("escape", "deselect", "Deselect", show=False),
    ]

    _findings: reactive[list[Finding]] = reactive([], always_update=True)
    _current_finding_index: int = 0
    _panel_mode: str = "contextual"       # "contextual" | "global"
    _selected_artifact_id: str | None = None

    def __init__(self, project_root: Path):
        super().__init__()
        self._project_root = project_root
        self._inventory: Inventory | None = None
        self._spec: Spec | None = None
        self._lock = threading.Lock()
        self._watcher_stop: threading.Event | None = None
        self._decisions: dict[tuple, bool] = {}  # (category, frozenset(artifact_ids)) → approved
        self._scanning: bool = False

    @staticmethod
    def _finding_key(f: Finding) -> tuple:
        return (f.category, frozenset(f.affected_artifact_ids))

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("", id="status")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Tree("Project", id="agent-tree")
            with Vertical(id="right"):
                yield ArtifactViewer(id="viewer", highlight=False, markup=True, auto_scroll=False)
                yield FindingsPanel(id="findings", highlight=False, markup=True, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        self._do_scan()
        self._start_watcher()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _do_scan(self, changed: set[Path] | None = None) -> None:
        with self._lock:
            self._scanning = True
        self._set_status("scanning...")
        threading.Thread(target=self._scan_worker, args=(changed,), daemon=True).start()

    def _scan_worker(self, changed: set[Path] | None) -> None:
        try:
            artifacts = discover(self._project_root)
            inventory = Inventory(artifacts)
            spec = load_spec(self._project_root)

            with self._lock:
                self._inventory = inventory
                self._spec = spec

            self.call_from_thread(self._rebuild_tree, inventory)
            self._set_status(f"analyzing {len(artifacts)} artifact(s)...")
            self._run_analysis(inventory, spec)
        except Exception as e:
            with self._lock:
                self._scanning = False
            self._set_status(f"[red]scan error: {e}[/red]")

    def _run_analysis(self, inventory: Inventory, spec: Spec) -> None:
        try:
            from .agents.analyzers import run_analysis
            from .agents.proposer import generate_proposals

            findings = run_analysis(inventory, spec)
            findings = generate_proposals(findings, inventory, spec)
            findings = [f for f in findings if f.proposed_changes]

            with self._lock:
                for f in findings:
                    key = self._finding_key(f)
                    if key in self._decisions:
                        f.approved = self._decisions[key]
                self._findings = findings
                self._current_finding_index = 0
                self._scanning = False

            self.call_from_thread(self._refresh_findings)
            self.call_from_thread(self._update_status)
            with self._lock:
                inv = self._inventory
            if inv:
                self.call_from_thread(self._rebuild_tree, inv)
        except Exception as e:
            with self._lock:
                self._scanning = False
            self._set_status(f"[red]analysis error: {e}[/red]")

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def _pending_artifact_ids(self) -> set[str]:
        """Return artifact IDs that have at least one pending (unapproved) finding."""
        with self._lock:
            findings = list(self._findings)
        result: set[str] = set()
        for f in findings:
            if f.approved is None:
                for c in f.proposed_changes:
                    result.add(c.artifact_id)
        return result

    def _rebuild_tree(self, inventory: Inventory) -> None:
        tree: Tree = self.query_one("#agent-tree", Tree)
        tree.clear()

        pending_ids = self._pending_artifact_ids()

        sections: dict[str, list[tuple[str, list]]] = {
            "Agents": [],
            "Tools": [],
            "Handlers": [],
            "Skills": [],
        }

        for agent_name, artifacts in sorted(inventory.by_agent().items()):
            skill_artifacts = [a for a in artifacts if a.type == ArtifactType.SKILL]
            other_artifacts = [a for a in artifacts if a.type != ArtifactType.SKILL]

            if skill_artifacts:
                sections["Skills"].append((agent_name, skill_artifacts))

            if other_artifacts:
                types = {a.type for a in other_artifacts}
                if ArtifactType.SYSTEM_PROMPT in types:
                    section = "Agents"
                elif ArtifactType.HANDLER_PROMPT in types:
                    section = "Handlers"
                else:
                    section = "Tools"
                sections[section].append((agent_name, other_artifacts))

        section_styles = {
            "Agents":   "bold green",
            "Tools":    "bold yellow",
            "Handlers": "bold magenta",
            "Skills":   "bold blue",
        }

        for section_name, groups in sections.items():
            if not groups:
                continue
            section_node: TreeNode = tree.root.add(
                f"[{section_styles[section_name]}]{section_name}[/]",
                expand=True,
            )
            for agent_name, artifacts in groups:
                agent_pending = sum(1 for a in artifacts if a.id in pending_ids)
                badge = f"  [yellow bold]{agent_pending}●[/yellow bold]" if agent_pending else ""
                agent_node = section_node.add(
                    f"[cyan]{agent_name}[/cyan]{badge}", expand=True
                )
                for artifact in artifacts:
                    label = _TYPE_LABEL.get(artifact.type, artifact.type.value)
                    if artifact.id in pending_ids:
                        leaf_label = f"[yellow]{label}  ●[/yellow]"
                    else:
                        leaf_label = f"[dim]{label}[/dim]"
                    agent_node.add_leaf(leaf_label, data=artifact.id)

        tree.root.expand()

    def _on_artifact_node_focused(self, artifact_id: str) -> None:
        if not self._inventory:
            return
        artifact = self._inventory.get(artifact_id)
        if not artifact:
            return

        self._selected_artifact_id = artifact_id

        viewer: ArtifactViewer = self.query_one("#viewer", ArtifactViewer)
        title = f"{artifact.agent_name} / {_TYPE_LABEL.get(artifact.type, artifact.type.value)}"
        viewer.show(title, artifact.content)

        if self._panel_mode == "contextual":
            self._refresh_findings()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if event.node.data:
            self._on_artifact_node_focused(event.node.data)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data:
            self._on_artifact_node_focused(event.node.data)

    # ------------------------------------------------------------------
    # Findings panel
    # ------------------------------------------------------------------

    def action_toggle_panel(self) -> None:
        self._panel_mode = "global" if self._panel_mode == "contextual" else "contextual"
        self._refresh_findings()
        self._update_status()

    def action_deselect(self) -> None:
        if self._panel_mode == "contextual":
            self._selected_artifact_id = None
            viewer: ArtifactViewer = self.query_one("#viewer", ArtifactViewer)
            viewer.clear()
            self._refresh_findings()

    def _refresh_findings(self) -> None:
        if self._panel_mode == "contextual":
            self._render_contextual()
        else:
            self._render_global()

    def _render_contextual(self) -> None:
        panel: FindingsPanel = self.query_one("#findings", FindingsPanel)
        panel.clear()

        with self._lock:
            findings = list(self._findings)
            artifact_id = self._selected_artifact_id

        panel.write(
            "[dim]contextual  [/dim][bold]f[/bold][dim]=global queue[/dim]\n"
        )

        if not artifact_id:
            panel.write("[dim]Select an artifact to see its findings.[/dim]")
            return

        relevant = [
            f for f in findings
            if any(c.artifact_id == artifact_id for c in f.proposed_changes)
        ]

        if not relevant:
            panel.write("[green]✓  All good — no findings for this artifact.[/green]")
            return

        for f in relevant:
            sev_color = "red" if f.severity.value == "error" else "yellow"
            sev_label = f"[{sev_color} dim]{f.severity.value}[/{sev_color} dim]"
            status = ""
            if f.approved is True:
                status = "  [green]✓ approved[/green]"
            elif f.approved is False:
                status = "  [dim]✗ rejected[/dim]"

            panel.write(
                f"[{sev_color} bold]{f.category.value.upper()}[/{sev_color} bold]  "
                f"{sev_label}{status}"
            )
            panel.write(f"[white]{f.description}[/white]\n")

            if f.approved is None:
                for change in f.proposed_changes:
                    if change.artifact_id != artifact_id:
                        continue
                    before_markup, after_markup = _diff_markup(
                        textwrap.dedent(change.original).strip(),
                        textwrap.dedent(change.proposed).strip(),
                    )
                    panel.write("[dim red]Before:[/dim red]")
                    panel.write(f"{before_markup}\n")
                    panel.write("[dim green]After:[/dim green]")
                    panel.write(f"{after_markup}\n")
                    panel.write(f"[dim italic]{change.reasoning}[/dim italic]\n")
                panel.write("[dim]  [bold]y[/bold]=approve  [bold]n[/bold]=reject[/dim]\n")

        panel.scroll_home(animate=False)

    def _render_global(self) -> None:
        panel: FindingsPanel = self.query_one("#findings", FindingsPanel)
        panel.clear()

        with self._lock:
            findings = list(self._findings)
            idx = self._current_finding_index

        panel.write(
            "[dim]global queue  [/dim][bold]f[/bold][dim]=contextual[/dim]\n"
        )

        if not findings:
            panel.write("[green]✓  No findings.[/green]")
            panel.scroll_home(animate=False)
            return

        pending = sum(1 for f in findings if f.approved is None)
        panel.write(
            f"[bold]{len(findings)} finding(s)[/bold]  "
            f"[dim]{pending} pending  [bold]y[/bold]=approve  [bold]n[/bold]=reject  [bold]a[/bold]=apply[/dim]\n"
        )

        for i, f in enumerate(findings):
            marker = "[bold yellow]▶[/bold yellow]" if i == idx else " "
            sev_color = "red" if f.severity.value == "error" else "yellow"
            sev_label = f"[{sev_color} dim]{f.severity.value}[/{sev_color} dim]"
            status = ""
            if f.approved is True:
                status = "  [green]✓[/green]"
            elif f.approved is False:
                status = "  [dim]✗[/dim]"

            panel.write(
                f"{marker} [{sev_color} bold]{f.category.value.upper()}[/{sev_color} bold]  "
                f"{sev_label}{status}  [white]{f.description}[/white]"
            )

            if f.proposed_changes:
                inv = self._inventory
                artifact = inv.get(f.proposed_changes[0].artifact_id) if inv else None
                if artifact:
                    label = f"{artifact.agent_name} / {_TYPE_LABEL.get(artifact.type, artifact.type.value)}"
                    panel.write(f"  [cyan dim]→ {label}[/cyan dim]")

            panel.write("")

        panel.scroll_home(animate=False)

    # ------------------------------------------------------------------
    # Approval actions
    # ------------------------------------------------------------------

    def action_approve(self) -> None:
        with self._lock:
            if self._scanning:
                return
        self._apply_decision(True)

    def action_reject(self) -> None:
        with self._lock:
            if self._scanning:
                return
        self._apply_decision(False)

    def action_undo_decision(self) -> None:
        with self._lock:
            if self._scanning:
                return
        self._apply_decision(None)

    def action_apply(self) -> None:
        with self._lock:
            if self._scanning:
                return
        self._write_approved()

    def _apply_decision(self, approved: bool | None) -> None:
        with self._lock:
            findings = self._findings
            artifact_id = self._selected_artifact_id

        decided: Finding | None = None

        if self._panel_mode == "contextual" and artifact_id:
            for f in findings:
                if any(c.artifact_id == artifact_id for c in f.proposed_changes):
                    f.approved = approved
                    decided = f
                    break
        else:
            with self._lock:
                idx = self._current_finding_index
                if not findings or idx >= len(findings):
                    return
                findings[idx].approved = approved
                decided = findings[idx]
                if approved is None:
                    pass  # stay on current finding when undoing
                else:
                    for offset in range(1, len(findings)):
                        next_idx = (idx + offset) % len(findings)
                        if findings[next_idx].approved is None:
                            self._current_finding_index = next_idx
                            break

        if decided is None:
            return

        with self._lock:
            key = self._finding_key(decided)
            if approved is None:
                self._decisions.pop(key, None)
            else:
                self._decisions[key] = approved

        self._refresh_findings()
        self._update_status()

    def _write_approved(self) -> None:
        with self._lock:
            findings = [f for f in self._findings if f.approved is True]
            inventory = self._inventory

        if not inventory or not findings:
            return

        threading.Thread(
            target=self._apply_worker, args=(findings, inventory), daemon=True
        ).start()

    def _apply_worker(self, findings: list[Finding], inventory: Inventory) -> None:
        from .cli import _apply_findings
        try:
            applied = _apply_findings(findings, inventory, self._project_root)
            self._set_status(f"[green]{applied} change(s) written[/green]")
            written_artifact_ids = {
                c.artifact_id for f in findings for c in f.proposed_changes
            }
            with self._lock:
                self._decisions = {
                    k: v for k, v in self._decisions.items()
                    if not k[1] & written_artifact_ids
                }
        except Exception as e:
            self._set_status(f"[red]apply error: {e}[/red]")

    # ------------------------------------------------------------------
    # Watcher
    # ------------------------------------------------------------------

    def _start_watcher(self) -> None:
        from . import watcher
        self._watcher_stop = watcher.start(self._project_root, self._on_files_changed)

    def _on_files_changed(self, changed: set[Path]) -> None:
        self._set_status("[yellow]change detected — rescanning...[/yellow]")
        self.call_from_thread(self._do_scan, changed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def action_quit(self) -> None:
        self._set_status("exiting...")
        if self._watcher_stop:
            self._watcher_stop.set()
        self.exit()

    def on_unmount(self) -> None:
        if self._watcher_stop:
            self._watcher_stop.set()

    def action_refresh(self) -> None:
        self._do_scan()

    def _update_status(self) -> None:
        with self._lock:
            findings = list(self._findings)
            inventory = self._inventory

        if not inventory:
            return

        agent_count = len(inventory.by_agent())
        pending = sum(1 for f in findings if f.approved is None)
        mode = "contextual" if self._panel_mode == "contextual" else "global queue"
        pending_str = (
            f"[yellow]{pending} finding(s) pending[/yellow]"
            if pending else "[green]all clear[/green]"
        )
        self._set_status(
            f"watching · {agent_count} agent(s) · {pending_str} · [dim]{mode}[/dim]"
        )

    def _set_status(self, text: str) -> None:
        try:
            label = self.query_one("#status", Label)
            label.update(text)
        except Exception:
            pass
