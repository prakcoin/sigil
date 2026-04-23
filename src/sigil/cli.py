from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .core.discovery import discover
from .core.inventory import Inventory
from .core.models import Finding
from .core.spec import load_spec, spec_exists, write_spec

app = typer.Typer(help="Sigil — text artifact manager for agentic AI projects", add_completion=False)
console = Console()

_PENDING_FILE = ".sigil/pending.json"


# ---------------------------------------------------------------------------
# sigil scan
# ---------------------------------------------------------------------------

@app.command()
def scan(
    project: Path = typer.Argument(default=Path("."), help="Path to project root"),
    detail: bool = typer.Option(False, "--detail", "-d", help="Show all artifacts in full tables"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Show detail for one agent only"),
):
    """Inventory all text artifacts grouped by agent.

    Default view: one summary row per agent.
    Use --detail to see every artifact, or --agent <name> to drill into one.
    """
    artifacts = discover(project)
    inventory = Inventory(artifacts)

    if not artifacts:
        console.print("[yellow]No artifacts found.[/yellow]")
        raise typer.Exit()

    if agent:
        _scan_agent_detail(inventory, agent)
    elif detail:
        _scan_full_detail(inventory)
    else:
        _scan_summary(inventory)

    if not spec_exists(project):
        console.print(
            "\n[yellow]No sigil.yaml found. Run [bold]sigil init[/bold] to create one.[/yellow]"
        )


def _scan_summary(inventory: Inventory) -> None:
    from .core.models import ArtifactType

    _AGENT_TYPES = {ArtifactType.SYSTEM_PROMPT, ArtifactType.HANDLER_PROMPT, ArtifactType.SKILL}
    type_order = [
        ArtifactType.SYSTEM_PROMPT,
        ArtifactType.TOOL_DESCRIPTION,
        ArtifactType.HANDLER_PROMPT,
        ArtifactType.SKILL,
    ]
    type_labels = {
        ArtifactType.SYSTEM_PROMPT:    "sys",
        ArtifactType.TOOL_DESCRIPTION: "tool",
        ArtifactType.HANDLER_PROMPT:   "handler",
        ArtifactType.SKILL:            "skill",
    }

    # Split entries: agents (have sys/handler/skill) vs standalone tools (only tool_description)
    agent_rows = []
    standalone_tool_count = 0

    for agent_name, arts in sorted(inventory.by_agent().items()):
        counts = {t: sum(1 for a in arts if a.type == t) for t in type_order}
        has_agent_artifacts = any(counts[t] for t in _AGENT_TYPES)
        if has_agent_artifacts:
            agent_rows.append((agent_name, counts, len(arts)))
        else:
            standalone_tool_count += counts[ArtifactType.TOOL_DESCRIPTION]

    table = Table(show_lines=False, box=None, pad_edge=False)
    table.add_column("Agent", style="bold cyan", min_width=24)
    for t in type_order:
        table.add_column(type_labels[t], style="dim", justify="right", width=8)
    table.add_column("Total", justify="right", width=6)

    for agent_name, counts, total in agent_rows:
        table.add_row(
            agent_name,
            *[str(counts[t]) if counts[t] else "[dim]–[/dim]" for t in type_order],
            f"[bold]{total}[/bold]",
        )

    console.print(table)

    footer_parts = [
        f"[bold]{len(inventory)}[/bold] artifact(s)",
        f"[bold]{len(agent_rows)}[/bold] agent(s)",
    ]
    if standalone_tool_count:
        footer_parts.append(
            f"[dim]{standalone_tool_count} standalone tool(s) hidden — use --detail to see all[/dim]"
        )
    else:
        footer_parts.append("[dim]--detail for full view · --agent <name> to drill in[/dim]")
    console.print("\n" + "  ·  ".join(footer_parts))


def _scan_full_detail(inventory: Inventory) -> None:
    for agent_name, agent_artifacts in sorted(inventory.by_agent().items()):
        _print_agent_table(agent_name, agent_artifacts)


def _scan_agent_detail(inventory: Inventory, agent_name: str) -> None:
    by_agent = inventory.by_agent()
    # Case-insensitive match
    matches = {k: v for k, v in by_agent.items() if k.lower() == agent_name.lower()}
    if not matches:
        candidates = ", ".join(sorted(by_agent.keys()))
        console.print(f"[yellow]Agent '{agent_name}' not found. Known agents: {candidates}[/yellow]")
        raise typer.Exit(1)
    for name, arts in matches.items():
        _print_agent_table(name, arts)


def _print_agent_table(agent_name: str, artifacts: list) -> None:
    table = Table(title=f"[bold cyan]{agent_name}[/bold cyan]", show_lines=False)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Type", style="cyan", width=18)
    table.add_column("File", style="blue")
    table.add_column("Lines", style="dim", width=8)
    table.add_column("Preview")
    for a in artifacts:
        table.add_row(
            a.id,
            a.type.value,
            a.file_path,
            f"{a.line_start}–{a.line_end}",
            a.preview(),
        )
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# sigil init
# ---------------------------------------------------------------------------

@app.command()
def init(
    project: Path = typer.Argument(default=Path("."), help="Path to project root"),
):
    """Draft sigil.yaml from existing artifacts, then open for review."""
    from .agents.spec_drafter import draft_spec

    if spec_exists(project):
        overwrite = typer.confirm("sigil.yaml already exists. Overwrite?", default=False)
        if not overwrite:
            raise typer.Exit()

    artifacts = discover(project)
    if not artifacts:
        console.print("[yellow]No artifacts found — cannot draft spec.[/yellow]")
        raise typer.Exit(1)

    inventory = Inventory(artifacts)
    console.print(f"Drafting spec from [bold]{len(artifacts)}[/bold] artifact(s)...")

    draft = draft_spec(inventory)

    console.print("\n[bold]Proposed sigil.yaml:[/bold]\n")
    console.print(Panel(draft, border_style="dim"))

    if typer.confirm("\nWrite this to sigil.yaml?", default=True):
        write_spec(project, draft)
        console.print("[green]sigil.yaml written.[/green]")
    else:
        console.print("Aborted.")


# ---------------------------------------------------------------------------
# sigil review
# ---------------------------------------------------------------------------

@app.command()
def review(
    project: Path = typer.Argument(default=Path("."), help="Path to project root"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category: tone|vocabulary|constraints|routing"),
):
    """Run analysis agents and interactively approve or reject findings."""
    from .agents.analyzers import run_analysis
    from .agents.proposer import generate_proposals

    artifacts = discover(project)
    if not artifacts:
        console.print("[yellow]No artifacts found.[/yellow]")
        raise typer.Exit()

    inventory = Inventory(artifacts)
    spec = load_spec(project)

    if not spec_exists(project):
        console.print("[yellow]No sigil.yaml found. Run [bold]sigil init[/bold] first for better results.[/yellow]\n")

    console.print(f"Analyzing [bold]{len(artifacts)}[/bold] artifact(s) across 4 dimensions...")
    findings = run_analysis(inventory, spec)

    if not findings:
        console.print("[green]No issues found.[/green]")
        raise typer.Exit()

    console.print(f"Generating proposals for [bold]{len(findings)}[/bold] finding(s)...")
    findings = generate_proposals(findings, inventory)

    if category:
        findings = [f for f in findings if f.category.value == category]
        if not findings:
            console.print(f"[yellow]No findings in category '{category}'.[/yellow]")
            raise typer.Exit()

    _interactive_review(findings, inventory)

    pending_path = project / _PENDING_FILE
    approved = [f for f in findings if f.approved is True]
    if approved:
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pending_path, "w") as fp:
            json.dump([f.to_dict() for f in approved], fp, indent=2)
        console.print(
            f"\n[green]{len(approved)}[/green] finding(s) approved. "
            f"Run [bold]sigil apply[/bold] to write changes."
        )
    else:
        console.print("\nNo findings approved.")


def _interactive_review(findings: list[Finding], inventory: Inventory) -> None:
    console.print()
    for i, finding in enumerate(findings):
        header = (
            f"[bold]Finding {i + 1}/{len(findings)}[/bold]  "
            f"[cyan]{finding.category.value}[/cyan]  "
            f"[{'red' if finding.severity.value == 'error' else 'yellow'}]{finding.severity.value}[/]"
        )
        console.print(Panel(finding.description, title=header, border_style="dim"))

        for change in finding.proposed_changes:
            artifact = inventory.get(change.artifact_id)
            label = f"{artifact.agent_name} / {artifact.type.value}" if artifact else change.artifact_id
            console.print(f"  [dim]Artifact:[/dim] {label}")
            console.print(
                Columns([
                    Panel(change.original, title="Before", border_style="red", width=60),
                    Panel(change.proposed, title="After", border_style="green", width=60),
                ])
            )
            console.print(f"  [dim]Reason:[/dim] {change.reasoning}\n")

        choice = typer.prompt(
            "  [y] approve  [n] reject  [s] skip category",
            default="n",
        ).strip().lower()

        if choice == "y":
            finding.approved = True
            console.print("  [green]Approved.[/green]\n")
        elif choice == "s":
            # Skip remaining findings in this category
            cat = finding.category
            for remaining in findings[i:]:
                if remaining.category == cat and remaining.approved is None:
                    remaining.approved = False
            console.print(f"  [yellow]Skipped all {cat.value} findings.[/yellow]\n")
            break
        else:
            finding.approved = False
            console.print("  [dim]Rejected.[/dim]\n")


# ---------------------------------------------------------------------------
# sigil apply
# ---------------------------------------------------------------------------

@app.command()
def apply(
    project: Path = typer.Argument(default=Path("."), help="Path to project root"),
):
    """Write all approved changes to source files."""
    pending_path = project / _PENDING_FILE
    if not pending_path.exists():
        console.print("[yellow]Nothing pending. Run [bold]sigil review[/bold] first.[/yellow]")
        raise typer.Exit()

    with open(pending_path) as f:
        findings = [Finding.from_dict(d) for d in json.load(f)]

    artifacts = discover(project)
    inventory = Inventory(artifacts)

    applied = 0
    errors = 0

    for finding in findings:
        for change in finding.proposed_changes:
            artifact = inventory.get(change.artifact_id)
            if artifact is None:
                console.print(f"[yellow]Artifact {change.artifact_id} not found — skipped.[/yellow]")
                errors += 1
                continue

            file_path = project / artifact.file_path
            try:
                source = file_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                console.print(f"[red]File not found: {artifact.file_path}[/red]")
                errors += 1
                continue

            if artifact.source_segment not in source:
                console.print(
                    f"[yellow]Source segment no longer found in {artifact.file_path} "
                    f"(artifact {artifact.id}) — skipped. File may have changed since scan.[/yellow]"
                )
                errors += 1
                continue

            if artifact.file_path.endswith(".md"):
                # SKILL.md: source_segment is plain body text — replace directly
                new_source = source.replace(artifact.source_segment, change.proposed, 1)
            else:
                # Python file: source_segment is a quoted string literal — preserve quote style
                seg = artifact.source_segment
                if seg.startswith('"""') or seg.startswith("'''"):
                    quote = seg[:3]
                elif seg.startswith('"') or seg.startswith("'"):
                    quote = seg[0]
                else:
                    quote = '"'
                new_source = source.replace(seg, quote + change.proposed + quote, 1)

            file_path.write_text(new_source, encoding="utf-8")

            console.print(f"[green]Updated[/green] {artifact.file_path}  [dim]{artifact.id}[/dim]")
            applied += 1

    pending_path.unlink()
    console.print(f"\n{applied} change(s) applied, {errors} skipped.")


def main():
    app()
