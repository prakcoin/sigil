#!/usr/bin/env python3
"""
Sigil eval runner.

Usage:
    python eval/run_eval.py                  # run all fixtures
    python eval/run_eval.py bad_tone         # run one fixture by name

Each fixture in eval/fixtures/ must contain:
  - agent.py (or any .py files) — the synthetic project to analyze
  - sigil.yaml                  — the spec the project is evaluated against
  - expected.json               — assertions about the findings

expected.json fields:
  required_categories  — category values that MUST appear in findings
  forbidden_categories — category values that MUST NOT appear
  min_findings         — minimum number of total findings (default 0)
  max_findings         — maximum number of total findings (null = no limit)
  notes                — human-readable description (not evaluated)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sigil.core.discovery import discover
from sigil.core.inventory import Inventory
from sigil.core.spec import load_spec
from sigil.agents.analyzers import run_analysis
from sigil.agents.proposer import generate_proposals

FIXTURES_DIR = Path(__file__).parent / "fixtures"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def check_fixture(fixture_dir: Path) -> tuple[bool, str]:
    expected_path = fixture_dir / "expected.json"
    if not expected_path.exists():
        return False, "missing expected.json"

    expected = json.loads(expected_path.read_text())
    required_cats  = set(expected.get("required_categories", []))
    forbidden_cats = set(expected.get("forbidden_categories", []))
    min_findings   = expected.get("min_findings", 0)
    max_findings   = expected.get("max_findings", None)

    artifacts = discover(fixture_dir)
    inventory = Inventory(artifacts)
    spec      = load_spec(fixture_dir)
    findings  = run_analysis(inventory, spec)

    found_cats = {f.category.value for f in findings}
    n = len(findings)

    failures = []

    missing = required_cats - found_cats
    if missing:
        failures.append(f"required categories not found: {', '.join(sorted(missing))}")

    unexpected = forbidden_cats & found_cats
    if unexpected:
        failures.append(f"forbidden categories triggered: {', '.join(sorted(unexpected))}")

    if n < min_findings:
        failures.append(f"too few findings: got {n}, expected >= {min_findings}")

    if max_findings is not None and n > max_findings:
        failures.append(f"too many findings: got {n}, expected <= {max_findings}")

    detail = f"categories={sorted(found_cats)}, n={n}"
    if failures:
        return False, "; ".join(failures) + f"  [{detail}]"
    return True, detail


def report_fixture(fixture_dir: Path) -> None:
    """Print full findings + proposals for a fixture."""
    artifacts = discover(fixture_dir)
    inventory = Inventory(artifacts)
    spec      = load_spec(fixture_dir)
    findings  = run_analysis(inventory, spec)
    if findings:
        findings = generate_proposals(findings, inventory)

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{fixture_dir.name}{RESET}  {DIM}({len(artifacts)} artifacts, {len(findings)} findings){RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")

    if not findings:
        print(f"  {DIM}No findings.{RESET}\n")
        return

    sev_color = {"error": RED, "warning": YELLOW, "info": DIM}

    for i, f in enumerate(findings, 1):
        col = sev_color.get(f.severity.value, RESET)
        print(f"\n  {BOLD}[{i}] {f.category.value.upper()}{RESET}  {col}{f.severity.value}{RESET}")
        print(f"  {f.description}")

        for change in f.proposed_changes:
            artifact = inventory.get(change.artifact_id)
            label = f"{artifact.agent_name} / {artifact.type.value}" if artifact else change.artifact_id
            print(f"\n    {DIM}artifact:{RESET} {label}")
            orig_preview  = change.original[:120].replace("\n", "↵ ")
            prop_preview  = change.proposed[:120].replace("\n", "↵ ")
            print(f"    {DIM}before:{RESET}   {orig_preview}{'…' if len(change.original) > 120 else ''}")
            print(f"    {DIM}after:{RESET}    {GREEN}{prop_preview}{'…' if len(change.proposed) > 120 else ''}{RESET}")
            print(f"    {DIM}reason:{RESET}   {change.reasoning}")

    print()


def main(target: str | None = None, report: bool = False) -> None:
    fixtures = sorted(
        d for d in FIXTURES_DIR.iterdir() if d.is_dir()
    ) if FIXTURES_DIR.exists() else []

    if target:
        fixtures = [f for f in fixtures if f.name == target]
        if not fixtures:
            print(f"{RED}Fixture '{target}' not found in {FIXTURES_DIR}{RESET}")
            sys.exit(1)

    if not fixtures:
        print(f"{YELLOW}No fixtures found in {FIXTURES_DIR}{RESET}")
        sys.exit(1)

    if report:
        for fixture_dir in fixtures:
            report_fixture(fixture_dir)
        return

    print(f"\n{BOLD}Sigil eval — {len(fixtures)} fixture(s){RESET}\n")
    results: list[tuple[str, bool, str]] = []

    for fixture_dir in fixtures:
        print(f"  {DIM}running {fixture_dir.name}...{RESET}", end="", flush=True)
        t0 = time.time()
        try:
            passed, detail = check_fixture(fixture_dir)
        except Exception as e:
            passed, detail = False, f"exception: {e}"
        elapsed = time.time() - t0

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"\r  {status}  {fixture_dir.name:<28}  {DIM}{detail}{RESET}  {DIM}({elapsed:.1f}s){RESET}")
        results.append((fixture_dir.name, passed, detail))

    total    = len(results)
    passed_n = sum(1 for _, p, _ in results if p)
    print()

    if passed_n == total:
        print(f"{BOLD}{GREEN}{passed_n}/{total} passed.{RESET}\n")
    else:
        failed = [name for name, p, _ in results if not p]
        print(f"{BOLD}{RED}{passed_n}/{total} passed.{RESET}  Failed: {', '.join(failed)}\n")
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    _report = "--report" in args
    args = [a for a in args if a != "--report"]
    main(args[0] if args else None, report=_report)
