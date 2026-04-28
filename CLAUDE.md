# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (required before first use)
pip install -e .

# Run the CLI against a target project
sigil scan [path]          # inventory artifacts
sigil init [path]          # draft sigil.yaml via agent
sigil review [path]        # run analysis + interactive approval
sigil apply [path]         # write approved changes to source files

# Override the model used by all agents
SIGIL_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0 sigil review .
```

There are no linter configs yet. See the Evaluation section below for the eval suite.

## Architecture

Sigil is a CLI tool that manages the "text layer" of Strands-based agentic Python projects — inventorying system prompts, tool descriptions, and handler prompts, then proposing consistency improvements with human approval before any file is written.

### Pipeline

```
discover() → Inventory → run_analysis() → generate_proposals() → _interactive_review() → apply
   (AST)                  (4 agents +         (N agents,             (CLI y/n/s)        (file write)
                          vocab check)         one/finding)
```

1. **Discovery** (`core/discovery.py`) — Pure AST walk; no LLM. Finds:
   - `@tool` docstrings → `ArtifactType.TOOL_DESCRIPTION`
   - `Agent(system_prompt=...)` kwargs → `ArtifactType.SYSTEM_PROMPT`
   - `*Handler(system_prompt=...)` kwargs → `ArtifactType.HANDLER_PROMPT`
   - `SKILL.md` files → `ArtifactType.SKILL`
   
   Agent name resolution priority: `@tool` function name > `name=` kwarg > variable assignment > class name > file stem. Module-level handler prompts are re-attributed to whichever agent consumes them via `Agent(plugins=[...])`.

2. **Analysis** (`agents/analyzers.py`) — Four agents run in parallel via `ThreadPoolExecutor`, each scoped to one `FindingCategory`: tone, constraints, routing, flow. Vocabulary is a deterministic regex check (`core/checks.py`) that runs before the agents. All agents use `structured_output_model` (Pydantic) instead of JSON parsing.

3. **Proposal** (`agents/proposer.py`) — One agent per finding, parallelised. Each agent receives only the affected artifacts (not the full inventory). Uses `structured_output_model`. The system prompt constrains agents to minimum edits — rewording only, no new content unless a required constraint is explicitly missing.

4. **Approval** (`cli.py:_interactive_review`) — Per-finding `y/n/s` (approve / reject / skip category). Approved findings are written to `.sigil/pending.json`.

5. **Apply** (`cli.py:apply`) — Content-based replacement: `Artifact.source_segment` is the exact quoted literal text found in the source file. Python files preserve the original quote style (`"""`, `'''`, `"`, `'`). Markdown files (SKILL.md) replace the body text directly.

### Key data model

```
Artifact      — discovered text artifact (id, type, agent_name, content, source_segment, file_path, line_start/end)
Inventory     — dict-backed store with by_agent() / by_type() / to_prompt_text() helpers
Spec          — loaded from sigil.yaml (tone, vocabulary, required_constraints, examples)
Finding       — analysis result (category, severity, affected_artifact_ids, proposed_changes, approved)
ProposedChange — (artifact_id, original, proposed, reasoning)
```

`Artifact.id` is a deterministic MD5 prefix of `(file_path, line_start, artifact_type)`. It is the stable key used across discovery, analysis, proposal, and apply.

### sigil.yaml

Human-owned spec file at the project root. Agents read it but never write it. `sigil init` drafts an initial version for human review. Fields: `tone` (prose), `vocabulary` (canonical/avoid pairs), `required_constraints` (list), `examples` (dict).

### Model

All agents default to `us.amazon.nova-2-lite-v1:0` via AWS Bedrock. Override with `SIGIL_MODEL_ID`. Temperature is always `0.0`.

## Evaluation

The eval suite lives in `eval/` and tests the analysis pipeline against synthetic agent projects with known issues seeded in.

```bash
python3 eval/run_eval.py                        # pass/fail for all fixtures
python3 eval/run_eval.py <fixture_name>         # single fixture
python3 eval/run_eval.py --report               # full findings + proposals for manual review
python3 eval/run_eval.py --report <fixture_name>
```

### Fixtures

Each fixture in `eval/fixtures/` is a self-contained mini-project: one or more Python agent files, a `sigil.yaml` spec, and an `expected.json` defining what findings must (or must not) appear.

| Fixture | Issue seeded | Category tested |
|---|---|---|
| `bad_tone` | Casual docstrings and system prompt against a professional tone spec | `tone` |
| `missing_constraints` | Two agents with contradictory constraint language — one refuses to fabricate, one estimates from context | `constraints` |
| `vocab_violations` | "utilize", "leverage", "in order to" throughout tool docs and system prompt | `vocabulary` |
| `routing_issues` | `shopping_assistant` system prompt claims research capability that contradicts the orchestrator's routing rules | `routing` |
| `skill_issues` | SKILL.md body uses casual language against a professional tone spec | `tone` (via skill artifact) |
| `clean` | Well-written agents that fully match the spec | none — must produce 0 findings |

### expected.json schema

```json
{
  "required_categories": ["tone"],   // categories that MUST appear in findings
  "forbidden_categories": [],        // categories that MUST NOT appear
  "min_findings": 1,                 // minimum finding count
  "max_findings": null,              // maximum finding count (null = no limit)
  "notes": "..."                     // human-readable description, not evaluated
}
```

Assertions are intentionally loose — they check category and count, not exact text, since LLM outputs are non-deterministic. The `--report` flag is for manual inspection of finding quality and proposal correctness.

### Known limitations

- **Constraints detection is cross-agent only** — a single agent missing a required constraint won't reliably trigger; the analyzer needs two agents with contradictory constraint language to compare.
- **Vocabulary check does not match morphological variants** — `\butilize\b` catches "utilize" but not "utilizing". Suffixed forms must be listed separately in `sigil.yaml` if needed.
