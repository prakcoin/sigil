from __future__ import annotations

import re
import yaml
from concurrent.futures import ThreadPoolExecutor
from strands import Agent

from ..core.inventory import Inventory
from ..core.model import make_model

_NUM_DRAFTS = 3

_SYSTEM_PROMPT = """\
You are a technical writer auditing text artifacts from an agentic AI system.
Write an ASPIRATIONAL sigil.yaml — a spec that sets a higher bar than the current
state of the artifacts, so that a review pass will produce actionable findings.

Tone: use the most precise, professional artifacts as the standard. If any artifacts
use casual, vague, or wordy language, write the tone spec to explicitly exclude it.

Vocabulary: a list of candidate terms extracted from the artifacts will be provided
at the end of the prompt. Use it as your starting point — create an entry for each
term that would benefit from standardization, and skip any that don't need it.
Each entry maps exactly one canonical to its synonyms — never group terms that have
different preferred forms into the same entry. The definition should be specific
enough that a reviewer can tell whether a given occurrence of an avoid-term is
actually a synonym for the canonical or is being used in a different sense.

Required constraints: list only rules that must apply to every agent regardless of
its specific function — safety rules, epistemic rules (never fabricate, state
limitations explicitly), and output format rules that are missing from at least one
agent. Do NOT include tool-specific steps, workflow instructions, or anything that
only makes sense for one agent's domain. If it would not belong in every single
system prompt, leave it out.

The sigil.yaml format is:

tone: |
  <prose describing the target tone — be specific about what to avoid>

vocabulary:
  - canonical: <preferred term>
    definition: <one sentence — what this term means in this project's specific context>
    avoid:
      - <term to avoid>

required_constraints:
  - <constraint every agent should enforce, one per line>

Output ONLY valid YAML. No explanations, no markdown fences, no preamble.\
"""

_MERGE_SYSTEM_PROMPT = """\
You are a technical writer merging multiple independent sigil.yaml drafts into one
authoritative final version. Each draft was generated from the same set of artifacts.

Tone: produce a single tone description that captures the most specific and
comprehensive guidance across all drafts. If drafts emphasize different aspects,
combine them into one coherent paragraph.

Vocabulary: include every canonical term that appears in any draft. For each
canonical, use the most specific definition — the one most grounded in this
project's context. Combine and deduplicate the avoid lists from all drafts.
Do not invent canonicals not present in any draft.

Required constraints: apply a two-step filter before including any constraint.

Step 1 — universality test: only include a constraint if it would belong
verbatim in the system prompt of a completely unrelated agent — one with
different tools, a different domain, and a different purpose. Ask: "Would
this rule still make sense for a customer support agent? A code review agent?
A data extraction agent?" If the answer is no for any of them, discard it.

Discard rules that are:
- Tool-specific: reference a named tool or function (e.g. "use the stop tool",
  "call the validation tool before returning")
- Workflow-specific: reference a pipeline step or data source (e.g. "cite
  source URLs for search results", "pass the ID to the lookup tool first")
- Domain-specific: reference the project's subject matter (e.g. "refuse
  queries outside the scope of this domain", "prefer image evidence over
  text metadata")
- Response-string-specific: prescribe a fixed output phrase tied to one
  agent's context (e.g. "respond with '[specific phrase]' when data is
  unavailable")

Keep only epistemic rules (never fabricate, state limitations explicitly),
tone rules (neutral and factual, no internal monologue), and output structure
rules (no meta-commentary, deliver directly to user) that hold for any agent.

Step 2 — deduplication: treat two constraints as duplicates if they encode
the same underlying rule, even in different words. "Never fabricate
information" and "do not infer or guess" are the same rule — keep the most
specific wording, drop the other. "No internal monologue" and "omit reasoning
steps from output" are the same rule — merge them. Aim for 4–7 constraints
total.

Do not invent constraints not present in any draft.

The sigil.yaml format is:

tone: |
  <prose describing the target tone>

vocabulary:
  - canonical: <preferred term>
    definition: <one sentence>
    avoid:
      - <term to avoid>

required_constraints:
  - <constraint>

Output ONLY valid YAML. No explanations, no markdown fences, no preamble.\
"""


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def _repair_yaml(yaml_str: str) -> str:
    """Quote list-item values that contain unquoted colons — the most common LLM YAML mistake."""
    lines = []
    for line in yaml_str.splitlines():
        m = re.match(r'^(\s*-\s+)([^"\'{][^:]*\s[^:]*:.+)$', line)
        if m:
            prefix, value = m.group(1), m.group(2)
            lines.append(f"{prefix}\"{value.replace(chr(34), chr(92) + chr(34))}\"")
        else:
            lines.append(line)
    return "\n".join(lines)


def _clean_vocabulary(yaml_str: str) -> str:
    """Merge duplicate canonicals and deduplicate avoid terms."""
    try:
        doc = yaml.safe_load(yaml_str)
    except Exception:
        yaml_str = _repair_yaml(yaml_str)
        try:
            doc = yaml.safe_load(yaml_str)
        except Exception:
            return yaml_str
    if not isinstance(doc, dict) or not isinstance(doc.get("vocabulary"), list):
        return yaml_str
    merged: dict[str, set[str]] = {}
    definitions: dict[str, str] = {}
    for entry in doc["vocabulary"]:
        canonical = str(entry.get("canonical", "")).strip()
        avoid_raw = entry.get("avoid") or []
        if isinstance(avoid_raw, str):
            avoid_raw = [avoid_raw]
        merged.setdefault(canonical, set()).update(str(t).strip() for t in avoid_raw)
        if canonical not in definitions and entry.get("definition"):
            definitions[canonical] = str(entry["definition"]).strip()
    doc["vocabulary"] = [
        {"canonical": c, "definition": definitions.get(c, ""), "avoid": sorted(a)}
        for c, a in merged.items() if a
    ]
    return yaml.dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _single_draft(prompt: str) -> str:
    agent = Agent(system_prompt=_SYSTEM_PROMPT, model=make_model(), callback_handler=None)
    return _strip_fences(str(agent(prompt)).strip())


def _merge_drafts(drafts: list[str]) -> str:
    agent = Agent(system_prompt=_MERGE_SYSTEM_PROMPT, model=make_model(), callback_handler=None)
    drafts_text = "\n\n---\n\n".join(
        f"Draft {i + 1}:\n{draft}" for i, draft in enumerate(drafts)
    )
    result = _strip_fences(str(agent(
        f"Here are {len(drafts)} independently generated sigil.yaml drafts. "
        f"Merge them into a single authoritative spec.\n\n{drafts_text}"
    )).strip())
    return _clean_vocabulary(result)


def draft_spec(inventory: Inventory) -> str:
    """Run N parallel drafts then merge into a single stable spec."""
    from ..core.checks import extract_vocabulary_candidates

    candidates = extract_vocabulary_candidates(inventory)
    candidate_section = ""
    if candidates:
        candidate_section = (
            "\n\nVocabulary candidates (terms appearing across multiple artifacts — "
            "use as your starting point for the vocabulary section):\n"
            + ", ".join(candidates)
        )

    prompt = (
        "Here are all text artifacts from the project. "
        "Analyze them and draft the sigil.yaml specification.\n\n"
        + inventory.to_prompt_text()
        + candidate_section
    )

    with ThreadPoolExecutor(max_workers=_NUM_DRAFTS) as executor:
        drafts = list(executor.map(_single_draft, [prompt] * _NUM_DRAFTS))

    return _merge_drafts(drafts)
