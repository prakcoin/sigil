from __future__ import annotations

import os

import yaml
from strands import Agent
from strands.models import BedrockModel

from ..core.inventory import Inventory

_SYSTEM_PROMPT = """\
You are a technical writer auditing text artifacts from an agentic AI system.
Write an ASPIRATIONAL sigil.yaml — a spec that sets a higher bar than the current
state of the artifacts, so that a review pass will produce actionable findings.

Tone: use the most precise, professional artifacts as the standard. If any artifacts
use casual, vague, or wordy language, write the tone spec to explicitly exclude it.

Vocabulary: identify terms in the artifacts where a simpler or more precise word
exists. For each such term, create one entry with the preferred canonical, a concise
definition of what it means in this project's context, and the term(s) to avoid.
The definition should be specific enough that a reviewer can tell whether a given
occurrence of an avoid-term is actually a synonym for the canonical or is being
used in a different sense. Each entry maps exactly one canonical to its synonyms —
never group terms that have different preferred forms into the same entry.

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


def _clean_vocabulary(yaml_str: str) -> str:
    """Merge duplicate canonicals and deduplicate avoid terms."""
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


def draft_spec(inventory: Inventory) -> str:
    """Run the spec drafter agent and return raw YAML string for human review."""
    model = BedrockModel(
        model_id=os.environ.get("SIGIL_MODEL_ID", "us.amazon.nova-2-lite-v1:0"),
        temperature=0.0,
        max_tokens=12000,
    )
    agent = Agent(system_prompt=_SYSTEM_PROMPT, model=model, callback_handler=None)

    prompt = (
        "Here are all text artifacts from the project. "
        "Analyze them and draft the sigil.yaml specification.\n\n"
        + inventory.to_prompt_text()
    )

    result = str(agent(prompt)).strip()
    # Strip markdown code fences if the model ignores the "no fences" instruction
    if result.startswith("```"):
        lines = result.splitlines()
        result = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return _clean_vocabulary(result.strip())
