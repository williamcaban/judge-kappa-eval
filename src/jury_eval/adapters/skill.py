"""
SkillAdapter — reads SKILL.md + evals/evals.json (agent-skills-eval spec).
Returns (skill_context, cases) so the caller can attach skill_context to
the treatment Variant.
"""

from __future__ import annotations

import json
from pathlib import Path

from jury_eval.adapters.base import InputAdapter
from jury_eval.models import Assertion, EvalCase


class SkillAdapter(InputAdapter):
    def load(self, source: object) -> list[EvalCase]:
        skill_dir, _ = self.load_with_context(source)  # type: ignore[arg-type]
        return skill_dir

    def load_with_context(self, skill_dir: str | Path) -> tuple[list[EvalCase], str]:
        """Return (cases, skill_markdown) — the caller injects skill_markdown into
        the treatment Variant's skill_context field."""
        skill_dir = Path(skill_dir)
        skill_md = (skill_dir / "SKILL.md").read_text()
        raw = json.loads((skill_dir / "evals" / "evals.json").read_text())

        cases = [
            EvalCase(
                id=entry["id"],
                prompt=entry["prompt"],
                assertions=[Assertion(text=a) for a in entry.get("assertions", [])],
                expected_output=entry.get("expected_output"),
                files=entry.get("files", []),
                metadata={"name": entry.get("name", "")},
            )
            for entry in raw["evals"]
        ]
        return cases, skill_md
