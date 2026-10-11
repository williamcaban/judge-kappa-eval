"""Input adapters: turn external sources into ``EvalCase`` lists."""

from judge_kappa.adapters.base import InputAdapter
from judge_kappa.adapters.dataset import DatasetAdapter
from judge_kappa.adapters.skill import SkillAdapter

__all__ = ["InputAdapter", "SkillAdapter", "DatasetAdapter"]
