"""
Guards against the package being renamed in code but not in user-facing text.

The distribution is ``judge-kappa`` (module ``judge_kappa``, command ``judge-kappa``).
An earlier name survived in CLI usage text, install hints in ImportError messages and
two example scripts; the install hints pointed users at the wrong PyPI package and the
examples failed with ModuleNotFoundError.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import patch

import judge_kappa

ROOT = Path(__file__).resolve().parent.parent
_OLD_NAME = re.compile("jury" + r"[-_]" + "eval", re.IGNORECASE)   # built so this file does not match itself
_SCAN_DIRS = ("src", "examples", "docs")
_SCAN_FILES = ("README.md", "AGENTS.md", "pyproject.toml")
_TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json"}


def _shipped_text_files() -> list[Path]:
    files = [ROOT / f for f in _SCAN_FILES if (ROOT / f).exists()]
    for d in _SCAN_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in _TEXT_SUFFIXES]
    return files


def test_no_stale_package_name_in_shipped_text():
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in _shipped_text_files()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if _OLD_NAME.search(line)
    ]
    assert not offenders, f"stale package name found: {offenders}"


def test_example_imports_resolve():
    """Every ``from judge_kappa import X`` in examples/ names a real export."""
    problems: list[str] = []
    for path in sorted((ROOT / "examples").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "judge_kappa":
                problems += [
                    f"{path.name}: {alias.name}"
                    for alias in node.names
                    if not hasattr(judge_kappa, alias.name)
                ]
    assert not problems, f"examples import names judge_kappa does not export: {problems}"


def test_cli_usage_names_the_installed_command(capsys):
    from judge_kappa.cli.main import app

    with patch("sys.argv", ["judge-kappa"]):
        app()
    out = capsys.readouterr().out
    assert "judge-kappa run" in out
    assert "judge-kappa validate" in out
    assert not _OLD_NAME.search(out)


def test_unknown_command_message_names_the_installed_command(capsys):
    import pytest

    from judge_kappa.cli.main import app

    with patch("sys.argv", ["judge-kappa", "nope"]), pytest.raises(SystemExit):
        app()
    err = capsys.readouterr().err
    assert "judge-kappa --help" in err
