#!/usr/bin/env python3
"""Validate the public release structure and delegate to the scientific validator."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CODE = REPOSITORY / "code"
REQUIRED_ROOT = {".gitignore", "LICENSE", "README.md", "index.html", "main.tex", "main.pdf", "code"}
REQUIRED_CODE = {"README.md", "pyproject.toml", "src", "tests", "scripts", "configs", "data", "results", "manuscript_assets"}
DELEGATE = ["python","code/manuscript_assets/validate_manuscript.py"]


def main() -> None:
    root_names = {p.name for p in REPOSITORY.iterdir() if p.name != ".git"}
    missing_root = sorted(REQUIRED_ROOT - root_names)
    extra_root = sorted(root_names - REQUIRED_ROOT)
    code_names = {p.name for p in CODE.iterdir()}
    missing_code = sorted(REQUIRED_CODE - code_names)
    if missing_root or extra_root or missing_code:
        raise SystemExit(
            json.dumps(
                {"missing_root": missing_root, "extra_root": extra_root, "missing_code": missing_code},
                indent=2,
            )
        )
    manuscript = (REPOSITORY / "main.tex").read_text(encoding="utf-8")
    generic_class = r"\documentclass[10pt,letterpaper,twoside]{article}"
    if generic_class not in manuscript:
        raise SystemExit("one-column preprint class contract changed")
    if r"\usepackage{code/manuscript_assets/researchpreprint}" not in manuscript:
        raise SystemExit("shared preprint style is missing")
    if DELEGATE:
        command = list(DELEGATE)
        if command[0] == "python":
            command[0] = sys.executable
        elif shutil.which(command[0]) is None:
            raise SystemExit(
                f"{command[0]!r} is not installed. Run: python -m pip install -e code"
            )
        subprocess.run(command, cwd=REPOSITORY, check=True)
    print("release contract: PASS")


if __name__ == "__main__":
    main()
