#!/usr/bin/env python3
"""Regenerate evidence figures from the locked hierarchical-QAOA audit."""
from __future__ import annotations

import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def main() -> None:
    from hqml_hierarchical_gctr.design_study import write_figures

    result_path = REPOSITORY / "code" / "results" / "design_study.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    write_figures(result, REPOSITORY / "code" / "results")
    print("regenerated figures from", result_path)


if __name__ == "__main__":
    main()
