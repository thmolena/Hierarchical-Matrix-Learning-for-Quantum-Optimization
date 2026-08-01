#!/usr/bin/env python3
"""Static manuscript, evidence, and public-release checks."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Auditing Hierarchical Subspace Learning for Depth-One QAOA Landscapes"
FIGURES = {
    "code/results/compression_acceptance.pdf",
    "code/results/probe_performance.pdf",
    "code/results/subspace_baselines.pdf",
    "code/results/gset_target_audit.pdf",
    "code/results/noise_certificate.pdf",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    source = (ROOT / "main.tex").read_text(encoding="utf-8")
    result = json.loads(
        (ROOT / "code/results/design_study.json").read_text(encoding="utf-8")
    )
    locked = json.loads(
        (ROOT / "code/src/hqml_hierarchical_gctr/locked/design_study.json").read_text(
            encoding="utf-8"
        )
    )

    require(r"\title{Auditing Hierarchical Subspace Learning for Depth-One QAOA Landscapes}" in source, "title mismatch")
    require(r"\author{Molena Huynh}" in source, "author mismatch")
    require(r"\email{molena.huynh@jmp.com}" in source, "email mismatch")
    require(r"\affiliation{North Carolina State University}" in source, "affiliation mismatch")
    for statement in (
        "Exact graph--angle factorization",
        "Compression-regret theorem",
        "Action-only learning contract",
        "Row-Gram recovery of the right singular subspace",
        "Greedy design guarantee",
        "Affine recovery and decision bound",
    ):
        require(statement in source, f"missing formal result: {statement}")
    require(r"\bibliography{code/manuscript_assets/references}" in source, "bibliography path mismatch")

    require(result == locked, "source and installed design-study locks differ")
    require(result["schema_version"] == 2, "design-study schema mismatch")
    require(len(result["data"]["graphs"]) == 8, "Gset suite is incomplete")
    require(result["data"]["public_test_rows"] == 8, "Gset test-row count changed")
    require(result["compression"]["selected_candidate"] == 2, "compression selection changed")
    selected = result["compression"]["frontier"][2]
    require(selected["storage_fraction"] < 0.16, "selected storage threshold failed")
    require(selected["relative_error"] < 5.0e-3, "selected compression error failed")
    require(selected["maximum_grid_regret"] <= 2.0e-4, "selected compression regret failed")
    require(selected["certificate_violation"] == 0.0, "compression theorem check failed")
    require(all(row["all_errors_bounded"] for row in result["noise_study"]), "noise bound failed")
    audit = result["audit_strategies"]
    random_audit = next(
        row for row in result["random_probe_distribution"] if row["probes"] == 16
    )
    require(random_audit["replicates"] == 64, "random baseline replication changed")
    require(audit["decision_weighted"]["median_relative_error"] < random_audit["median_relative_error"]["median"], "decision-weighted design does not beat median random error")
    baselines = result["subspace"]["subspace_baselines"]
    require(baselines["dense_input_materialized_by_builder"], "dense-builder baseline status changed")
    require(baselines["methods"]["row_gram"]["projector_distance_to_thin_svd"] < 1.0e-9, "row-Gram projector check failed")
    require(baselines["methods"]["hierarchical_action_iteration"]["probe_overlap_with_thin_svd"] == 16, "subspace routes no longer select the same audit probes")
    require("runtime_benchmark" in result and result["runtime_benchmark"]["repetitions"] >= 9, "runtime benchmark missing")

    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    require(set(figures) == FIGURES and len(figures) == len(FIGURES), "figure set differs from the five registered evidence plots")
    require(len(re.findall(r"\\begin\{table\*?\}", source)) == 3, "manuscript must contain three evidence tables")
    for name in FIGURES | {
        "main.pdf",
        "code/results/design_study.json",
        "code/results/design_study.tex",
        "code/results/study_constants.tex",
        "code/manuscript_assets/references.bib",
        "code/manuscript_assets/main.bbl",
    }:
        require((ROOT / name).is_file(), f"missing artifact: {name}")

    bib = (ROOT / "code/manuscript_assets/references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", source)
        for key in group.split(",")
    }
    require(len(cited) >= 60, "fewer than 60 distinct sources are cited")
    require(cited <= bib_keys, f"missing bibliography keys: {sorted(cited - bib_keys)}")

    require(r"\onecolumngrid" not in source, "manuscript leaves the two-column grid")
    require(
        source.startswith(
            r"\documentclass[twocolumn,amsmath,amssymb,longbibliography,nofootinbib,floatfix]{revtex4-2}"
        ),
        "generic two-column class contract changed",
    )

    index = (ROOT / "index.html").read_text(encoding="utf-8")
    require("tex-svg.js" in index and "displayMath" in index, "MathJax configuration missing")
    for match in re.findall(r'(?:href|src)="([^"]+)"', index):
        if match.startswith(("http://", "https://", "data:", "#")):
            continue
        require((ROOT / match).is_file(), f"broken local website path: {match}")

    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        metadata = subprocess.run(
            [pdfinfo, str(ROOT / "main.pdf")], check=True, capture_output=True, text=True
        ).stdout
        pages_match = re.search(r"^Pages:\s+(\d+)", metadata, flags=re.MULTILINE)
        require(pages_match is not None and 20 <= int(pages_match.group(1)) <= 32, "main.pdf is not 20--32 pages")

    required_root = {".gitignore", "LICENSE", "README.md", "code", "index.html", "main.pdf", "main.tex"}
    visible_root = {path.name for path in ROOT.iterdir() if path.name != ".git"}
    require(visible_root == required_root, f"root layout mismatch: {sorted(visible_root ^ required_root)}")

    forbidden_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}
    for path in (ROOT / "code").rglob("*"):
        require(not path.is_symlink(), f"symlink in public code tree: {path}")
        if path.is_dir():
            require(path.name not in forbidden_dirs and not path.name.endswith(".egg-info"), f"generated directory: {path}")
        if path.is_file():
            require(path.suffix not in {".pyc", ".pyo"}, f"generated bytecode: {path}")

    implementation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "code/src/hqml_hierarchical_gctr").glob("*.py")
    )
    for primitive in (".scatter(", ".hist(", ".imshow(", ".boxplot(", ".violinplot(", ".pie(", ".hexbin(", ".contour("):
        require(primitive not in implementation, f"forbidden plot primitive: {primitive}")

    print(
        "manuscript contract: PASS",
        "graphs=8",
        "tables=3",
        "figure_files=5",
        f"references={len(cited)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
