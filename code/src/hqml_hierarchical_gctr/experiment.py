"""Reproduce the error-controlled causal-landscape compression study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import sys
from typing import Any
import urllib.request
from urllib.parse import urlparse

import numpy as np

from .causal import (
    CausalType,
    benchmark_graphs,
    causal_histogram,
    parse_gset,
    statevector_audit,
    type_operator,
)
from .matrix_free import (
    ButterflyResult,
    HNode,
    build_hmatrix,
    butterfly_truncate,
    matrix_free_right_subspace,
    morton_order,
)

PACKAGE_NAME = "hqml-hierarchical-gctr"
TITLE = "Auditing Hierarchical Subspace Learning for Depth-One QAOA Landscapes"
SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GSET_SOURCES = {
    "G1": "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665",
    "G6": "e3dd8b29205280f1ad5806849321cdff6f11324c75cb95647d5783f3e47a1103",
    "G11": "c2a760d2926db4fefd23b25c098dcd6311f711b355dbd1cc689fa25660c73174",
    "G14": "dc769b978a40d458f693d5bd2cf8b8cceabd430b8e976204746696179c3d5945",
    "G18": "6185af31fae0f7d59e232dad91178753574b7caa111dd6e6bf52f11c80e928d8",
    "G22": "9baeee06eb147b1c9ca42b43be86592d4e6fc60784a85af9be5b63d1362ef28e",
    "G32": "9760fce6b601832db1b1173c1a09c8556d64bfb27ba603ef52ca24ee6c8527cc",
    "G43": "9af5445b4b066cbf1eabe218d4e0d907cb6f211651cae557c761ec344dc37be8",
}
GSET_URLS = {
    name: f"https://web.stanford.edu/~yyye/yyye/Gset/{name}"
    for name in GSET_SOURCES
}
SEED = 260724

PROFILES = {
    "quick": {
        "sizes": (32, 64, 128),
        "replicates": 2,
        "grid": 16,
        "candidates": (
            (1e-8, 1e-6, 20),
            (1e-5, 1e-4, 16),
            (1e-4, 1e-3, 12),
        ),
    },
    "paper": {
        "sizes": (64, 128, 256, 512, 1024, 2048, 4096, 8192),
        "replicates": 4,
        "grid": 32,
        "candidates": (
            (1e-9, 1e-7, 28),
            (1e-6, 1e-5, 24),
            (1e-4, 5e-4, 18),
            (1e-3, 5e-3, 12),
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gset_path(name: str) -> Path:
    if name not in GSET_SOURCES:
        raise ValueError(f"unknown Gset instance: {name}")
    configured_cache = os.environ.get("HQML_GSET_CACHE")
    if configured_cache:
        cache = Path(configured_cache).expanduser()
    else:
        cache_base = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        ).expanduser()
        cache = cache_base / "hqml-hierarchical-gctr" / "gset"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / name
    expected = GSET_SOURCES[name]
    if destination.is_file() and sha256(destination) == expected:
        return destination

    url = GSET_URLS[name]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "web.stanford.edu":
        raise ValueError("Gset downloads must use the allowlisted Stanford host")
    temporary = cache / f".{name}.part"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hqml-hierarchical-gctr provenance audit"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        observed = sha256(temporary)
        if observed != expected:
            raise ValueError(
                f"Gset {name} SHA-256 mismatch: expected {expected}, "
                f"observed {observed}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, np.floating):
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def _row_key(
    features: np.ndarray, training_rows: np.ndarray | None = None
) -> np.ndarray:
    """Return a scalar row-ordering coordinate fitted on training rows only."""

    values = np.asarray(features, dtype=float)
    fit = values if training_rows is None else values[training_rows]
    center = fit.mean(axis=0)
    scale = fit.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (values - center) / scale
    _, _, right = np.linalg.svd((fit - center) / scale, full_matrices=False)
    key = standardized @ right[0]
    fitted_key = key if training_rows is None else key[training_rows]
    pivot = int(np.argmax(np.abs(fitted_key)))
    if fitted_key[pivot] < 0:
        key = -key
    return key


def _rank_summary(node: HNode) -> dict[str, float | int | list[int]]:
    ranks = node.ranks
    return {
        "compressed_blocks": len(ranks),
        "rank_min": min(ranks) if ranks else 0,
        "rank_median": float(np.median(ranks)) if ranks else 0.0,
        "rank_mean": float(np.mean(ranks)) if ranks else 0.0,
        "rank_max": max(ranks) if ranks else 0,
        "ranks": ranks,
        "dense_leaf_entries": node.dense_leaf_entries,
    }


def _compress_rows(
    exact: np.ndarray,
    butterfly_landscapes: np.ndarray,
    rows: np.ndarray,
    row_key: np.ndarray,
    column_order: np.ndarray,
    *,
    h_tolerance: float,
    max_rank: int,
    leaf_size: int = 24,
) -> tuple[np.ndarray, HNode, dict[str, float | int | list[int]]]:
    local_order = np.argsort(row_key[rows], kind="mergesort")
    ordered = butterfly_landscapes[rows[local_order]][:, column_order]
    hierarchy = build_hmatrix(
        ordered,
        relative_tolerance=h_tolerance,
        max_rank=max_rank,
        leaf_size=leaf_size,
    )
    reconstructed = hierarchy.reconstruct()
    inverse_columns = np.empty_like(column_order)
    inverse_columns[column_order] = np.arange(len(column_order))
    local_approximation = reconstructed[:, inverse_columns]
    approximation = np.empty_like(local_approximation)
    approximation[local_order] = local_approximation

    truth = exact[rows]
    difference = approximation - truth
    denominator = max(float(np.linalg.norm(truth)), np.finfo(float).tiny)
    exact_optimum = truth.max(axis=1)
    selected = np.argmax(approximation, axis=1)
    regret = exact_optimum - truth[np.arange(len(rows)), selected]
    epsilon = float(np.max(np.abs(difference), initial=0.0))
    margins = np.partition(truth, -2, axis=1)[:, -2:]
    margins = margins[:, 1] - margins[:, 0]
    summary: dict[str, float | int | list[int]] = {
        "relative_frobenius_error": float(np.linalg.norm(difference) / denominator),
        "max_abs_error": epsilon,
        "mean_grid_regret": float(np.mean(regret)),
        "max_grid_regret": float(np.max(regret, initial=0.0)),
        "certificate_limit": 2.0 * epsilon,
        "certificate_violation": float(np.max(regret - 2.0 * epsilon, initial=0.0)),
        "margin_preserved_fraction": float(np.mean(margins > 2.0 * epsilon)),
        "storage_fraction": float(hierarchy.storage / max(ordered.size, 1)),
        "certificate_frobenius": float(
            hierarchy.certificate_frobenius
            / max(float(np.linalg.norm(ordered)), np.finfo(float).tiny)
        ),
        **_rank_summary(hierarchy),
    }
    return approximation, hierarchy, summary


def _candidate_loss(summary: dict[str, Any], butterfly: ButterflyResult) -> float:
    """Certificate-aware hierarchical compression loss on development rows."""

    return float(
        summary["relative_frobenius_error"] ** 2
        + summary["mean_grid_regret"]
        + 0.01 * summary["storage_fraction"]
        + 0.002
        * butterfly.retained_coefficients
        / max(butterfly.total_coefficients, 1)
        + 100.0 * summary["certificate_violation"]
    )


def _recover_real_landscapes(
    development_operator: HNode,
    exact: np.ndarray,
    public_indices: np.ndarray,
    column_order: np.ndarray,
    records: list[Any],
    *,
    profile: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Learn a matrix-free angle basis and recover real held-out landscapes."""

    rank = min(12, development_operator.rows, development_operator.cols)
    learned = matrix_free_right_subspace(
        development_operator,
        rank,
        oversampling=4,
        iterations=4,
        seed=SEED + 177,
    )
    basis = learned.basis
    orthogonality_error = float(
        np.linalg.norm(basis.T @ basis - np.eye(rank), ord=2)
    )
    probe_counts = (16, 32) if profile == "quick" else (16, 32, 64, 128)
    audit_count = probe_counts[-1] if profile == "quick" else 64
    probe_order = np.random.default_rng(SEED).permutation(development_operator.cols)
    ridge = 1e-6

    mean_landscape, mean_work = development_operator.transpose_matmat(
        np.full(development_operator.rows, 1.0 / development_operator.rows)
    )
    baseline_regrets = []
    recovery_rows = []
    per_graph: dict[str, dict[str, float | int]] = {}
    recovered_at_32: list[np.ndarray] = []
    exact_real: list[np.ndarray] = []
    for count in probe_counts:
        probes = probe_order[:count]
        relative_errors = []
        regrets = []
        exact_optimizer_count = 0
        for public_index in public_indices:
            truth = exact[public_index][column_order]
            sampled_basis = basis[probes]
            coefficients = np.linalg.solve(
                sampled_basis.T @ sampled_basis + ridge * np.eye(rank),
                sampled_basis.T @ truth[probes],
            )
            recovered = basis @ coefficients
            relative_error = float(
                np.linalg.norm(recovered - truth)
                / max(float(np.linalg.norm(truth)), np.finfo(float).tiny)
            )
            selected = int(np.argmax(recovered))
            exact_selected = int(np.argmax(truth))
            regret = float(truth[exact_selected] - truth[selected])
            relative_errors.append(relative_error)
            regrets.append(regret)
            exact_optimizer_count += int(selected == exact_selected)
            if count == audit_count:
                record = records[int(public_index)]
                per_graph[record.name] = {
                    "vertices": record.n,
                    "edges": len(record.edges),
                    "relative_error": relative_error,
                    "grid_regret": regret,
                    "exact_optimizer_recovered": int(selected == exact_selected),
                }
                recovered_at_32.append(recovered)
                exact_real.append(truth)
        recovery_rows.append(
            {
                "probes": count,
                "median_relative_error": float(np.median(relative_errors)),
                "mean_relative_error": float(np.mean(relative_errors)),
                "mean_grid_regret": float(np.mean(regrets)),
                "max_grid_regret": float(np.max(regrets, initial=0.0)),
                "exact_optimizer_count": exact_optimizer_count,
            }
        )

    for public_index in public_indices:
        truth = exact[public_index][column_order]
        baseline_selected = int(np.argmax(mean_landscape))
        baseline_regrets.append(float(truth.max() - truth[baseline_selected]))

    column_gram_work = (
        development_operator.rows
        * development_operator.cols
        * development_operator.cols
    )
    row_gram_work = (
        development_operator.rows
        * development_operator.rows
        * development_operator.cols
        + development_operator.rows
        * development_operator.cols
        * rank
    )
    summary = {
        "training_rows": development_operator.rows,
        "angle_settings": development_operator.cols,
        "basis_rank": rank,
        "oversampling": 4,
        "power_iterations": learned.iterations,
        "ridge": ridge,
        "operator_work": learned.operator_work + mean_work,
        "column_gram_formation_work": column_gram_work,
        "row_gram_leading_work": row_gram_work,
        "row_gram_to_action_product_ratio": row_gram_work
        / max(learned.operator_work + mean_work, 1),
        "peak_working_scalars": learned.peak_working_scalars,
        "column_gram_scalars": learned.dense_gram_scalars_avoided,
        "row_gram_scalars": development_operator.rows**2,
        "baseline_warning": (
            "A^T A is dominated when rows << columns; row-Gram and thin-SVD "
            "baselines are required for performance claims"
        ),
        "stored_training_operator_scalars": development_operator.storage,
        "dense_training_operator_scalars": (
            development_operator.rows * development_operator.cols
        ),
        "basis_orthogonality_error": orthogonality_error,
        "probe_seed": SEED,
        "probe_counts": recovery_rows,
        "audit_probes": audit_count,
        "mean_landscape_baseline_mean_regret": float(np.mean(baseline_regrets)),
        "mean_landscape_baseline_max_regret": float(
            np.max(baseline_regrets, initial=0.0)
        ),
        "per_graph_at_audit_probes": per_graph,
    }
    artifacts = {
        "exact_real": np.asarray(exact_real),
        "recovered_real": np.asarray(recovered_at_32),
    }
    return summary, artifacts


def _build_landscapes(profile: str) -> dict[str, Any]:
    settings = PROFILES[profile]
    gset_paths = {name: _gset_path(name) for name in GSET_SOURCES}
    for name, expected in GSET_SOURCES.items():
        if sha256(gset_paths[name]) != expected:
            raise ValueError(f"Gset {name} SHA-256 mismatch")

    records = list(
        benchmark_graphs(
            settings["sizes"],
            int(settings["replicates"]),
            seed=SEED,
        )
    )
    records.extend(parse_gset(gset_paths[name], name) for name in GSET_SOURCES)

    raw_histograms = []
    graph_features = []
    type_union: set[CausalType] = set()
    scaling: dict[int, list[dict[str, float]]] = {}
    for record in records:
        histogram, features = causal_histogram(record)
        raw_histograms.append(histogram)
        type_union.update(histogram)
        graph_features.append(
            [
                math.log2(record.n),
                features["mean_degree"],
                features["degree_std"],
                features["max_degree"],
                math.log1p(features["triangles"]),
                features["causal_types"],
            ]
        )
        if not record.public:
            scaling.setdefault(record.n, []).append(features)

    causal_types = sorted(type_union)
    type_index = {causal_type: index for index, causal_type in enumerate(causal_types)}
    histograms = np.zeros((len(records), len(causal_types)), dtype=float)
    for row, (record, histogram) in enumerate(zip(records, raw_histograms)):
        denominator = max(len(record.edges), 1)
        for causal_type, multiplicity in histogram.items():
            histograms[row, type_index[causal_type]] = multiplicity / denominator

    grid = int(settings["grid"])
    type_landscapes, gamma, beta = type_operator(causal_types, grid, grid)
    landscapes = histograms @ type_landscapes
    column_order = morton_order(grid, grid)
    development = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if not record.public and record.replicate < int(settings["replicates"]) - 1
        ],
        dtype=int,
    )
    test = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if record.public or record.replicate == int(settings["replicates"]) - 1
        ],
        dtype=int,
    )
    row_key = _row_key(np.asarray(graph_features, dtype=float), development)
    scaling_rows = []
    for size in sorted(scaling):
        rows = scaling[size]
        scaling_rows.append(
            {
                "vertices": size,
                "graphs": len(rows),
                "mean_edges": float(np.mean([row["edges"] for row in rows])),
                "mean_causal_types": float(
                    np.mean([row["causal_types"] for row in rows])
                ),
                "max_degree": int(max(row["max_degree"] for row in rows)),
            }
        )
    return {
        "records": records,
        "causal_types": causal_types,
        "histograms": histograms,
        "histogram_nonzeros": int(np.count_nonzero(histograms)),
        "type_landscapes": type_landscapes,
        "landscapes": landscapes,
        "row_key": row_key,
        "column_order": column_order,
        "development": development,
        "test": test,
        "gamma": gamma,
        "beta": beta,
        "scaling": scaling_rows,
    }


def reproduce(profile: str = "paper", *, return_artifacts: bool = False) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    data = _build_landscapes(profile)
    settings = PROFILES[profile]
    exact = data["landscapes"]
    frontier = []
    butterfly_cache: dict[float, ButterflyResult] = {}
    development_nodes: dict[int, HNode] = {}

    for index, (fourier_tolerance, h_tolerance, max_rank) in enumerate(
        settings["candidates"]
    ):
        butterfly = butterfly_cache.setdefault(
            fourier_tolerance,
            butterfly_truncate(
                data["type_landscapes"],
                (int(settings["grid"]), int(settings["grid"])),
                fourier_tolerance,
            ),
        )
        butterfly_landscapes = data["histograms"] @ butterfly.approximation
        _, node, summary = _compress_rows(
            exact,
            butterfly_landscapes,
            data["development"],
            data["row_key"],
            data["column_order"],
            h_tolerance=h_tolerance,
            max_rank=max_rank,
        )
        loss = _candidate_loss(summary, butterfly)
        frontier.append(
            {
                "candidate": index,
                "fourier_tolerance": fourier_tolerance,
                "hierarchical_tolerance": h_tolerance,
                "max_rank": max_rank,
                "development_loss": loss,
                "development_relative_error": summary["relative_frobenius_error"],
                "development_mean_regret": summary["mean_grid_regret"],
                "development_storage_fraction": summary["storage_fraction"],
                "fourier_storage_fraction": butterfly.retained_coefficients
                / max(butterfly.total_coefficients, 1),
            }
        )
        development_nodes[index] = node

    selected_index = min(
        range(len(frontier)), key=lambda index: frontier[index]["development_loss"]
    )
    selected = frontier[selected_index]
    butterfly = butterfly_cache[selected["fourier_tolerance"]]
    butterfly_landscapes = data["histograms"] @ butterfly.approximation
    test_approximation, _, test_summary = _compress_rows(
        exact,
        butterfly_landscapes,
        data["test"],
        data["row_key"],
        data["column_order"],
        h_tolerance=selected["hierarchical_tolerance"],
        max_rank=int(selected["max_rank"]),
    )
    full_approximation, full_node, full_summary = _compress_rows(
        exact,
        butterfly_landscapes,
        np.arange(len(exact), dtype=int),
        data["row_key"],
        data["column_order"],
        h_tolerance=selected["hierarchical_tolerance"],
        max_rank=int(selected["max_rank"]),
    )

    ordered_rows = np.argsort(data["row_key"], kind="mergesort")
    ordered_operator = butterfly_landscapes[ordered_rows][:, data["column_order"]]
    rng = np.random.default_rng(SEED + 91)
    batch = rng.normal(size=(ordered_operator.shape[1], 8))
    applied, hierarchical_work = full_node.matmat(batch)
    direct = ordered_operator @ batch
    application_error = float(
        np.linalg.norm(applied - direct)
        / max(float(np.linalg.norm(direct)), np.finfo(float).tiny)
    )
    dense_work = ordered_operator.shape[0] * ordered_operator.shape[1] * batch.shape[1]
    exact_factor_work = (
        data["type_landscapes"].shape[0]
        * data["type_landscapes"].shape[1]
        * batch.shape[1]
        + data["histogram_nonzeros"] * batch.shape[1]
    )

    public_indices = np.asarray(
        [index for index, record in enumerate(data["records"]) if record.public],
        dtype=int,
    )
    public_rows = []
    for public_index in public_indices:
        test_location = int(np.flatnonzero(data["test"] == public_index)[0])
        public_truth = exact[public_index]
        public_approximation = test_approximation[test_location]
        public_selected = int(np.argmax(public_approximation))
        record = data["records"][int(public_index)]
        public_rows.append(
            {
                "name": record.name,
                "vertices": record.n,
                "edges": len(record.edges),
                "sha256": GSET_SOURCES[record.name.removeprefix("Gset-")],
                "selected_gamma": float(
                    data["gamma"][public_selected // int(settings["grid"])]
                ),
                "selected_beta": float(
                    data["beta"][public_selected % int(settings["grid"])]
                ),
                "grid_regret": float(
                    public_truth.max() - public_truth[public_selected]
                ),
            }
        )

    recovery, recovery_artifacts = _recover_real_landscapes(
        development_nodes[selected_index],
        exact,
        public_indices,
        data["column_order"],
        data["records"],
        profile=profile,
    )

    audit = statevector_audit()
    result: dict[str, Any] = {
        "schema_version": 2,
        "title": TITLE,
        "package": PACKAGE_NAME,
        "profile": profile,
        "seed": SEED,
        "dataset": {
            "synthetic_families": 5,
            "synthetic_graphs": sum(
                not record.public for record in data["records"]
            ),
            "public_graphs": len(public_indices),
            "development_graphs": len(data["development"]),
            "test_graphs": len(data["test"]),
            "minimum_vertices": min(record.n for record in data["records"]),
            "maximum_vertices": max(record.n for record in data["records"]),
            "maximum_edges": max(len(record.edges) for record in data["records"]),
            "gset_sha256": GSET_SOURCES,
            "gset_graphs": [
                {
                    "name": row["name"],
                    "vertices": row["vertices"],
                    "edges": row["edges"],
                    "sha256": row["sha256"],
                }
                for row in public_rows
            ],
        },
        "operator": {
            "causal_types": len(data["causal_types"]),
            "histogram_nonzeros": data["histogram_nonzeros"],
            "angle_grid_side": int(settings["grid"]),
            "angle_settings": int(settings["grid"]) ** 2,
            "matrix_rows": exact.shape[0],
            "matrix_columns": exact.shape[1],
            "dense_entries": int(exact.size),
        },
        "statevector_audit": audit,
        "selection": {
            "loss_definition": "L_CHC = relative_frobenius_error^2 + mean_grid_regret + 0.01*h_storage_fraction + 0.002*fourier_storage_fraction + 100*certificate_violation",
            "selected_candidate": selected_index,
            "selected_fourier_tolerance": selected["fourier_tolerance"],
            "selected_hierarchical_tolerance": selected[
                "hierarchical_tolerance"
            ],
            "selected_max_rank": selected["max_rank"],
            "frontier": frontier,
        },
        "butterfly": {
            "retained_coefficients": butterfly.retained_coefficients,
            "total_coefficients": butterfly.total_coefficients,
            "storage_fraction": butterfly.retained_coefficients
            / max(butterfly.total_coefficients, 1),
            "relative_error": butterfly.relative_error,
            "max_abs_error": butterfly.max_abs_error,
            "counted_work": butterfly.counted_work,
            "dense_fourier_work": int(
                len(data["causal_types"])
                * int(settings["grid"]) ** 4
            ),
        },
        "hierarchy": {
            "storage_fraction": full_summary["storage_fraction"],
            "relative_frobenius_error": full_summary[
                "relative_frobenius_error"
            ],
            "max_abs_error": full_summary["max_abs_error"],
            "certificate_frobenius": full_summary["certificate_frobenius"],
            "application_relative_error_against_butterfly_operator": application_error,
            "dense_matmat_work": dense_work,
            "hierarchical_matmat_work": hierarchical_work,
            "dense_to_hierarchical_work_ratio": dense_work
            / max(hierarchical_work, 1),
            "exact_causal_factor_matmat_work": exact_factor_work,
            "dense_to_exact_factor_work_ratio": dense_work
            / max(exact_factor_work, 1),
            **_rank_summary(full_node),
        },
        "held_out": {
            key: test_summary[key]
            for key in (
                "relative_frobenius_error",
                "max_abs_error",
                "mean_grid_regret",
                "max_grid_regret",
                "certificate_limit",
                "certificate_violation",
                "margin_preserved_fraction",
                "storage_fraction",
            )
        },
        "public_gset": {
            "graphs": public_rows,
            "mean_grid_regret": float(
                np.mean([row["grid_regret"] for row in public_rows])
            ),
            "max_grid_regret": float(
                np.max([row["grid_regret"] for row in public_rows], initial=0.0)
            ),
        },
        "matrix_free_recovery": recovery,
        "scaling": data["scaling"],
    }
    result = _rounded(result)
    if return_artifacts:
        result["_artifacts"] = {
            "exact": exact,
            "approximation": full_approximation,
            "public_indices": public_indices,
            "grid": int(settings["grid"]),
            "ranks": full_node.ranks,
            **recovery_artifacts,
        }
    return result


def _tex_macros(result: dict[str, Any]) -> str:
    dataset = result["dataset"]
    operator = result["operator"]
    butterfly = result["butterfly"]
    hierarchy = result["hierarchy"]
    held_out = result["held_out"]
    public = result["public_gset"]
    recovery = result["matrix_free_recovery"]
    recovery_audit = next(
        row
        for row in recovery["probe_counts"]
        if row["probes"] == recovery["audit_probes"]
    )
    macros = {
        "SyntheticGraphs": dataset["synthetic_graphs"],
        "PublicGraphs": dataset["public_graphs"],
        "DevelopmentGraphs": dataset["development_graphs"],
        "TestGraphs": dataset["test_graphs"],
        "MaximumVertices": dataset["maximum_vertices"],
        "MaximumEdges": dataset["maximum_edges"],
        "CausalTypes": operator["causal_types"],
        "AngleSettings": operator["angle_settings"],
        "OperatorEntries": operator["dense_entries"],
        "HistogramNonzeros": operator["histogram_nonzeros"],
        "StatevectorChecks": result["statevector_audit"]["statevector_comparisons"],
        "StatevectorError": f'{result["statevector_audit"]["statevector_max_abs_error"]:.2e}',
        "FourierStorage": f'{100 * butterfly["storage_fraction"]:.2f}',
        "FourierError": f'{butterfly["relative_error"]:.2e}',
        "ButterflyWorkRatio": f'{butterfly["dense_fourier_work"] / butterfly["counted_work"]:.2f}',
        "HStorage": f'{100 * hierarchy["storage_fraction"]:.2f}',
        "HError": f'{hierarchy["relative_frobenius_error"]:.2e}',
        "HMaxRank": hierarchy["rank_max"],
        "HMedianRank": f'{hierarchy["rank_median"]:.1f}',
        "HWorkRatio": f'{hierarchy["dense_to_hierarchical_work_ratio"]:.2f}',
        "FactorWorkRatio": f'{hierarchy["dense_to_exact_factor_work_ratio"]:.2f}',
        "DenseMatmatWork": hierarchy["dense_matmat_work"],
        "HierarchyMatmatWork": hierarchy["hierarchical_matmat_work"],
        "FactorMatmatWork": hierarchy["exact_causal_factor_matmat_work"],
        "HeldoutError": f'{held_out["relative_frobenius_error"]:.2e}',
        "HeldoutMeanRegret": f'{held_out["mean_grid_regret"]:.2e}',
        "HeldoutMaxRegret": f'{held_out["max_grid_regret"]:.2e}',
        "HeldoutEpsilon": f'{held_out["max_abs_error"]:.2e}',
        "HeldoutCertificate": f'{held_out["certificate_limit"]:.2e}',
        "HeldoutMargin": f'{100 * held_out["margin_preserved_fraction"]:.1f}',
        "PublicMeanRegret": f'{public["mean_grid_regret"]:.2e}',
        "PublicMaxRegret": f'{public["max_grid_regret"]:.2e}',
        "RecoveryRank": recovery["basis_rank"],
        "RecoveryProbes": recovery["audit_probes"],
        "RecoveryMedianError": f'{recovery_audit["median_relative_error"]:.2e}',
        "RecoveryMeanRegret": f'{recovery_audit["mean_grid_regret"]:.2e}',
        "RecoveryMaxRegret": f'{recovery_audit["max_grid_regret"]:.2e}',
        "RecoveryExactCount": recovery_audit["exact_optimizer_count"],
        "RecoveryWorkRatio": f'{recovery["row_gram_to_action_product_ratio"]:.2f}',
        "RecoveryPeakKiB": f'{8 * recovery["peak_working_scalars"] / 1024:.1f}',
        "RecoveryGramMiB": f'{8 * recovery["row_gram_scalars"] / 2**20:.3f}',
    }
    return "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    ) + "\n"


def _write_figures(result: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = result["_artifacts"]
    frontier = result["selection"]["frontier"]
    plt.figure(figsize=(5.2, 3.2))
    x = [100 * row["development_storage_fraction"] for row in frontier]
    y = [row["development_mean_regret"] for row in frontier]
    plt.plot(x, y, "o-", color="#285f9e")
    selected = result["selection"]["selected_candidate"]
    plt.plot(
        [x[selected]],
        [y[selected]],
        marker="*",
        markersize=13,
        linestyle="none",
        color="#b52b2b",
        zorder=3,
    )
    for row, x_value, y_value in zip(frontier, x, y):
        plt.annotate(str(row["candidate"]), (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("hierarchical storage (% of dense)")
    plt.ylabel("development grid regret")
    plt.tight_layout()
    plt.savefig(output / "compression_frontier.pdf")
    plt.close()

    ranks = artifacts["ranks"]
    rank_values, rank_counts = np.unique(ranks, return_counts=True)
    plt.figure(figsize=(5.2, 3.2))
    plt.bar(rank_values, rank_counts, color="#4c956c", edgecolor="white")
    plt.xlabel("adaptive block rank")
    plt.ylabel("compressed blocks")
    plt.tight_layout()
    plt.savefig(output / "rank_distribution.pdf")
    plt.close()

    exact = artifacts["exact_real"][0]
    recovered = artifacts["recovered_real"][0]
    order = np.argsort(exact, kind="mergesort")
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.65), constrained_layout=True)
    axes[0].plot(exact[order], label="exact", color="#1f4e79", linewidth=1.4)
    axes[0].plot(
        recovered[order],
        label=f'{result["matrix_free_recovery"]["audit_probes"]}-probe recovery',
        color="#b33c2e",
        linewidth=1.1,
        alpha=0.9,
    )
    axes[0].set_xlabel("G1 angle settings, sorted by exact value")
    axes[0].set_ylabel("normalized expected cut")
    axes[0].legend(frameon=False)
    axes[1].plot(
        np.abs(exact[order] - recovered[order]),
        color="#5b8e55",
        linewidth=1.2,
    )
    axes[1].set_xlabel("same ordered angle settings")
    axes[1].set_ylabel("absolute recovery error")
    figure.savefig(output / "gset_landscape_audit.pdf")
    plt.close(figure)

    recovery = result["matrix_free_recovery"]
    probe_rows = recovery["probe_counts"]
    probes = [row["probes"] for row in probe_rows]
    median_error = [row["median_relative_error"] for row in probe_rows]
    maximum_regret = [row["max_grid_regret"] for row in probe_rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.65), constrained_layout=True)
    axes[0].plot(probes, median_error, "o-", color="#285f9e")
    axes[0].set_xlabel("observed angle values per real graph")
    axes[0].set_ylabel("median relative recovery error")
    axes[0].set_yscale("log")
    axes[1].plot(probes, maximum_regret, "o-", color="#b33c2e")
    axes[1].set_xlabel("observed angle values per real graph")
    axes[1].set_ylabel("maximum grid regret")
    axes[1].set_yscale("log")
    figure.savefig(output / "real_recovery.pdf")
    plt.close(figure)

    names = list(recovery["per_graph_at_audit_probes"])
    errors = [
        recovery["per_graph_at_audit_probes"][name]["relative_error"]
        for name in names
    ]
    regrets = [
        recovery["per_graph_at_audit_probes"][name]["grid_regret"]
        for name in names
    ]
    short_names = [name.removeprefix("Gset-") for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.65), constrained_layout=True)
    axes[0].bar(short_names, errors, color="#4c78a8")
    axes[0].set_xlabel("Stanford Gset target")
    axes[0].set_ylabel("relative recovery error")
    axes[1].bar(short_names, regrets, color="#e07a5f")
    axes[1].set_xlabel("Stanford Gset target")
    axes[1].set_ylabel("grid regret")
    figure.savefig(output / "real_recovery_bars.pdf")
    plt.close(figure)


def _write_tables(result: dict[str, Any], output: Path) -> None:
    recovery = result["matrix_free_recovery"]
    real_rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Target & $|V|$ & $|E|$ & rel. error & grid regret & exact grid optimum\\",
        r"\midrule",
    ]
    for name, row in recovery["per_graph_at_audit_probes"].items():
        real_rows.append(
            f"{name.removeprefix('Gset-')} & {row['vertices']} & "
            f"{row['edges']} & {row['relative_error']:.3e} & "
            f"{row['grid_regret']:.3e} & "
            f"{'yes' if row['exact_optimizer_recovered'] else 'no'}\\\\"
        )
    real_rows.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "real_recovery_table.tex").write_text(
        "\n".join(real_rows) + "\n",
        encoding="utf-8",
    )

    provenance_rows = [
        r"\begin{tabular}{lrrll}",
        r"\toprule",
        r"Instance & $|V|$ & $|E|$ & SHA-256 prefix & source\\",
        r"\midrule",
    ]
    for row in result["dataset"]["gset_graphs"]:
        label = row["name"].removeprefix("Gset-")
        provenance_rows.append(
            f"{label} & {row['vertices']} & {row['edges']} & "
            rf"\texttt{{{row['sha256'][:12]}}} & Stanford Gset\\"
        )
    provenance_rows.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "gset_provenance_table.tex").write_text(
        "\n".join(provenance_rows) + "\n",
        encoding="utf-8",
    )

    profile_rows = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"probes & median rel. error & mean regret & max regret & exact optima\\",
        r"\midrule",
    ]
    for row in recovery["probe_counts"]:
        profile_rows.append(
            f"{row['probes']} & {row['median_relative_error']:.3e} & "
            f"{row['mean_grid_regret']:.3e} & {row['max_grid_regret']:.3e} & "
            f"{row['exact_optimizer_count']}/{result['dataset']['public_graphs']}\\\\"
        )
    profile_rows.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "recovery_profile_table.tex").write_text(
        "\n".join(profile_rows) + "\n",
        encoding="utf-8",
    )

    resource_rows = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"quantity & counted value\\",
        r"\midrule",
        f"development rows & {recovery['training_rows']}\\\\",
        f"angle settings & {recovery['angle_settings']}\\\\",
        f"stored hierarchical scalars & {recovery['stored_training_operator_scalars']}\\\\",
        f"dense training-operator scalars & {recovery['dense_training_operator_scalars']}\\\\",
        f"matrix-free peak working scalars & {recovery['peak_working_scalars']}\\\\",
        f"row-Gram scalars & {recovery['row_gram_scalars']}\\\\",
        f"column-Gram scalars (dominated) & {recovery['column_gram_scalars']}\\\\",
        f"matrix-free counted products & {recovery['operator_work']}\\\\",
        f"row-Gram leading products & {recovery['row_gram_leading_work']}\\\\",
        f"column-Gram formation products (dominated) & {recovery['column_gram_formation_work']}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (output / "matrix_free_resources_table.tex").write_text(
        "\n".join(resource_rows) + "\n",
        encoding="utf-8",
    )


def write_outputs(result: dict[str, Any]) -> None:
    output = PROJECT_ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if key != "_artifacts"}
    locked_text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    (output / "results.json").write_text(locked_text, encoding="utf-8")
    (SOURCE_ROOT / "locked" / "results.json").write_text(
        locked_text,
        encoding="utf-8",
    )
    (output / "results.tex").write_text(_tex_macros(serializable), encoding="utf-8")
    with (output / "compression_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable["selection"]["frontier"][0]))
        writer.writeheader()
        writer.writerows(serializable["selection"]["frontier"])
    _write_figures(result, output)
    _write_tables(serializable, output)

    manifest = {
        "schema_version": 2,
        "title": TITLE,
        "package": PACKAGE_NAME,
        "claim_status": (
            "application-specific audit; no new generic matrix-free, hierarchical, "
            "subspace-iteration, or log-determinant algorithm is claimed"
        ),
        "weighted_gset": False,
        "seed": SEED,
        "profile": serializable["profile"],
        "public_data": [
            {
                "name": f"Stanford Gset {name}",
                "url": GSET_URLS[name],
                "sha256": digest,
            }
            for name, digest in GSET_SOURCES.items()
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
            "processor": platform.processor(),
        },
        "commands": {
            "install": "python -m pip install .",
            "reproduce": (
                "HQML_GSET_CACHE=/tmp/hqml-gset "
                "hierarchical-qaoa-reproduce --profile paper --verify"
            ),
            "test": "python -m unittest discover -s tests -v",
        },
    }
    (PROJECT_ROOT / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compare(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in expected:
            _compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path}: list shape differs")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare(expected_item, actual_item, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(expected, float(actual), rel_tol=2e-8, abs_tol=2e-10):
            raise AssertionError(f"{path}: {actual} != {expected}")
    elif expected != actual:
        raise AssertionError(f"{path}: {actual} != {expected}")


def verify(result: dict[str, Any]) -> None:
    result = {key: value for key, value in result.items() if key != "_artifacts"}
    locked = PROJECT_ROOT / "results" / "results.json"
    if not locked.is_file():
        locked = SOURCE_ROOT / "locked" / "results.json"
    expected = json.loads(locked.read_text(encoding="utf-8"))
    if expected.get("profile") != result.get("profile"):
        if result["held_out"]["certificate_violation"] != 0.0:
            raise AssertionError("quick-profile compression-regret certificate failed")
        if result["matrix_free_recovery"]["basis_orthogonality_error"] > 1e-10:
            raise AssertionError("quick-profile learned basis is not orthonormal")
        return
    _compare(expected, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="paper")
    parser.add_argument("--write", action="store_true", help="regenerate paper artifacts")
    parser.add_argument("--verify", action="store_true", help="compare with locked results")
    arguments = parser.parse_args(argv)
    result = reproduce(arguments.profile, return_artifacts=arguments.write)
    if arguments.verify:
        verify(result)
    if arguments.write:
        write_outputs(result)
    json.dump(
        {key: value for key, value in result.items() if key != "_artifacts"},
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
