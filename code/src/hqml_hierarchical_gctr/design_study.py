"""Baseline-audited affine design study on authenticated Stanford Gset graphs."""

from __future__ import annotations

import argparse
import hashlib
from importlib import resources
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from .experiment import (
    GSET_SOURCES,
    PROFILES,
    SEED,
    _build_landscapes,
    _compress_rows,
    _rounded,
)
from .matrix_free import (
    CenteredOperator,
    butterfly_truncate,
    greedy_logdet_probes,
    matrix_free_right_subspace,
    projector_distance,
    relative_projection_residual,
    row_gram_right_subspace,
    thin_svd_right_subspace,
)

COUNTS = (12, 16, 24, 32, 48, 64)
DECISION_WEIGHTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
RIDGE = 1.0e-6
DESIGN_RIDGE = 1.0e-8
RANK = 12
RANDOM_REPLICATES = 64
RUNTIME_RANKS = (4, 8, 12, 16, 24)
RUNTIME_WARMUPS = 2
RUNTIME_REPETITIONS = 15


def _mean_and_basis(node, *, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    mean, mean_work = node.transpose_matmat(np.full(node.rows, 1.0 / node.rows))
    centered = CenteredOperator(node, mean)
    learned = matrix_free_right_subspace(
        centered,
        RANK,
        oversampling=4,
        iterations=4,
        seed=seed,
    )
    return mean, learned.basis, learned.operator_work + mean_work


def _importance(training_rows: np.ndarray) -> np.ndarray:
    threshold = np.quantile(training_rows, 0.95, axis=1, keepdims=True)
    frequency = np.mean(training_rows >= threshold, axis=0)
    maximum = float(np.max(frequency, initial=0.0))
    return frequency / maximum if maximum > 0.0 else frequency


def _recover(
    basis: np.ndarray,
    mean: np.ndarray,
    rows: np.ndarray,
    probes: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    sampled = basis[probes]
    system = sampled.T @ sampled + RIDGE * np.eye(basis.shape[1])
    relative_errors: list[float] = []
    regrets: list[float] = []
    exact: list[int] = []
    details: list[dict[str, Any]] = []
    recovered_rows: list[np.ndarray] = []
    for row in rows:
        coefficients = np.linalg.solve(
            system,
            sampled.T @ (row[probes] - mean[probes]),
        )
        recovered = mean + basis @ coefficients
        relative_error = float(
            np.linalg.norm(recovered - row)
            / max(float(np.linalg.norm(row)), np.finfo(float).tiny)
        )
        selected = int(np.argmax(recovered))
        exact_selected = int(np.argmax(row))
        regret = float(row[exact_selected] - row[selected])
        relative_errors.append(relative_error)
        regrets.append(regret)
        exact.append(int(selected == exact_selected))
        recovered_rows.append(recovered)
        details.append(
            {
                "relative_error": relative_error,
                "grid_regret": regret,
                "exact_optimizer_recovered": int(selected == exact_selected),
            }
        )
    summary = {
        "median_relative_error": float(np.median(relative_errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
        "mean_grid_regret": float(np.mean(regrets)),
        "maximum_grid_regret": float(np.max(regrets, initial=0.0)),
        "exact_optimizer_count": int(np.sum(exact)),
        "minimum_sampled_singular_value": float(
            np.linalg.svd(sampled, compute_uv=False)[-1]
        ),
    }
    return summary, details, np.asarray(recovered_rows)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _random_probe_distribution(
    basis: np.ndarray,
    mean: np.ndarray,
    rows: np.ndarray,
    counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Evaluate a predeclared distribution of random observation sets.

    A single random permutation is an unstable baseline.  Each registered
    seed creates an independent permutation, and every metric is summarized
    by its five-number distribution without selecting a favorable seed.
    """

    records: list[dict[str, Any]] = []
    for count in counts:
        summaries: list[dict[str, Any]] = []
        for replicate in range(RANDOM_REPLICATES):
            generator = np.random.default_rng(SEED + 50_000 + replicate)
            probes = generator.permutation(len(mean))[:count]
            summary, _, _ = _recover(basis, mean, rows, probes)
            summaries.append(summary)
        records.append(
            {
                "probes": count,
                "replicates": RANDOM_REPLICATES,
                "seed_first": SEED + 50_000,
                "seed_last": SEED + 50_000 + RANDOM_REPLICATES - 1,
                "median_relative_error": _distribution(
                    [row["median_relative_error"] for row in summaries]
                ),
                "maximum_grid_regret": _distribution(
                    [row["maximum_grid_regret"] for row in summaries]
                ),
                "minimum_sampled_singular_value": _distribution(
                    [row["minimum_sampled_singular_value"] for row in summaries]
                ),
                "exact_optimizer_count": _distribution(
                    [row["exact_optimizer_count"] for row in summaries]
                ),
            }
        )
    return records


def _timed_distribution(function, repetitions: int) -> dict[str, Any]:
    for _ in range(RUNTIME_WARMUPS):
        function()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) * 1.0e-9)
    return {
        "seconds": samples,
        **_distribution(samples),
    }


def _runtime_benchmark(
    node,
    mean: np.ndarray,
    dense_centered: np.ndarray,
    *,
    repetitions: int,
) -> dict[str, Any]:
    """Time three subspace routes on the same already-built operator.

    The benchmark starts after hierarchy construction.  The row-Gram and thin
    SVD routes are given the dense matrix that the reference builder already
    materialized; hiding that available baseline would bias the comparison.
    Timings are descriptive and are excluded from cross-machine verification.
    """

    centered = CenteredOperator(node, mean)
    by_rank: list[dict[str, Any]] = []
    for rank in RUNTIME_RANKS:
        methods = {
            "hierarchical_action_iteration": lambda rank=rank: matrix_free_right_subspace(
                centered,
                rank,
                oversampling=4,
                iterations=4,
                seed=SEED + 177,
            ),
            "row_gram": lambda rank=rank: row_gram_right_subspace(
                dense_centered, rank
            ),
            "thin_svd": lambda rank=rank: thin_svd_right_subspace(
                dense_centered, rank
            ),
        }
        by_rank.append(
            {
                "rank": rank,
                "methods": {
                    name: _timed_distribution(function, repetitions)
                    for name, function in methods.items()
                },
            }
        )
    return {
        "scope": "learning phase only; hierarchy construction and data loading excluded",
        "dense_input_status": "available from the dense-local reference builder",
        "clock": "time.perf_counter_ns",
        "warmups": RUNTIME_WARMUPS,
        "repetitions": repetitions,
        "thread_policy": "ambient NumPy/BLAS threading; no cross-machine speed claim",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
            "processor": platform.processor(),
        },
        "by_rank": by_rank,
    }


def _uniform_order(size: int, count: int) -> np.ndarray:
    return np.unique(np.rint(np.linspace(0, size - 1, count)).astype(int))


def _compression_frontier(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int, Any]:
    exact = data["landscapes"]
    settings = PROFILES["paper"]
    frontier: list[dict[str, Any]] = []
    nodes: list[Any] = []
    for index, (fourier_tolerance, h_tolerance, max_rank) in enumerate(
        settings["candidates"]
    ):
        butterfly = butterfly_truncate(
            data["type_landscapes"],
            (int(settings["grid"]), int(settings["grid"])),
            fourier_tolerance,
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
        frontier.append(
            {
                "candidate": index,
                "fourier_tolerance": fourier_tolerance,
                "hierarchical_tolerance": h_tolerance,
                "maximum_rank": max_rank,
                "relative_error": summary["relative_frobenius_error"],
                "maximum_grid_regret": summary["max_grid_regret"],
                "storage_fraction": summary["storage_fraction"],
                "certificate_violation": summary["certificate_violation"],
                "observed_maximum_rank": summary["rank_max"],
            }
        )
        nodes.append(node)
    feasible = [
        record
        for record in frontier
        if record["relative_error"] <= 5.0e-3
        and record["maximum_grid_regret"] <= 2.0e-4
        and record["certificate_violation"] <= 1.0e-12
    ]
    if not feasible:
        raise AssertionError("no compression candidate satisfies the registered constraints")
    selected = min(feasible, key=lambda record: record["storage_fraction"])
    selected_index = int(selected["candidate"])
    return frontier, selected_index, nodes[selected_index]


def _noise_study(
    basis: np.ndarray,
    mean: np.ndarray,
    rows: np.ndarray,
    probes: np.ndarray,
) -> list[dict[str, Any]]:
    sampled = basis[probes]
    singular_minimum = float(np.linalg.svd(sampled, compute_uv=False)[-1])
    sampled_norm = float(np.linalg.norm(sampled, ord=2))
    denominator = singular_minimum**2 + RIDGE
    records: list[dict[str, Any]] = []
    for noise_standard_deviation in (0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        actual_values: list[float] = []
        bounds: list[float] = []
        for noise_seed in range(8):
            generator = np.random.default_rng(SEED + 1000 + noise_seed)
            for row in rows:
                coefficients = basis.T @ (row - mean)
                residual = row - mean - basis @ coefficients
                noise = generator.normal(
                    scale=noise_standard_deviation, size=len(probes)
                )
                estimate = np.linalg.solve(
                    sampled.T @ sampled + RIDGE * np.eye(basis.shape[1]),
                    sampled.T @ (row[probes] + noise - mean[probes]),
                )
                recovered = mean + basis @ estimate
                actual_values.append(float(np.max(np.abs(recovered - row))))
                bound = float(
                    np.max(np.abs(residual))
                    + (
                        RIDGE * np.linalg.norm(coefficients)
                        + sampled_norm
                        * (np.linalg.norm(residual[probes]) + np.linalg.norm(noise))
                    )
                    / denominator
                )
                bounds.append(bound)
        records.append(
            {
                "noise_standard_deviation": noise_standard_deviation,
                "maximum_observed_error": float(np.max(actual_values)),
                "maximum_bound": float(np.max(bounds)),
                "all_errors_bounded": bool(
                    np.all(np.asarray(actual_values) <= np.asarray(bounds) + 1.0e-14)
                ),
            }
        )
    return records


def run_design_study(
    *,
    record_timings: bool = False,
    timing_repetitions: int = RUNTIME_REPETITIONS,
) -> dict[str, Any]:
    data = _build_landscapes("paper")
    exact = data["landscapes"]
    column_order = data["column_order"]
    records = data["records"]
    frontier, selected_index, development_node = _compression_frontier(data)

    calibration_train = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if not record.public and record.replicate < 2
        ],
        dtype=int,
    )
    calibration_validation = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if not record.public and record.replicate == 2
        ],
        dtype=int,
    )
    public_indices = np.asarray(
        [index for index, record in enumerate(records) if record.public], dtype=int
    )
    public_rows = exact[public_indices][:, column_order]

    fourier_tolerance, h_tolerance, max_rank = PROFILES["paper"]["candidates"][selected_index]
    butterfly = butterfly_truncate(
        data["type_landscapes"], (32, 32), fourier_tolerance
    )
    butterfly_landscapes = data["histograms"] @ butterfly.approximation
    _, calibration_node, _ = _compress_rows(
        exact,
        butterfly_landscapes,
        calibration_train,
        data["row_key"],
        column_order,
        h_tolerance=h_tolerance,
        max_rank=max_rank,
    )
    calibration_mean, calibration_basis, _ = _mean_and_basis(
        calibration_node, seed=SEED + 377
    )
    calibration_training_rows = exact[calibration_train][:, column_order]
    calibration_validation_rows = exact[calibration_validation][:, column_order]
    calibration_importance = _importance(calibration_training_rows)

    selected_weights: dict[int, float] = {}
    calibration_records: dict[int, list[dict[str, Any]]] = {}
    for count in COUNTS:
        candidates: list[dict[str, Any]] = []
        for decision_weight in DECISION_WEIGHTS:
            probes = greedy_logdet_probes(
                calibration_basis,
                count,
                weights=calibration_importance,
                decision_weight=decision_weight,
                regularization=DESIGN_RIDGE,
            )
            summary, _, _ = _recover(
                calibration_basis,
                calibration_mean,
                calibration_validation_rows,
                probes,
            )
            selection_loss = (
                summary["mean_relative_error"]
                + 100.0 * summary["mean_grid_regret"]
            )
            candidates.append(
                {
                    "decision_weight": decision_weight,
                    "selection_loss": selection_loss,
                    **summary,
                }
            )
        best_loss = min(record["selection_loss"] for record in candidates)
        numerically_tied = [
            record
            for record in candidates
            if record["selection_loss"] <= best_loss + 1.0e-12
        ]
        selected = min(numerically_tied, key=lambda record: record["decision_weight"])
        selected_weights[count] = float(selected["decision_weight"])
        calibration_records[count] = candidates

    mean, basis, operator_work = _mean_and_basis(
        development_node, seed=SEED + 177
    )
    development_rows = exact[data["development"]][:, column_order]
    importance = _importance(development_rows)
    leverage_order = np.argsort(np.sum(basis * basis, axis=1))[::-1]

    count_records: list[dict[str, Any]] = []
    proposed_details: dict[int, list[dict[str, Any]]] = {}
    proposed_recovered: dict[int, np.ndarray] = {}
    for count in COUNTS:
        unweighted = greedy_logdet_probes(
            basis, count, regularization=DESIGN_RIDGE
        )
        proposed = greedy_logdet_probes(
            basis,
            count,
            weights=importance,
            decision_weight=selected_weights[count],
            regularization=DESIGN_RIDGE,
        )
        strategies = {
            "morton_spaced": _uniform_order(len(mean), count),
            "leverage": leverage_order[:count],
            "unweighted_logdet": unweighted,
            "decision_weighted": proposed,
        }
        strategy_results: dict[str, Any] = {}
        for name, probes in strategies.items():
            summary, details, recovered = _recover(
                basis, mean, public_rows, probes
            )
            strategy_results[name] = summary
            if name == "decision_weighted":
                proposed_details[count] = details
                proposed_recovered[count] = recovered
        count_records.append(
            {
                "probes": count,
                "selected_decision_weight": selected_weights[count],
                "strategies": strategy_results,
            }
        )

    random_distribution = _random_probe_distribution(
        basis, mean, public_rows, COUNTS
    )

    audit_count = 16
    audit_record = next(record for record in count_records if record["probes"] == audit_count)
    audit_details = proposed_details[audit_count]
    per_graph = []
    for index, (public_index, detail) in enumerate(zip(public_indices, audit_details)):
        record = records[int(public_index)]
        per_graph.append(
            {
                "name": record.name,
                "vertices": record.n,
                "edges": len(record.edges),
                **detail,
            }
        )
    proposed_probes = greedy_logdet_probes(
        basis,
        audit_count,
        weights=importance,
        decision_weight=selected_weights[audit_count],
        regularization=DESIGN_RIDGE,
    )
    noise = _noise_study(basis, mean, public_rows, proposed_probes)

    dense_operator = development_node.reconstruct()
    dense_centered = dense_operator - dense_operator.mean(axis=0, keepdims=True)
    if not np.allclose(dense_operator.mean(axis=0), mean, rtol=1.0e-12, atol=1.0e-13):
        raise AssertionError("action-computed and dense development means differ")
    thin = thin_svd_right_subspace(dense_centered, RANK)
    row_gram = row_gram_right_subspace(dense_centered, RANK)
    subspace_bases = {
        "hierarchical_action_iteration": basis,
        "row_gram": row_gram.basis,
        "thin_svd": thin.basis,
    }
    subspace_methods: dict[str, Any] = {}
    for name, method_basis in subspace_bases.items():
        probes = greedy_logdet_probes(
            method_basis,
            audit_count,
            weights=importance,
            decision_weight=selected_weights[audit_count],
            regularization=DESIGN_RIDGE,
        )
        recovery_summary, _, _ = _recover(
            method_basis, mean, public_rows, probes
        )
        subspace_methods[name] = {
            "relative_projection_residual": relative_projection_residual(
                dense_centered, method_basis
            ),
            "projector_distance_to_thin_svd": projector_distance(
                method_basis, thin.basis
            ),
            "probe_overlap_with_thin_svd": int(
                len(
                    np.intersect1d(
                        probes,
                        greedy_logdet_probes(
                            thin.basis,
                            audit_count,
                            weights=importance,
                            decision_weight=selected_weights[audit_count],
                            regularization=DESIGN_RIDGE,
                        ),
                    )
                )
            ),
            "audit_recovery": recovery_summary,
        }

    rows, columns = dense_centered.shape
    width = RANK + 4
    matrix_free_working_scalars = int(
        2 * columns * width + rows * width + width * width
    )
    subspace_baselines = {
        "comparison_scope": "same selected hierarchical approximation after construction",
        "dimensions": {"m": rows, "q": columns, "rank": RANK},
        "dense_input_materialized_by_builder": True,
        "methods": subspace_methods,
        "resource_ledger": {
            "hierarchical_action_iteration": {
                "operator_products": operator_work,
                "stored_operator_scalars": development_node.storage + columns,
                "working_scalars": matrix_free_working_scalars,
                "qr_and_small_svd_excluded_from_product_count": True,
            },
            "row_gram": {
                "leading_products": row_gram.leading_products,
                "input_scalars": row_gram.input_scalars,
                "auxiliary_scalars": row_gram.auxiliary_scalars,
                "gram_shape": [rows, rows],
            },
            "thin_svd": {
                "leading_order": "O(m^2 q) for m <= q; library workspace is implementation dependent",
                "input_scalars": thin.input_scalars,
                "reported_auxiliary_scalars_lower_bound": thin.auxiliary_scalars,
            },
            "column_gram_A_transpose_A": {
                "formation_products": int(rows * columns * columns),
                "gram_scalars": int(columns * columns),
                "scientific_status": "dominated comparator for m << q; not a headline baseline",
            },
        },
    }

    result = {
        "schema_version": 2,
        "study": "audited affine subspace recovery on authenticated Stanford Gset graph topologies",
        "claim_status": (
            "application-specific audit; no new generic matrix-free, hierarchical, "
            "subspace-iteration, or log-determinant algorithm is claimed"
        ),
        "data": {
            "source": "https://web.stanford.edu/~yyye/yyye/Gset/",
            "graphs": [records[int(index)].name for index in public_indices],
            "sha256": {
                record.name: GSET_SOURCES[record.name.removeprefix("Gset-")]
                for record in records
                if record.public
            },
            "weight_field_used": False,
            "development_rows": len(data["development"]),
            "calibration_training_rows": len(calibration_train),
            "calibration_validation_rows": len(calibration_validation),
            "public_test_rows": len(public_indices),
            "angle_settings": len(mean),
        },
        "compression": {
            "selection_rule": "minimum storage subject to development relative error <= 0.005, maximum grid regret <= 0.0002, and zero certificate violation",
            "selected_candidate": selected_index,
            "frontier": frontier,
        },
        "subspace": {
            "rank": RANK,
            "ridge": RIDGE,
            "design_regularization": DESIGN_RIDGE,
            "operator_work": operator_work,
            "basis_orthogonality_error": float(
                np.linalg.norm(basis.T @ basis - np.eye(RANK), ord=2)
            ),
            "subspace_baselines": subspace_baselines,
        },
        "calibration": {
            "selection_loss": "mean relative validation error + 100 times mean validation grid regret",
            "candidate_decision_weights": list(DECISION_WEIGHTS),
            "tie_rule": "smallest decision weight within absolute loss tolerance 1e-12",
            "by_probe_count": calibration_records,
        },
        "probe_study": count_records,
        "random_probe_distribution": random_distribution,
        "audit_probe_count": audit_count,
        "audit_strategies": audit_record["strategies"],
        "per_graph_decision_weighted": per_graph,
        "noise_study": noise,
    }
    if record_timings:
        result["runtime_benchmark"] = _runtime_benchmark(
            development_node,
            mean,
            dense_centered,
            repetitions=timing_repetitions,
        )

    return _rounded(result)


def semantic_sha256(result: dict[str, Any]) -> str:
    scientific = {key: value for key, value in result.items() if key != "runtime_benchmark"}
    payload = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_tex(result: dict[str, Any], destination: Path) -> None:
    compression_rows = []
    for record in result["compression"]["frontier"]:
        compression_rows.append(
            f"{record['candidate']} & {record['fourier_tolerance']:.0e} & "
            f"{record['hierarchical_tolerance']:.0e} & "
            f"{100 * record['storage_fraction']:.2f} & "
            f"{record['relative_error']:.2e} & "
            f"{record['maximum_grid_regret']:.2e} \\\\" 
        )
    audit = result["audit_strategies"]
    labels = {
        "morton_spaced": "Morton-spaced",
        "leverage": "largest leverage",
        "unweighted_logdet": "unweighted log-det",
        "decision_weighted": "decision-weighted log-det",
    }
    method_rows = []
    random_audit = next(
        row
        for row in result["random_probe_distribution"]
        if row["probes"] == result["audit_probe_count"]
    )
    random_error = random_audit["median_relative_error"]
    random_regret = random_audit["maximum_grid_regret"]
    random_exact = random_audit["exact_optimizer_count"]
    random_sigma = random_audit["minimum_sampled_singular_value"]
    method_rows.append(
        f"random (64 seeds) & {random_error['median']:.3e} & "
        f"[{random_error['q25']:.2e}, {random_error['q75']:.2e}] & "
        f"{random_regret['median']:.3e} & "
        f"{random_exact['median']:.1f}/8 & {random_sigma['median']:.3e} \\\\"
    )
    for name in labels:
        record = audit[name]
        method_rows.append(
            f"{labels[name]} & {record['median_relative_error']:.3e} & -- & "
            f"{record['maximum_grid_regret']:.3e} & "
            f"{record['exact_optimizer_count']}/8 & "
            f"{record['minimum_sampled_singular_value']:.3e} \\\\"
        )
    count_rows = []
    for record in result["probe_study"]:
        proposed = record["strategies"]["decision_weighted"]
        count_rows.append(
            f"{record['probes']} & {record['selected_decision_weight']:.2g} & "
            f"{proposed['median_relative_error']:.3e} & "
            f"{proposed['maximum_grid_regret']:.3e} & "
            f"{proposed['exact_optimizer_count']}/8 \\\\"
        )
    graph_rows = []
    for record in result["per_graph_decision_weighted"]:
        graph_rows.append(
            f"{record['name'].removeprefix('Gset-')} & {record['vertices']} & "
            f"{record['edges']} & {record['relative_error']:.3e} & "
            f"{record['grid_regret']:.3e} & "
            f"{'yes' if record['exact_optimizer_recovered'] else 'no'} \\\\"
        )
    proposed = audit["decision_weighted"]
    baselines = result["subspace"]["subspace_baselines"]
    runtime = result.get("runtime_benchmark")
    if runtime is None:
        raise ValueError("runtime benchmark is required when writing paper artifacts")
    rank_runtime = next(row for row in runtime["by_rank"] if row["rank"] == RANK)
    baseline_labels = {
        "hierarchical_action_iteration": "hierarchical action iteration",
        "row_gram": r"row Gram $AA^{\mathsf{T}}$",
        "thin_svd": "dense thin SVD",
    }
    subspace_rows = []
    for name, label in baseline_labels.items():
        diagnostics = baselines["methods"][name]
        timing = rank_runtime["methods"][name]
        recovery = diagnostics["audit_recovery"]
        subspace_rows.append(
            f"{label} & {diagnostics['relative_projection_residual']:.3e} & "
            f"{diagnostics['projector_distance_to_thin_svd']:.3e} & "
            f"{1000 * timing['median']:.2f} "
            f"[{1000 * timing['q25']:.2f}, {1000 * timing['q75']:.2f}] & "
            f"{recovery['median_relative_error']:.3e} & "
            f"{recovery['maximum_grid_regret']:.3e} & "
            f"{recovery['exact_optimizer_count']}/8 \\\\"
        )
    resources = baselines["resource_ledger"]
    lines = [
        "\\newcommand{\\DesignCompressionRows}{%",
        "\n".join(compression_rows),
        "}",
        "\\newcommand{\\DesignMethodRows}{%",
        "\n".join(method_rows),
        "}",
        "\\newcommand{\\DesignSubspaceRows}{%",
        "\n".join(subspace_rows),
        "}",
        "\\newcommand{\\DesignCountRows}{%",
        "\n".join(count_rows),
        "}",
        "\\newcommand{\\DesignGraphRows}{%",
        "\n".join(graph_rows),
        "}",
        f"\\newcommand{{\\DesignMedianError}}{{{proposed['median_relative_error']:.3e}}}",
        f"\\newcommand{{\\DesignMaxRegret}}{{{proposed['maximum_grid_regret']:.3e}}}",
        f"\\newcommand{{\\DesignExactCount}}{{{proposed['exact_optimizer_count']}}}",
        f"\\newcommand{{\\RandomMedianError}}{{{random_error['median']:.3e}}}",
        f"\\newcommand{{\\RandomErrorQOne}}{{{random_error['q25']:.3e}}}",
        f"\\newcommand{{\\RandomErrorQThree}}{{{random_error['q75']:.3e}}}",
        f"\\newcommand{{\\MFOperatorProducts}}{{{resources['hierarchical_action_iteration']['operator_products']}}}",
        f"\\newcommand{{\\RowGramProducts}}{{{resources['row_gram']['leading_products']}}}",
        f"\\newcommand{{\\ColumnGramProducts}}{{{resources['column_gram_A_transpose_A']['formation_products']}}}",
        f"\\newcommand{{\\MFMedianMillis}}{{{1000 * rank_runtime['methods']['hierarchical_action_iteration']['median']:.2f}}}",
        f"\\newcommand{{\\RowGramMedianMillis}}{{{1000 * rank_runtime['methods']['row_gram']['median']:.2f}}}",
        f"\\newcommand{{\\ThinSVDMedianMillis}}{{{1000 * rank_runtime['methods']['thin_svd']['median']:.2f}}}",
        f"\\newcommand{{\\DesignHash}}{{{semantic_sha256(result)}}}",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(result: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    colors = {
        "random": "#666666",
        "morton_spaced": "#9467bd",
        "leverage": "#d97706",
        "unweighted_logdet": "#2a6fbb",
        "decision_weighted": "#9b2c2c",
    }
    labels = {
        "morton_spaced": "Morton-spaced",
        "leverage": "largest leverage",
        "unweighted_logdet": "unweighted log-det",
        "decision_weighted": "decision-weighted log-det",
    }
    counts = [record["probes"] for record in result["probe_study"]]
    random = result["random_probe_distribution"]
    figure, axes = plt.subplots(1, 3, figsize=(10.3, 3.05), constrained_layout=True)
    for axis, (field, ylabel) in zip(
        axes,
        (
            ("median_relative_error", "median relative error"),
            ("maximum_grid_regret", "maximum grid regret"),
            ("minimum_sampled_singular_value", r"$\sigma_{\min}(Q_S)$"),
        ),
    ):
        random_median = np.asarray([row[field]["median"] for row in random])
        random_q25 = np.asarray([row[field]["q25"] for row in random])
        random_q75 = np.asarray([row[field]["q75"] for row in random])
        floor = 1.0e-8 if field == "minimum_sampled_singular_value" else 1.0e-12
        axis.plot(counts, np.maximum(random_median, floor), "o--", color=colors["random"], label="random median (64 seeds)")
        axis.fill_between(
            counts,
            np.maximum(random_q25, floor),
            np.maximum(random_q75, floor),
            color=colors["random"],
            alpha=0.18,
            linewidth=0,
            label="random IQR",
        )
        for name in labels:
            values = [record["strategies"][name][field] for record in result["probe_study"]]
            axis.plot(counts, np.maximum(values, floor), "o-", color=colors[name], label=labels[name])
        axis.set_xlabel("observed angle values per Gset graph")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        if field == "minimum_sampled_singular_value":
            axis.set_ylim(5.0e-9, 3.0e-1)
        axis.grid(alpha=0.25, which="both")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    figure.savefig(destination / "probe_performance.pdf", bbox_inches="tight")
    plt.close(figure)

    # Primary diagnostic for the rewritten manuscript: reconstruction and
    # decision quality are shown separately, together with finite-seed random
    # variation and the graphwise location of the remaining regret.
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    fields = (
        ("median_relative_error", "median relative error"),
        ("maximum_grid_regret", "maximum grid regret"),
    )
    for axis, (field, ylabel) in zip(axes[0], fields):
        rmed = np.asarray([row[field]["median"] for row in random])
        rq25 = np.asarray([row[field]["q25"] for row in random])
        rq75 = np.asarray([row[field]["q75"] for row in random])
        axis.plot(counts, rmed, "o--", color=colors["random"], label="random median")
        axis.fill_between(counts, rq25, rq75, color=colors["random"], alpha=0.18, label="random IQR")
        for name in ("unweighted_logdet", "decision_weighted"):
            values = [record["strategies"][name][field] for record in result["probe_study"]]
            axis.plot(counts, values, "o-", color=colors[name], label=labels[name])
        axis.axvline(result["audit_probe_count"], color="#222222", lw=0.8, ls=":")
        axis.set(xlabel="observed angle values", ylabel=ylabel, yscale="log")
        axis.grid(alpha=0.23, which="both")
    axes[0, 0].legend(frameon=False, fontsize=7)

    for name in ("unweighted_logdet", "decision_weighted"):
        exact = [record["strategies"][name]["exact_optimizer_count"] for record in result["probe_study"]]
        axes[1, 0].plot(counts, exact, "o-", color=colors[name], label=labels[name])
    axes[1, 0].plot(
        counts,
        [row["exact_optimizer_count"]["median"] for row in random],
        "o--", color=colors["random"], label="random median",
    )
    axes[1, 0].set(xlabel="observed angle values", ylabel="exact grid optima recovered (of 8)", ylim=(-0.2, 8.2))
    axes[1, 0].grid(alpha=0.23)

    graph_rows = result["per_graph_decision_weighted"]
    graph_names = [row["name"].removeprefix("Gset-") for row in graph_rows]
    graph_regret = [row["grid_regret"] for row in graph_rows]
    axes[1, 1].bar(graph_names, graph_regret, color="#9b2c2c", alpha=0.86)
    axes[1, 1].set(xlabel="authenticated Gset target", ylabel="grid regret at 16 probes")
    axes[1, 1].grid(axis="y", alpha=0.23)
    figure.savefig(destination / "decision_probe_audit.pdf")
    figure.savefig(destination / "decision_probe_audit.png", dpi=180)
    plt.close(figure)

    compression = result["compression"]["frontier"]
    storage = np.asarray([100 * record["storage_fraction"] for record in compression])
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
    for axis, field, threshold, ylabel in (
        (axes[0], "relative_error", 5.0e-3, "relative Frobenius error"),
        (axes[1], "maximum_grid_regret", 2.0e-4, "maximum development regret"),
    ):
        values = np.asarray([record[field] for record in compression])
        axis.plot(storage, np.maximum(values, 1.0e-12), "o-", color="#2a6fbb")
        axis.axhline(threshold, color="#9b2c2c", linestyle="--", linewidth=1.0, label="registered limit")
        for record, x_value, y_value in zip(compression, storage, values):
            marker = "*" if record["candidate"] == result["compression"]["selected_candidate"] else "o"
            axis.plot(x_value, max(y_value, 1.0e-12), marker=marker, color="#111111", markersize=9 if marker == "*" else 4)
            axis.annotate(str(record["candidate"]), (x_value, max(y_value, 1.0e-12)), xytext=(4, 4), textcoords="offset points")
        axis.set_xlabel("hierarchy storage (% of dense)")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    figure.savefig(destination / "compression_acceptance.pdf")
    plt.close(figure)

    runtime = result["runtime_benchmark"]
    method_labels = {
        "hierarchical_action_iteration": "hierarchical actions",
        "row_gram": r"row Gram $AA^{\mathsf{T}}$",
        "thin_svd": "thin SVD",
    }
    method_colors = {
        "hierarchical_action_iteration": "#9b2c2c",
        "row_gram": "#2a6fbb",
        "thin_svd": "#4c956c",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)
    ranks = [row["rank"] for row in runtime["by_rank"]]
    for name, label in method_labels.items():
        medians = np.asarray([row["methods"][name]["median"] for row in runtime["by_rank"]])
        q25 = np.asarray([row["methods"][name]["q25"] for row in runtime["by_rank"]])
        q75 = np.asarray([row["methods"][name]["q75"] for row in runtime["by_rank"]])
        axes[0].plot(ranks, 1000 * medians, "o-", label=label, color=method_colors[name])
        axes[0].fill_between(ranks, 1000 * q25, 1000 * q75, color=method_colors[name], alpha=0.16, linewidth=0)
    axes[0].set_xlabel("requested subspace rank")
    axes[0].set_ylabel("learning time (ms)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    diagnostics = result["subspace"]["subspace_baselines"]["methods"]
    x = np.arange(len(method_labels))
    residuals = [diagnostics[name]["relative_projection_residual"] for name in method_labels]
    distances = [diagnostics[name]["projector_distance_to_thin_svd"] for name in method_labels]
    axes[1].plot(x, np.maximum(residuals, 1.0e-16), "o-", label="rank-12 residual", color="#2a6fbb")
    axes[1].plot(x, np.maximum(distances, 1.0e-16), "s--", label="projector distance to SVD", color="#9b2c2c")
    axes[1].set_xticks(x, ["hier.\nactions", "row\nGram", "thin\nSVD"])
    axes[1].set_ylabel("relative diagnostic")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=7)
    figure.savefig(destination / "subspace_baselines.pdf")
    web_figures = destination.parent / "manuscript_assets" / "figures"
    web_figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(web_figures / "web_overview.png", dpi=180)
    plt.close(figure)

    noise = result["noise_study"]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
    x = [record["noise_standard_deviation"] for record in noise]
    x_plot = [max(value, 1.0e-7) for value in x]
    observed = np.asarray([record["maximum_observed_error"] for record in noise])
    bounds = np.asarray([record["maximum_bound"] for record in noise])
    axes[0].loglog(x_plot, observed, "o-", color="#2a6fbb", label="observed error")
    axes[0].loglog(x_plot, bounds, "s--", color="#9b2c2c", label="deterministic bound")
    axes[0].set_xlabel("observation-noise standard deviation")
    axes[0].set_ylabel("maximum landscape error")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].semilogx(x_plot, bounds / observed, "o-", color="#6a4c93")
    axes[1].axhline(1.0, color="#111111", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("observation-noise standard deviation")
    axes[1].set_ylabel("bound / observed error")
    axes[1].grid(alpha=0.25, which="both")
    figure.savefig(destination / "noise_certificate.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), constrained_layout=True)
    axes[0].loglog(x_plot, observed, "o-", color="#2a6fbb", label="observed maximum")
    axes[0].loglog(x_plot, bounds, "s--", color="#9b2c2c", label="deterministic bound")
    axes[0].set(xlabel="observation-noise standard deviation", ylabel="maximum landscape error")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    rank_rows = result["runtime_benchmark"]["by_rank"]
    for name, label, color in (
        ("hierarchical_action_iteration", "hierarchical actions", "#9b2c2c"),
        ("row_gram", r"row Gram $AA^{\mathsf{T}}$", "#2a6fbb"),
        ("thin_svd", "thin SVD", "#4c956c"),
    ):
        median = 1000 * np.asarray([row["methods"][name]["median"] for row in rank_rows])
        q25 = 1000 * np.asarray([row["methods"][name]["q25"] for row in rank_rows])
        q75 = 1000 * np.asarray([row["methods"][name]["q75"] for row in rank_rows])
        axes[1].plot([row["rank"] for row in rank_rows], median, "o-", color=color, label=label)
        axes[1].fill_between([row["rank"] for row in rank_rows], q25, q75, color=color, alpha=0.14)
    axes[1].set(xlabel="requested rank", ylabel="learning time (ms)", yscale="log")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=7)
    figure.savefig(destination / "certificate_runtime_audit.pdf")
    figure.savefig(destination / "certificate_runtime_audit.png", dpi=180)
    plt.close(figure)

    graphs = result["per_graph_decision_weighted"]
    names = [record["name"].removeprefix("Gset-") for record in graphs]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.8), constrained_layout=True)
    axes[0].plot(names, [record["relative_error"] for record in graphs], "o-", color="#2a6fbb")
    axes[0].set_xlabel("Stanford Gset graph")
    axes[0].set_ylabel("relative recovery error")
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].plot(names, [record["grid_regret"] for record in graphs], "o-", color="#9b2c2c")
    axes[1].set_xlabel("Stanford Gset graph")
    axes[1].set_ylabel("grid regret")
    axes[1].grid(alpha=0.25, axis="y")
    figure.savefig(destination / "gset_target_audit.pdf")
    plt.close(figure)


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output / "design_study.json").write_text(serialized, encoding="utf-8")
    (output / "locked_results.json").write_text(serialized, encoding="utf-8")
    (Path(__file__).resolve().parent / "locked" / "design_study.json").write_text(
        serialized, encoding="utf-8"
    )
    write_tex(result, output / "design_study.tex")
    write_figures(result, output)


def locked_design_study() -> dict[str, Any]:
    return json.loads(
        resources.files("hqml_hierarchical_gctr")
        .joinpath("locked/design_study.json")
        .read_text(encoding="utf-8")
    )


def verify_design_study() -> dict[str, Any]:
    expected = locked_design_study()
    observed = run_design_study()
    expected_hash = semantic_sha256(expected)
    observed_hash = semantic_sha256(observed)

    def compare(left: Any, right: Any, path: str = "result") -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            left_by_key = {str(key): value for key, value in left.items()}
            right_by_key = {str(key): value for key, value in right.items()}
            if left_by_key.keys() != right_by_key.keys():
                raise AssertionError(f"key mismatch at {path}")
            for key in left_by_key:
                compare(left_by_key[key], right_by_key[key], f"{path}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise AssertionError(f"length mismatch at {path}")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            if not np.isclose(float(left), float(right), rtol=2.0e-8, atol=2.0e-10):
                raise AssertionError(f"numeric mismatch at {path}: {left!r} != {right!r}")
            return
        if left != right:
            raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")

    expected_scientific = {
        key: value for key, value in expected.items() if key != "runtime_benchmark"
    }
    observed_scientific = {
        key: value for key, value in observed.items() if key != "runtime_benchmark"
    }
    compare(expected_scientific, observed_scientific)
    runtime = expected.get("runtime_benchmark")
    if not isinstance(runtime, dict):
        raise AssertionError("locked runtime benchmark is missing")
    if runtime.get("repetitions", 0) < 9:
        raise AssertionError("runtime benchmark has too few repetitions")
    for row in runtime.get("by_rank", []):
        for timing in row["methods"].values():
            if timing["minimum"] <= 0.0 or len(timing["seconds"]) != runtime["repetitions"]:
                raise AssertionError("runtime benchmark is malformed")
    return {
        "status": "verified",
        "locked_semantic_sha256": expected_hash,
        "observed_semantic_sha256": observed_hash,
        "graphs": expected["data"]["graphs"],
        "all_noise_bounds_hold": all(
            record["all_errors_bounded"] for record in expected["noise_study"]
        ),
        "row_gram_baseline_present": True,
        "random_probe_replicates": expected["random_probe_distribution"][0]["replicates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.output is not None:
            parser.error("--verify cannot be combined with --output")
        print(json.dumps(verify_design_study(), indent=2, sort_keys=True))
        return
    result = run_design_study(record_timings=args.output is not None)
    if args.output is not None:
        write_outputs(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
