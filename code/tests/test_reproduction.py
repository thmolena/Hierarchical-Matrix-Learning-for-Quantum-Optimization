from __future__ import annotations

import unittest

import numpy as np

from hqml_hierarchical_gctr import experiment
from hqml_hierarchical_gctr.design_study import locked_design_study
from hqml_hierarchical_gctr.matrix_free import (
    CenteredOperator,
    HNode,
    greedy_logdet_probes,
    projector_distance,
    row_gram_right_subspace,
    thin_svd_right_subspace,
)


class ReproductionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = experiment.reproduce("quick")

    def test_closed_form_matches_statevector(self) -> None:
        audit = self.result["statevector_audit"]
        self.assertEqual(audit["statevector_comparisons"], 24)
        self.assertLess(audit["statevector_max_abs_error"], 1e-11)

    def test_compression_regret_certificate(self) -> None:
        held_out = self.result["held_out"]
        self.assertLessEqual(
            held_out["max_grid_regret"],
            held_out["certificate_limit"] + 1e-12,
        )
        self.assertLessEqual(held_out["certificate_violation"], 1e-12)

    def test_structures_are_used_by_the_landscape_operator(self) -> None:
        butterfly = self.result["butterfly"]
        hierarchy = self.result["hierarchy"]
        self.assertLess(butterfly["storage_fraction"], 1.0)
        self.assertLess(hierarchy["storage_fraction"], 1.0)
        self.assertGreater(hierarchy["compressed_blocks"], 0)
        self.assertGreater(hierarchy["dense_to_hierarchical_work_ratio"], 1.0)
        self.assertLess(
            hierarchy["application_relative_error_against_butterfly_operator"],
            0.02,
        )

    def test_public_and_large_graph_contract(self) -> None:
        dataset = self.result["dataset"]
        self.assertEqual(dataset["public_graphs"], 8)
        self.assertEqual(len(dataset["gset_graphs"]), 8)
        self.assertEqual(dataset["gset_graphs"][0]["vertices"], 800)
        self.assertEqual(dataset["gset_graphs"][0]["edges"], 19176)
        self.assertGreaterEqual(dataset["maximum_vertices"], 800)

    def test_matrix_free_recovery_contract(self) -> None:
        recovery = self.result["matrix_free_recovery"]
        self.assertEqual(recovery["basis_rank"], 12)
        self.assertLess(recovery["basis_orthogonality_error"], 1e-10)
        self.assertGreater(recovery["row_gram_leading_work"], 0)
        self.assertEqual(recovery["row_gram_scalars"], recovery["training_rows"] ** 2)
        self.assertEqual(recovery["column_gram_scalars"], 256**2)
        self.assertIn("dominated", recovery["baseline_warning"])
        self.assertLess(
            recovery["probe_counts"][-1]["median_relative_error"],
            0.02,
        )

    def test_row_gram_matches_thin_svd(self) -> None:
        generator = np.random.default_rng(1204)
        matrix = generator.normal(size=(18, 64))
        row_gram = row_gram_right_subspace(matrix, 8)
        thin = thin_svd_right_subspace(matrix, 8)
        self.assertLess(projector_distance(row_gram.basis, thin.basis), 1.0e-10)
        np.testing.assert_allclose(
            row_gram.basis.T @ row_gram.basis,
            np.eye(8),
            atol=1.0e-12,
        )
        self.assertEqual(row_gram.auxiliary_scalars, 18**2 + 64 * 8)

    def test_centered_operator_matches_dense_centering(self) -> None:
        matrix = np.arange(20, dtype=float).reshape(4, 5)
        mean = matrix.mean(axis=0)
        operator = CenteredOperator(HNode(4, 5, dense=matrix), mean)
        block = np.arange(15, dtype=float).reshape(5, 3)
        forward, _ = operator.matmat(block)
        np.testing.assert_allclose(forward, (matrix - mean) @ block)
        left = np.arange(12, dtype=float).reshape(4, 3)
        transpose, _ = operator.transpose_matmat(left)
        np.testing.assert_allclose(transpose, (matrix - mean).T @ left)

    def test_logdet_probe_design_is_unique_and_well_conditioned(self) -> None:
        coordinate = np.linspace(-1.0, 1.0, 25)
        basis, _ = np.linalg.qr(
            np.column_stack((np.ones_like(coordinate), coordinate, coordinate**2))
        )
        probes = greedy_logdet_probes(basis, 6)
        self.assertEqual(len(np.unique(probes)), 6)
        self.assertGreater(np.linalg.svd(basis[probes], compute_uv=False)[-1], 0.0)

    def test_locked_design_study_contract(self) -> None:
        result = locked_design_study()
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["audit_probe_count"], 16)
        self.assertEqual(result["compression"]["selected_candidate"], 2)
        self.assertEqual(len(result["data"]["graphs"]), 8)
        self.assertFalse(result["data"]["weight_field_used"])
        self.assertTrue(all(row["all_errors_bounded"] for row in result["noise_study"]))
        self.assertTrue(result["subspace"]["subspace_baselines"]["dense_input_materialized_by_builder"])
        self.assertEqual(result["random_probe_distribution"][0]["replicates"], 64)
        audit = result["audit_strategies"]
        self.assertLess(
            audit["decision_weighted"]["median_relative_error"],
            result["random_probe_distribution"][1]["median_relative_error"]["median"],
        )


if __name__ == "__main__":
    unittest.main()
