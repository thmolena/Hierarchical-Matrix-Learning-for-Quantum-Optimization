# Decision-Weighted D-Optimal Probes for Depth-One QAOA Landscape Recovery

This repository contains a deterministic study of angle-probe selection for recovering depth-one MaxCut QAOA landscapes. A development-only rank-12 subspace and optimizer-sensitivity score define a monotone submodular design objective. Public evaluation uses eight checksum-authenticated Stanford Gset graphs.

At the predeclared budget of 16 observations from a 1024-angle grid, decision-weighted D-optimal selection recovers six of eight grid optimizers with maximum regret \(1.734\times10^{-3}\). Unweighted D-optimal selection recovers four with maximum regret \(1.997\times10^{-3}\). A 64-seed random comparator has median maximum regret \(3.378\times10^{-2}\). The conditioned design does not uniformly minimize surface error, and the repository retains that distinction.

## Reproduction

    python -m venv .venv
    .venv/bin/pip install -e 'code[test]'
    .venv/bin/hierarchical-qaoa-design --verify
    .venv/bin/pytest code/tests
    tectonic main.tex

The locked record includes source hashes, calibration choices, probe indices, random seeds, graphwise outcomes, noise tests, runtime repetitions, operation counts, and theorem checks. Edge weights are not used, and the results concern the fixed depth-one angle grid.
