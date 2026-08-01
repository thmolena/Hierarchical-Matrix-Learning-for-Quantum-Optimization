# CPU reproduction package

`hqml-hierarchical-gctr` reproduces the experiments for **Auditing
Hierarchical Subspace Learning for Depth-One QAOA Landscapes**.

The implementation performs the following computation:

1. authenticate eight original Stanford Gset graph files by SHA-256;
2. evaluate exact depth-one unweighted-MaxCut causal types and graph histograms;
3. verify the causal formula against an independent statevector evaluator;
4. select the minimum-storage Fourier and hierarchical representation that
   satisfies the registered development error and regret constraints;
5. learn a centered rank-12 angle basis through hierarchical forward and
   transpose products;
6. compare that basis with direct thin SVD and the strong `A @ A.T` row-Gram
   baseline required by the `120 x 1024` matrix shape;
7. select target-independent observations by greedy maximization of a
   decision-weighted log-determinant objective;
8. compare 64 random seeds, Morton-spaced, largest-leverage, unweighted
   log-determinant, and decision-weighted designs on eight untouched Gset
   landscapes; and
9. evaluate the affine recovery bound under five noise levels and eight seeds.

## Installation and registered run

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
export HQML_GSET_CACHE=/tmp/hqml-gset
hierarchical-qaoa-design --verify
python -m unittest discover -s tests -v
```

Editable installation is supported with `python -m pip install -e .`.
Regeneration in a source checkout uses:

```bash
hierarchical-qaoa-design --output results
```

The package requires NumPy and Matplotlib. It uses no GPU, remote quantum
service, proprietary input, or dataset mirror.

## Implementation map

- `src/hqml_hierarchical_gctr/causal.py`: graph parsing, controlled graph
  families, causal histograms, exact landscapes, and statevector audit.
- `src/hqml_hierarchical_gctr/matrix_free.py`: Fourier truncation,
  hierarchical construction, centered forward/transpose products, block
  subspace iteration, and greedy log-determinant design.
- `src/hqml_hierarchical_gctr/experiment.py`: authenticated sources, corpus
  construction, compression profiles, and the original component experiment.
- `src/hqml_hierarchical_gctr/design_study.py`: registered fit/calibration/test
  split, compression constraints, probe baselines, noise study, figures, and
  locked verification.
- `src/hqml_hierarchical_gctr/locked/design_study.json`: installed numerical
  lock used by `--verify`.
- `tests/test_reproduction.py`: causal, action, centering, design, recovery,
  and provenance tests.

## Scope

The registered results concern unweighted MaxCut at QAOA depth one on a
`32 x 32` angle grid. Gset weights are not used. The reference builder uses
dense local SVDs and already materializes the development matrix. Row Gram is
faster than the recursive action learner in the locked learning-phase timings
and produces the same downstream result. The surviving claim is limited to
compressed retained storage and fewer counted action products after
construction; no generic novelty or end-to-end acceleration is claimed.

## Standard entry points

From the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
python code/scripts/reproduce.py
python code/scripts/make_figures.py
python code/scripts/validate_release.py
```

Dataset provenance is recorded under `data/`, locked numerical evidence under
`results/`, and manuscript bibliography and figures under `manuscript_assets/`.
