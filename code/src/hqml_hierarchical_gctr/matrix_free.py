"""Butterfly evaluation and adaptive hierarchical compression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ButterflyResult:
    approximation: np.ndarray
    supports: tuple[np.ndarray, ...]
    coefficients: tuple[np.ndarray, ...]
    retained_coefficients: int
    total_coefficients: int
    relative_error: float
    max_abs_error: float
    counted_work: int


@dataclass(frozen=True)
class MatrixFreeSubspace:
    """Right singular subspace learned without forming a Gram matrix."""

    basis: np.ndarray
    operator_work: int
    iterations: int
    block_width: int
    peak_working_scalars: int
    dense_gram_scalars_avoided: int


@dataclass(frozen=True)
class DenseSubspaceBaseline:
    """A dense right-subspace baseline with an explicit resource ledger."""

    basis: np.ndarray
    singular_values: np.ndarray
    input_scalars: int
    auxiliary_scalars: int
    leading_products: int


def thin_svd_right_subspace(matrix: np.ndarray, rank: int) -> DenseSubspaceBaseline:
    """Return the optimal rank-``rank`` right subspace of a dense matrix.

    This is the numerically preferred dense reference when the complete
    ``m x q`` input is already resident.  It is intentionally kept alongside
    the action-only routine: omitting it would make a ``q x q`` covariance
    comparator look artificially weak when ``m << q``.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two dimensional")
    if rank <= 0 or rank > min(values.shape):
        raise ValueError("rank must be positive and no larger than the matrix")
    _, singular_values, right = np.linalg.svd(values, full_matrices=False)
    rows, cols = values.shape
    return DenseSubspaceBaseline(
        basis=right[:rank].T.copy(),
        singular_values=singular_values[:rank].copy(),
        input_scalars=int(values.size),
        auxiliary_scalars=int(min(rows, cols) ** 2 + cols * rank),
        leading_products=int(rows * rows * cols),
    )


def row_gram_right_subspace(matrix: np.ndarray, rank: int) -> DenseSubspaceBaseline:
    """Recover right singular vectors through the smaller row Gram matrix.

    For ``matrix.shape == (m, q)`` with ``m <= q``, this routine forms
    ``matrix @ matrix.T`` (``m x m``), diagonalizes it, and maps the positive
    left singular vectors back through ``matrix.T``.  A final QR removes
    roundoff-level loss of orthogonality.  The routine rejects a requested
    rank beyond the numerical rank rather than dividing by tiny eigenvalues.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two dimensional")
    rows, cols = values.shape
    if rows > cols:
        raise ValueError("row-Gram baseline is intended for m <= q")
    if rank <= 0 or rank > rows:
        raise ValueError("rank must be positive and no larger than m")

    gram = values @ values.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    scale = max(float(eigenvalues[0]), np.finfo(float).tiny)
    numerical_rank = int(
        np.count_nonzero(eigenvalues > np.finfo(float).eps * max(rows, cols) * scale)
    )
    if rank > numerical_rank:
        raise ValueError(
            f"requested rank {rank} exceeds numerical row rank {numerical_rank}"
        )
    singular_values = np.sqrt(eigenvalues[:rank])
    basis = values.T @ left[:, :rank]
    basis /= singular_values[None, :]
    basis, _ = np.linalg.qr(basis, mode="reduced")
    return DenseSubspaceBaseline(
        basis=basis,
        singular_values=singular_values,
        input_scalars=int(values.size),
        auxiliary_scalars=int(rows * rows + cols * rank),
        leading_products=int(rows * rows * cols + rows * cols * rank),
    )


def relative_projection_residual(matrix: np.ndarray, basis: np.ndarray) -> float:
    """Return ``||A - AQQ^T||_F / ||A||_F`` for an orthonormal basis ``Q``."""

    values = np.asarray(matrix, dtype=float)
    vectors = np.asarray(basis, dtype=float)
    denominator = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    residual = values - (values @ vectors) @ vectors.T
    return float(np.linalg.norm(residual) / denominator)


def projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the spectral distance between two equal-dimensional projectors."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape:
        raise ValueError("bases must have the same shape")
    return float(
        np.linalg.norm(first @ first.T - second @ second.T, ord=2)
    )


@dataclass(frozen=True)
class CenteredOperator:
    """Affine centering wrapper using only forward and transpose actions."""

    base: "HNode"
    mean: np.ndarray

    @property
    def rows(self) -> int:
        return self.base.rows

    @property
    def cols(self) -> int:
        return self.base.cols

    @property
    def storage(self) -> int:
        return self.base.storage + self.cols

    def matmat(self, batch: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(batch, dtype=float)
        vector = values.ndim == 1
        output, work = self.base.matmat(values)
        if vector:
            correction = float(self.mean @ values)
            output = output - correction
            width = 1
        else:
            correction = self.mean @ values
            output = output - np.ones((self.rows, 1)) * correction
            width = values.shape[1]
        return output, int(work + (self.rows + self.cols) * width)

    def transpose_matmat(self, batch: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(batch, dtype=float)
        vector = values.ndim == 1
        output, work = self.base.transpose_matmat(values)
        if vector:
            output = output - self.mean * float(np.sum(values))
            width = 1
        else:
            output = output - self.mean[:, None] * np.sum(values, axis=0)
            width = values.shape[1]
        return output, int(work + (self.rows + self.cols) * width)


def greedy_logdet_probes(
    basis: np.ndarray,
    count: int,
    *,
    weights: np.ndarray | None = None,
    decision_weight: float = 0.0,
    regularization: float = 1.0e-8,
) -> np.ndarray:
    """Select rows by greedy regularized log-determinant gain.

    The optional nonnegative weights add a modular decision-importance term.
    The inverse information matrix is updated with Sherman--Morrison, so no
    angle-by-angle covariance matrix is formed.
    """

    values = np.asarray(basis, dtype=float)
    if values.ndim != 2:
        raise ValueError("basis must be a matrix")
    if not 0 < count <= values.shape[0]:
        raise ValueError("probe count must lie between one and the number of rows")
    if regularization <= 0.0 or decision_weight < 0.0:
        raise ValueError("regularization must be positive and decision weight nonnegative")
    if weights is None:
        importance = np.zeros(values.shape[0], dtype=float)
    else:
        importance = np.asarray(weights, dtype=float)
        if importance.shape != (values.shape[0],):
            raise ValueError("weights must have one value per basis row")
        if np.any(importance < 0.0):
            raise ValueError("weights must be nonnegative")

    inverse = np.eye(values.shape[1]) / regularization
    selected = np.zeros(values.shape[0], dtype=bool)
    order = np.empty(count, dtype=int)
    for step in range(count):
        transformed = values @ inverse
        leverage = np.einsum("ij,ij->i", transformed, values)
        gains = np.log1p(np.maximum(leverage, 0.0))
        gains += decision_weight * importance
        gains[selected] = -np.inf
        index = int(np.argmax(gains))
        order[step] = index
        selected[index] = True
        row = values[index]
        direction = inverse @ row
        inverse -= np.outer(direction, direction) / (1.0 + row @ direction)
    return order


def butterfly_truncate(
    operator: np.ndarray,
    grid_shape: tuple[int, int],
    relative_tolerance: float,
) -> ButterflyResult:
    """Apply an FFT butterfly and retain the smallest support meeting the energy bound."""

    values = np.asarray(operator, dtype=float)
    if values.shape[1] != grid_shape[0] * grid_shape[1]:
        raise ValueError("operator width and grid shape disagree")
    reshaped = values.reshape(len(values), *grid_shape)
    coefficients = np.fft.fft2(reshaped, axes=(1, 2), norm="ortho")
    sparse = np.zeros_like(coefficients)
    supports: list[np.ndarray] = []
    retained_values: list[np.ndarray] = []
    retained = 0
    for row in range(len(values)):
        flat = coefficients[row].reshape(-1)
        energy = np.abs(flat) ** 2
        total = float(energy.sum())
        if total <= 0.0:
            supports.append(np.empty(0, dtype=np.int32))
            retained_values.append(np.empty(0, dtype=complex))
            continue
        order = np.argsort(energy)[::-1]
        cumulative = np.cumsum(energy[order])
        target = (1.0 - relative_tolerance**2) * total
        count = int(np.searchsorted(cumulative, target, side="left") + 1)
        selected = order[:count]
        sparse[row].reshape(-1)[selected] = flat[selected]
        supports.append(selected.astype(np.int32, copy=True))
        retained_values.append(flat[selected].copy())
        retained += count
    approximation = np.fft.ifft2(sparse, axes=(1, 2), norm="ortho").real.reshape(values.shape)
    difference = approximation - values
    denominator = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    grid_size = grid_shape[0] * grid_shape[1]
    stages = int(np.ceil(np.log2(max(grid_shape)))) * 2
    counted_work = int(len(values) * grid_size * max(stages, 1))
    return ButterflyResult(
        approximation=approximation,
        supports=tuple(supports),
        coefficients=tuple(retained_values),
        retained_coefficients=retained,
        total_coefficients=int(values.size),
        relative_error=float(np.linalg.norm(difference) / denominator),
        max_abs_error=float(np.max(np.abs(difference), initial=0.0)),
        counted_work=counted_work,
    )


@dataclass
class HNode:
    rows: int
    cols: int
    dense: np.ndarray | None = None
    left_factor: np.ndarray | None = None
    right_factor: np.ndarray | None = None
    children: tuple[tuple[slice, slice, "HNode"], ...] = ()
    discarded_frobenius: float = 0.0

    @property
    def rank(self) -> int:
        if self.left_factor is None:
            return 0
        return int(self.left_factor.shape[1])

    @property
    def storage(self) -> int:
        if self.dense is not None:
            return int(self.dense.size)
        if self.left_factor is not None and self.right_factor is not None:
            return int(self.left_factor.size + self.right_factor.size)
        return int(sum(child.storage for _, _, child in self.children))

    @property
    def ranks(self) -> list[int]:
        if self.left_factor is not None:
            return [self.rank]
        result: list[int] = []
        for _, _, child in self.children:
            result.extend(child.ranks)
        return result

    @property
    def dense_leaf_entries(self) -> int:
        if self.dense is not None:
            return int(self.dense.size)
        return int(sum(child.dense_leaf_entries for _, _, child in self.children))

    @property
    def certificate_frobenius(self) -> float:
        if self.left_factor is not None:
            return float(self.discarded_frobenius)
        return float(
            np.sqrt(
                sum(child.certificate_frobenius**2 for _, _, child in self.children)
            )
        )

    def reconstruct(self) -> np.ndarray:
        if self.dense is not None:
            return self.dense.copy()
        if self.left_factor is not None and self.right_factor is not None:
            return self.left_factor @ self.right_factor
        output = np.zeros((self.rows, self.cols), dtype=float)
        for row_slice, col_slice, child in self.children:
            output[row_slice, col_slice] = child.reconstruct()
        return output

    def matmat(self, batch: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(batch, dtype=float)
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        if values.shape[0] != self.cols:
            raise ValueError("right-hand side has incompatible leading dimension")
        width = values.shape[1]
        if self.dense is not None:
            output = self.dense @ values
            work = self.rows * self.cols * width
        elif self.left_factor is not None and self.right_factor is not None:
            output = self.left_factor @ (self.right_factor @ values)
            work = self.rank * (self.rows + self.cols) * width
        else:
            output = np.zeros((self.rows, width), dtype=float)
            work = 0
            for row_slice, col_slice, child in self.children:
                partial, child_work = child.matmat(values[col_slice])
                output[row_slice] += partial
                work += child_work
        return (output[:, 0] if vector else output), int(work)

    def transpose_matmat(self, batch: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(batch, dtype=float)
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        if values.shape[0] != self.rows:
            raise ValueError("left-hand side has incompatible leading dimension")
        width = values.shape[1]
        if self.dense is not None:
            output = self.dense.T @ values
            work = self.rows * self.cols * width
        elif self.left_factor is not None and self.right_factor is not None:
            output = self.right_factor.T @ (self.left_factor.T @ values)
            work = self.rank * (self.rows + self.cols) * width
        else:
            output = np.zeros((self.cols, width), dtype=float)
            work = 0
            for row_slice, col_slice, child in self.children:
                partial, child_work = child.transpose_matmat(values[row_slice])
                output[col_slice] += partial
                work += child_work
        return (output[:, 0] if vector else output), int(work)


def matrix_free_right_subspace(
    operator: HNode,
    rank: int,
    *,
    oversampling: int = 4,
    iterations: int = 4,
    seed: int = 0,
) -> MatrixFreeSubspace:
    """Learn a right subspace using only ``A @ X`` and ``A.T @ Y`` actions.

    A block power iteration followed by a Rayleigh--Ritz extraction is used.
    At no point is the dense operator or its ``cols x cols`` covariance
    materialized by this routine.
    """

    if rank <= 0 or rank > min(operator.rows, operator.cols):
        raise ValueError("rank must be positive and no larger than the operator")
    if oversampling < 0 or iterations < 1:
        raise ValueError("oversampling and iterations are invalid")
    width = min(rank + oversampling, operator.rows, operator.cols)
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(operator.cols, width)), mode="reduced")
    work = 0
    left = np.empty((operator.rows, width), dtype=float)
    dual = np.empty((operator.cols, width), dtype=float)
    for _ in range(iterations):
        left, local_work = operator.matmat(basis)
        work += local_work
        left, _ = np.linalg.qr(left, mode="reduced")
        dual, local_work = operator.transpose_matmat(left)
        work += local_work
        basis, _ = np.linalg.qr(dual, mode="reduced")

    left, local_work = operator.matmat(basis)
    work += local_work
    left, _ = np.linalg.qr(left, mode="reduced")
    dual, local_work = operator.transpose_matmat(left)
    work += local_work
    _, _, right = np.linalg.svd(dual.T, full_matrices=False)
    learned = right[:rank].T
    peak = int(
        operator.cols * width
        + operator.rows * width
        + operator.cols * width
        + width * width
    )
    return MatrixFreeSubspace(
        basis=learned,
        operator_work=int(work),
        iterations=iterations,
        block_width=width,
        peak_working_scalars=peak,
        dense_gram_scalars_avoided=int(operator.cols**2),
    )


def _adaptive_rank(singular_values: np.ndarray, relative_tolerance: float) -> tuple[int, float]:
    energy = singular_values**2
    total = float(energy.sum())
    if total <= 0.0:
        return 0, 0.0
    tail = np.sqrt(np.maximum(total - np.cumsum(energy), 0.0))
    threshold = relative_tolerance * np.sqrt(total)
    accepted = np.flatnonzero(tail <= threshold)
    rank = int(accepted[0] + 1) if len(accepted) else len(singular_values)
    discarded = float(tail[rank - 1]) if rank > 0 else float(np.sqrt(total))
    return rank, discarded


def build_hmatrix(
    matrix: np.ndarray,
    *,
    relative_tolerance: float,
    max_rank: int,
    leaf_size: int,
    compression_margin: float = 0.85,
    minimum_depth: int = 2,
    _depth: int = 0,
) -> HNode:
    """Build a rectangular block hierarchy with locally adaptive SVD ranks."""

    values = np.asarray(matrix, dtype=float)
    rows, cols = values.shape
    if rows == 0 or cols == 0:
        return HNode(rows, cols, dense=values.copy())
    if _depth >= minimum_depth and min(rows, cols) > 1:
        left, singular, right = np.linalg.svd(values, full_matrices=False)
        rank, discarded = _adaptive_rank(singular, relative_tolerance)
        compressed_storage = rank * (rows + cols)
        if (
            rank > 0
            and rank <= max_rank
            and compressed_storage < compression_margin * rows * cols
        ):
            root = left[:, :rank] * singular[:rank]
            return HNode(
                rows,
                cols,
                left_factor=root,
                right_factor=right[:rank],
                discarded_frobenius=discarded,
            )
    if rows <= leaf_size and cols <= leaf_size:
        return HNode(rows, cols, dense=values.copy())

    children: list[tuple[slice, slice, HNode]] = []
    if rows > leaf_size and cols > leaf_size:
        row_mid, col_mid = rows // 2, cols // 2
        row_slices = (slice(0, row_mid), slice(row_mid, rows))
        col_slices = (slice(0, col_mid), slice(col_mid, cols))
    elif rows > leaf_size:
        row_mid = rows // 2
        row_slices = (slice(0, row_mid), slice(row_mid, rows))
        col_slices = (slice(0, cols),)
    else:
        col_mid = cols // 2
        row_slices = (slice(0, rows),)
        col_slices = (slice(0, col_mid), slice(col_mid, cols))
    for row_slice in row_slices:
        for col_slice in col_slices:
            child_values = values[row_slice, col_slice]
            child = build_hmatrix(
                child_values,
                relative_tolerance=relative_tolerance,
                max_rank=max_rank,
                leaf_size=leaf_size,
                compression_margin=compression_margin,
                minimum_depth=minimum_depth,
                _depth=_depth + 1,
            )
            children.append((row_slice, col_slice, child))
    return HNode(rows, cols, children=tuple(children))


def morton_order(rows: int, cols: int) -> np.ndarray:
    """Return a deterministic bit-interleaved order for a rectangular grid."""

    width = max(rows, cols).bit_length()

    def key(index: int) -> int:
        row, col = divmod(index, cols)
        result = 0
        for bit in range(width):
            result |= ((row >> bit) & 1) << (2 * bit)
            result |= ((col >> bit) & 1) << (2 * bit + 1)
        return result

    return np.asarray(sorted(range(rows * cols), key=key), dtype=int)
