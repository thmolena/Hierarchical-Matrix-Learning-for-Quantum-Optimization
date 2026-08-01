"""Exact depth-one MaxCut causal types and deterministic graph benchmarks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

Edge = tuple[int, int]
CausalType = tuple[int, int, int]


@dataclass(frozen=True)
class GraphRecord:
    name: str
    family: str
    n: int
    edges: tuple[Edge, ...]
    replicate: int
    public: bool = False


def canonical_edges(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    cleaned = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edges
        if int(u) != int(v)
    }
    return tuple(sorted(cleaned))


def parse_gset(path: Path, name: str | None = None) -> GraphRecord:
    with path.open(encoding="utf-8") as handle:
        n, declared = map(int, handle.readline().split()[:2])
        edges = []
        for line in handle:
            u, v, _ = line.split()
            edges.append((int(u) - 1, int(v) - 1))
    canonical = canonical_edges(edges)
    if len(edges) != declared or len(canonical) != declared:
        raise ValueError("Gset edge count does not match its header")
    label = name or path.name
    return GraphRecord(f"Gset-{label}", "public-Gset", n, canonical, 0, True)


def ring_lattice(n: int, half_width: int) -> tuple[Edge, ...]:
    return canonical_edges(
        (u, (u + offset) % n)
        for u in range(n)
        for offset in range(1, half_width + 1)
    )


def small_world(n: int, seed: int, half_width: int = 3) -> tuple[Edge, ...]:
    """A connected rewired ring with a deterministic pseudorandom seed."""

    rng = np.random.default_rng(seed)
    edges = set(ring_lattice(n, half_width))
    forward = sorted(edges)
    for u, v in forward:
        if rng.random() >= 0.18:
            continue
        edges.discard((u, v))
        for _ in range(32):
            w = int(rng.integers(0, n))
            candidate = (min(u, w), max(u, w))
            if w != u and candidate not in edges:
                edges.add(candidate)
                break
        else:
            edges.add((u, v))
    return tuple(sorted(edges))


def sparse_random(n: int, seed: int, average_degree: int = 8) -> tuple[Edge, ...]:
    """A connected sparse graph sampled without an O(n^2) pair scan."""

    rng = np.random.default_rng(seed)
    edges = set(ring_lattice(n, 1))
    target = average_degree * n // 2
    while len(edges) < target:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n - 1))
        if v >= u:
            v += 1
        edges.add((min(u, v), max(u, v)))
    return tuple(sorted(edges))


def preferential_attachment(n: int, seed: int, width: int = 3) -> tuple[Edge, ...]:
    """A compact Barabasi--Albert-style generator without external packages."""

    if n <= width:
        return canonical_edges((u, v) for u in range(n) for v in range(u + 1, n))
    rng = np.random.default_rng(seed)
    initial = width + 1
    edges = {(u, v) for u in range(initial) for v in range(u + 1, initial)}
    degree_pool = [vertex for edge in edges for vertex in edge]
    for new_vertex in range(initial, n):
        targets: set[int] = set()
        while len(targets) < width:
            targets.add(int(degree_pool[int(rng.integers(0, len(degree_pool)))]))
        for target in sorted(targets):
            edges.add((target, new_vertex))
            degree_pool.extend((target, new_vertex))
    return tuple(sorted(edges))


def two_block_graph(n: int, seed: int) -> tuple[Edge, ...]:
    """Two sparse communities joined by a controlled number of bridges."""

    rng = np.random.default_rng(seed)
    split = n // 2
    edges = set()
    for start, stop in ((0, split), (split, n)):
        width = stop - start
        for u in range(start, stop):
            edges.add((min(u, start + (u - start + 1) % width), max(u, start + (u - start + 1) % width)))
        target = 3 * width
        while sum(start <= u < stop and start <= v < stop for u, v in edges) < target:
            u = int(rng.integers(start, stop))
            v = int(rng.integers(start, stop - 1))
            if v >= u:
                v += 1
            edges.add((min(u, v), max(u, v)))
    for _ in range(max(2, n // 128)):
        u = int(rng.integers(0, split))
        v = int(rng.integers(split, n))
        edges.add((u, v))
    return tuple(sorted(edges))


def benchmark_graphs(
    sizes: tuple[int, ...],
    replicates: int,
    *,
    seed: int,
) -> Iterator[GraphRecord]:
    generators = (
        ("ring", lambda n, s: ring_lattice(n, 2 + s % 3)),
        ("small-world", small_world),
        ("sparse-random", sparse_random),
        ("preferential", preferential_attachment),
        ("two-block", two_block_graph),
    )
    for size in sizes:
        for replicate in range(replicates):
            for family_index, (family, generator) in enumerate(generators):
                local_seed = seed + 100_003 * family_index + 997 * replicate + size
                yield GraphRecord(
                    name=f"{family}-n{size}-r{replicate}",
                    family=family,
                    n=size,
                    edges=generator(size, local_seed),
                    replicate=replicate,
                )


def adjacency(n: int, edges: Iterable[Edge]) -> list[set[int]]:
    result = [set() for _ in range(n)]
    for u, v in edges:
        result[u].add(v)
        result[v].add(u)
    return result


def causal_histogram(record: GraphRecord) -> tuple[Counter[CausalType], dict[str, float]]:
    adj = adjacency(record.n, record.edges)
    degrees = np.asarray([len(neighbors) for neighbors in adj], dtype=float)
    counts: Counter[CausalType] = Counter()
    triangle_incidence = 0
    for u, v in record.edges:
        common = len(adj[u].intersection(adj[v]))
        degree_u, degree_v = len(adj[u]), len(adj[v])
        counts[(min(degree_u, degree_v), max(degree_u, degree_v), common)] += 1
        triangle_incidence += common
    edge_count = len(record.edges)
    features = {
        "vertices": float(record.n),
        "edges": float(edge_count),
        "mean_degree": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "max_degree": float(degrees.max(initial=0.0)),
        "triangles": float(triangle_incidence / 3.0),
        "causal_types": float(len(counts)),
    }
    return counts, features


def edge_landscape(
    causal_type: CausalType,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Exact p=1 expected cut value of one unweighted edge."""

    degree_u, degree_v, triangles = causal_type
    cosine = np.cos(gamma)
    first = 0.25 * np.sin(4.0 * beta) * np.sin(gamma) * (
        cosine ** (degree_u - 1) + cosine ** (degree_v - 1)
    )
    second = (
        0.25
        * np.sin(2.0 * beta) ** 2
        * cosine ** (degree_u + degree_v - 2 * triangles - 2)
        * (1.0 - np.cos(2.0 * gamma) ** triangles)
    )
    return 0.5 + first - second


def type_operator(
    causal_types: list[CausalType],
    gamma_points: int,
    beta_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma = np.linspace(0.0, math.pi, gamma_points, endpoint=False)
    beta = np.linspace(0.0, math.pi / 2.0, beta_points, endpoint=False)
    gamma_mesh, beta_mesh = np.meshgrid(gamma, beta, indexing="ij")
    operator = np.vstack(
        [
            edge_landscape(causal_type, gamma_mesh, beta_mesh).reshape(-1)
            for causal_type in causal_types
        ]
    )
    return operator, gamma, beta


def statevector_energy(edges: tuple[Edge, ...], gamma: float, beta: float, n: int) -> float:
    states = np.arange(1 << n, dtype=np.uint32)
    cost = np.zeros(len(states), dtype=float)
    for u, v in edges:
        cost += ((states >> u) ^ (states >> v)) & 1
    psi = np.exp(-1j * gamma * cost) / math.sqrt(len(states))
    cosine, sine = math.cos(beta), -1j * math.sin(beta)
    for qubit in range(n):
        block = 1 << qubit
        for start in range(0, len(states), 2 * block):
            left = psi[start : start + block].copy()
            right = psi[start + block : start + 2 * block].copy()
            psi[start : start + block] = cosine * left + sine * right
            psi[start + block : start + 2 * block] = sine * left + cosine * right
    return float(np.dot(np.abs(psi) ** 2, cost))


def formula_energy(edges: tuple[Edge, ...], gamma: float, beta: float, n: int) -> float:
    record = GraphRecord("audit", "audit", n, edges, 0)
    histogram, _ = causal_histogram(record)
    return float(
        sum(
            multiplicity
            * edge_landscape(
                causal_type,
                np.asarray(gamma),
                np.asarray(beta),
            )
            for causal_type, multiplicity in histogram.items()
        )
    )


def statevector_audit() -> dict[str, float | int]:
    graphs = (
        canonical_edges([(0, 1)]),
        canonical_edges([(0, 1), (1, 2)]),
        canonical_edges([(0, 1), (1, 2), (2, 0)]),
        canonical_edges([(0, 1), (1, 2), (2, 3), (3, 0)]),
        canonical_edges([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
        canonical_edges([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]),
    )
    angles = ((0.17, 0.11), (0.63, 0.29), (1.21, 0.51), (2.34, 0.73))
    maximum = 0.0
    comparisons = 0
    for edges in graphs:
        n = max(max(edge) for edge in edges) + 1
        for gamma, beta in angles:
            exact = statevector_energy(edges, gamma, beta, n)
            formula = formula_energy(edges, gamma, beta, n)
            maximum = max(maximum, abs(exact - formula))
            comparisons += 1
    return {"statevector_comparisons": comparisons, "statevector_max_abs_error": maximum}
