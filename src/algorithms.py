"""
Algorithms for online inverse linear optimization over M-convex sets.

Implements:
- Algorithm 1 (basic): O(d^2) regret via topological sort prediction
- Algorithm 1 (CoG): O(d log d) regret via center-of-gravity prediction
- Algorithm 2: O((C+1) d log d) corruption-robust with restart mechanism

Reference: arXiv:2602.01682 (Oki & Sakaue, ICML 2026)
"""

import numpy as np
from typing import Set, Tuple, Optional
from .matroids import UniformMatroid
from .graph_utils import is_acyclic, topological_sort


def compute_center_of_gravity(
    d: int,
    A: Set[Tuple[int, int]],
    rng: np.random.RandomState,
    n_samples: int = 30000,
) -> np.ndarray:
    """Approximate center of gravity of P_t = {w in [0,1]^d : w(i) >= w(j) for all (i,j) in A}.

    Uses Monte Carlo sampling to approximate the centroid.
    """
    if len(A) == 0:
        return np.ones(d) / 2.0

    samples = rng.rand(n_samples, d)
    mask = np.ones(n_samples, dtype=bool)
    for i, j in A:
        mask &= samples[:, i] >= samples[:, j]

    if mask.any():
        return samples[mask].mean(axis=0)
    else:
        # Fallback: topological-sort-based point
        order = topological_sort(list(A), d)
        if not order:
            return np.ones(d) / 2.0
        q = np.zeros(d)
        for rank, node in enumerate(order):
            q[node] = 1.0 - (rank + 1) / (d + 1)
        return q


def algorithm1_basic(
    T: int,
    matroid: UniformMatroid,
    w_star: np.ndarray,
    seed: int = 42,
) -> dict:
    """Algorithm 1 (basic): O(d^2) regret via topological sort.

    At each round:
    1. Choose w_hat with distinct components satisfying arc constraints
    2. Play x_hat = argmax_{x in X_t} <w_hat, x>
    3. Observe agent's action x_star
    4. Update arc set A

    Returns dict with:
    - total_regret: cumulative regret
    - regrets: per-round regret list
    - n_mistakes: number of rounds where x_hat != x_star
    - arcs_added_over_time: arc set size over time
    """
    d = matroid.d
    rng = np.random.RandomState(seed)
    A = set()
    total_regret = 0.0
    regrets = []
    n_mistakes = 0
    arcs_over_time = []

    for t in range(T):
        # Step 3: Build w_hat satisfying arc constraints
        if A:
            order = topological_sort(list(A), d)
            if order:
                w_hat = np.zeros(d)
                for rank, node in enumerate(order):
                    w_hat[node] = d - rank
            else:
                w_hat = rng.randn(d)
        else:
            w_hat = rng.randn(d)
            # Sort descending so matroid picks first k indices
            idx = np.argsort(-w_hat)
            w_hat = np.zeros(d)
            for rank, i in enumerate(idx):
                w_hat[i] = d - rank

        w_hat += rng.randn(d) * 1e-10  # Break ties

        # Step 4: Compute learner's action
        x_hat = matroid.optimize(w_hat)

        # Step 5: Observe agent's action
        x_star = matroid.optimize(w_star)

        # Compute regret
        regret = max(0.0, float(np.dot(w_star, x_star) - np.dot(w_star, x_hat)))
        total_regret += regret
        regrets.append(regret)
        if regret > 1e-10:
            n_mistakes += 1

        # Step 6: Update arc set
        A |= matroid.exchange_arcs(x_star)
        arcs_over_time.append(len(A))

    return {
        'total_regret': total_regret,
        'regrets': regrets,
        'n_mistakes': n_mistakes,
        'arcs_over_time': arcs_over_time,
        'final_arcs': len(A),
    }


def algorithm1_cog(
    T: int,
    matroid: UniformMatroid,
    w_star: np.ndarray,
    seed: int = 42,
    n_samples: int = 30000,
) -> dict:
    """Algorithm 1 (center-of-gravity): O(d log d) regret.

    Uses the center of gravity of P_t for prediction, enabling the volume
    contraction argument that gives O(d log d) regret.

    Returns dict with:
    - total_regret: cumulative regret
    - regrets: per-round regret list
    - n_mistakes: number of rounds where x_hat != x_star
    - arcs_added_over_time: arc set size over time
    """
    d = matroid.d
    rng = np.random.RandomState(seed)
    A = set()
    total_regret = 0.0
    regrets = []
    n_mistakes = 0
    arcs_over_time = []

    for t in range(T):
        # Step 3: Approximate center of gravity
        w_hat = compute_center_of_gravity(d, A, rng, n_samples)
        w_hat += rng.randn(d) * 1e-10

        # Step 4: Compute learner's action
        x_hat = matroid.optimize(w_hat)

        # Step 5: Observe agent's action
        x_star = matroid.optimize(w_star)

        # Compute regret
        regret = max(0.0, float(np.dot(w_star, x_star) - np.dot(w_star, x_hat)))
        total_regret += regret
        regrets.append(regret)
        if regret > 1e-10:
            n_mistakes += 1

        # Step 6: Update arc set
        A |= matroid.exchange_arcs(x_star)
        arcs_over_time.append(len(A))

    return {
        'total_regret': total_regret,
        'regrets': regrets,
        'n_mistakes': n_mistakes,
        'arcs_over_time': arcs_over_time,
        'final_arcs': len(A),
    }


def algorithm2_corrupt(
    T: int,
    matroid: UniformMatroid,
    w_star: np.ndarray,
    corrupted_rounds: Set[int],
    seed: int = 42,
    n_samples: int = 30000,
) -> dict:
    """Algorithm 2: O((C+1) d log d) corruption-robust with restart.

    Monitors acyclicity of ([d], A_t). When a cycle is detected (indicating
    corruption), restarts by clearing A. No prior knowledge of C required.

    Returns dict with:
    - total_regret: cumulative regret
    - regrets: per-round regret list
    - restarts: number of restarts triggered
    - cycle_detection_rounds: rounds where cycles were detected
    """
    d = matroid.d
    rng = np.random.RandomState(seed)
    A = set()
    total_regret = 0.0
    regrets = []
    restarts = 0
    cycle_detection_rounds = []

    for t in range(T):
        # Step 3: Approximate center of gravity
        w_hat = compute_center_of_gravity(d, A, rng, n_samples)
        w_hat += rng.randn(d) * 1e-10

        # Step 4: Compute learner's action
        x_hat = matroid.optimize(w_hat)

        # Step 5: Observe agent's action (possibly corrupted)
        if t in corrupted_rounds:
            # Provide worst possible action
            x_star = min(matroid.bases, key=lambda b: np.dot(w_star, b))
        else:
            x_star = matroid.optimize(w_star)

        # Compute regret (always relative to true optimal)
        x_true = matroid.optimize(w_star)
        regret = max(0.0, float(np.dot(w_star, x_true) - np.dot(w_star, x_hat)))
        total_regret += regret
        regrets.append(regret)

        # Step 6: Update arc set
        new_arcs = matroid.exchange_arcs(x_star)
        A_new = A | new_arcs

        # Step 7: Restart if cycle detected
        if not is_acyclic(A_new, d):
            A = set()
            restarts += 1
            cycle_detection_rounds.append(t)
        else:
            A = A_new

    return {
        'total_regret': total_regret,
        'regrets': regrets,
        'restarts': restarts,
        'cycle_detection_rounds': cycle_detection_rounds,
    }
