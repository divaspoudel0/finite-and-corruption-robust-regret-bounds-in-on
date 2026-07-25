"""
M-convex action sets for online inverse linear optimization.

Implements uniform matroids U(k, d) and partition matroids, both subclasses
of M-convex sets as defined in Murota (2003).
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple


class UniformMatroid:
    """Uniform matroid U(k, d): choose exactly k elements from [d].

    This is a canonical M-convex set. The characteristic vectors of bases
    form the feasible set X ⊆ {0,1}^d with sum(x) = k.
    """

    def __init__(self, d: int, k: int):
        assert 1 <= k <= d, f"k must be in [1, d], got k={k}, d={d}"
        self.d = d
        self.k = k
        self._bases: List[np.ndarray] = None
        self._arcs: set = None

    @property
    def bases(self) -> List[np.ndarray]:
        """Enumerate all basis characteristic vectors."""
        if self._bases is None:
            self._bases = []
            for cols in combinations(range(self.d), self.k):
                b = np.zeros(self.d)
                for c in cols:
                    b[c] = 1.0
                self._bases.append(b)
        return self._bases

    @property
    def num_bases(self) -> int:
        return len(self.bases)

    def optimize(self, w: np.ndarray) -> np.ndarray:
        """Return argmax_{x in bases} <w, x>."""
        return max(self.bases, key=lambda b: np.dot(w, b))

    def exchange_arcs(self, x: np.ndarray) -> set:
        """Extract M-convex exchange arcs from action x.

        For x in X, returns {(i,j) : x - e_i + e_j in X}.
        These are the arcs (i,j) where i is in the support and j is not.
        """
        x_int = np.round(x).astype(int)
        ones = [i for i in range(self.d) if x_int[i] == 1]
        zeros = [j for j in range(self.d) if x_int[j] == 0]
        return {(i, j) for i in ones for j in zeros}

    def regret(self, w: np.ndarray, x_hat: np.ndarray) -> float:
        """Compute regret: <w, x*> - <w, x_hat> where x* = argmax."""
        x_star = self.optimize(w)
        return max(0.0, float(np.dot(w, x_star) - np.dot(w, x_hat)))

    def __repr__(self):
        return f"UniformMatroid(d={self.d}, k={self.k}, |bases|={self.num_bases})"


class PartitionMatroid:
    """Partition matroid: at most c_i elements from each block i.

    A subclass of M-convex sets.
    """

    def __init__(self, block_sizes: List[int], capacities: List[int]):
        assert len(block_sizes) == len(capacities)
        self.block_sizes = block_sizes
        self.capacities = capacities
        self.d = sum(block_sizes)
        self.blocks = []
        idx = 0
        for s in block_sizes:
            self.blocks.append(list(range(idx, idx + s)))
            idx += s
        self._bases = None

    @property
    def bases(self) -> List[np.ndarray]:
        if self._bases is None:
            self._bases = []
            from itertools import product
            block_feasibles = []
            for block, cap in zip(self.blocks, self.capacities):
                feas = []
                for count in range(min(cap + 1, len(block) + 1)):
                    for combo in combinations(range(len(block)), count):
                        vec = np.zeros(len(block))
                        for c in combo:
                            vec[c] = 1.0
                        feas.append(vec)
                block_feasibles.append(feas)

            result = [np.array([])]
            for feas in block_feasibles:
                new_result = []
                for prev in result:
                    for curr in feas:
                        new_result.append(np.concatenate([prev, curr]) if prev.size else curr.copy())
                result = new_result
            self._bases = result
        return self._bases

    def optimize(self, w: np.ndarray) -> np.ndarray:
        return max(self.bases, key=lambda b: np.dot(w, b))

    def exchange_arcs(self, x: np.ndarray) -> set:
        x_int = np.round(x).astype(int)
        ones = [i for i in range(self.d) if x_int[i] == 1]
        zeros = [j for j in range(self.d) if x_int[j] == 0]
        return {(i, j) for i in ones for j in zeros}

    def __repr__(self):
        return f"PartitionMatroid(d={self.d}, blocks={len(self.blocks)})"
