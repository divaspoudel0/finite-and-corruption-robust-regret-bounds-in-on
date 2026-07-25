# Claim 4: Lower bound Ω(d) for M-convex instances (Theorem 6.1)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e352b510b5b3", "created_at": "2026-07-25T14:00:42+00:00", "title": "Claim 4: Lower bound Ω(d) for M-convex instances (Theorem 6.1)"}
-->
## Claim 4: Lower bound Ω(d) for M-convex instances (Theorem 6.1)

**Theorem 6.1:** There exists an instance of online inverse linear optimization over M-convex sets such that any randomized algorithm incurs regret $\mathbb{E}[R_T] = \Omega(d)$.

**Note on verification:** This is a worst-case lower bound over ALL possible algorithms. Our numerical test uses Algorithm 1 on random instances, which is NOT the hardest instance for this algorithm. The lower bound proof constructs specific adversarial instances (axis-aligned segments embedded as M-convex sets) designed to be hard for any algorithm. Empirical testing cannot fully verify a worst-case lower bound.

**Evidence (regret of Algorithm 1 on random instances):**
| d | Regret |
|---|---|
| 3 | 1.72 |
| 5 | 1.10 |
| 8 | 2.05 |
| 12 | 2.54 |
| 15 | 0.90 |

The regret values are small and do not show clear linear growth, which is expected: Algorithm 1 is a good algorithm that avoids the hard instances constructed in the lower bound proof. The Ω(d) lower bound means that NO algorithm can do better than linear regret on some instances, not that every algorithm achieves linear regret on every instance.

**Theoretical justification:** The proof reduces to the lower bound of Sakaue et al. (2025b) by embedding M♮-convex sets into M-convex sets via the standard coordinate-sum transformation. The construction uses axis-aligned line segments (a special case of M-convex sets) where each round reveals one coordinate of the hidden objective vector, requiring Ω(d) rounds to learn all d coordinates.

**Code:** [numerical_audit.py](numerical_audit.py) (algorithm test), paper proof for lower bound construction

**Verdict: SUPPORTED (theoretically)** — The Ω(d) lower bound is a rigorous mathematical proof that cannot be falsified by empirical testing on random instances.
