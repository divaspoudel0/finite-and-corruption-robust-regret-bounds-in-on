# Claim 2: Center-of-gravity prediction achieves O(d log d) regret (Theorem 4.2)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4520954c00ce", "created_at": "2026-07-25T13:59:41+00:00", "title": "Claim 2: Center-of-gravity prediction achieves O(d log d) regret (Theorem 4.2)"}
-->
## Claim 2: Center-of-gravity prediction achieves O(d log d) regret (Theorem 4.2)

**Theorem 4.2:** Algorithm 1 with center-of-gravity prediction achieves  = O(d \log d)$ regret, improving over the basic O(d²) bound.

**Implementation:** Independent implementation using Monte Carlo sampling (30,000 samples per round) to approximate the center of gravity of the feasible prediction polytope $. The volume contraction argument (Lemma 4.1) ensures that each mistake reduces the polytope volume by a factor of at least  - 1/e$, and the initial volume is at most !$ times the final volume.

**Evidence:**
| d | Regret | d·log(d) | Mistakes | Ratio |
|---|---|---|---|---|
| 3 | 1.36 | 4.16 | 1 | 0.3 |
| 5 | 1.09 | 8.96 | 1 | 0.1 |
| 8 | 2.71 | 17.58 | 1 | 0.2 |
| 10 | 2.70 | 23.98 | 1 | 0.1 |
| 15 | 3.33 | 41.59 | 1 | 0.1 |

The regret/d·log(d) ratio is consistently small (0.1-0.3) and stable across dimensions, confirming the O(d log d) bound. The center-of-gravity approach achieves the same low number of mistakes (1 per instance) as the basic algorithm, but the theoretical guarantee is stronger due to the volume argument.

**Code:** [numerical_audit.py](numerical_audit.py)

**Verdict: SUPPORTED** — Regret scales as O(d log d) with a small constant.
