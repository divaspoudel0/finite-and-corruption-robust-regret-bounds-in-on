# Reproduction: Finite and Corruption-Robust Regret Bounds in Online Inverse Linear Optimization under M-Convex Action Sets

Numerical audit of **arXiv 2602.01682** (Oki & Sakaue, ICML 2026).

**Paper:** [arXiv:2602.01682](https://arxiv.org/abs/2602.01682) | **OpenReview:** [g7LE4mukGq](https://openreview.net/forum?id=g7LE4mukGq) | **Trackio Logbook:** [HuggingFace](https://huggingface.co/spaces/divaspoudel/repro-finite-and-corruption-robust-regret-bounds-in-online-inverse-linear-optimization-under-m-c)

---

## Claims Audited

| # | Claim | Theorem | Verdict |
|---|-------|---------|---------|
| 1 | O(d²) regret bound | Theorem 3.1 | **SUPPORTED** — regret/d² ratio decreases with d |
| 2 | O(d log d) regret bound | Theorem 4.2 | **SUPPORTED** — stable low ratio |
| 3 | O((C+1) d log d) corruption-robust | Theorem 5.3 | **SUPPORTED** — restarts match C exactly |
| 4 | Omega(d) lower bound | Theorem 6.1 | **SUPPORTED** theoretically (worst-case) |
| 5 | Volume contraction (1 - 1/e) | Lemma 4.1 | **SUPPORTED** — monotone volume decrease |

## Repository Structure

```
├── src/
│   ├── __init__.py
│   ├── matroids.py          # UniformMatroid, PartitionMatroid implementations
│   ├── graph_utils.py       # DAG checking, topological sort
│   └── algorithms.py        # Algorithms 1 (basic, CoG) and 2 (corruption-robust)
├── experiments/
│   ├── experiment1_claim1.py  # O(d²) regret
│   ├── experiment2_claim2.py  # O(d log d) regret
│   ├── experiment3_claim3.py  # Corruption-robust
│   ├── experiment4_claim4.py  # Lower bound
│   ├── experiment5_claim5.py  # Volume contraction
│   └── generate_plots.py      # Generate all figures
├── results/
│   ├── experiment*.json       # Raw results
│   └── plots/                 # Generated figures (PDF + PNG)
├── notebooks/
│   └── reproduce_claims.ipynb # Colab-ready notebook
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run all experiments
python3 experiments/experiment1_claim1.py
python3 experiments/experiment2_claim2.py
python3 experiments/experiment3_claim3.py
python3 experiments/experiment4_claim4.py
python3 experiments/experiment5_claim5.py

# Generate plots
python3 experiments/generate_plots.py
```

Or open the **Colab notebook** directly: `notebooks/reproduce_claims.ipynb`

## Implementation Details

- **Matroid**: Uniform matroid U(k, d) with k = floor(d/3)
- **Optimization**: Exhaustive enumeration over all bases (feasible for small d)
- **Center-of-gravity**: Monte Carlo approximation (30,000 samples)
- **Corruption model**: Adversary provides worst-case action on corrupted rounds
- **Restart detection**: Acyclicity check on the arc set ([d], A_t)

## Reference

```
@inproceedings{oki2026finite,
  title={Finite and Corruption-Robust Regret Bounds in Online Inverse Linear Optimization under M-Convex Action Sets},
  author={Oki, Taihei and Sakaue, Shinsaku},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
