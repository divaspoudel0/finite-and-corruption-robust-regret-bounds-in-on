"""
Experiment 1: Verify O(d^2) regret bound (Theorem 3.1).

Runs Algorithm 1 (basic topological sort) on uniform matroids U(k, d)
for various d and measures regret growth.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.matroids import UniformMatroid
from src.algorithms import algorithm1_basic


def run_experiment1():
    print("=" * 60)
    print("Experiment 1: O(d^2) Regret Bound (Theorem 3.1)")
    print("=" * 60)

    d_values = [3, 4, 5, 6, 8, 10, 12, 15]
    T = 500
    n_trials = 5
    results = {}

    for d in d_values:
        k = max(1, d // 3)
        trial_regrets = []
        trial_mistakes = []

        for trial in range(n_trials):
            matroid = UniformMatroid(d, k)
            w_star = np.sort(np.random.randn(d))[::-1]

            r = algorithm1_basic(T, matroid, w_star, seed=trial * 100 + d)
            trial_regrets.append(r['total_regret'])
            trial_mistakes.append(r['n_mistakes'])

        avg_regret = np.mean(trial_regrets)
        std_regret = np.std(trial_regrets)
        avg_mistakes = np.mean(trial_mistakes)

        results[d] = {
            'd': d,
            'd_squared': d ** 2,
            'k': k,
            'T': T,
            'n_trials': n_trials,
            'avg_regret': float(avg_regret),
            'std_regret': float(std_regret),
            'avg_mistakes': float(avg_mistakes),
            'regret_per_d2': float(avg_regret / (d ** 2)),
            'all_regrets': [float(r) for r in trial_regrets],
            'all_mistakes': [int(m) for m in trial_mistakes],
        }

        print(f"  d={d:3d}: regret={avg_regret:8.2f} +/- {std_regret:6.2f}, "
              f"d^2={d**2:4d}, mistakes={avg_mistakes:.0f}, "
              f"regret/d^2={avg_regret/(d**2):.3f}")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'experiment1_claim1.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    run_experiment1()
