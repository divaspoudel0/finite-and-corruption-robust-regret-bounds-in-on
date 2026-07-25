"""
Experiment 3: Verify O((C+1) d log d) corruption-robust bound (Theorem 5.3).

Runs Algorithm 2 with restart mechanism on uniform matroids with corrupted rounds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.matroids import UniformMatroid
from src.algorithms import algorithm2_corrupt


def run_experiment3():
    print("=" * 60)
    print("Experiment 3: O((C+1)d log d) Corruption-Robust (Theorem 5.3)")
    print("=" * 60)

    d = 8
    k = max(1, d // 3)
    T = 500
    C_values = [0, 2, 5, 10, 15, 20]
    n_trials = 5
    n_samples = 30000
    results = {}

    for C in C_values:
        trial_regrets = []
        trial_restarts = []

        for trial in range(n_trials):
            matroid = UniformMatroid(d, k)
            w_star = np.sort(np.random.randn(d))[::-1]

            # Generate C corrupted rounds
            rng = np.random.RandomState(trial * 100 + C)
            if C > 0:
                corrupted = set(rng.choice(T, size=min(C, T), replace=False))
            else:
                corrupted = set()

            r = algorithm2_corrupt(T, matroid, w_star, corrupted,
                                   seed=trial * 100 + C, n_samples=n_samples)
            trial_regrets.append(r['total_regret'])
            trial_restarts.append(r['restarts'])

        avg_regret = np.mean(trial_regrets)
        std_regret = np.std(trial_regrets)
        avg_restarts = np.mean(trial_restarts)
        expected_bound = (C + 1) * d * np.log(d + 1)

        results[C] = {
            'C': C,
            'd': d,
            'k': k,
            'T': T,
            'n_trials': n_trials,
            'n_samples': n_samples,
            'avg_regret': float(avg_regret),
            'std_regret': float(std_regret),
            'avg_restarts': float(avg_restarts),
            'expected_bound': float(expected_bound),
            'regret_vs_bound': float(avg_regret / max(expected_bound, 1e-10)),
            'all_regrets': [float(r) for r in trial_regrets],
            'all_restarts': [int(r) for r in trial_restarts],
        }

        print(f"  C={C:3d}: regret={avg_regret:8.2f} +/- {std_regret:6.2f}, "
              f"restarts={avg_restarts:.0f}, bound={expected_bound:.1f}")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'experiment3_claim3.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    run_experiment3()
