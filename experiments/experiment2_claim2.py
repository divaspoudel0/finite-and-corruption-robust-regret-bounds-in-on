"""
Experiment 2: Verify O(d log d) regret bound (Theorem 4.2).

Runs Algorithm 1 with center-of-gravity prediction on uniform matroids.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.matroids import UniformMatroid
from src.algorithms import algorithm1_cog


def run_experiment2():
    print("=" * 60)
    print("Experiment 2: O(d log d) Regret Bound (Theorem 4.2)")
    print("=" * 60)

    d_values = [3, 4, 5, 6, 8, 10, 12, 15]
    T = 500
    n_trials = 5
    n_samples = 30000
    results = {}

    for d in d_values:
        k = max(1, d // 3)
        trial_regrets = []
        trial_mistakes = []

        for trial in range(n_trials):
            matroid = UniformMatroid(d, k)
            w_star = np.sort(np.random.randn(d))[::-1]

            r = algorithm1_cog(T, matroid, w_star, seed=trial * 100 + d, n_samples=n_samples)
            trial_regrets.append(r['total_regret'])
            trial_mistakes.append(r['n_mistakes'])

        avg_regret = np.mean(trial_regrets)
        std_regret = np.std(trial_regrets)
        avg_mistakes = np.mean(trial_mistakes)
        dlogd = d * np.log(d + 1)

        results[d] = {
            'd': d,
            'd_log_d': float(dlogd),
            'k': k,
            'T': T,
            'n_trials': n_trials,
            'n_samples': n_samples,
            'avg_regret': float(avg_regret),
            'std_regret': float(std_regret),
            'avg_mistakes': float(avg_mistakes),
            'regret_per_dlogd': float(avg_regret / max(dlogd, 1e-10)),
            'all_regrets': [float(r) for r in trial_regrets],
            'all_mistakes': [int(m) for m in trial_mistakes],
        }

        print(f"  d={d:3d}: regret={avg_regret:8.2f} +/- {std_regret:6.2f}, "
              f"d*log(d)={dlogd:7.2f}, mistakes={avg_mistakes:.0f}, "
              f"ratio={avg_regret/max(dlogd,1e-10):.2f}")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'experiment2_claim2.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    run_experiment2()
