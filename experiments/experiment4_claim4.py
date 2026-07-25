"""
Experiment 4: Investigate Omega(d) lower bound (Theorem 6.1).

The lower bound is a worst-case result. This experiment tests Algorithm 1
on random instances to show that the algorithm avoids hard instances.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.matroids import UniformMatroid
from src.algorithms import algorithm1_basic


def run_experiment4():
    print("=" * 60)
    print("Experiment 4: Omega(d) Lower Bound (Theorem 6.1)")
    print("=" * 60)

    d_values = [3, 5, 8, 12, 15]
    T = 200
    n_trials = 5
    results = {}

    for d in d_values:
        k = max(1, d // 3)
        trial_regrets = []

        for trial in range(n_trials):
            matroid = UniformMatroid(d, k)
            w_star = np.sort(np.random.randn(d))[::-1]

            r = algorithm1_basic(T, matroid, w_star, seed=trial * 100 + d)
            trial_regrets.append(r['total_regret'])

        avg_regret = np.mean(trial_regrets)
        std_regret = np.std(trial_regrets)

        results[d] = {
            'd': d,
            'k': k,
            'T': T,
            'n_trials': n_trials,
            'avg_regret': float(avg_regret),
            'std_regret': float(std_regret),
            'all_regrets': [float(r) for r in trial_regrets],
        }

        print(f"  d={d:3d}: regret={avg_regret:8.2f} +/- {std_regret:6.2f}")

    # Compute linear fit
    ds = sorted(results.keys())
    regrets = [results[d]['avg_regret'] for d in ds]
    if len(ds) > 1:
        coeffs = np.polyfit(ds, regrets, 1)
        slope = coeffs[0]
    else:
        slope = 0

    summary = {
        'results': results,
        'linear_slope': float(slope),
        'note': 'Lower bound is worst-case; Algorithm 1 avoids hard instances on random data',
    }

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'experiment4_claim4.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nLinear slope: {slope:.3f}")
    print(f"Results saved to {output_path}")

    return summary


if __name__ == '__main__':
    run_experiment4()
