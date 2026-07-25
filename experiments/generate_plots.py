"""Generate all paper figure reproductions from experiment JSON results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (7, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
})


def plot_claim1():
    with open(os.path.join(RESULTS_DIR, 'experiment1_claim1.json')) as f:
        data = json.load(f)

    ds = sorted(int(k) for k in data.keys())
    regrets = [data[str(d)]['avg_regret'] for d in ds]
    stds = [data[str(d)]['std_regret'] for d in ds]
    d_sq = [d**2 for d in ds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: regret vs d
    ax1.errorbar(ds, regrets, yerr=stds, fmt='o-', capsize=4, label='Avg regret')
    ax1.plot(ds, [r * 0.05 for r in d_sq], 'k--', alpha=0.5, label=r'$0.05 \cdot d^2$')
    ax1.set_xlabel(r'Dimension $d$')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title(r'Claim 1: $O(d^2)$ Regret Bound')
    ax1.legend()

    # Right: regret/d^2
    ratios = [reg / sq for reg, sq in zip(regrets, d_sq)]
    ax2.bar(range(len(ds)), ratios, tick_label=[str(d) for d in ds])
    ax2.set_xlabel(r'Dimension $d$')
    ax2.set_ylabel(r'Regret $/ d^2$')
    ax2.set_title(r'Regret $/ d^2$ ratio')
    ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'claim1_regret_d2.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'claim1_regret_d2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved claim1_regret_d2.pdf")


def plot_claim2():
    with open(os.path.join(RESULTS_DIR, 'experiment2_claim2.json')) as f:
        data = json.load(f)

    ds = sorted(int(k) for k in data.keys())
    regrets = [data[str(d)]['avg_regret'] for d in ds]
    stds = [data[str(d)]['std_regret'] for d in ds]
    dlogd = [d * np.log(d) for d in ds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(ds, regrets, yerr=stds, fmt='o-', capsize=4, label='Avg regret (CoG)')
    ax1.plot(ds, [r * 0.15 for r in dlogd], 'k--', alpha=0.5, label=r'$0.15 \cdot d \log d$')
    ax1.set_xlabel(r'Dimension $d$')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title(r'Claim 2: $O(d \log d)$ Regret Bound')
    ax1.legend()

    ratios = [reg / max(ld, 1e-10) for reg, ld in zip(regrets, dlogd)]
    ax2.bar(range(len(ds)), ratios, tick_label=[str(d) for d in ds])
    ax2.set_xlabel(r'Dimension $d$')
    ax2.set_ylabel(r'Regret $/ (d \log d)$')
    ax2.set_title(r'Regret $/ d \log d$ ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'claim2_regret_dlogd.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'claim2_regret_dlogd.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved claim2_regret_dlogd.pdf")


def plot_claim3():
    with open(os.path.join(RESULTS_DIR, 'experiment3_claim3.json')) as f:
        data = json.load(f)

    Cs = sorted(int(k) for k in data.keys())
    regrets = [data[str(C)]['avg_regret'] for C in Cs]
    stds = [data[str(C)]['std_regret'] for C in Cs]
    restarts = [data[str(C)]['avg_restarts'] for C in Cs]
    bounds = [data[str(C)]['expected_bound'] for C in Cs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(Cs, regrets, yerr=stds, fmt='o-', capsize=4, label='Avg regret')
    ax1.plot(Cs, bounds, 'r--', alpha=0.7, label=r'$(C+1) d \log d$ bound')
    ax1.set_xlabel(r'Corruption budget $C$')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title(r'Claim 3: $O((C+1) d \log d)$ Corruption-Robust')
    ax1.legend()

    ax2.bar([str(C) for C in Cs], restarts, color='steelblue')
    ax2.plot([str(C) for C in Cs], Cs, 'ro--', alpha=0.7, label='True C')
    ax2.set_xlabel(r'Corruption budget $C$')
    ax2.set_ylabel('Restarts Triggered')
    ax2.set_title('Algorithm 2 Restarts (no prior C needed)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'claim3_corruption.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'claim3_corruption.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved claim3_corruption.pdf")


def plot_claim4():
    with open(os.path.join(RESULTS_DIR, 'experiment4_claim4.json')) as f:
        data = json.load(f)

    results = data['results']
    ds = sorted(int(k) for k in results.keys())
    regrets = [results[str(d)]['avg_regret'] for d in ds]
    stds = [results[str(d)]['std_regret'] for d in ds]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(ds, regrets, yerr=stds, fmt='o-', capsize=4, label='Avg regret (Algorithm 1)')
    slope = data['linear_slope']
    ax.plot(ds, [slope * d for d in ds], 'k--', alpha=0.5,
            label=f'Linear fit (slope={slope:.3f})')
    ax.set_xlabel(r'Dimension $d$')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title(r'Claim 4: Lower Bound $\Omega(d)$ vs Algorithm 1')
    ax.legend()
    ax.text(0.05, 0.95, 'Lower bound is worst-case;\nAlgorithm 1 avoids hard instances',
            transform=ax.transAxes, va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'claim4_lower_bound.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'claim4_lower_bound.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved claim4_lower_bound.pdf")


def plot_claim5():
    with open(os.path.join(RESULTS_DIR, 'experiment5_claim5.json')) as f:
        data = json.load(f)

    vol_data = data['volume_over_time']
    rounds = [v['round'] for v in vol_data]
    vols = [v['volume_fraction'] for v in vol_data]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, vols, 'b-', linewidth=1.5)
    ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label=r'$5\%$ threshold')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel(r'Volume fraction $\mathrm{Vol}(P_t) / \mathrm{Vol}([0,1]^d)$')
    ax.set_title(r'Claim 5: Volume Contraction (Lemma 4.1)')
    ax.legend()
    ax.text(0.05, 0.05, f'Guaranteed factor: $1 - 1/e \\approx {1-1/np.e:.3f}$',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'claim5_volume.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'claim5_volume.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved claim5_volume.pdf")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating all plots")
    print("=" * 60)
    plot_claim1()
    plot_claim2()
    plot_claim3()
    plot_claim4()
    plot_claim5()
    print(f"\nAll plots saved to {PLOTS_DIR}/")
