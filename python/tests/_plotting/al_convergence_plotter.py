"""Convergence curves for the Augmented Lagrangian (AL) solver.

The tendon-hand grasp trajectory solve runs on the AL path (terminal contact =
hard constraint). Each AL *outer iteration* produces a full trajectory plus a
scalar (cost, constraint-violation, penalty mu). This plots those three series
vs. outer iteration so you can see how the solver converges and where it spends
its iterations -- the AL analogue of the finger's SolverDiagnosticsPlotter.

Populated from ``result.meta.al_iteration_costs / _violations / _mus`` (only set
when ``config.base.record_iterations = True``).
"""

import numpy as np
import matplotlib.pyplot as plt


def _semilogy(ax, y, color, label, ylabel):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    # semilogy needs strictly positive values; clamp non-positive (e.g. cost/
    # violation that hit exactly 0 at convergence) to a small floor for display.
    pos = y[y > 0]
    floor = (pos.min() * 1e-3) if pos.size else 1e-12
    ax.semilogy(x, np.clip(y, floor, None), "o-", color=color, label=label, ms=4)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)


def plot_al_convergence(costs, violations, mus, title="AL Convergence",
                        save_path=None, show=False):
    """Render cost / constraint-violation / penalty-mu vs. AL outer iteration.

    Returns the matplotlib Figure. Saves to save_path when given.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(title)

    _semilogy(axes[0], costs, "tab:blue", "cost", "objective cost")
    _semilogy(axes[1], violations, "tab:red", "violation", "constraint violation")
    _semilogy(axes[2], mus, "tab:green", "mu", "penalty $\\mu$")

    axes[-1].set_xlabel("AL outer iteration")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig
