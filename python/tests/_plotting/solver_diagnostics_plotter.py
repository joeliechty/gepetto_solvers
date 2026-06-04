import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class SolverDiagnosticsPlotter:
    """Accumulates per-iteration solver data and renders convergence diagnostics.

    Optional inputs collected via record():
      - solution                  : per-iteration error, trust region, step norm
      - factor_error_summary      : per-factor-type total error (bar chart)
      - factor_errors_by_type     : per-factor-type individual residuals (histograms)
      - initial_factor_summary    : per-factor-type error at the initial guess (vs. final)
      - hessian                   : dense Hessian matrix (spy plot + cond/SV annotations)
    """

    def __init__(self, title: str = "Solver Diagnostics"):
        self.title = title

        # Iteration-resolved series.
        self._errors: list[float] = []
        self._trust_region: list[float] = []
        self._step_norms: list[float] = []

        # Most-recently recorded factor data.
        self._factor_errors: list[tuple[str, int, float]] = []
        self._factor_errors_by_type: list[tuple[str, list[float]]] = []
        self._initial_factor_summary: list[tuple[str, int, float]] = []

        # Most recent Hessian snapshot.
        self._hessian: np.ndarray | None = None

    # ------------------------------------------------------------------

    def record(
        self,
        solution=None,
        factor_error_summary=None,
        factor_errors_by_type=None,
        initial_factor_summary=None,
        hessian=None,
    ):
        if solution is not None:
            meta = solution.meta
            if meta.iteration_errors:
                self._errors.extend(meta.iteration_errors)
                self._trust_region.extend(meta.iteration_trust_region)
                # Step norms have length = errors - 1 (first error is pre-iterate).
                # Pad with NaN at the start so x-axis lines up with the error curve.
                pad = [float("nan")] * (
                    len(meta.iteration_errors) - len(meta.iteration_step_norms)
                )
                self._step_norms.extend(pad + list(meta.iteration_step_norms))
            else:
                self._errors.append(meta.error)
                self._trust_region.append(float("nan"))
                self._step_norms.append(float("nan"))

        if factor_error_summary is not None:
            self._factor_errors = list(factor_error_summary)
        if factor_errors_by_type is not None:
            self._factor_errors_by_type = list(factor_errors_by_type)
        if initial_factor_summary is not None:
            self._initial_factor_summary = list(initial_factor_summary)
        if hessian is not None:
            self._hessian = np.asarray(hessian)

    # ------------------------------------------------------------------

    def build(self):
        """Build (or rebuild) the diagnostics figure. Returns the matplotlib Figure."""
        self._fig = self._build_figure()
        return self._fig

    def show(self, block: bool = True):
        if not hasattr(self, "_fig"):
            self.build()
        plt.show(block=block)
        # Pump the event loop so the window actually renders even with block=False.
        if not block:
            plt.pause(0.1)
        return self._fig

    def save(self, path: str, dpi: int = 200):
        if not hasattr(self, "_fig"):
            self.build()
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved solver diagnostics to: {path}")

    # ------------------------------------------------------------------

    def _build_figure(self):
        has_trust = any(not np.isnan(v) for v in self._trust_region)
        has_step = any(not np.isnan(v) for v in self._step_norms)
        has_summary = bool(self._factor_errors)
        has_initial = bool(self._initial_factor_summary)
        has_hist = bool(self._factor_errors_by_type)
        has_hessian = self._hessian is not None

        # Build a 2-column layout: left = iteration time series (error, trust, step),
        # right = factor diagnostics + Hessian. Both columns get stacked subplots.
        left_rows = [
            ("error", True),
            ("trust", has_trust),
            ("step", has_step),
        ]
        right_rows = [
            ("summary", has_summary),
            ("initial_vs_final", has_initial),
            ("histograms", has_hist),
            ("hessian", has_hessian),
        ]
        left_active = [k for k, on in left_rows if on]
        right_active = [k for k, on in right_rows if on]

        n_rows = max(len(left_active), len(right_active), 1)
        fig = plt.figure(figsize=(15, 3.2 * n_rows))
        fig.suptitle(self.title, fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(n_rows, 2, hspace=0.55, wspace=0.3,
                                width_ratios=[1.0, 1.2])

        iters = np.arange(len(self._errors))

        # --- Left column: iteration time series ---
        for row_idx, name in enumerate(left_active):
            ax = fig.add_subplot(gs[row_idx, 0])
            if name == "error":
                ax.semilogy(iters, self._errors, color="tab:blue", linewidth=1.5)
                ax.set_ylabel("Total error (log)")
                ax.set_title("Objective error vs. iteration")
            elif name == "trust":
                ax.semilogy(iters, self._trust_region, color="tab:orange", linewidth=1.5)
                ax.set_ylabel("Trust region (log)")
                ax.set_title("Trust region (Dogleg δ / LM λ) vs. iteration")
            elif name == "step":
                ax.semilogy(iters, self._step_norms, color="tab:red", linewidth=1.5)
                ax.set_ylabel("‖Δx‖ (log)")
                ax.set_title("Update step norm vs. iteration")
            ax.set_xlabel("Iteration")
            ax.grid(True, which="both", alpha=0.3)

        # --- Right column: factor + Hessian diagnostics ---
        for row_idx, name in enumerate(right_active):
            ax = fig.add_subplot(gs[row_idx, 1])
            if name == "summary":
                self._render_factor_summary(ax)
            elif name == "initial_vs_final":
                self._render_initial_vs_final(ax)
            elif name == "histograms":
                self._render_factor_histograms(fig, gs[row_idx, 1])
            elif name == "hessian":
                self._render_hessian(ax)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return fig

    # ------------------------------------------------------------------

    def _render_factor_summary(self, ax):
        names, counts, totals = zip(*self._factor_errors)
        short = [_shorten_type(n) for n in names]
        y = np.arange(len(short))
        bars = ax.barh(y, totals, color="tab:green", alpha=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("Total factor error")
        ax.set_title("Per-factor-type error (final)")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        for bar, count, total in zip(bars, counts, totals):
            ax.text(
                bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2,
                f"×{count}  {total:.3g}",
                va="center", ha="left", fontsize=6,
            )

    def _render_initial_vs_final(self, ax):
        # Align initial and final summaries by type name.
        final_by_name = {n: (c, t) for n, c, t in self._factor_errors}
        init_by_name = {n: (c, t) for n, c, t in self._initial_factor_summary}
        all_names = sorted(set(final_by_name) | set(init_by_name),
                            key=lambda n: -max(final_by_name.get(n, (0, 0))[1],
                                                init_by_name.get(n, (0, 0))[1]))
        short = [_shorten_type(n) for n in all_names]
        y = np.arange(len(short))
        width = 0.4
        init_totals = [init_by_name.get(n, (0, 0))[1] for n in all_names]
        final_totals = [final_by_name.get(n, (0, 0))[1] for n in all_names]
        ax.barh(y - width / 2, init_totals, height=width,
                color="tab:gray", alpha=0.7, label="initial")
        ax.barh(y + width / 2, final_totals, height=width,
                color="tab:green", alpha=0.8, label="final")
        ax.set_yticks(y)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Total factor error (symlog)")
        ax.set_title("Initial vs. final error per factor type")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    def _render_factor_histograms(self, fig, subplot_spec):
        # Replace the placeholder Axes with a nested GridSpec of small-multiples.
        # The parent ax was already created — remove it first.
        ax_placeholder = fig.axes[-1]
        fig.delaxes(ax_placeholder)

        items = self._factor_errors_by_type
        n = len(items)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        sub_gs = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=subplot_spec, hspace=0.55, wspace=0.35)

        for idx, (name, errs) in enumerate(items):
            r, c = divmod(idx, cols)
            ax = fig.add_subplot(sub_gs[r, c])
            arr = np.asarray(errs, dtype=float)
            if arr.size == 0:
                ax.set_visible(False)
                continue
            ax.hist(arr, bins=min(20, max(5, arr.size // 3)),
                    color="tab:purple", alpha=0.75)
            ax.set_title(_shorten_type(name), fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            if r == rows - 1 or idx >= n - cols:
                ax.set_xlabel("Factor error", fontsize=6)
            if c == 0:
                ax.set_ylabel("Count", fontsize=6)

    def _render_hessian(self, ax):
        H = self._hessian
        # Sparsity / structure spy plot
        ax.spy(np.abs(H) > 1e-12, markersize=0.6, color="tab:blue")
        ax.set_title("Hessian sparsity pattern (|H| > 1e-12)")
        ax.set_xlabel("Variable column index")
        ax.set_ylabel("Variable row index")

        # Compute condition number + 5 smallest singular values.
        try:
            sv = np.linalg.svd(H, compute_uv=False)
            cond = sv[0] / sv[-1] if sv[-1] > 0 else float("inf")
            smallest = sv[-5:][::-1]
            small_str = ", ".join(f"{s:.2e}" for s in smallest)
            txt = (
                f"shape: {H.shape[0]}×{H.shape[1]}    "
                f"cond: {cond:.2e}\n"
                f"5 smallest σ: {small_str}"
            )
            ax.text(
                0.02, -0.18, txt,
                transform=ax.transAxes, fontsize=7,
                family="monospace", verticalalignment="top",
            )
        except Exception as exc:
            ax.text(0.02, -0.18, f"SVD failed: {exc}",
                    transform=ax.transAxes, fontsize=7,
                    family="monospace", verticalalignment="top")


# ---------------------------------------------------------------------------

def _shorten_type(name: str) -> str:
    if "::" in name:
        name = name.rsplit("::", 1)[-1]
    if "<" in name:
        name = name[: name.index("<")]
    return name
