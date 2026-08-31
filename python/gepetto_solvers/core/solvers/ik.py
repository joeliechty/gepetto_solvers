""":class:`HandIKSolver` -- a single terminal grasp, one shot.

Each fingertip is driven onto the shared object surface by a hard contact
constraint, which routes the C++ side onto the Augmented Lagrangian path.
"""


import gepetto_solvers

from .base import HandSolverBase
from .capabilities import _set_if
from .result import HandResult, _make_frame


class HandIKSolver(HandSolverBase):
    """Single terminal grasp: each fingertip driven onto the shared object surface
    by a hard contact constraint (Augmented Lagrangian). ``ik_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_environment()

        cfg = gepetto_solvers.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.al_initial_mu = self.params.al_mu
        cfg.base.al_mu_increase_rate = self.params.al_rate
        cfg.base.al_max_iterations = self.params.al_iters
        _set_if(cfg.base, "record_iterations", self.params.record_iterations)
        if self.params.record_iterations:
            # On the AL path an interval of 0 already means "snapshot every outer
            # iteration", but the plain LM/Dogleg path (a contact-free solve)
            # stores nothing unless the interval is > 0. Setting 1 records every
            # iteration either way.
            _set_if(cfg.base, "iteration_sample_interval", 1)

        solver = gepetto_solvers.TendonHandSolver(self.configs, cfg)
        sol = solver.solve(self._tension_priors(self._flexor_tension_cov()),
                           self._tip_wrenches())
        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        iterates, iterate_states = self._collect_iterates(solver, sol)
        return self._result([frame], sol.meta, self.params.contact_fingers,
                            [sol.marginals], iterates, iterate_states)

    def _collect_iterates(self, solver, sol):
        """The solve's convergence snapshots as ``(iterates, iterate_states)``,
        or ``(None, None)`` when nothing was recorded.

        The sequence is initial guess, then one entry per recorded iteration,
        then the final solution. That last entry is what ``sol.marginals``
        already holds, so it may repeat the last AL iterate -- it is appended
        anyway so the end of the scrubber always shows exactly the state the
        rest of the result reports on."""
        if not (self.params.record_iterations
                and hasattr(solver, "get_intermediate_solutions")):
            return None, None
        states = ([solver.get_initial_solution()]
                  + list(solver.get_intermediate_solutions())
                  + [sol.marginals])
        iterates = [[_make_frame(self.finger_names, hm, sol.meta)] for hm in states]
        return iterates, [[hm] for hm in states]
