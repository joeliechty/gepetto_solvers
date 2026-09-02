""":class:`HandPlannerSolver` -- a K+1-step trajectory.

Owns K+1 hand models tied by GP temporal priors on the wrist pose and the finger
tensions, with terminal contact constraints.
"""

import numpy as np

import gepetto_solvers

from .base import HandSolverBase
from .capabilities import _set_if
from .result import HandResult, _make_frame


class HandPlannerSolver(HandSolverBase):
    """A K+1-step grasp trajectory tied by GP temporal priors on the wrist pose and
    finger tensions, with terminal contact constraints. ``traj_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_environment()

        act = self.hand.actuation
        n = act.n
        pc = gepetto_solvers.HandTrajectoryPlannerConfig()
        pc.K = self.params.K
        pc.dt = self.params.dt
        pc.wrist_pose = self.params.wrist_pose
        pc.sigma_wrist_pos = self.params.sigma_wrist_pos
        pc.sigma_wrist_rot = self.params.sigma_wrist_rot
        pc.gp_wrist_Qc = self.params.gp_wrist * np.eye(6)
        pc.gp_actuation_Qc = self.params.gp_tense * np.eye(n)
        pc.gp_displacement_Qc = (self.params.gp_len * np.eye(n)
                                 if self.params.gp_len > 0.0 else np.zeros((0, 0)))
        pc.base.linear_solver_type = "MULTIFRONTAL_QR"
        pc.base.al_initial_mu = self.params.al_mu
        pc.base.al_mu_increase_rate = self.params.al_rate
        pc.base.al_max_iterations = self.params.al_iters
        # Inexact-AL tuning and slide-grasp scheduling exist only on newer builds.
        _set_if(pc.base, "al_inner_rel_tol_initial", self.params.al_inner_tol)
        _set_if(pc.base, "al_abs_cost_tol", self.params.al_abs_cost_tol)
        _set_if(pc.base, "record_iterations", self.params.record_iterations)
        if self.params.table and self.params.k_touch is not None:
            _set_if(pc, "k_touch", self.params.k_touch)

        planner = gepetto_solvers.HandTrajectoryPlanner(self._hand_spec(), pc)

        # Target actuation at k>=1 (tight passive / loose driven), plus the
        # measured k=0 start (open hand at start_flexor, all pinned) that the
        # trajectory closes from.
        cov = self._flexor_tension_cov()
        start_cov = np.diag([1e-6] * n)
        starts = []
        for _ in self.configs:
            sm = np.full(n, self.params.passive_tension)
            act.set_drive(sm, self.params.start_flexor)
            starts.append(gepetto_solvers.VectorXGaussian(sm, start_cov))

        result = planner.plan(self._tension_priors(cov), self._tip_wrenches(),
                              start_tensions=starts)
        frames = [_make_frame(self.finger_names, hm, result.meta)
                  for hm in result.trajectory]
        return self._result(frames, result.meta, self.params.contact_fingers,
                            list(result.trajectory))
