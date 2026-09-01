""":class:`HandIKStepper` -- one Augmented Lagrangian OUTER iteration per call.

The same problem :class:`HandIKSolver` poses, advanced one step at a time so a
GUI can scrub it. Carries mu and the multipliers between calls, which is what
makes a stepped solve reproduce the one-shot one rather than restarting.
"""

from typing import NamedTuple

import numpy as np

import gepetto_solvers

from .base import HandSolverBase
from .capabilities import _set_if, capabilities
from .params import HandSolveParams
from .result import HandResult, _make_frame

# The tight-passive/loose-flexor prior itself now lives on
# HandSolverBase._flexor_tension_cov() (params.flexor_tension_sigma,
# params.passive_tension_sigma), shared
# by the one-shot IK solve, the stepper and the planner so the three cannot
# drift into solving subtly different problems -- the whole point of the
# stepper is that it advances *this* solve.

# The same prior with the FLEXOR pinned as hard as the passives, used only to
# settle a cold start (HandIKStepper.step, params.ik_settle_steps).
#
# The C++ initial values seed every tension at ZERO on a rod already at its rest
# shape (TendonRobotModel::get_initial_values) -- a guess that looks right but is
# statically inconsistent. Against a commanded 0.5 N passive at sigma 1e-3 that
# is 0.5 * (0.5/1e-3)^2 * 25 = 3.125e6 units of prior cost, which is exactly the
# cost the AL trace reports at its iteration 0. The first inner LM solve spends
# its whole budget hauling the 25 passive tensions home, and the flexor -- whose
# prior is 1e5x weaker in weight -- is the cheap direction that absorbs the
# leftover inconsistency: it swings to about -0.9 N. Negative tension is
# HYPEREXTENSION, so the hand visibly bends backwards, and since the inner LM is
# capped at 100 iterations per outer step it then takes ~13 steps to crawl back
# to the FK pose the GUI was already showing. Measured: it is one continuous LM
# descent chopped into 100-iteration slices, not the constraints pulling (a 400-
# iteration budget reaches step 4's state in one step, and the excursion is
# identical for al_mu from 1e-6 to 10, and for contact-only, collision-only or
# both).
#
# Pinning the flexor for the settling step removes the soft direction, so the
# transient is absorbed by the passives alone: step 1 lands exactly on the FK
# pose and the solve reaches the same converged grasp in the same number of steps
# and less wall time.
_IK_SETTLE_TENSION_COV = np.diag([1e-6] * 6)


class StepStatus(NamedTuple):
    """Where a stepped IK solve stands after the last outer iteration."""
    state: str        # "running" | "converged" | "stalled"
    violation: float  # worst constraint violation
    cost: float       # objective (constraint penalty excluded)
    mu: float         # current AL penalty weight
    steps: int        # outer iterations taken so far

    @property
    def done(self):
        return self.state != "running"


class HandIKStepper(HandSolverBase):
    """The IK solve of :class:`HandIKSolver`, advanced one Augmented Lagrangian
    outer iteration per :meth:`step` call instead of run to convergence in one go.

    Every step solves the *identical* graph the one-shot solve builds -- same
    contact constraints, same priors (``_flexor_tension_cov()``), same tolerances. The
    only difference is that the outer loop is told to stop after one iteration
    and resume on the next call: ``al_max_iterations = 1`` with
    ``al_warm_start_duals`` carries mu, the Lagrange multipliers and the values
    across calls, so N steps are the N iterations the one-shot solve would have
    run internally. Holding the loop counter is the whole trick.

    The one deliberate departure is the leading ``params.ik_settle_steps`` steps,
    which pin the flexor prior to settle the cold start before releasing it (see
    ``_IK_SETTLE_TENSION_COV``). Without it the first steps are spent watching the
    hand hyperextend and crawl back, which is a solver transient rather than
    anything the solve is being asked to do.

    Like :class:`HandFKSolver` this owns its ``gepetto_solvers.HandSolver`` for
    its lifetime -- that is what lets anything carry at all, since a solver
    rebuilt per call cold-starts even its values. Tensions and the wrist pose are
    passed per step, so they stay live between steps; anything that changes the
    CONSTRAINT SET (object, contact mask, collision, table) needs :meth:`reset`,
    because the carried duals describe the old constraints.
    """

    def __init__(self, params: HandSolveParams | None = None, hand=None):
        super().__init__(params, hand)
        self._build()

    # -- construction / restart --

    def _build(self):
        self._attach_environment()

        cfg = gepetto_solvers.HandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.al_initial_mu = self.params.al_mu
        cfg.base.al_mu_increase_rate = self.params.al_rate
        # One outer iteration per solve() call, resumed from the last one.
        cfg.base.al_max_iterations = 1
        _set_if(cfg.base, "al_warm_start_duals", True)
        # Effectively uncapped mu, and it has to be. al_warm_mu_max exists for
        # the Section 1.8 controller, where mu compounds forever across ticks of
        # a MOVING problem and would eventually swamp the rod physics -- but the
        # one-shot solve this stepper reproduces runs optimize() with no clamp at
        # all, so its mu reaches ~2^28 by the time the contact closes. Leaving
        # the 1e4 default in place stalls the stepped solve at a ~60 mm gap while
        # the one-shot reaches 0.1 mm: the cap, not the method, is the difference.
        _set_if(cfg.base, "al_warm_mu_max", 1e12)
        # ...but a mu carried in from ANOTHER solver is clamped: see
        # HandSolveParams.al_transfer_mu_max.
        _set_if(cfg.base, "al_transfer_mu_max", self.params.al_transfer_mu_max)
        # The al_iteration_* arrays are the only readout of what a step did, and
        # they are populated only when recording is on -- so this is not optional
        # here the way it is for the one-shot solve.
        _set_if(cfg.base, "record_iterations", True)
        # Means-only extraction: the covariance factorization is the most
        # expensive step after the optimizer itself, and nothing downstream of a
        # step reads a covariance. Gated, because a binding without the
        # means-only branch would read an empty marginals object.
        if capabilities()["ik_stepping"]:
            _set_if(cfg.base, "skip_marginals", True)
        # Warm-start posture, when the caller committed one. Read here rather
        # than in __init__ so a reset() picks up whatever params carries now:
        # seeding IS the mechanism for carrying a solve across the rebuild that
        # a changed constraint set forces.
        if self.params.initial_state is not None:
            _set_if(cfg, "initial_state", self.params.initial_state)

        # Read the stopping tolerances back off the config rather than keeping a
        # second copy: status() has to mirror the C++ convergence test (which
        # reports no stop reason of its own), and reading the same object it was
        # configured from is the only way the two cannot drift.
        self._tols = (cfg.base.al_abs_violation_tol, cfg.base.al_abs_cost_tol,
                      cfg.base.al_rel_violation_tol, cfg.base.al_rel_cost_tol)

        self._solver = gepetto_solvers.HandSolver(self._hand_spec(), cfg)
        # Multipliers carried in from a previous solver, re-seated onto THIS
        # solver's constraints by identity on its first solve. Set on the solver
        # rather than the config because the remap needs this graph, which does
        # not exist until that solve runs.
        if (self.params.initial_duals is not None
                and hasattr(self._solver, "set_initial_duals")):
            self._solver.set_initial_duals(self.params.initial_duals)
        self._history = []      # HandState per step (initial guess first)
        self._frames = []       # the same states as render frames
        self._notes = []        # one readout line per history entry
        self._steps = 0
        self._status = StepStatus("running", float("inf"), float("inf"),
                                  self.params.al_mu, 0)
        self._prev = None       # (violation, cost) of the previous step

    def reset(self):
        """Full cold start: straight-hand values, zero duals, mu back to
        ``al_mu``, and the scene/envs re-derived from the current params.

        Rebuilding is what makes it a *cold* start -- ``get_initial_values()``
        runs only in the C++ constructor, so nothing short of a new solver
        restores the initial posture. Re-running the base init also gives
        :meth:`_build` clean finger configs, since ``_attach_*`` mutate them in
        place and would otherwise stack a second environment on the first."""
        super().__init__(self.params)
        self._build()

    def restart_al(self):
        """Re-run the penalty schedule from the CURRENT posture: drops the duals
        and mu but keeps the pose reached so far. The weaker sibling of
        :meth:`reset`; no-op on a binding without the accessor."""
        if hasattr(self._solver, "reset_al_duals"):
            self._solver.reset_al_duals()
            self._prev = None
            self._status = self._status._replace(state="running",
                                                 mu=self.params.al_mu)
            return True
        return False

    # -- stepping --

    def status(self) -> StepStatus:
        return self._status

    def al_duals(self):
        """This solve's AL multipliers + penalty weight, tagged by constraint
        identity -- what a differently-constrained rebuild takes to continue
        holding the constraints the two problems share. None on a binding
        without the accessor."""
        if not hasattr(self._solver, "get_al_duals"):
            return None
        return self._solver.get_al_duals()

    def dual_transfer(self):
        """How much of the incoming transfer matched (or None). A 0/N here on a
        rebuild that should have matched means the constraint TAGS drifted, not
        that the problem genuinely changed."""
        if not hasattr(self._solver, "al_transfer_report"):
            return None
        rep = self._solver.al_transfer_report()
        return rep if rep.total else None

    def factor_errors(self):
        """``[(factor_family, count, total_error)]`` at the CURRENT values.

        Read-only; the one readout that says which part of the graph the inner LM
        is spending its budget on, which is what decides whether a stalled step is
        a constraint problem or a scaling problem. Empty on a binding without the
        accessor."""
        if not hasattr(self._solver, "get_factor_error_summary"):
            return []
        return [(str(name), int(count), float(err))
                for name, count, err in self._solver.get_factor_error_summary()]

    def _settling(self):
        """True while this step should pin the flexor to settle the cold start.

        Counted off ``self._steps`` (steps already taken), not off a flag, so a
        :meth:`reset` re-settles and :meth:`restart_al` -- which keeps the posture
        -- correctly does not.

        NEVER when the solve was seeded (``params.initial_state``). Settling pins
        every tendon at its COMMANDED mean, which is the right way to absorb a
        cold start's statically inconsistent Q = 0 guess -- and exactly the wrong
        thing to do to a warm one: a contact solve drives the flexor well away
        from its commanded value (1.28 N against a commanded 0.6 N is typical),
        so pinning it back hauls the fingers open by ~57 mm on step 1 and throws
        away the posture the seed existed to preserve. A seeded start is already
        consistent, which is the only thing settling was ever for."""
        if self.params.initial_state is not None:
            return False
        return self._steps < max(int(self.params.ik_settle_steps), 0)

    def step(self) -> HandResult:
        """Advance the AL outer loop by exactly one iteration."""
        # Re-aimed every step so the wrist slider stays live mid-solve; the
        # tension priors are rebuilt from params for the same reason.
        self._solver.set_wrist_pose(self.params.wrist_pose)
        settling = self._settling()
        cov = _IK_SETTLE_TENSION_COV if settling else self._flexor_tension_cov()
        sol = self._solver.solve(self._tension_priors(cov), self._tip_wrenches())

        if not self._history:
            # The pre-step values of the first step: the true initial guess, and
            # the only frame the history cannot get from a solve result.
            self._append(self._solver.get_initial_solution(), sol.meta,
                         "initial guess")
        self._steps += 1
        self._update_status(sol.meta, settling=settling)
        s = self._status
        self._append(sol.marginals, sol.meta,
                     f"step {s.steps} &nbsp; violation={s.violation:.3e} "
                     f"&nbsp; cost={s.cost:.4g} &nbsp; mu={s.mu:.3g}"
                     + (" &nbsp; *(settling: flexor pinned)*" if settling else ""))

        return self._result(
            [self._frames[-1]], sol.meta, self.params.contact_fingers,
            [self._history[-1]],
            [[f] for f in self._frames], [[hm] for hm in self._history],
            list(self._notes), duals=self.al_duals(),
            dual_transfer=self.dual_transfer())

    def _append(self, hand_marginals, meta, note):
        self._history.append(hand_marginals)
        self._frames.append(_make_frame(self.finger_names, hand_marginals, meta))
        self._notes.append(note)

    def _update_status(self, meta, settling=False):
        """Mirror ``ConstrainedOptimizer::checkConvergence`` on this step's trace.

        With ``al_max_iterations = 1`` the C++ loop always exits on its iteration
        test before evaluating the tolerances, and it records no stop reason, so
        the caller has to apply the same two tests itself. Each call logs a seed
        state plus the one iterate; the last entry is the state we just reached.

        A ``settling`` step is exempt from both verdicts and leaves no baseline
        behind: it minimizes a DIFFERENT objective (the flexor is pinned), so its
        cost is not comparable to the released steps on either side of it, and a
        settled cold start that stops moving is the step doing its job rather than
        the solve giving up."""
        def last(name, default=float("nan")):
            arr = list(getattr(meta, name, []) or [])
            return float(arr[-1]) if arr else default

        viol, cost, mu = (last("al_iteration_violations"),
                          last("al_iteration_costs"),
                          last("al_iteration_mus", self._status.mu))
        abs_v, abs_c, rel_v, rel_c = self._tols
        if settling:
            state = "running"
        elif viol < abs_v and cost < abs_c:
            state = "converged"
        elif (self._prev is not None
              and abs(viol - self._prev[0]) < rel_v
              and abs(cost - self._prev[1]) < rel_c):
            state = "stalled"
        else:
            state = "running"
        self._prev = None if settling else (viol, cost)
        self._status = StepStatus(state, viol, cost, mu, self._steps)

    def run(self, max_steps=200, on_step=None, should_stop=None) -> StepStatus:
        """Step until the solve converges, stalls, is stopped, or hits the cap.

        ``on_step(result, status)`` fires after every iteration (that is what
        lets a caller animate the convergence) and ``should_stop()`` is polled
        between iterations, so an interactive caller can break out without
        waiting for the whole run."""
        for _ in range(max_steps):
            if should_stop is not None and should_stop():
                break
            result = self.step()
            if on_step is not None:
                on_step(result, self._status)
            if self._status.done:
                break
        return self._status

    # A single "solve" of a stepper is one iteration, mirroring the controller's
    # solve == one tick, so it can stand in wherever a solver is expected.
    solve = step
