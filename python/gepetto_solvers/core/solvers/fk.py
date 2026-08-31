""":class:`HandFKSolver` -- pure kinematics driven by tensions, no contact.

Re-commands the wrist each solve so repeated calls warm-start from the previous
solution, and cold-restarts when the wrist jumps too far for that to be safe.
"""


import numpy as np

import gepetto_solvers

from .base import HandSolverBase
from .capabilities import _set_if
from .frames import solved_wrist_pose
from .params import HandSolveParams
from .result import HandResult, _make_frame, _tip_points


class HandFKSolver(HandSolverBase):
    """Pure-kinematics hand solve driven by tensions (no contact). Re-commands the
    wrist each solve so repeated calls warm-start from the previous solution
    (``fk_5f_sweep.py``), and cold-restarts when the wrist jumps too far for that
    to be safe -- see :attr:`_WARM_START_MAX_POS_M`."""

    # What a warm start actually retains is the previous solve's ROD NODES, and
    # set_wrist_pose moves only the PRIOR. So a wrist that jumps leaves the whole
    # hand sitting where it used to be, a long way from the pose a sigma-1e-4
    # prior is now pulling it to, and the optimizer has to drag every node across
    # that gap before it can do any kinematics. Measured on this hand, jumping
    # from the default hover: fine to 0.15 m; from 0.2 m it throws
    # IndeterminantLinearSystem on W0 (the wrist), and jumps that do NOT throw can
    # instead stall at max_iterations having left the hand 96-131 mm from the pose
    # it was told to be at -- silently, since nothing in the result says
    # "not converged". Rotation goes the same way past ~2 rad.
    #
    # A COLD start does not have the problem at all: every node is seeded at
    # T_wrist o offset, so it begins at the commanded pose and the iteration count
    # comes out identical whether the jump was 0.2 m or 1.0 m. It costs one graph
    # rebuild, and the warm start only buys anything for the small moves a slider
    # drag makes, so these sit well under where the trouble starts rather than
    # near it.
    #
    # The caller that jumps is a robot readback (viz_interactive's "Get robot
    # state"), where the measured wrist can be a third of a metre from wherever
    # the app's hand happened to be.
    _WARM_START_MAX_POS_M = 0.05
    _WARM_START_MAX_ROT_RAD = 0.5

    # How far the SOLVED wrist may sit from the commanded one before the solve is
    # treated as failed. Nothing pulls on the wrist in an FK solve -- no contact,
    # so nothing to trade against the prior -- and a healthy solve lands within a
    # few microns of the commanded pose, where a stalled one lands tens of
    # millimetres away. Anything between the two separates them; 1 mm is nowhere
    # near either.
    _WRIST_TRACKING_TOL_M = 1e-3

    def __init__(self, params: HandSolveParams | None = None):
        super().__init__(params)
        # A posture the next rebuild starts from, committed by seed_posture()
        # and consumed by the next solve. None -- the case every caller had
        # before phase 4 needed one -- is the straight-rod cold guess.
        self._seed = None
        self._build()

    def _build(self, seed=None):
        """Build (or rebuild) the underlying solver, cold-started at the params'
        current wrist pose -- or, with ``seed``, at that posture.

        ``seed`` is a :meth:`HandResult.state` from any solver over this same
        hand; the C++ side merges it over the cold guess, so a state that does
        not carry every variable still works."""
        cfg = gepetto_solvers.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.max_iterations = 500
        if seed is not None:
            _set_if(cfg, "initial_state", seed)
        self._solver = gepetto_solvers.TendonHandSolver(self.configs, cfg)
        # Where the values this solver is holding actually sit. None = nothing
        # worth warm-starting from (a solve that failed left them wherever it
        # gave up), which forces the next solve to rebuild.
        # Optional: cleared to None wherever the retained values stop being
        # something a warm start may begin from.
        self._warm_wrist: np.ndarray | None = np.array(
            self.params.wrist_pose, float)

    def _warm_start_holds(self, T):
        """Whether the retained values are close enough to ``T`` to start from."""
        if self._warm_wrist is None:
            return False
        if (np.linalg.norm(T[:3, 3] - self._warm_wrist[:3, 3])
                > self._WARM_START_MAX_POS_M):
            return False
        # Rotation angle of the residual R_warm^T R, via the trace. Clipped
        # because a cosine a rounding step outside [-1, 1] is an ordinary result
        # for two nearly equal rotations, not a bad matrix.
        cos = 0.5 * (np.trace(self._warm_wrist[:3, :3].T @ T[:3, :3]) - 1.0)
        return float(np.arccos(np.clip(cos, -1.0, 1.0))) <= self._WARM_START_MAX_ROT_RAD

    def seed_posture(self, state):
        """Start the next solve from ``state`` instead of the cold guess.

        For the caller crossing INTO an FK ramp from a solve this solver never
        ran -- the phase-4 close, the phase-5 lift, both handed a hand that an
        IK solve posed. Its own retained values are whatever its last FK solve
        left, which may be a phase and a scene ago, so without this the ramp's
        first pose drags the whole hand across that gap.

        Forces the rebuild that applies it (``_warm_wrist = None``): a wrist
        that has barely moved would otherwise warm-start off those retained
        values and the seed would never be read.

        Note what an FK solve can and cannot do with it. Nothing is enforced
        here, so the seed cannot make the hand HOLD the posture it came from:
        the tensions decide where this solve settles, and the seed only says
        where it starts looking. It buys continuity and the iterations that go
        with it -- not the contact the IK solve was maintaining.

        ``state`` of None is a no-op, so a caller with nothing solved yet (or a
        result from before ``HandResult.states`` existed) can call it blind.
        """
        if state is None:
            return
        self._seed = state
        self._warm_wrist = None

    # How far a re-solve may move a fingertip before a seeded pose counts as
    # settled, and how many re-solves it gets. Both measured (see _settle): a
    # phase-2 grasp carried into an FK solve lands up to 6 mm short, one more
    # solve reaches the fixed point, and a third does not move at all.
    _SETTLE_TOL_M = 1e-4
    _SETTLE_MAX_SOLVES = 3

    def _settle(self, frame):
        """Re-solve a SEEDED pose until it stops moving. Returns the settled
        ``(frame, sol)``.

        An unseeded FK solve is at its fixed point when it returns -- solve it
        again at the same wrist and tensions and nothing moves. A seeded one is
        not: it starts near a posture some other solver produced, and the
        optimizer's convergence test can call it done several millimetres short
        of the tension equilibrium the same wrist and tensions reach from cold
        (up to 6 mm on a phase-2 grasp carried into a close).

        That difference matters to the caller doing the carrying, because it
        MEASURES this pose. :func:`synchronized_close` reads the starting tendon
        lengths off it and then probes for a slope; a pose still settling hands
        the probe a few millimetres of travel the tension nudge did not buy, and
        the whole ramp is spaced off that slope. Better to arrive settled.

        Costs one extra FK solve (~100 ms) on the single seeded solve at the
        head of a ramp; nothing else in the app ever seeds one."""
        for _ in range(self._SETTLE_MAX_SOLVES):
            before = _tip_points(frame)
            frame, sol = self._solve_once()
            moved = float(np.max(np.linalg.norm(
                _tip_points(frame) - before, axis=1)))
            if moved <= self._SETTLE_TOL_M:
                break
        return frame, sol

    def _solve_once(self):
        # Uniform prior on every tendon: a tight-passive/loose-flexor prior is
        # underdetermined without contact (IndeterminantLinearSystem on the
        # tension variable) -- see fk_5f_sweep.py.
        cov = (1e-2) ** 2 * np.eye(6)
        sol = self._solver.solve(self._tension_priors(cov), self._tip_wrenches())
        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        return frame, sol

    def solve(self) -> HandResult:
        T = np.asarray(self.params.wrist_pose, float)
        if self._warm_start_holds(T):
            self._solver.set_wrist_pose(T)   # re-aim the prior; keep the posture
        else:
            # Too far to drag the hand: start there -- from a committed posture
            # if one is pending, else cold.
            self._build(self._seed)
        # One-shot, whether or not it was used: it describes the hand at the
        # moment the caller committed it, and the retry below is supposed to be
        # a genuine cold start rather than the same seed a second time.
        seeded, self._seed = self._seed is not None, None

        # The thresholds above are where the trouble STARTS, not a proof, and a
        # bad warm start can be a bad one for reasons that have nothing to do with
        # the wrist. So the result is checked and a cold restart tried once --
        # which is cheap, and turns the whole failure mode into a slower solve
        # rather than a raised exception or, worse, a hand drawn somewhere the
        # robot is not.
        for last_attempt in (False, True):
            try:
                frame, sol = self._solve_once()
                offset = float(np.linalg.norm(
                    solved_wrist_pose(self.configs, frame)[:3, 3] - T[:3, 3]))
                if offset <= self._WRIST_TRACKING_TOL_M:
                    if seeded:
                        frame, sol = self._settle(frame)
                    self._warm_wrist = T.copy()
                    # FK constrains no finger, but the mask still rides along: it
                    # is read live off params (not baked in at construction), and
                    # the goal overlays drawn over an FK pose -- p_bar, the
                    # opposition split, the support-plane equalities -- are all
                    # statements about the DESIGNATED contact set.
                    return self._result([frame], sol.meta,
                                        self.params.contact_fingers, [sol.marginals])
                why = (f"stalled with the wrist {offset * 1e3:.0f} mm from the "
                       f"commanded pose after {sol.meta.iterations} iterations")
            except RuntimeError as exc:
                why = f"failed ({str(exc).strip().splitlines()[0]})"
            self._warm_wrist = None          # these values are not startable-from
            if last_attempt:
                raise RuntimeError(
                    f"FK solve {why}. This was already a cold start, so it is not "
                    f"the warm-start jump -- check the tensions and the wrist pose "
                    f"being commanded.")
            self._build()                    # drop the values, retry cold

        # Unreachable: the loop above either returns on success or raises on
        # last_attempt. Stated rather than implied, so the totality of this
        # function does not rest on the reader noticing the literal 2-tuple.
        raise AssertionError("unreachable")
