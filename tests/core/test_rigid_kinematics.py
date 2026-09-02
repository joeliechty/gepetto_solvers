"""The rigid-body kinematics, driven end to end on the Allegro hand.

This is the first time the ``HandKinematics`` seam carries something that is not
a Cosserat rod, so these tests are mostly about the seam holding: that a hand
described by a URDF loads by name, builds a graph, solves, and comes back through
the same ``HandState`` the tendon hand uses.

The posterior being solved is

    p(Theta) ~ p(T_w) p(q) p(T | T_w, q)

so the assertions are about those three terms: the wrist prior HandModel emits,
the joint prior per digit, and the kinematics likelihood tying each site to
(T_w, q). No task constraints are attached -- those are the same code for every
hand and are covered elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

import gepetto_solvers

pytest.importorskip(
    "pinocchio",
    reason="pinocchio is a conda C++ dependency; see conda_setup_*.sh")

from gepetto_solvers.core.hands.allegro import spec as allegro  # noqa: E402


def _hand_spec(q_init=None, sigma_fk=None):
    s = gepetto_solvers.HandSpec()
    s.kinematics = "rigid_urdf"
    s.digit_names = list(allegro.DIGIT_NAMES)
    s.opposing_digit = allegro.DIGIT_NAMES.index(allegro.OPPOSING_DIGIT)
    n = len(allegro.DIGIT_NAMES)
    s.env = [None] * n
    s.sphere_contact = [None] * n
    s.kinematics_config = allegro.kinematics_config(q_init=q_init,
                                                    sigma_fk=sigma_fk)
    return s


def _solver(hand_spec, max_iterations=200):
    cfg = gepetto_solvers.HandSolverConfig()
    cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
    cfg.base.max_iterations = max_iterations
    return gepetto_solvers.HandSolver(hand_spec, cfg)


#: A mildly flexed posture: the base joint neutral, the three flexion joints bent.
Q_TARGET = np.array([0.0, 0.4, 0.4, 0.4])
Q_COV = (1e-2) ** 2 * np.eye(allegro.DOF_PER_DIGIT)


def _priors(q=Q_TARGET, cov=Q_COV):
    n = len(allegro.DIGIT_NAMES)
    actuation = [gepetto_solvers.VectorXGaussian(np.asarray(q, float), cov)
                 for _ in range(n)]
    wrenches = [gepetto_solvers.Vector6Gaussian(np.zeros(6), 1e-4 * np.eye(6))
                for _ in range(n)]
    return actuation, wrenches


@pytest.fixture(scope="module")
def solved():
    """One Allegro solve, seeded at the prior mean -- the normal case."""
    s = _solver(_hand_spec(q_init=[Q_TARGET] * len(allegro.DIGIT_NAMES)))
    return s, s.solve(*_priors())


# ---------------------------------------------------------------------------
# Loading, by name, through the registry.
# ---------------------------------------------------------------------------

def test_the_rigid_kinematics_is_registered():
    assert "rigid_urdf" in gepetto_solvers.registered_hand_kinematics()


def test_both_kinematics_coexist():
    """The seam's whole point: two mechanisms behind one graph builder."""
    known = set(gepetto_solvers.registered_hand_kinematics())
    assert {"rigid_urdf", "tendon"} <= known


def test_the_wrong_payload_is_refused():
    """A HandSpec naming rigid_urdf but carrying a tendon payload must say so,
    not downcast to garbage."""
    s = _hand_spec()
    s.kinematics_config = gepetto_solvers.TendonHandKinematicsConfig()
    with pytest.raises(ValueError, match="RigidHandKinematicsConfig"):
        _solver(s)


def test_a_missing_frame_names_itself():
    cfg = allegro.kinematics_config()
    digits = cfg.digits
    d = digits[0]
    d.site_frames = ["no_such_link"]
    digits[0] = d
    cfg.digits = digits
    s = _hand_spec()
    s.kinematics_config = cfg
    with pytest.raises(ValueError, match="no frame named"):
        _solver(s)


# ---------------------------------------------------------------------------
# The solve.
# ---------------------------------------------------------------------------

def test_it_solves_the_allegro_hand(solved):
    _, sol = solved
    assert len(sol.marginals.digits) == 4
    assert list(sol.marginals.digit_names) == ["index", "middle", "ring", "thumb"]


def test_seeding_at_the_prior_mean_converges_immediately(solved):
    """q_init == q_S means the solve starts at zero residual on every term, so
    it should take one iteration and land exactly there. This is why
    `kinematics_config` tells callers to seed at the prior mean."""
    _, sol = solved
    assert sol.meta.iterations <= 2
    assert sol.meta.error < 1e-12
    for d in sol.marginals.digits:
        np.testing.assert_allclose(d.actuation.mean, Q_TARGET, atol=1e-9)


def test_it_converges_from_a_cold_seed_too():
    """Seeded a full 0.4 rad per joint from the prior mean it still reaches the
    same answer -- it just costs iterations. A stiffer sigma_fk costs more; see
    the measured table on RigidHandKinematicsConfig::sigma_fk."""
    s = _solver(_hand_spec(q_init=[[0.0] * 4] * 4))
    sol = s.solve(*_priors())
    assert sol.meta.iterations < 200, "should converge well inside the cap"
    for d in sol.marginals.digits:
        np.testing.assert_allclose(d.actuation.mean, Q_TARGET, atol=1e-6)


def test_the_joint_prior_actually_moves_the_hand():
    """p(q) is what poses the hand here. Two different q_S must give two
    different postures -- otherwise the prior is being built but not enforced."""
    a = _solver(_hand_spec(q_init=[Q_TARGET] * 4)).solve(*_priors())
    bent = np.array([0.0, 0.9, 0.9, 0.9])
    b = _solver(_hand_spec(q_init=[bent] * 4)).solve(*_priors(q=bent))

    tip_a = np.asarray(a.marginals.digits[0].sites[-1].pose.mean)[:3, 3]
    tip_b = np.asarray(b.marginals.digits[0].sites[-1].pose.mean)[:3, 3]
    assert np.linalg.norm(tip_a - tip_b) > 0.01, "0.5 rad should move the tip"


# ---------------------------------------------------------------------------
# The state bundle a rigid hand fills.
# ---------------------------------------------------------------------------

def test_the_state_has_the_neutral_shape(solved):
    _, sol = solved
    for d in sol.marginals.digits:
        assert len(d.sites) == allegro.SITES_PER_DIGIT
        assert np.asarray(d.actuation.mean).shape == (allegro.DOF_PER_DIGIT,)
        assert d.collision_sites == list(range(1, allegro.SITES_PER_DIGIT))


def test_a_rigid_hand_carries_no_tendon_extras(solved):
    """`extras` is None rather than an empty tendon payload: this hand has no
    routing, and a reader must be able to tell that by asking."""
    _, sol = solved
    for d in sol.marginals.digits:
        assert d.extras is None


def test_a_rigid_hand_has_no_displacement(solved):
    """Actuation IS position here, so there is no second variable and no
    displacement readout."""
    _, sol = solved
    for d in sol.marginals.digits:
        assert list(d.displacement) == []


def test_the_bundle_carries_the_wrist(solved):
    _, sol = solved
    T = np.asarray(sol.marginals.wrist_pose, float)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-6)


def test_site_zero_is_the_fixed_mount(solved):
    """The interface's invariant: T_0 = T_wrist o digit_base_offset, so a caller
    can recover the wrist from a frame alone. Here the wrist is identity, so each
    digit's site 0 IS its mount -- and the four differ, which is what says the
    mount is per digit rather than one shared palm frame."""
    _, sol = solved
    mounts = [np.asarray(d.sites[0].pose.mean)[:3, 3] for d in sol.marginals.digits]
    for a in range(len(mounts)):
        for b in range(a + 1, len(mounts)):
            assert np.linalg.norm(mounts[a] - mounts[b]) > 1e-4


def test_the_four_digits_reach_different_places(solved):
    """A sanity check that the per-digit joint index mapping is not aliased --
    if every digit read the same joints, the tips would coincide."""
    _, sol = solved
    tips = [np.asarray(d.sites[-1].pose.mean)[:3, 3] for d in sol.marginals.digits]
    for a in range(len(tips)):
        for b in range(a + 1, len(tips)):
            assert np.linalg.norm(tips[a] - tips[b]) > 1e-3


def test_the_thumb_opposes_the_fingers(solved):
    """Not a constraint, just geometry: the thumb tip should sit clearly off the
    plane the three fingers lie in, or nothing about opposition would work."""
    _, sol = solved
    names = list(sol.marginals.digit_names)
    tips = {n: np.asarray(d.sites[-1].pose.mean)[:3, 3]
            for n, d in zip(names, sol.marginals.digits)}
    fingers = np.array([tips[n] for n in ("index", "middle", "ring")])
    assert np.linalg.norm(tips["thumb"] - fingers.mean(axis=0)) > 0.05


# ---------------------------------------------------------------------------
# Round-tripping a posture.
# ---------------------------------------------------------------------------

def test_a_solved_state_can_seed_another_solver(solved):
    """`initial_state` is how a solve carries into the next one. It goes through
    insert_from_state, which for this hand reads q and the site poses back onto
    fresh variables."""
    _, sol = solved
    cfg = gepetto_solvers.HandSolverConfig()
    cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
    cfg.initial_state = sol.marginals
    seeded = gepetto_solvers.HandSolver(_hand_spec(q_init=[Q_TARGET] * 4), cfg)
    again = seeded.solve(*_priors())

    np.testing.assert_allclose(again.marginals.digits[0].actuation.mean,
                               sol.marginals.digits[0].actuation.mean, atol=1e-9)


def test_a_tendon_state_cannot_seed_a_rigid_hand():
    """Postures are not interchangeable between mechanisms, and saying so beats
    silently seeding a hand from a different robot."""
    empty = gepetto_solvers.HandState()
    empty.digit_names = list(allegro.DIGIT_NAMES)
    empty.digits = [gepetto_solvers.DigitState() for _ in allegro.DIGIT_NAMES]
    cfg = gepetto_solvers.HandSolverConfig()
    cfg.initial_state = empty
    with pytest.raises(ValueError):
        gepetto_solvers.HandSolver(_hand_spec(), cfg)


# ---------------------------------------------------------------------------
# What this hand refuses.
# ---------------------------------------------------------------------------

def test_a_displacement_gp_is_refused():
    """A joint-space hand has no displacement variable, so a caller asking for
    one is told rather than having it quietly dropped."""
    pc = gepetto_solvers.HandTrajectoryPlannerConfig()
    pc.K = 2
    pc.base.linear_solver_type = "MULTIFRONTAL_QR"
    pc.gp_actuation_Qc = 1e-2 * np.eye(allegro.DOF_PER_DIGIT)
    pc.gp_displacement_Qc = 1e-2 * np.eye(allegro.DOF_PER_DIGIT)
    planner = gepetto_solvers.HandTrajectoryPlanner(_hand_spec(), pc)
    with pytest.raises(ValueError, match="no displacement variable"):
        planner.plan(*_priors())


def test_a_mismatched_joint_prior_is_refused():
    s = _hand_spec()
    solver = _solver(s)
    bad = [gepetto_solvers.VectorXGaussian(np.zeros(3), np.eye(3))
           for _ in allegro.DIGIT_NAMES]
    _, wrenches = _priors()
    with pytest.raises(ValueError, match="joints"):
        solver.solve(bad, wrenches)
