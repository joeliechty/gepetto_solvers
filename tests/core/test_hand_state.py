"""The shape of a solved hand, and what is neutral about it.

``HandState`` is the transport between a finished solve and everything that
reads one. It used to be rod-and-tendon shaped -- ``digits[i].rod.states[j]``,
``.tendon_lengths``, ``.tendon_config`` -- which meant a hand that is not a
Cosserat rod could not fill it.

These tests pin the split that fixed that: a neutral part every mechanism has
(sites, actuation, collision sites, the wrist) and an ``extras`` payload only the
mechanism that owns it understands. The neutral half is what
``_FingerSol``'s accessors expose, and what a second kinematics has to satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest

import gepetto_solvers
from _pkg import solvers


@pytest.fixture(scope="module")
def hand():
    """The tendon hand on its BUNDLED dimensions, so these assertions do not
    depend on whether gepetto_core is installed (see tests/README.md)."""
    from gepetto_solvers.core.hands.tendon_5f import (
        DEFAULT_HAND_DIMENSIONS,
        TendonHand5F,
    )
    return TendonHand5F(DEFAULT_HAND_DIMENSIONS)


@pytest.fixture(scope="module")
def fk_solver(hand):
    return solvers.HandFKSolver(solvers.HandSolveParams(), hand)


@pytest.fixture(scope="module")
def fk_result(fk_solver):
    """One cheap FK solve, shared -- these are all structural assertions."""
    return fk_solver.solve()


# ---------------------------------------------------------------------------
# The neutral half.
# ---------------------------------------------------------------------------

def test_a_digit_exposes_sites_not_rod_nodes(fk_result):
    state = fk_result.state(0)
    assert len(state.digits) == 5
    for d in state.digits:
        assert len(d.sites) > 0
        assert np.asarray(d.sites[0].pose.mean).shape == (4, 4)


def test_a_digit_exposes_its_actuation_and_displacement(fk_result):
    """Named for what they DO, not for what drives this particular hand: the
    tendon hand's actuation is tension and its displacement is length, and a
    position-controlled hand fills actuation and leaves displacement empty."""
    state = fk_result.state(0)
    for d in state.digits:
        assert np.asarray(d.actuation.mean).shape == (6,)
        assert len(d.displacement) == 6


def test_collision_sites_come_off_the_state(fk_result):
    """Which sites carry a sphere is read off the SOLVE, not off a config, so an
    overlay can never mark a sphere the graph did not actually carry."""
    state = fk_result.state(0)
    for d in state.digits:
        assert len(d.collision_sites) > 0
        assert max(d.collision_sites) < len(d.sites)
        assert all(i >= 0 for i in d.collision_sites)


def test_the_bundle_carries_the_wrist(fk_result):
    """The wrist used to be recoverable only by inverting digit 0's mounting
    offset in Python -- a trick that holds for this hand and would not for a
    mechanism that owns its wrist variable outright. Each kinematics now answers
    it, and the answer rides on the bundle."""
    state = fk_result.state(0)
    T = np.asarray(state.wrist_pose, float)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)


def test_the_carried_wrist_agrees_with_the_offset_inversion(fk_result, fk_solver):
    """The two routes to the wrist must give the same answer on this hand, which
    is what says the new one is right rather than merely new.

    They are independent: one reads the wrist variable in C++, the other inverts
    digit 0's mounting offset out of its base site in Python."""
    carried = fk_result.wrist_pose(0)
    inverted = solvers.solved_wrist_pose(fk_solver.configs, fk_result.frames[0])
    np.testing.assert_allclose(carried, inverted, atol=1e-9)


# ---------------------------------------------------------------------------
# The mechanism-specific half.
# ---------------------------------------------------------------------------

def test_tendon_state_lives_behind_extras(fk_result):
    """Routing, per-disc wrenches and the tension Jacobian have no analogue on a
    rigid-body hand, so they are not fields on the neutral struct."""
    state = fk_result.state(0)
    for d in state.digits:
        assert isinstance(d.extras, gepetto_solvers.TendonDigitExtras)
        assert d.extras.tendon_config.num_tendons == 6
        assert len(d.extras.tendon_config.disc_pose_idx) > 0


def test_the_neutral_struct_names_no_tendon_field(fk_result):
    """A guard against the fields creeping back: if `tendon_config` or
    `tendon_lengths` reappears on DigitState, the split has been undone."""
    d = fk_result.state(0).digits[0]
    for gone in ("rod", "tendon_config", "tendon_lengths", "tensions",
                 "external_wrenches", "J_pose_tensions"):
        assert not hasattr(d, gone), f"DigitState should not expose {gone!r}"


def test_collision_sites_match_the_tendon_disc_set(fk_result):
    """On THIS hand the disc set is the collision-sphere set. That equality is a
    tendon-hand fact, which is why the neutral field is populated from it rather
    than the readers going to the routing for it."""
    for d in fk_result.state(0).digits:
        assert d.collision_sites == list(d.extras.tendon_config.disc_pose_idx)


# ---------------------------------------------------------------------------
# The accessors every reader is supposed to go through.
# ---------------------------------------------------------------------------

def test_finger_sol_accessors_agree_with_the_raw_fields(fk_result):
    frame = fk_result.frames[0]
    for name in fk_result.finger_names:
        fs = frame[name]
        np.testing.assert_allclose(
            fs.tip_pose(), np.asarray(fs.marginals.sites[-1].pose.mean, float))
        np.testing.assert_allclose(fs.tip_point(), fs.tip_pose()[:3, 3])
        assert fs.num_sites() == len(fs.marginals.sites)
        assert fs.sphere_sites() == list(fs.marginals.collision_sites)
        np.testing.assert_allclose(
            fs.actuation(), np.asarray(fs.marginals.actuation.mean, float))


def test_tendon_accessor_returns_the_payload_here(fk_result):
    """`.tendon()` is the guarded route to the mechanism-specific half: a caller
    checks it for None instead of catching AttributeError."""
    fs = fk_result.frames[0][fk_result.finger_names[0]]
    assert fs.tendon() is not None
    assert fs.tendon().tendon_config.num_tendons == 6
