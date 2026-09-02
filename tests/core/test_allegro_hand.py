"""The Allegro hand driven through the ordinary Python solver stack.

Phase 3 proved the rigid kinematics against the C++ interface. This is the layer
above: an ``AllegroHand`` handed to ``HandFKSolver`` / ``HandIKSolver`` exactly
as ``TendonHand5F`` is, with the same params object and the same task
constraints.

What makes these tests worth having is that they are the same calls the tendon
hand makes. If the seam leaked -- if a solver reached for a tendon field, a
scalar commanded value, or the tendon hand's measured wrist pose -- it would show
up here and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import hands, solvers

pytest.importorskip(
    "pinocchio",
    reason="pinocchio is a conda C++ dependency; see conda_setup_*.sh")


@pytest.fixture(scope="module")
def hand():
    return hands.get_hand("allegro")


@pytest.fixture
def params(hand):
    """Params posed at the hand's own default, with the three-digit grasp set."""
    wrist, means = hand.default_pose()
    p = solvers.HandSolveParams()
    p.wrist_pose = wrist
    p.joint_targets = [list(m) for m in means]
    p.contact_fingers = [n in hand.default_contact_digits
                         for n in hand.digit_names]
    return p


# ---------------------------------------------------------------------------
# The hand itself.
# ---------------------------------------------------------------------------

def test_it_is_registered(hand):
    assert hand.name == "allegro"
    assert hand.kinematics == "rigid_urdf"
    assert hand.digit_names == ["index", "middle", "ring", "thumb"]


def test_every_joint_is_driven(hand):
    """The first hand to drive more than one actuator per digit. Anything
    reading ``drive_indices[0]`` is wrong here, which is what the
    ``single_drive`` feature gates."""
    assert hand.actuation.n == 4
    assert hand.actuation.drive_indices == (0, 1, 2, 3)
    assert hand.actuation.passive_indices == ()
    assert "single_drive" not in hand.features


def test_it_declares_no_tendon_features(hand):
    """Empty rather than partial: no tendons, no pinch table, no calibration
    landmarks, no hardware bridge. Each gates a workbench panel off."""
    assert hand.features == frozenset()
    assert hand.features <= hands.FEATURES


def test_asking_for_a_single_driven_value_is_refused(hand):
    """``drive_value`` would have to pick one of four. It says so instead."""
    with pytest.raises(ValueError, match="4 actuators per digit"):
        hand.actuation.drive_value(np.zeros(4))


def test_it_has_no_measured_pinch_table(hand):
    """None is the honest answer, and the pre-grasp centroid constraint has to
    be able to get it rather than an exception or a wrong default."""
    assert hand.pinch_pose([True, True, False, True]) is None


def test_digit_configs_are_fresh_each_call(hand):
    """The attach_* family mutates these in place."""
    a, b = hand.digit_configs(), hand.digit_configs()
    assert a[0][1] is not b[0][1]


# ---------------------------------------------------------------------------
# Commanding it.
# ---------------------------------------------------------------------------

def test_joint_targets_command_the_posture(hand, params):
    means = hand.actuation_means(params)
    assert len(means) == 4
    np.testing.assert_allclose(means[0], hand.DEFAULT_FINGER_Q)
    np.testing.assert_allclose(means[3], hand.DEFAULT_THUMB_Q)


def test_an_uncommanded_hand_falls_back_to_neutral(hand):
    p = solvers.HandSolveParams()
    p.joint_targets = None
    for m in hand.actuation_means(p):
        np.testing.assert_allclose(m, np.zeros(4))


def test_a_wrongly_shaped_joint_target_is_refused(hand):
    p = solvers.HandSolveParams()
    p.joint_targets = [[0.0, 0.1, 0.2]] * 4      # three joints, not four
    with pytest.raises(ValueError, match="expected"):
        hand.actuation_means(p)


def test_the_wrong_number_of_digits_is_refused(hand):
    p = solvers.HandSolveParams()
    p.joint_targets = [[0.0] * 4] * 5
    with pytest.raises(ValueError, match="5 entries"):
        hand.actuation_means(p)


# ---------------------------------------------------------------------------
# Through the solvers.
# ---------------------------------------------------------------------------

def test_fk_poses_it_exactly(hand, params):
    """Seeded at the same posture the joint prior is centred on, FK has nothing
    to solve: zero residual in one iteration."""
    result = solvers.HandFKSolver(params, hand).solve()
    assert result.meta.iterations <= 2
    assert result.meta.error < 1e-12
    for i, name in enumerate(result.finger_names):
        np.testing.assert_allclose(result.frames[0][name].actuation(),
                                   params.joint_targets[i], atol=1e-9)


def test_fk_reports_the_commanded_wrist(hand, params):
    result = solvers.HandFKSolver(params, hand).solve()
    np.testing.assert_allclose(result.wrist_pose(0), params.wrist_pose, atol=1e-6)


def test_the_default_pose_puts_the_grasp_digits_on_the_object(hand, params):
    """A default that aims the hand nowhere is worse than none. The tendon
    hand's measured wrist pose points THIS hand away from the object -- its
    fingers extend +z where the tendon palm lies along -x -- so Allegro carries
    its own."""
    result = solvers.HandFKSolver(params, hand).solve()
    gaps = result.surface_gaps(0)
    for name in hand.default_contact_digits:
        assert abs(gaps[name]) < 0.02, (name, gaps[name])


def test_ik_drives_the_grasp_digits_onto_the_object(hand, params):
    """The whole point: the same contact constraint the tendon hand uses,
    applied to a mechanism that is not a rod."""
    result = solvers.HandIKSolver(params, hand).solve()
    gaps = result.surface_gaps(0)
    for name in hand.default_contact_digits:
        assert abs(gaps[name]) < 2e-3, (name, gaps[name])


def test_ik_leaves_the_uncommanded_digit_alone(hand, params):
    """`ring` is not in the contact set, so nothing should be driving it onto
    the object -- a masked-off digit keeps collision avoidance and no more."""
    result = solvers.HandIKSolver(params, hand).solve()
    assert result.surface_gaps(0)["ring"] > 0.01


def test_the_solved_state_has_the_neutral_shape(hand, params):
    result = solvers.HandFKSolver(params, hand).solve()
    state = result.state(0)
    assert len(state.digits) == 4
    for d in state.digits:
        assert d.extras is None            # no tendon payload
        assert list(d.displacement) == []  # actuation IS position
        assert len(d.sites) == 5           # mount + four links


def test_the_finger_sol_accessors_work_on_a_rigid_hand(hand, params):
    """The accessors are what let a reader stay mechanism-neutral. `tendon()`
    returning None is how one that cannot finds out."""
    result = solvers.HandFKSolver(params, hand).solve()
    fs = result.frames[0]["index"]
    assert fs.num_sites() == 5
    assert fs.sphere_sites() == [1, 2, 3, 4]
    assert fs.tendon() is None
    assert fs.tip_pose().shape == (4, 4)
    np.testing.assert_allclose(fs.tip_point(), fs.tip_pose()[:3, 3])


# ---------------------------------------------------------------------------
# Both hands, side by side.
# ---------------------------------------------------------------------------

def test_the_two_hands_report_different_defaults():
    """They are different robots; a shared default would aim one of them wrong."""
    a_wrist, _ = hands.get_hand("allegro").default_pose()
    t_wrist, _ = hands.get_hand("tendon_5f").default_pose()
    assert not np.allclose(a_wrist, t_wrist)


def test_both_hands_satisfy_the_protocol():
    for name in hands.registered_hands():
        hand = hands.get_hand(name)
        assert isinstance(hand, hands.Hand), name
        assert hand.features <= hands.FEATURES, name
        assert hand.opposing_index in range(len(hand.digit_names)) or \
            hand.opposing_index == -1, name
