"""The flat encoding a plan crosses a transport in: ``flatten_plan`` /
``unflatten_plan``.

Pure numpy -- no solver, no ROS, no hardware::

    pytest tests/core/test_plan_wire.py

WHY THIS IS HERE AND NOT IN THE ROS PACKAGE. ``gepetto_ros``'s ``PlayPlan`` goal
is exactly these fields, and its bridge and executor are a field-by-field copy
onto and off them. Written inside that package the encoding would have to be
written twice -- once to send, once to receive -- with the two copies as the only
check on each other, and neither testable without a built colcon workspace. Here
there is one copy, and this file is what proves it round-trips at both widths a
plan is ever built at.

THE ALLEGRO ORDER CHECK at the bottom is the one with teeth. Wonik's driver reads
``sensor_msgs/JointState`` POSITIONALLY -- ``desired_position[i] =
msg->position[i]``, with ``msg.name`` ignored entirely -- so a permutation between
the solver's digit blocks and the driver's joint numbering does not fail, it
drives the wrong joints. That cannot be caught by a round trip, only by pinning
our table against the driver's own.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import hands, robot_plan

SolvePlan = robot_plan.SolvePlan
Waypoint = robot_plan.Waypoint

DIGITS = ["index", "middle", "thumb"]

#: (width, command_kind) -- the two shapes a real plan takes. Width 1 is the
#: tendon hand's flexor displacement in metres; width 4 is the Allegro's joint
#: positions in radians.
SHAPES = [
    (1, robot_plan.TENDON_DISPLACEMENT_M),
    (4, robot_plan.JOINT_POSITION_RAD),
]


def _pose(i):
    """A pose that is different for every waypoint, in translation AND rotation.

    Rotation matters: the encoding hands 4x4s straight through, so a bug that
    flattened only the translation would pass against identity rotations.
    """
    angle = 0.1 * i
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                          [np.sin(angle), np.cos(angle), 0.0],
                          [0.0, 0.0, 1.0]])
    T[:3, 3] = [0.01 * i, -0.02 * i, 0.3 + 0.001 * i]
    return T


def _plan(width, kind, n=4):
    """A plan whose every commanded number is DISTINCT.

    Deliberately not random and not repeated: values are ``digit_index.actuator``
    style, so a transposed reshape or an off-by-one digit block shows up as a
    number landing under the wrong name rather than as a near-miss.
    """
    rng = np.random.default_rng(11)
    waypoints = [
        Waypoint(wrist_pose=_pose(i),
                 digit_cmd={name: rng.uniform(-0.5, 0.5, width)
                            for name in DIGITS},
                 note=f"iterate {i}")
        for i in range(n)
    ]
    return SolvePlan(waypoints=waypoints,
                     corner_viz=np.array([-0.2, -0.2, 0.0]),
                     digit_names=list(DIGITS),
                     open_lengths={name: 0.1 for name in DIGITS},
                     command_kind=kind)


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("width", "kind"), SHAPES)
def test_round_trip_is_exact(width, kind):
    plan = _plan(width, kind)
    restored = robot_plan.unflatten_plan(robot_plan.flatten_plan(plan))

    assert restored.digit_names == plan.digit_names
    assert restored.command_kind == plan.command_kind
    assert restored.dof_per_digit == width
    assert len(restored.waypoints) == len(plan.waypoints)
    assert np.allclose(restored.corner_viz, plan.corner_viz)
    for before, after in zip(plan.waypoints, restored.waypoints, strict=True):
        assert np.allclose(after.wrist_pose, before.wrist_pose)
        assert after.note == before.note
        assert set(after.digit_cmd) == set(before.digit_cmd)
        for name in before.digit_cmd:
            # Exact, not close: nothing in the encoding does arithmetic, so a
            # difference of one ulp would mean a value took a path through a
            # narrower dtype.
            assert np.array_equal(after.digit_cmd[name], before.digit_cmd[name])


@pytest.mark.parametrize(("width", "kind"), SHAPES)
def test_the_flat_block_is_row_major_over_waypoint_digit_actuator(width, kind):
    """The layout, pinned. It is the one thing a receiver has to assume, and it
    is assumed by an interface file rather than by code that could be read."""
    plan = _plan(width, kind)
    fields = robot_plan.flatten_plan(plan)
    flat = np.asarray(fields["digit_cmd"], float)

    assert flat.size == len(plan.waypoints) * len(DIGITS) * width
    for i, waypoint in enumerate(plan.waypoints):
        for j, name in enumerate(DIGITS):
            start = (i * len(DIGITS) + j) * width
            assert np.array_equal(flat[start:start + width],
                                  waypoint.digit_cmd[name])


def test_every_field_is_declared():
    """``FIELDS`` is what a marshaller loops over, so it must not fall behind
    what ``flatten_plan`` actually produces."""
    fields = robot_plan.flatten_plan(_plan(1, robot_plan.TENDON_DISPLACEMENT_M))
    assert set(fields) == set(robot_plan.FIELDS)


def test_a_block_of_the_wrong_size_raises():
    """Rather than reshaping to something that fits. A silently reshaped block
    drives the right motors with the wrong values, which no downstream check
    could distinguish from a strange solve."""
    fields = robot_plan.flatten_plan(_plan(4, robot_plan.JOINT_POSITION_RAD))
    fields["digit_cmd"] = np.asarray(fields["digit_cmd"])[:-1]
    with pytest.raises(ValueError, match="digit_cmd has"):
        robot_plan.unflatten_plan(fields)


def test_a_ragged_plan_is_refused_before_it_reaches_the_wire():
    """A plan whose digits carry different widths has no flat representation.
    It must fail HERE, where the shapes are still visible, not as a length
    mismatch on the far side of a transport."""
    plan = _plan(4, robot_plan.JOINT_POSITION_RAD)
    plan.waypoints[1].digit_cmd["thumb"] = np.zeros(3)
    with pytest.raises(ValueError, match="ragged plan"):
        robot_plan.flatten_plan(plan)


# ---------------------------------------------------------------------------
# The Allegro's positional command order
# ---------------------------------------------------------------------------

#: ``jointNames[]`` from ``allegro_hand_controllers/src/allegro_node.cpp``, and
#: the same list ``allegro_mock_node.py`` declares. Copied here rather than
#: imported: that package is a ROS workspace sibling, not a Python dependency of
#: this one, and a test that silently skipped when it was absent would be no
#: check at all. If Wonik renumbers the hand, this list is what has to change,
#: and the failure will say so.
DRIVER_JOINT_NAMES = [
    "joint_0_0", "joint_1_0", "joint_2_0", "joint_3_0",
    "joint_4_0", "joint_5_0", "joint_6_0", "joint_7_0",
    "joint_8_0", "joint_9_0", "joint_10_0", "joint_11_0",
    "joint_12_0", "joint_13_0", "joint_14_0", "joint_15_0",
]


def test_allegro_driver_order_matches_the_hand_that_will_be_commanded():
    from gepetto_solvers.core.hands.allegro import spec

    assert spec.DRIVER_JOINT_ORDER == DRIVER_JOINT_NAMES


def test_allegro_digit_blocks_tile_the_driver_order_exactly():
    """Each digit's four joints are CONTIGUOUS in the driver's list, in our digit
    order. That is what lets the bridge write a digit's command vector into a
    slice rather than scattering it by name -- and it is a fact about Allegro's
    numbering (thumb 12-15, last) that happens to coincide with ours, so it is
    asserted rather than assumed."""
    from gepetto_solvers.core.hands.allegro import spec

    hand = hands.get_hand("allegro")
    per_digit = hand.driver_joint_names()

    flat = [joint for name in hand.digit_names for joint in per_digit[name]]
    assert flat == spec.DRIVER_JOINT_ORDER
    for name in hand.digit_names:
        assert len(per_digit[name]) == hand.actuation.n
        start = spec.DRIVER_JOINT_ORDER.index(per_digit[name][0])
        assert spec.DRIVER_JOINT_ORDER[start:start + hand.actuation.n] == \
            per_digit[name]


def test_the_allegro_plan_width_is_the_driver_joint_count():
    """16 numbers reach the hand, and a plan must carry exactly 16. A plan
    narrower than the driver's positional read is not a smaller command, it is a
    rejected message (the driver drops anything under 16)."""
    hand = hands.get_hand("allegro")
    assert (len(hand.digit_names) * len(hand.actuation.drive_indices)
            == len(DRIVER_JOINT_NAMES))
    assert robot_plan.command_kind(hand) == robot_plan.JOINT_POSITION_RAD
