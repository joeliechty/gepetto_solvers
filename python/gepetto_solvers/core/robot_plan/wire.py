"""A plan flattened to rectangular arrays, and back.

This is what a :class:`~.types.SolvePlan` looks like on a wire: no dataclasses, no
dicts of arrays, just lists and one flat float block. ``gepetto_ros``'s
``PlayPlan`` action carries exactly these fields, and its bridge and its executor
are a field-by-field copy onto and off these two functions.

WHY IT LIVES HERE AND NOT IN THE ROS PACKAGE. The encoding is pure numpy and has
nothing to do with ROS -- it is the same reshape whether it crosses an action, a
socket or a ``.npz``. Written inside the ROS package it would have to be written
TWICE (once in the client's ``_to_goal``, once in the server's ``_to_plan``), the
two copies would be the only check on each other, and neither could be tested
without a built workspace. Here there is one copy and
``tests/core/test_plan_wire.py`` proves the round trip is exact.

RECTANGULAR BY CONSTRUCTION. ``digit_cmd`` is row-major over
``(waypoint, digit, actuator)``. A digit missing from a waypoint goes out as zeros
rather than being dropped, because a ragged block has no representation here; a
plan whose digits genuinely differ between waypoints is a bug upstream and
:attr:`~.types.SolvePlan.dof_per_digit` raises on it before this is reached.
"""

import numpy as np

from .types import SolvePlan, Waypoint

#: Every key :func:`flatten_plan` produces and :func:`unflatten_plan` consumes.
#: Named so a marshaller can be written as a loop over this rather than as nine
#: hand-copied lines that can quietly lose one.
FIELDS = ("wrist_poses", "digit_names", "dof_per_digit", "command_kind",
          "digit_cmd", "notes", "corner_viz")


def flatten_plan(plan) -> dict:
    """``plan`` as plain arrays and lists. The inverse of :func:`unflatten_plan`.

    ``wrist_poses`` comes back as an ``(N, 4, 4)`` array rather than as poses:
    turning a 4x4 into whatever pose type the transport wants is the transport's
    job, and it is the one part of this that a non-ROS consumer would do
    differently.
    """
    names = list(plan.digit_names)
    width = plan.dof_per_digit
    n = len(plan.waypoints)

    block = np.zeros((n, len(names), width), float)
    for i, waypoint in enumerate(plan.waypoints):
        for j, name in enumerate(names):
            value = waypoint.digit_cmd.get(name)
            if value is not None:
                block[i, j] = value

    return {
        "wrist_poses": np.array([np.asarray(w.wrist_pose, float)
                                 for w in plan.waypoints],
                                float).reshape(n, 4, 4),
        "digit_names": names,
        "dof_per_digit": int(width),
        "command_kind": plan.command_kind,
        "digit_cmd": block.reshape(-1),
        "notes": [w.note for w in plan.waypoints],
        "corner_viz": np.asarray(plan.corner_viz, float).reshape(3),
    }


def unflatten_plan(fields) -> SolvePlan:
    """The exact inverse of :func:`flatten_plan`.

    ``open_lengths`` is NOT carried: it is the zero a tendon plan's displacements
    were measured FROM, and the executor never needs it -- it commands the
    displacements themselves. Leaving it off the wire keeps the goal from
    implying that a consumer could re-derive lengths from it, which it cannot
    without the same hand model the plan was built against.

    Raises rather than truncating on a block of the wrong size. That check is the
    whole reason the width and the digit names travel alongside the numbers: a
    silently reshaped block would drive the right motors with the wrong values.
    """
    names = list(fields["digit_names"])
    width = int(fields["dof_per_digit"])
    poses = np.asarray(fields["wrist_poses"], float).reshape(-1, 4, 4)
    notes = list(fields["notes"])
    flat = np.asarray(fields["digit_cmd"], float).ravel()

    expected = len(poses) * len(names) * width
    if flat.size != expected:
        raise ValueError(
            f"digit_cmd has {flat.size} values; {len(poses)} waypoints x "
            f"{len(names)} digits x {width} actuators needs {expected}")
    block = flat.reshape(len(poses), len(names), width)

    waypoints = [
        Waypoint(wrist_pose=poses[i],
                 digit_cmd={name: block[i, j] for j, name in enumerate(names)},
                 note=(notes[i] if i < len(notes) else f"waypoint {i}"))
        for i in range(len(poses))
    ]

    return SolvePlan(
        waypoints=waypoints,
        corner_viz=np.asarray(fields["corner_viz"], float).reshape(3),
        digit_names=names,
        open_lengths={},
        command_kind=str(fields["command_kind"]))
