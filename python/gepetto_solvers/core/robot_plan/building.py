"""Turn a solve into a plan, and describe one.

``build_plan`` samples the solve: one :class:`Waypoint` per recorded Augmented
Lagrangian OUTER ITERATION -- the same snapshots the *Solve steps* scrubber
replays. Nothing is interpolated and nothing is timed.

THE ITERATES ARE OPTIMIZER ITERATIONS, NOT A PLANNED PATH. They converge to a
grasp; they do not promise to stay collision-free or monotonic on the way, and a
cold start with ``ik_settle_steps = 0`` visibly hyperextends before it recovers.
``source="final"`` exists for when you want the destination without the journey.

THE ONE BRANCH IN THIS PACKAGE lives here, in :func:`_digit_command`, and it is a
branch on what the hand's state MEANS rather than on which hand it is:

* a hand with the ``"displacement"`` feature has a length readout distinct from
  its actuation, and the robot is commanded on the CHANGE in that length --
  ``open_lengths[digit] - length[drive_index]``;
* a hand without it is commanded on its actuation directly, which for the
  joint-space hands is the solved joint vector in radians.

Everything downstream of this file works in whichever of the two the plan says it
is carrying, and never asks again.
"""

from dataclasses import replace

import numpy as np

from .hardware import _drive_index, _hand, _solvers
from .se3 import _rotation_error
from .types import (
    COMMAND_UNITS,
    JOINT_POSITION_RAD,
    TENDON_DISPLACEMENT_M,
    SolvePlan,
    Waypoint,
    as_command,
)


def command_kind(hand) -> str:
    """Which units :func:`build_plan` will write for ``hand``.

    Public because the ROS side has to declare the same thing on the wire before
    it has a plan in hand (the executor's driver is built at goal time), and
    deriving it in two places is how the two would come to disagree.
    """
    return (TENDON_DISPLACEMENT_M if "displacement" in hand.features
            else JOINT_POSITION_RAD)


def _digit_command(view, hand, open_lengths):
    """``{digit: (K,) command}`` for one solved state. See the module docstring."""
    if "displacement" in hand.features:
        idx = _drive_index(hand)
        return {name: as_command(open_lengths[name] - float(lengths[idx]))
                for name, lengths in zip(view.finger_names, view.displacements(0))
                if name in open_lengths}
    # The actuation variable IS the command on a joint-space hand: q, in radians,
    # as the solve left it. Read off the result rather than off the commanded
    # means (`hand.actuation_means`) for the reason `_report_actuation` gives --
    # past the first iterate the posture is the solver's, not the slider's, and
    # replaying the commanded means would play a path the solve never took.
    frames = view.frames[0]
    return {name: as_command(np.asarray(frames[name].actuation(), float))
            for name in view.finger_names}


def _solved_wrist(view, configs):
    """The wrist pose one solved state actually ended at, as a 4x4.

    The state bundle FIRST, because that is the hand-agnostic answer: each
    kinematics reports its own wrist, so nothing here has to know that the tendon
    hand's node 0 is not a variable. ``solved_wrist_pose`` is the fallback, and it
    is only usable at all on a hand whose configs carry ``hand_base_offset`` --
    the Allegro's ``DigitEnv`` does not, since its mounts live in the URDF, so
    that path raises rather than answering wrongly.
    """
    pose = view.wrist_pose(0)
    if pose is not None:
        return np.asarray(pose, float)
    return np.asarray(_solvers().solved_wrist_pose(configs, view.frames[0]), float)


def build_plan(result, configs, corner_viz, open_lengths=None, source="history",
               hand=None):
    """A :class:`SolvePlan` from a solved :class:`~.solvers.HandResult`.

    ``source``:
      ``"history"``  EVERY recorded AL outer iteration, in order, from the first.
      ``"final"``    the converged state alone, as a single waypoint.

    A result with no recorded iterates -- an FK pose, or a solve that never
    stepped -- yields the single final waypoint whatever ``source`` says, because
    there is no history to play.

    ``open_lengths`` is required for a hand with the ``"displacement"`` feature
    and ignored for one without: a joint-space plan commands positions, which have
    no hand-open zero to be measured from. It defaults to None rather than ``{}``
    so that omitting it on a tendon hand fails here, loudly, instead of silently
    building a plan with no digits in it.

    THERE IS DELIBERATELY NO ``start``. This used to take the convergence
    scrubber's index, to play "from where you are looking". That reads well and
    was silently useless: `_rebuild_iter_slider` opens the scrubber at the LAST
    iterate, so after any solve the index was ``n - 1``, the slice was
    ``range(n - 1, n)``, and "recorded path" meant one waypoint -- a single hop to
    the final pose with the whole trajectory dropped. The scrubber decides what is
    DRAWN; it does not decide what is played. Playing a tail is what the plan
    slicing in the ROS bridge's resume path is for, and it is reached by being
    interrupted rather than by looking at a frame.

    ``configs`` is the solver's ``(name, cfg)`` list, needed by
    :func:`~.solvers.solved_wrist_pose` to recover the wrist the solve actually
    reached (the wrist is a variable, and contact moves it off the commanded pose).
    """
    hand = _hand(hand)
    kind = command_kind(hand)
    if kind == TENDON_DISPLACEMENT_M and open_lengths is None:
        raise ValueError(
            f"build_plan: hand {hand.name!r} is commanded on tendon DISPLACEMENT, "
            f"so it needs open_lengths (see robot_plan.open_tendon_lengths).")
    open_lengths = open_lengths or {}

    n = result.num_iterates()
    if source == "final" or n <= 1:
        views = [result]
        notes = ["converged state"]
    else:
        views = [result.at_iterate(i) for i in range(n)]
        raw = result.iterate_notes
        notes = [(raw[i] if raw is not None and i < len(raw) else f"iterate {i}")
                 for i in range(n)]

    waypoints = [
        Waypoint(
            wrist_pose=_solved_wrist(view, configs),
            digit_cmd=_digit_command(view, hand, open_lengths),
            note=note)
        for view, note in zip(views, notes)
    ]

    return SolvePlan(waypoints=waypoints,
                     corner_viz=np.asarray(corner_viz, float).reshape(3),
                     digit_names=list(result.finger_names),
                     open_lengths=dict(open_lengths),
                     command_kind=kind)


def prepend_current(plan, wrist_pose, digit_cmd):
    """Put the robot's CURRENT state on the front of the plan as waypoint 0.

    Without this the first tick is a step change. A plan's own first waypoint is
    not where the robot is: it is the solve's initial guess, which already carries
    whatever the sliders were commanding (~2 mm of tendon displacement on a
    default grasp scene, or a whole pre-grasp posture on a joint hand) and a wrist
    at the commanded start pose. Playing it cold asks the hand to be somewhere it
    is not, instantly -- which an integrating hand node absorbs as one saturated
    ramp and the arm's resolved-rate loop absorbs as a burst of maximum twist.

    Prepending the measured state instead makes the approach an ordinary segment,
    so it gets a duration from the same speed ceilings as every other one and the
    hand moves onto the start of the solve at a controlled rate.

    ``digit_cmd`` may name only some of the digits (one that failed to read is
    left out of the measured state); anything missing falls back to the plan's own
    first waypoint, i.e. that digit is assumed to be where the plan wants it and
    simply is not moved by the approach segment. Scalars are accepted for a
    width-1 hand, so a caller with one number per finger need not wrap them.
    """
    if not plan.waypoints:
        return plan
    first = plan.waypoints[0]
    current = Waypoint(
        wrist_pose=np.asarray(wrist_pose, float),
        digit_cmd={name: as_command(digit_cmd.get(name, value))
                   for name, value in first.digit_cmd.items()},
        note="current robot state")
    return replace(plan, waypoints=[current] + list(plan.waypoints))


def summarize(plan, samples=None):
    """A one-paragraph markdown description of a plan, for the GUI status line."""
    lines = [f"**{len(plan.waypoints)} waypoint(s)**"]
    if len(plan.waypoints) >= 2:
        first, last = plan.waypoints[0], plan.waypoints[-1]
        travel = float(np.linalg.norm(last.wrist_pose[:3, 3] - first.wrist_pose[:3, 3]))
        rotation = float(np.linalg.norm(
            _rotation_error(last.wrist_pose[:3, :3], first.wrist_pose[:3, :3])))
        digit = max(
            (float(np.max(np.abs(value - first.digit_cmd[name])))
             for name, value in last.digit_cmd.items() if name in first.digit_cmd),
            default=0.0)
        suffix, scale = COMMAND_UNITS.get(plan.command_kind, ("", 1.0))
        channel = "tendon" if plan.command_kind == TENDON_DISPLACEMENT_M else "joint"
        lines.append(f"wrist {travel * 1e3:.0f} mm / {np.degrees(rotation):.0f}°, "
                     f"{channel} up to {digit * scale:.1f} {suffix}")
    if samples:
        lines.append(f"{len(samples)} ticks, {samples[-1].t:.1f} s")
    return " &nbsp; ".join(lines)
