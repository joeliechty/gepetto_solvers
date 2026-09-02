"""Turn a solve into a plan, and describe one.

``build_plan`` samples the solve: one :class:`Waypoint` per recorded Augmented
Lagrangian OUTER ITERATION -- the same snapshots the *Solve steps* scrubber
replays. Nothing is interpolated and nothing is timed.

THE ITERATES ARE OPTIMIZER ITERATIONS, NOT A PLANNED PATH. They converge to a
grasp; they do not promise to stay collision-free or monotonic on the way, and a
cold start with ``ik_settle_steps = 0`` visibly hyperextends before it recovers.
``source="final"`` exists for when you want the destination without the journey.
"""

from dataclasses import replace

import numpy as np

from .hardware import _drive_index, _hand, _solvers
from .se3 import _rotation_error
from .types import SolvePlan, Waypoint


def build_plan(result, configs, corner_viz, open_lengths, source="history",
               hand=None):
    """A :class:`SolvePlan` from a solved :class:`~.solvers.HandResult`.

    ``source``:
      ``"history"``  EVERY recorded AL outer iteration, in order, from the first.
      ``"final"``    the converged state alone, as a single waypoint.

    A result with no recorded iterates -- an FK pose, or a solve that never
    stepped -- yields the single final waypoint whatever ``source`` says, because
    there is no history to play.

    THERE IS DELIBERATELY NO ``start``. This used to take the convergence
    scrubber's index, to play "from where you are looking". That reads well and
    was silently useless: `_rebuild_iter_slider` opens the scrubber at the LAST
    iterate, so after any solve the index was ``n - 1``, the slice was
    ``range(n - 1, n)``, and "recorded path" meant one waypoint -- a single hop to
    the final pose with the whole trajectory dropped. The scrubber decides what is
    DRAWN; it does not decide what is played. Playing a tail is what the plan
    slicing in `robot_bridge._apply_resume` is for, and it is reached by being
    interrupted rather than by looking at a frame.

    ``configs`` is the solver's ``(name, cfg)`` list, needed by
    :func:`~.solvers.solved_wrist_pose` to recover the wrist the solve actually
    reached (the wrist is a variable, and contact moves it off the commanded pose).
    """
    n = result.num_iterates()
    if source == "final" or n <= 1:
        views = [result]
        notes = ["converged state"]
    else:
        views = [result.at_iterate(i) for i in range(n)]
        raw = result.iterate_notes
        notes = [(raw[i] if raw is not None and i < len(raw) else f"iterate {i}")
                 for i in range(n)]

    solved_wrist_pose = _solvers().solved_wrist_pose
    idx = _drive_index(_hand(hand))
    waypoints = []
    for view, note in zip(views, notes):
        lengths = view.displacements(0)
        waypoints.append(Waypoint(
            wrist_pose=np.asarray(solved_wrist_pose(configs, view.frames[0]), float),
            tendon_disp={name: open_lengths[name] - float(length[idx])
                         for name, length in zip(view.finger_names, lengths)
                         if name in open_lengths},
            note=note))

    return SolvePlan(waypoints=waypoints,
                     corner_viz=np.asarray(corner_viz, float).reshape(3),
                     finger_names=list(result.finger_names),
                     open_lengths=dict(open_lengths))


def prepend_current(plan, wrist_pose, tendon_disp):
    """Put the robot's CURRENT state on the front of the plan as waypoint 0.

    Without this the first tick is a step change. A plan's own first waypoint is
    not where the robot is: it is the solve's initial guess, which already carries
    whatever flexor tension the sliders were commanding (~2 mm of displacement on
    a default grasp scene) and a wrist at the commanded start pose. Playing it
    cold asks the hand to be somewhere it is not, instantly -- which the tendon
    node absorbs as one saturated ramp and the arm's resolved-rate loop absorbs as
    a burst of maximum twist.

    Prepending the measured state instead makes the approach an ordinary segment,
    so it gets a duration from the same speed ceilings as every other one and the
    hand moves onto the start of the solve at a controlled rate.

    ``tendon_disp`` may name only some of the digits (a finger that failed to read
    is left out of ``measured_state``); anything missing falls back to the plan's
    own first waypoint, i.e. that finger is assumed to be where the plan wants it
    and simply is not moved by the approach segment.
    """
    if not plan.waypoints:
        return plan
    first = plan.waypoints[0]
    current = Waypoint(
        wrist_pose=np.asarray(wrist_pose, float),
        tendon_disp={name: float(tendon_disp.get(name, value))
                     for name, value in first.tendon_disp.items()},
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
        tendon = max((abs(last.tendon_disp[name] - first.tendon_disp.get(name, 0.0))
                      for name in last.tendon_disp), default=0.0)
        lines.append(f"wrist {travel * 1e3:.0f} mm / {np.degrees(rotation):.0f}°, "
                     f"tendon up to {tendon * 1e3:.1f} mm")
    if samples:
        lines.append(f"{len(samples)} ticks, {samples[-1].t:.1f} s")
    return " &nbsp; ".join(lines)
