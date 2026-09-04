"""Waypoints plus speed ceilings -> a fixed-rate sample stream.

Positions lerp, rotations slerp, digit commands lerp. A segment is walked along
``T_k @ se3_exp(V * t)`` for a CONSTANT body twist V, which is what makes the
feed-forward handed to the controller the exact derivative of the reference at
every instant rather than only at the segment edges.
"""

import numpy as np

from .pacing import segment_durations
from .se3 import se3_exp, se3_log
from .types import PathSchedule, Sample


def plan_schedule(plan, hz=100.0, max_linear=0.2, max_angular=0.4,
                  max_digit=0.0163, min_duration=0.05):
    """Time ``plan`` at the given speed ceilings, quantized to the ``hz`` grid.

    Speed arguments are CEILINGS, not setpoints -- see :func:`segment_durations`.
    A plan of fewer than two waypoints has no segments and yields an empty
    schedule of zero duration, which :func:`sample_at` handles as "go here".
    """
    period = 1.0 / float(hz)
    raw = segment_durations(plan, max_linear, max_angular, max_digit,
                            min_duration)

    durations, twists = [], []
    for k, duration in enumerate(raw):
        a, b = plan.waypoints[k], plan.waypoints[k + 1]
        duration = max(1, int(round(duration / period))) * period
        durations.append(duration)
        # The body twist carrying a onto b in exactly `duration`. Relative pose
        # first, then the log: this is a screw, so the wrist rotates and
        # translates as one motion rather than as two independently interpolated
        # channels that only agree at the ends.
        relative = np.linalg.inv(np.asarray(a.wrist_pose, float)) @ \
            np.asarray(b.wrist_pose, float)
        twists.append(se3_log(relative) / duration)

    edges = np.concatenate([[0.0], np.cumsum(durations)]) if durations \
        else np.zeros(1)
    return PathSchedule(durations=durations, edges=edges,
                        total=float(edges[-1]), body_twist=twists)


def _lerp(start, end, s):
    """One digit's command a fraction ``s`` along a segment.

    ``start`` is None for a digit the segment's first waypoint does not name --
    a partial readback, or a plan whose digits changed mid-path. Zero is the
    wrong answer there (it would ramp the finger in from fully open), so the
    segment simply starts where it ends: that digit is held, not moved.
    """
    end = np.asarray(end, float)
    if start is None:
        return end.copy()
    start = np.asarray(start, float)
    return start + s * (end - start)


def sample_at(plan, schedule, t):
    """Where the robot should be at time ``t`` along ``schedule``.

    ``t`` is clamped to ``[0, schedule.total]``: before the start is the first
    waypoint, at or past the end is the last one held with ZERO feed-forward,
    which is what the terminal hold wants to command.

    The wrist reference is ``T_k @ se3_exp(V * elapsed)`` -- the flow of the
    segment's constant body twist. At ``elapsed == duration`` that is exactly
    waypoint ``k+1`` by construction of ``V``, so no waypoint is ever missed, and
    at every instant between, the twist handed back IS the derivative of the pose
    handed back. Interpolating position and rotation separately would break that
    second property: the true body twist of a lerp-plus-slerp path varies along
    the segment, so a constant feed-forward would be subtly wrong everywhere
    except the ends.
    """
    if not plan.waypoints:
        raise ValueError("empty plan has no samples")

    if not schedule.durations:
        w = plan.waypoints[0]
        return Sample(0.0, np.asarray(w.wrist_pose, float),
                      {name: value.copy() for name, value in w.digit_cmd.items()},
                      np.zeros(6), 0)

    t = float(np.clip(t, 0.0, schedule.total))
    # -1 because searchsorted returns the insertion point; clipped to the last
    # segment so t == total lands on the end of it rather than off the end.
    k = int(np.clip(np.searchsorted(schedule.edges, t, side="right") - 1,
                    0, len(schedule.durations) - 1))
    duration = schedule.durations[k]
    elapsed = float(np.clip(t - schedule.edges[k], 0.0, duration))
    s = elapsed / duration

    a, b = plan.waypoints[k], plan.waypoints[k + 1]
    twist = schedule.body_twist[k]
    T = np.asarray(a.wrist_pose, float) @ se3_exp(twist * elapsed)

    # Held at the end, not still travelling: a feed-forward past the last
    # waypoint would walk the arm straight through it.
    at_end = t >= schedule.total
    return Sample(
        t=t,
        wrist_pose=T,
        # Digit commands stay a plain linear ramp in their own coordinate --
        # they are a point in R^K, not a pose, so there is no manifold to respect
        # and theta(t) = theta_k + theta_dot * t is already exact. That holds for
        # a joint vector exactly as it did for a scalar tendon displacement,
        # which is why this widened without changing.
        digit_cmd={name: _lerp(a.digit_cmd.get(name), value, s)
                   for name, value in b.digit_cmd.items()},
        body_twist=(np.zeros(6) if at_end else np.asarray(twist, float)),
        waypoint=k + 1)


def interpolate(plan, hz=100.0, max_linear=0.2, max_angular=0.4, max_digit=0.0163,
                min_duration=0.05):
    """Time the plan and sample it at ``hz``, with feed-forward rates.

    Speed arguments are CEILINGS, not setpoints -- see :func:`segment_durations`.
    Their defaults are the fractions the visualizer opens on for the TENDON hand:
    50% of MoveIt Servo's ``scale.linear``/``scale.rotational`` (0.4 m/s,
    0.8 rad/s) and 25% of ``HandConfig.max_tendon_speed`` (0.065 m/s). A
    joint-space hand's ``max_digit`` is in rad/s and has nothing to do with that
    default, so it passes its own.

    A single-waypoint plan yields one sample with zero velocity: "go here", which
    is what a resolved-rate controller needs to servo to a static target.

    This is now a walk of :func:`sample_at` over the fixed grid rather than its
    own copy of the interpolation -- the last tick of each segment still falls
    out naturally as the first tick of the next, and the final waypoint is still
    emitted exactly once, at ``schedule.total``.
    """
    if not plan.waypoints:
        return []

    schedule = plan_schedule(plan, hz, max_linear, max_angular, max_digit,
                             min_duration)
    if not schedule.durations:
        return [sample_at(plan, schedule, 0.0)]

    period = 1.0 / float(hz)
    ticks = int(round(schedule.total / period))
    samples = [sample_at(plan, schedule, i * period) for i in range(ticks)]
    samples.append(sample_at(plan, schedule, schedule.total))
    return samples
