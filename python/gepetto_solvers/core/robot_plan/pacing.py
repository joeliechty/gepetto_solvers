"""How long each segment must take, given the speed ceilings.

A plan is a path through configuration space and says nothing about how fast to
walk it; this is where that decision is made. Each segment gets the duration its
slowest channel needs.

THREE CHANNELS, AND THE THIRD IS WHATEVER DRIVES THE HAND. ``max_digit`` is a
ceiling on how fast any single driven actuator may move -- metres per second of
tendon on a tendon hand, radians per second of joint on a joint-space one. It is
the max over the whole per-digit vector rather than its norm, because each
actuator has its own motor and its own limit: a digit moving four joints by 0.1
rad each is not asking anything more of any one of them than a digit moving one.
"""

import numpy as np

from .se3 import _rotation_error
from .types import COMMAND_UNITS, TENDON_DISPLACEMENT_M


def _digit_travel(a, b):
    """The largest single-actuator move between two waypoints.

    A digit ``b`` names and ``a`` does not contributes NOTHING, because
    :func:`~.schedule._lerp` holds such a digit rather than ramping it in from an
    assumed zero. The two must agree: this function is what decides how long the
    segment takes, and timing travel that the interpolation never performs would
    stretch every such segment for a move that does not happen.
    """
    travel = 0.0
    for name, value in b.digit_cmd.items():
        previous = a.digit_cmd.get(name)
        if previous is None:
            continue
        travel = max(travel, float(np.max(np.abs(value - previous))))
    return travel


def segment_durations(plan, max_linear, max_angular, max_digit, min_duration=0.05):
    """How long each waypoint-to-waypoint segment must take, in seconds.

    The slowest channel sets the pace: a segment is only as fast as its linear
    travel, its rotation and its digit travel each allow. That coupling is the
    point -- running the wrist at full speed while the fingers lag would put the
    hand at the object with the grasp half closed.

    ``min_duration`` keeps a segment that barely moves from collapsing to zero
    ticks; consecutive AL iterates late in a converging solve differ by microns.
    """
    durations = []
    for a, b in zip(plan.waypoints, plan.waypoints[1:]):
        linear = float(np.linalg.norm(b.wrist_pose[:3, 3] - a.wrist_pose[:3, 3]))
        angular = float(np.linalg.norm(
            _rotation_error(b.wrist_pose[:3, :3], a.wrist_pose[:3, :3])))
        digit = _digit_travel(a, b)
        durations.append(max(min_duration,
                             linear / max_linear if max_linear > 0 else 0.0,
                             angular / max_angular if max_angular > 0 else 0.0,
                             digit / max_digit if max_digit > 0 else 0.0))
    return durations


def pacing_summary(plan, max_linear, max_angular, max_digit, min_duration=0.05):
    """Which channel decides each segment's duration, and what the path demands.

    Pure reporting -- `segment_durations` is the authority and this must never
    disagree with it. It exists because "the arm is too slow" and "the path is
    timed for a different channel entirely" look identical from the outside, and
    the difference decides whether the answer is a gain, a speed setting, or
    neither. A path whose every segment is paced by ROTATION cannot be made
    quicker by raising the linear ceiling, and one pinned at `min_duration` is
    not being paced by any ceiling at all.

    Returns a dict of counts and totals; see `describe_pacing` for the line.
    """
    counts = {"linear": 0, "angular": 0, "digit": 0, "floor": 0}
    totals = {"linear": 0.0, "angular": 0.0, "digit": 0.0, "duration": 0.0}
    for a, b in zip(plan.waypoints, plan.waypoints[1:]):
        linear = float(np.linalg.norm(b.wrist_pose[:3, 3] - a.wrist_pose[:3, 3]))
        angular = float(np.linalg.norm(
            _rotation_error(b.wrist_pose[:3, :3], a.wrist_pose[:3, :3])))
        digit = _digit_travel(a, b)
        needs = {
            "linear": linear / max_linear if max_linear > 0 else 0.0,
            "angular": angular / max_angular if max_angular > 0 else 0.0,
            "digit": digit / max_digit if max_digit > 0 else 0.0,
        }
        winner = max(needs, key=needs.__getitem__)
        counts[winner if needs[winner] > min_duration else "floor"] += 1
        totals["linear"] += linear
        totals["angular"] += angular
        totals["digit"] += digit
        totals["duration"] += max(min_duration, *needs.values())
    return {"counts": counts, "totals": totals,
            "segments": max(len(plan.waypoints) - 1, 0)}


def describe_pacing(plan, max_linear, max_angular, max_digit, min_duration=0.05):
    """`pacing_summary` as one line, for an operator reading a log.

    The digit channel is named and scaled off the plan's own ``command_kind``, so
    a joint-space run reads "0.8 rad of joint" rather than borrowing the tendon
    hand's millimetres.
    """
    s = pacing_summary(plan, max_linear, max_angular, max_digit, min_duration)
    c, t, n = s["counts"], s["totals"], s["segments"]
    if not n:
        return "single waypoint; nothing to pace"
    net = float(np.linalg.norm(plan.waypoints[-1].wrist_pose[:3, 3]
                               - plan.waypoints[0].wrist_pose[:3, 3]))
    kind = getattr(plan, "command_kind", TENDON_DISPLACEMENT_M)
    suffix, scale = COMMAND_UNITS.get(kind, ("", 1.0))
    channel = "tendon" if kind == TENDON_DISPLACEMENT_M else "joint"
    # Arc length AND net displacement, because an optimizer history wanders: the
    # arm can travel far less arc than the reference and still arrive, so arc
    # alone reads as a tracking failure that is not there.
    return (f"wrist {t['linear'] * 1e3:.0f} mm of arc for {net * 1e3:.0f} mm net, "
            f"{np.degrees(t['angular']):.0f}deg of rotation, "
            f"{t['digit'] * scale:.1f} {suffix} of {channel}; "
            f"paced by linear x{c['linear']} / angular x{c['angular']} / "
            f"{channel} x{c['digit']} / min_duration x{c['floor']} of {n} segments")
