"""How long each segment must take, given the speed ceilings.

A plan is a path through configuration space and says nothing about how fast to
walk it; this is where that decision is made. Each segment gets the duration its
slowest channel needs.
"""

import numpy as np

from .se3 import _rotation_error


def segment_durations(plan, max_linear, max_angular, max_tendon, min_duration=0.05):
    """How long each waypoint-to-waypoint segment must take, in seconds.

    The slowest channel sets the pace: a segment is only as fast as its linear
    travel, its rotation and its tendon travel each allow. That coupling is the
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
        tendon = max((abs(b.tendon_disp[name] - a.tendon_disp.get(name, 0.0))
                      for name in b.tendon_disp), default=0.0)
        durations.append(max(min_duration,
                             linear / max_linear if max_linear > 0 else 0.0,
                             angular / max_angular if max_angular > 0 else 0.0,
                             tendon / max_tendon if max_tendon > 0 else 0.0))
    return durations


def pacing_summary(plan, max_linear, max_angular, max_tendon, min_duration=0.05):
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
    counts = {"linear": 0, "angular": 0, "tendon": 0, "floor": 0}
    totals = {"linear": 0.0, "angular": 0.0, "tendon": 0.0, "duration": 0.0}
    for a, b in zip(plan.waypoints, plan.waypoints[1:]):
        linear = float(np.linalg.norm(b.wrist_pose[:3, 3] - a.wrist_pose[:3, 3]))
        angular = float(np.linalg.norm(
            _rotation_error(b.wrist_pose[:3, :3], a.wrist_pose[:3, :3])))
        tendon = max((abs(b.tendon_disp[name] - a.tendon_disp.get(name, 0.0))
                      for name in b.tendon_disp), default=0.0)
        needs = {
            "linear": linear / max_linear if max_linear > 0 else 0.0,
            "angular": angular / max_angular if max_angular > 0 else 0.0,
            "tendon": tendon / max_tendon if max_tendon > 0 else 0.0,
        }
        winner = max(needs, key=needs.get)
        counts[winner if needs[winner] > min_duration else "floor"] += 1
        totals["linear"] += linear
        totals["angular"] += angular
        totals["tendon"] += tendon
        totals["duration"] += max(min_duration, *needs.values())
    return {"counts": counts, "totals": totals,
            "segments": max(len(plan.waypoints) - 1, 0)}


def describe_pacing(plan, max_linear, max_angular, max_tendon, min_duration=0.05):
    """`pacing_summary` as one line, for an operator reading a log."""
    s = pacing_summary(plan, max_linear, max_angular, max_tendon, min_duration)
    c, t, n = s["counts"], s["totals"], s["segments"]
    if not n:
        return "single waypoint; nothing to pace"
    net = float(np.linalg.norm(plan.waypoints[-1].wrist_pose[:3, 3]
                               - plan.waypoints[0].wrist_pose[:3, 3]))
    # Arc length AND net displacement, because an optimizer history wanders: the
    # arm can travel far less arc than the reference and still arrive, so arc
    # alone reads as a tracking failure that is not there.
    return (f"wrist {t['linear'] * 1e3:.0f} mm of arc for {net * 1e3:.0f} mm net, "
            f"{np.degrees(t['angular']):.0f}deg of rotation, "
            f"{t['tendon'] * 1e3:.1f} mm of tendon; "
            f"paced by linear x{c['linear']} / angular x{c['angular']} / "
            f"tendon x{c['tendon']} / min_duration x{c['floor']} of {n} segments")
