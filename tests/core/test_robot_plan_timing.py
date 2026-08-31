"""Timing and sampling of a plan: `plan_schedule`, `sample_at`, `interpolate`.

Pure numpy -- no solver, no ROS, no hardware::

    pytest tests/core/test_robot_plan_timing.py

The property that matters most here is the LAST one: halving the speed ceilings
must stretch the schedule and leave the geometric path untouched. That is what
"the robot still interpolates fully between waypoints at any controller speed"
means, and it is the thing that was silently not true of the old executor.
"""

import numpy as np

from _pkg import robot_plan

SolvePlan = robot_plan.SolvePlan
Waypoint = robot_plan.Waypoint
_rotation_from_vector = robot_plan._rotation_from_vector
interpolate = robot_plan.interpolate
plan_schedule = robot_plan.plan_schedule
sample_at = robot_plan.sample_at
se3_exp = robot_plan.se3_exp
se3_log = robot_plan.se3_log
segment_durations = robot_plan.segment_durations

HZ = 100.0
PERIOD = 1.0 / HZ
SPEEDS = dict(max_linear=0.2, max_angular=0.4, max_tendon=0.0163)

FINGERS = ["index", "middle", "thumb"]


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _pose(xyz, axis=(0.0, 0.0, 1.0), angle=0.0):
    T = np.eye(4)
    T[:3, :3] = _rotation(axis, angle)
    T[:3, 3] = np.asarray(xyz, float)
    return T


def _plan(n=6):
    """A plan whose segments differ in which channel is the slowest -- long
    travel, pure rotation, and a segment that barely moves at all (the case
    `min_duration` exists for)."""
    rng = np.random.default_rng(7)
    waypoints = []
    for i in range(n):
        if i == 3:
            # Microns from its predecessor: a converged AL iterate.
            previous = waypoints[-1]
            waypoints.append(Waypoint(
                wrist_pose=_pose(previous.wrist_pose[:3, 3] + 1e-6),
                tendon_disp=dict(previous.tendon_disp),
                note=f"iterate {i}"))
            continue
        waypoints.append(Waypoint(
            wrist_pose=_pose(rng.uniform(-0.3, 0.3, 3),
                             axis=rng.uniform(-1.0, 1.0, 3),
                             angle=rng.uniform(-1.0, 1.0)),
            tendon_disp={name: float(rng.uniform(0.0, 0.015))
                         for name in FINGERS},
            note=f"iterate {i}"))
    return SolvePlan(waypoints=waypoints, corner_viz=np.zeros(3),
                     finger_names=list(FINGERS),
                     open_lengths={name: 0.1 for name in FINGERS})


# ---------------------------------------------------------------------------

def test_schedule_is_whole_control_periods():
    schedule = plan_schedule(_plan(), hz=HZ, **SPEEDS)
    for duration in schedule.durations:
        ticks = duration / PERIOD
        assert abs(ticks - round(ticks)) < 1e-9, duration
        assert ticks >= 1.0
    assert abs(schedule.total - sum(schedule.durations)) < 1e-9
    assert np.allclose(schedule.edges[1:] - schedule.edges[:-1],
                       schedule.durations)


def test_interpolate_is_a_walk_of_sample_at():
    """The one that keeps the two from ever drifting apart."""
    plan = _plan()
    schedule = plan_schedule(plan, hz=HZ, **SPEEDS)
    samples = interpolate(plan, hz=HZ, **SPEEDS)

    assert len(samples) == int(round(schedule.total / PERIOD)) + 1
    for i, sample in enumerate(samples[:-1]):
        direct = sample_at(plan, schedule, i * PERIOD)
        assert abs(sample.t - i * PERIOD) < 1e-9
        assert np.allclose(sample.wrist_pose, direct.wrist_pose)
        assert sample.waypoint == direct.waypoint

    # The final sample is the last waypoint, held, with no feed-forward.
    last = samples[-1]
    assert abs(last.t - schedule.total) < 1e-9
    assert np.allclose(last.wrist_pose, plan.waypoints[-1].wrist_pose)
    assert np.allclose(last.body_twist, 0.0)
    assert last.waypoint == len(plan.waypoints) - 1


def test_every_waypoint_is_landed_on():
    """No waypoint may be skipped: the pose at each segment edge IS the waypoint."""
    plan = _plan()
    schedule = plan_schedule(plan, hz=HZ, **SPEEDS)
    for k, edge in enumerate(schedule.edges):
        sample = sample_at(plan, schedule, float(edge))
        assert np.allclose(sample.wrist_pose, plan.waypoints[k].wrist_pose,
                           atol=1e-9), k


def test_sample_at_is_continuous_and_clamped():
    plan = _plan()
    schedule = plan_schedule(plan, hz=HZ, **SPEEDS)

    # Continuity across every interior segment boundary.
    for edge in schedule.edges[1:-1]:
        before = sample_at(plan, schedule, float(edge) - 1e-7)
        after = sample_at(plan, schedule, float(edge) + 1e-7)
        assert np.allclose(before.wrist_pose, after.wrist_pose, atol=1e-6)
        for name in FINGERS:
            assert abs(before.tendon_disp[name] - after.tendon_disp[name]) < 1e-9

    # Outside [0, total] is held, not extrapolated.
    assert np.allclose(sample_at(plan, schedule, -5.0).wrist_pose,
                       plan.waypoints[0].wrist_pose)
    assert np.allclose(sample_at(plan, schedule, schedule.total + 5.0).wrist_pose,
                       plan.waypoints[-1].wrist_pose)
    assert np.allclose(sample_at(plan, schedule, schedule.total + 5.0).body_twist,
                       0.0)


def test_feed_forward_matches_the_path_it_walks():
    """The twist handed to the controller must be the rate the target moves at.

    Flowing along the feed-forward for the segment's duration has to land on the
    next waypoint exactly; a twist computed from the unquantized duration misses
    by up to half a period per segment, which the arm then has to make up with
    error."""
    plan = _plan()
    schedule = plan_schedule(plan, hz=HZ, **SPEEDS)
    for k, duration in enumerate(schedule.durations):
        start = np.asarray(plan.waypoints[k].wrist_pose, float)
        landed = start @ se3_exp(schedule.body_twist[k] * duration)
        assert np.allclose(landed, plan.waypoints[k + 1].wrist_pose, atol=1e-9), k


def test_feed_forward_is_the_derivative_of_the_reference():
    """The property the separated lerp-plus-slerp reference did NOT have.

    A constant body twist is the exact derivative of `T_k @ se3_exp(V t)` at
    every instant, not just at the segment edges. Differentiating the sampled
    reference numerically must therefore reproduce the twist the same sample
    hands back -- otherwise the feed-forward is fighting the path it is meant to
    be riding, everywhere in between.
    """
    plan = _plan()
    schedule = plan_schedule(plan, hz=HZ, **SPEEDS)
    step = 1e-6
    for k, duration in enumerate(schedule.durations):
        for frac in (0.1, 0.37, 0.5, 0.9):
            t = schedule.edges[k] + frac * duration
            here = sample_at(plan, schedule, float(t))
            ahead = sample_at(plan, schedule, float(t) + step)
            # Body-frame finite difference: the twist that carries `here` to
            # `ahead`, per second.
            numeric = se3_log(np.linalg.inv(here.wrist_pose) @ ahead.wrist_pose) / step
            assert np.allclose(numeric, here.body_twist, atol=1e-6), (k, frac)


def test_se3_exp_log_round_trip():
    """The pair the whole reference is built on.

    Rotations are kept strictly inside pi: that is the domain the log is
    single-valued on, and beyond it a wrapped answer is correct rather than
    wrong. Segments between consecutive iterates are nowhere near it.
    """
    rng = np.random.default_rng(0)
    axis = np.array([0.3, 0.5, 0.81])
    axis = axis / np.linalg.norm(axis)
    for _ in range(500):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        xi = np.concatenate([rng.normal(size=3) * 0.4,
                             direction * rng.uniform(0.0, np.pi * 0.999)])
        assert np.allclose(se3_log(se3_exp(xi)), xi, atol=1e-9)

    # Tiny rotations are the regime consecutive iterates of a converged solve
    # live in, and the one a naive arccos/closed-form implementation gets wrong:
    # arccos loses half its digits as its argument approaches 1, and the
    # Jacobian coefficients are 0/0 there.
    for angle in (0.0, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3):
        xi = np.concatenate([np.array([0.1, -0.2, 0.35]), axis * angle])
        assert np.allclose(se3_log(se3_exp(xi)), xi, atol=1e-12), angle


def test_screw_segment_rotates_and_translates_as_one():
    """Simultaneous rotation and translation is where a screw and a separated
    lerp-plus-slerp genuinely differ, so it is worth pinning that the reference
    is the screw: the midpoint of a segment must be the half-twist, not the
    average of the endpoints."""
    a = np.eye(4)
    b = np.eye(4)
    b[:3, :3] = _rotation_from_vector(np.array([0.0, 0.0, np.pi / 2.0]))
    b[:3, 3] = np.array([1.0, 0.0, 0.0])
    plan = SolvePlan(waypoints=[Waypoint(a, {n: 0.0 for n in FINGERS}),
                                Waypoint(b, {n: 0.0 for n in FINGERS})],
                     corner_viz=np.zeros(3), finger_names=list(FINGERS),
                     open_lengths={name: 0.1 for name in FINGERS})
    schedule = plan_schedule(plan, hz=HZ, **SPEEDS)
    mid = sample_at(plan, schedule, schedule.total / 2.0)

    # The screw midpoint bows off the straight line between the endpoints.
    straight = 0.5 * (a[:3, 3] + b[:3, 3])
    assert not np.allclose(mid.wrist_pose[:3, 3], straight, atol=1e-3)
    # Two half-twists compose to the whole segment.
    half = se3_exp(schedule.body_twist[0] * (schedule.total / 2.0))
    assert np.allclose(a @ half @ half, b, atol=1e-9)


def test_single_waypoint_plan():
    plan = SolvePlan(waypoints=[_plan().waypoints[0]], corner_viz=np.zeros(3),
                     finger_names=list(FINGERS),
                     open_lengths={name: 0.1 for name in FINGERS})
    schedule = plan_schedule(plan, hz=HZ, **SPEEDS)
    assert schedule.durations == []
    assert schedule.total == 0.0

    samples = interpolate(plan, hz=HZ, **SPEEDS)
    assert len(samples) == 1
    assert np.allclose(samples[0].body_twist, 0.0)
    # Any t at all resolves to "go here".
    assert np.allclose(sample_at(plan, schedule, 12.0).wrist_pose,
                       plan.waypoints[0].wrist_pose)


def test_empty_plan():
    plan = SolvePlan(waypoints=[], corner_viz=np.zeros(3), finger_names=[],
                     open_lengths={})
    assert interpolate(plan, hz=HZ, **SPEEDS) == []


def test_speed_stretches_time_and_leaves_the_path_alone():
    """Halve every ceiling: the schedule doubles, the geometry is identical.

    This is the "fully interpolated at any speed" property. It is asserted PER
    SEGMENT rather than on a global fraction of the total, because segments
    pinned to `min_duration` are speed independent by definition -- halving the
    ceilings stretches the others around them, which genuinely reshapes where a
    given fraction of the total run falls. What must not change is the curve:
    the same fraction THROUGH A SEGMENT is the same pose at any speed.
    """
    plan = _plan()
    fast = plan_schedule(plan, hz=HZ, **SPEEDS)
    slow = plan_schedule(plan, hz=HZ,
                         **{k: v / 2.0 for k, v in SPEEDS.items()})

    # Not exactly 2x on segments pinned to min_duration -- those are speed
    # independent by definition -- so check per segment against which rule won.
    raw_fast = segment_durations(plan, min_duration=0.05, **SPEEDS)
    for k, (a, b) in enumerate(zip(fast.durations, slow.durations, strict=True)):
        if raw_fast[k] > 0.05 + 1e-9:
            assert abs(b - 2.0 * a) <= PERIOD, k     # quantization only
        else:
            assert b >= a, k
    assert slow.total > fast.total

    paired = zip(fast.durations, slow.durations, strict=True)
    for k, (quick, slower) in enumerate(paired):
        for s in np.linspace(0.0, 1.0, 21):
            here = sample_at(plan, fast, fast.edges[k] + s * quick)
            there = sample_at(plan, slow, slow.edges[k] + s * slower)
            assert np.allclose(here.wrist_pose, there.wrist_pose,
                               atol=1e-9), (k, s)
            for name in FINGERS:
                assert abs(here.tendon_disp[name]
                           - there.tendon_disp[name]) < 1e-9, (k, s, name)
