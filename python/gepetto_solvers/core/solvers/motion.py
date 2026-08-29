"""Open-loop motions that are not solves: closing the hand, and lifting it.

Both drive :class:`HandFKSolver` in small steps and rely on its warm start
carrying the hand across each one, so both check that it actually did rather
than trusting it.
"""

from dataclasses import replace

import numpy as np

from .capabilities import FLEXOR_IDX
from .frames import solved_wrist_pose

# ---------------------------------------------------------------------------
# Phase 4: the synchronized close.
# ---------------------------------------------------------------------------
#
# Phases 0-2 shut the fingers as a SIDE EFFECT: the object equality pulls each
# fingertip onto the surface and the flexor tension is whatever the optimizer
# had to spend getting it there, so the digits arrive one at a time, in whatever
# order the AL iteration happened to move them. Phase 4 is the opposite kind of
# thing -- a commanded close, on a schedule, every grasping finger held at the
# SAME fraction of its own travel at every instant.
#
# "In sync" is measured in TENDON DISPLACEMENT, not in tension, because
# displacement is what the hardware is commanded in (robot_plan's whole sign
# convention) and because the two are not interchangeable:
#
#   * The same pull buys different travel on different digits -- at 2.0 N the
#     index has taken in 5.8 mm, the middle 7.4 mm and the thumb 10.2 mm. A
#     shared tension ramp is only ACCIDENTALLY synchronized.
#   * The thumb reaches its hardware stop far sooner in tension than the others
#     (~8.0 mm of travel, but reached around 1.7 N against the index's 2.5 N), so
#     a tension ramp pushed far enough to shut the hand runs the thumb into
#     `robot_plan.clamp_to_travel`'s stop, where it sits still while the rest
#     keep closing. That failure appears only on the ROBOT -- the model has no
#     motor and happily draws the thumb past its stop.
#
# So the ramp is PLANNED in displacement and EXECUTED in tension, and the map
# between them is built as the close walks rather than tabulated up front. Two
# properties of the FK solver make that the only workable order:
#
#   * It warm-starts from the previous posture, and a big jump does not fully
#     unwind: a solve at 3.0 N followed by one back at 0.84 N returns a hand
#     1.3 mm more flexed than the same tensions reached gradually, because the
#     inner solve hits its 500-iteration cap (measured error 3.3e4, i.e. not
#     converged) instead of walking back. Sweeping the curve first and THEN
#     ramping from the bottom is exactly that jump, and it cost 3.9 mm of
#     tracking error in the first draft of this function.
#   * Walked gradually and upward, the same solver converges in one or two
#     iterations a step.
#
# Hence a monotone secant walk: hold a local slope d(T) per finger, predict the
# tension that lands the next displacement target, pose, and correct from the
# miss. Tension never decreases, every step is small, and a step that comes back
# wrong (a non-converged solve) is re-posed by the correction loop at nearly the
# same tension -- which is the very thing that unwinds it. The recorded poses go
# back as an ordinary HandResult with `iterates` filled, which is what lets the
# Solve-steps scrubber and the Robot folder's history playback handle a close
# with no plumbing of their own.

#: How many FK poses a close is recorded at, not counting the starting pose.
#: These become the plan's waypoints, and `robot_plan.sample_at` lerps tendon
#: displacement WITHIN a segment, so the count is about how faithfully the
#: curved tension schedule is followed, not about how smooth playback is -- the
#: interpolator fills the rest in at its own control rate either way.
CLOSE_STEPS = 12


#: How far into the travel each grasping finger HAS LEFT the close goes, by
#: default. Not 1.0: the last of the travel is where `clamp_to_travel` and the
#: servo node's own saturation live, and a close ending exactly on the stop
#: spends its final waypoints commanding a position the motors cannot reach.
CLOSE_FRACTION = 0.9


#: Tension step (N) used to measure the starting slope of each finger's
#: displacement curve. Small enough to be a local slope, large enough that the
#: resulting length change is well clear of solver noise (~0.4 mm on the index).
CLOSE_PROBE_STEP = 0.1


#: How close to its commanded displacement a finger must land, in metres, before
#: the walk accepts a step. 0.2 mm -- under half the tolerance
#: `_tensions_for_displacement` recovers the robot's measured state to, and well
#: under the ~1 mm the tendon nodes resolve.
CLOSE_TOL_M = 2e-4


#: How many secant corrections a single step may take before it is accepted as
#: it stands. The curve is nearly straight over one step, so one correction
#: normally does it; the cap is what stops a finger that has run out of curve
#: (already at the tension ceiling) from spinning.
CLOSE_REFINE = 3


#: How far a phase-5 lift raises the wrist, in metres of world +Z.
LIFT_HEIGHT_M = 0.15


#: How many FK poses a lift is recorded at, not counting the starting pose.
#: What matters here is the SIZE OF ONE STEP, not the total: 0.15 m over 12
#: steps is 12.5 mm a step, well under `HandFKSolver._WARM_START_MAX_POS_M`
#: (50 mm), so every pose warm-starts off the one before instead of rebuilding
#: the graph -- and, more to the point, the hand is MOVED rather than dragged.
#: A step past that bound is not merely slower: a warm start re-aims the wrist
#: prior while leaving every rod node where it was, so the optimizer has to haul
#: the whole hand across the gap and can land short of the commanded pose
#: without saying so. Raising the height without raising the steps is the way
#: to walk into that.
LIFT_STEPS = 12


def synchronized_close(fk_solver, open_lengths, fingers, travel,
                       fraction=CLOSE_FRACTION, steps=CLOSE_STEPS,
                       tension_ceiling=3.0, on_progress=None, should_stop=None):
    """Close ``fingers`` together, and hand back every pose along the way.

    Returns ``(result, notes)``. ``result`` is a :class:`HandResult` whose
    ``frames`` are the closed hand and whose ``iterates`` are the whole ramp,
    starting from the pose the hand is in NOW -- the same shape a stepped IK
    solve returns, so the iterate scrubber and ``robot_plan.build_plan``'s
    ``source="history"`` read it without knowing a close from a solve. ``notes``
    are markdown lines for the caller to print, including the MEASURED
    synchronization error, which is the number that says whether this function
    did what its name claims.

    ``open_lengths`` is ``robot_plan.open_tendon_lengths``' output -- the zero
    every displacement here is measured from -- taken as an argument rather than
    recomputed so the caller's cached copy is the only one in play. ``fingers``
    names the digits to close; every other digit is HELD at the tension it is
    already carrying. ``travel`` maps a finger to the flexion displacement it may
    not pass: pass the upper bounds of ``robot_plan.hardware_travel_limits()``.
    A digit missing from it is closed to the furthest the tension ceiling can
    take it instead, which is the model's limit rather than the motors', so the
    plan will meet ``clamp_to_travel`` on the way out -- fine for looking at,
    wrong for driving, and said so in the notes.

    The close ends at ``fraction`` of the travel each finger has LEFT, so a digit
    already half shut moves half as far as one still open and both finish at the
    same moment. Progress is the shared quantity: at ramp fraction ``s`` every
    closing finger is commanded to ``s`` of its own displacement target. What is
    REPORTED is how well that landed -- the worst spread, over the whole ramp,
    between the most and least advanced finger, as a percentage of the close.

    ``should_stop`` is polled between FK solves (the e-stop). A stop KEEPS
    everything solved so far and returns it as a shorter close rather than
    discarding it: a half-shut hand is a real state, and the caller may well want
    to play the part that ran.
    """
    params = fk_solver.params
    names = list(fk_solver.finger_names)
    index_of = {name: i for i, name in enumerate(names)}
    closing = [name for name in names if name in set(fingers)]
    if not closing:
        raise ValueError(
            "nothing to close -- no grasping finger is both selected and known "
            f"to the solver (selected {sorted(set(fingers))}, solver has {names})")
    missing = sorted(name for name in closing if name not in open_lengths)
    if missing:
        raise ValueError(
            f"no open-hand tendon length for {', '.join(missing)} -- every "
            f"displacement in a close is measured from it")

    # The tensions the hand is being pulled with right now. Digits outside the
    # closing set keep theirs for the whole ramp, and the closing ones only ever
    # go UP from here (see the clamp in the walk below).
    held = [float(t) for t in params.flexor_tensions]
    ceiling = float(tension_ceiling)
    steps = max(1, int(steps))
    travel = dict(travel or {})
    notes = []

    def _pose(tensions):
        """One FK pose, plus what each closing tendon had taken in at it."""
        params.flexor_tensions = list(tensions)
        res = fk_solver.solve()
        lengths = dict(zip(res.finger_names, res.tendon_lengths(0)))
        return res, {name: open_lengths[name] - float(lengths[name][FLEXOR_IDX])
                     for name in closing}

    # Where the close starts: the pose the sliders describe, kept as iterate 0 so
    # the recorded ramp begins at the hand the user is already looking at rather
    # than a step into the close.
    if on_progress is not None:
        on_progress("close: reading the starting pose")
    start_res, start = _pose(held)

    # -- where each finger is going --
    #
    # Done before anything moves, because it decides which digits the probe below
    # is allowed to nudge.
    targets, unbounded = {}, []
    for name in closing:
        if name in travel:
            stop = float(travel[name])
        else:
            # No motor limit for this digit. Closing it to the model's own reach
            # is the only thing left, and the tension ceiling is where that is --
            # resolved lazily in the walk, by simply letting it run to `ceiling`.
            stop = None
            unbounded.append(name)
        span = None if stop is None else stop - start[name]
        if span is not None and span <= 0.0:
            notes.append(
                f"**{name} is not closing**: it is already at "
                f"{start[name] * 1e3:.1f} mm, at or past the {stop * 1e3:.1f} mm "
                f"of travel it has -- it is held where it is while the rest close")
            targets[name] = start[name]
        elif span is None:
            # Marked with None and resolved on the first probe, below.
            targets[name] = None
        else:
            targets[name] = start[name] + float(fraction) * span
    if unbounded:
        notes.append(
            f"*no hardware travel limit for {', '.join(sorted(unbounded))}* -- "
            f"closed to the model's own reach at {ceiling:g} N instead, which "
            f"the motors may not have; the Robot folder will clamp it")

    moving = [name for name in closing if targets[name] != start[name]]
    if not moving:
        notes.append("**nothing moved** -- every selected finger is already shut")
        return replace(start_res, iterates=[start_res.frames],
                       iterate_states=[start_res.states],
                       iterate_notes=["close 0% -- starting pose"]), notes

    # -- the starting slope of each finger's displacement curve --
    #
    # One small upward nudge, measured on the digits that are actually going
    # somewhere. Everything after this is a secant update off the step before.
    if on_progress is not None:
        on_progress("close: measuring tendon travel per newton")
    probe = list(held)
    for name in moving:
        probe[index_of[name]] = min(held[index_of[name]] + CLOSE_PROBE_STEP,
                                    ceiling)
    _res, probed = _pose(probe)
    slope = {}
    for name in moving:
        rise = probed[name] - start[name]
        run = probe[index_of[name]] - held[index_of[name]]
        if run <= 0.0 or rise <= 0.0:
            raise RuntimeError(
                f"{name} took in no tendon for {run:.3g} N more flexor tension "
                f"({rise * 1e3:+.3f} mm) -- the close has no direction to walk "
                f"in. Check that its tension slider is not already at the "
                f"{ceiling:g} N ceiling.")
        slope[name] = rise / run
    # Any digit with no motor limit gets its target now: the probe fixed a slope,
    # so the reach at the ceiling is a straight-line extrapolation from here. It
    # is only a starting guess -- the walk measures the real thing as it goes --
    # but it is enough to space the ramp by.
    for name in unbounded:
        if targets[name] is None:
            reach = start[name] + slope[name] * (ceiling - held[index_of[name]])
            targets[name] = start[name] + float(fraction) * (reach - start[name])

    # -- walk one shared progress fraction from 0 to 1 --
    results, iterate_notes = [start_res], ["close 0% -- starting pose"]
    tension = {name: held[index_of[name]] for name in closing}
    reached = dict(start)
    worst_spread, worst_miss, stopped = 0.0, 0.0, False
    for k in range(1, steps + 1):
        if should_stop is not None and should_stop():
            stopped = True
            notes.append(
                f"**stopped** {k - 1}/{steps} of the way through the close -- "
                f"the poses already recorded are kept, so the scrubber and the "
                f"Robot folder can still play the part that ran")
            break
        s = k / float(steps)
        want = {name: start[name] + s * (targets[name] - start[name])
                for name in closing}
        if on_progress is not None:
            on_progress(f"close: {s * 100:.0f}% ({k}/{steps})")
        was = dict(tension)

        # Predict, pose, correct. The clamp to [tension so far, ceiling] is what
        # keeps the walk monotone: a correction is never allowed to pull the hand
        # back open, which is the move the warm start cannot undo.
        got, res = reached, None
        for _attempt in range(CLOSE_REFINE + 1):
            command = list(held)
            for name in closing:
                if name in slope:
                    step_up = (want[name] - got[name]) / slope[name]
                    tension[name] = float(np.clip(tension[name] + step_up,
                                                  tension[name], ceiling))
                command[index_of[name]] = tension[name]
            res, got = _pose(command)
            if max(abs(got[name] - want[name]) for name in moving) <= CLOSE_TOL_M:
                break
        results.append(res)

        # Update each secant off the step just accepted, ignoring an update that
        # is not a sane slope (a step that moved no tension, or a solve that came
        # back non-monotone): the previous slope is a better guess than a bad new
        # one, and the correction loop above absorbs the difference anyway.
        for name in moving:
            run, rise = tension[name] - was[name], got[name] - reached[name]
            if run > 1e-6 and rise > 0.0:
                slope[name] = rise / run
        reached = got

        # How far apart the fingers actually ARE, as a fraction of the close.
        progress = {name: (got[name] - start[name]) / (targets[name] - start[name])
                    for name in moving}
        worst_spread = max(worst_spread,
                           max(progress.values()) - min(progress.values()))
        worst_miss = max(worst_miss,
                         max(abs(got[name] - want[name]) for name in moving))
        iterate_notes.append(
            "close {:.0f}%  \n".format(s * 100)
            + ", ".join(f"{name} {got[name] * 1e3:.1f} mm" for name in closing))

    travelled = ", ".join(f"{name} {(reached[name] - start[name]) * 1e3:+.1f} mm"
                          for name in closing)
    notes.append(f"**closed** in {len(results) - 1} steps: {travelled}")
    notes.append(
        f"in sync to {worst_spread * 100:.1f}% of the close (the worst gap "
        f"between the most and least advanced finger at any step); every finger "
        f"landed within {worst_miss * 1e3:.2f} mm of the displacement it was "
        f"commanded to")

    # One HandResult carrying the whole ramp: the LAST pose as the state, every
    # pose as an iterate. `replace` off a real FK result rather than building one
    # from scratch, so the spec, object pose, tip radii and contact masks are
    # exactly the ones the scene produced and cannot drift from them.
    result = replace(results[-1],
                     iterates=[res.frames for res in results],
                     iterate_states=[res.states for res in results],
                     iterate_notes=iterate_notes)
    return result, notes


def lift_wrist(fk_solver, height=LIFT_HEIGHT_M, steps=LIFT_STEPS,
               on_progress=None, should_stop=None):
    """Raise the wrist ``height`` metres along world +Z, recording every pose.

    Phase 5, and the mirror image of :func:`synchronized_close`: that one walks
    the tendons and never touches the wrist, this one walks the wrist and never
    touches a tendon. The flexor tensions are left exactly as they are found, so
    a lift carries whatever grasp the close ended on -- and every finger outside
    it -- up unchanged.

    Returns ``(result, notes)`` in the close's shape. ``result`` is a
    :class:`HandResult` whose ``frames`` are the raised hand and whose
    ``iterates`` are the whole ramp starting from the pose the hand is in NOW,
    so the iterate scrubber and ``robot_plan.build_plan``'s ``source="history"``
    read it without knowing a lift from a solve -- ``build_plan`` takes the
    wrist off each iterate with :func:`solved_wrist_pose`, so the arm follows
    the ramp with no plumbing of its own. ``notes`` are markdown lines for the
    caller, including the MEASURED rise, which is the number that says whether
    this function did what its name claims.

    Straight up in the WORLD frame, not along any axis of the hand: only
    ``T[2, 3]`` moves, and the orientation the hand is holding is carried
    through untouched. Nothing here is a solve -- no constraint is enforced, and
    in particular nothing in the model holds the object being lifted.

    The rise is split into ``steps`` so each one stays inside the FK solver's
    warm-start bound (see :data:`LIFT_STEPS`). ``should_stop`` is polled between
    poses (the e-stop); a stop KEEPS everything solved so far and returns it as
    a shorter lift, like the close. On a stop or a failed solve, the params are
    left holding the last pose that actually came back, so they never describe a
    hand that was never drawn.
    """
    params = fk_solver.params
    T0 = np.array(params.wrist_pose, float)
    height = float(height)
    steps = max(1, int(steps))
    rise = height / steps
    notes = []

    def _pose(z):
        """One FK pose with the wrist commanded to height ``z``."""
        T = T0.copy()
        T[2, 3] = z
        params.wrist_pose = T
        return fk_solver.solve()

    def _wrist_z(res):
        """Where the wrist ACTUALLY landed -- the solved pose, not the command."""
        return float(solved_wrist_pose(fk_solver.configs, res.frames[0])[2, 3])

    if on_progress is not None:
        on_progress("lift: reading the starting pose")
    start_res = _pose(T0[2, 3])
    start_z = _wrist_z(start_res)

    if rise <= 0.0:
        notes.append("**nothing to lift** -- the height is zero")
        return replace(start_res, iterates=[start_res.frames],
                       iterate_states=[start_res.states],
                       iterate_notes=["lift 0% -- starting pose"]), notes

    # Said once, here, rather than left to be discovered as a hand that does not
    # quite reach: a caller is free to pass a height and a step count that put
    # the per-step move past what a warm start can carry.
    # Deferred: fk imports base, so a module-level import here would be
    # circular. This is the only reference either way.
    from .fk import HandFKSolver

    if rise > HandFKSolver._WARM_START_MAX_POS_M:
        notes.append(
            f"*{rise * 1e3:.0f} mm per step is past the {HandFKSolver._WARM_START_MAX_POS_M * 1e3:.0f} "
            f"mm a warm start carries* -- each pose rebuilds from cold, so the "
            f"lift is slower but no less correct; raise the step count to avoid it")

    results = [start_res]
    iterate_notes = ["lift 0% -- starting pose"]
    reached = T0[2, 3]
    for k in range(1, steps + 1):
        if should_stop is not None and should_stop():
            notes.append(
                f"**stopped** {k - 1}/{steps} of the way up -- the poses already "
                f"recorded are kept, so the scrubber and the Robot folder can "
                f"still play the part that ran")
            break
        z = T0[2, 3] + k * rise
        if on_progress is not None:
            on_progress(f"lift: {k / steps * 100:.0f}% ({k}/{steps}) "
                        f"-- {k * rise * 1e3:.0f} mm up")
        try:
            res = _pose(z)
        except Exception:
            # Back to the last pose that came back, so params, the drawn hand
            # and the panel's sliders cannot disagree about where the wrist is.
            T_last = T0.copy()
            T_last[2, 3] = reached
            params.wrist_pose = T_last
            raise
        results.append(res)
        reached = z
        iterate_notes.append(f"lift {k / steps * 100:.0f}%  \n"
                             f"{(z - T0[2, 3]) * 1e3:.0f} mm up")

    # The commanded rise is the easy number and the useless one -- an FK solve
    # that stalls short of its prior is exactly the failure this phase can have.
    # HandFKSolver.solve already refuses a solve that misses by more than a
    # millimetre, so this is a readout rather than a check, but it is the readout
    # that says the hand went where it was sent.
    measured = _wrist_z(results[-1]) - start_z
    notes.append(
        f"**lifted** in {len(results) - 1} steps: {measured * 1e3:+.1f} mm "
        f"measured at the wrist ({(reached - T0[2, 3]) * 1e3:+.1f} mm commanded)")

    result = replace(results[-1],
                     iterates=[res.frames for res in results],
                     iterate_states=[res.states for res in results],
                     iterate_notes=iterate_notes)
    return result, notes
