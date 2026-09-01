"""Headless self-checks of the solver-driving half of the app.

No viser, no browser, no hardware. These are the only coverage :class:`HandVizApp`
has, so they are wired into the test suite as well as reachable via ``--smoke``.
"""


import numpy as np

from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.geometry.scene import GRASP_FLEXOR_TENSION, table_corner
from gepetto_solvers.core.hands import get_hand
from gepetto_solvers.core.solvers import (
    HandFKSolver,
    HandIKStepper,
    HandSolveParams,
    apply_phase_preset,
    capabilities,
    disc_frame_error,
    disc_pose,
    euler_to_R,
    lift_wrist,
    resolve_scene,
    resolve_table_origin,
    solved_wrist_pose,
    synchronized_close,
    wrist_pose_for_disc_target,
    wrist_to_disc,
)

from .constants import (
    _CAL_ARTICULATED_MIN_MM,
    _CAL_RIGID_TOL_MM,
    _CAL_SMOKE_POS_MM,
    _CAL_SMOKE_ROT_DEG,
    _CLOSE_SYNC_TOL,
    _CLOSE_TRACK_TOL_M,
    _LIFT_ARRIVE_TOL_M,
    _LIFT_RIGID_TOL_M,
    CAL_DEFAULT_DISC,
    CAL_REFINE_PASSES,
)


def _smoke_close():
    """Check phase 4: that a synchronized close actually closes IN SYNC.

    The whole point of the phase is a claim about several fingers at once, and
    the claim is cheap to check and easy to break -- it rests on the FK solver
    warm-starting well over small upward tension steps, which is a property of
    the binding, not of this file. So it is measured here rather than trusted:
    re-derive each recorded pose's tendon displacements from the result itself,
    and fail on the worst disagreement between digits.
    """
    print("Smoke-testing the phase-4 synchronized close...")
    hand = get_hand()
    digits = list(hand.digit_names)
    drive = hand.actuation.drive_indices[0]
    fingers = list(hand.default_contact_digits)

    params = apply_phase_preset(HandSolveParams(), "phase4")
    _passive, open_flexors = robot_plan.open_pose_tensions()
    # From the calibrated OPEN hand, which is where the Close button starts from
    # on a freshly opened app and the only starting pose with a fixed meaning.
    params.flexor_tensions = [open_flexors.get(name, GRASP_FLEXOR_TENSION)
                              for name in digits]
    solver = HandFKSolver(params, hand)
    open_lengths = robot_plan.open_tendon_lengths(params, solver)
    limits = robot_plan.hardware_travel_limits()
    travel = None if limits is None else {name: hi
                                          for name, (_lo, hi) in limits.items()}

    result, notes = synchronized_close(solver, open_lengths, fingers, travel)
    n = result.num_iterates()

    # Progress per finger at every recorded pose, as a fraction of its own close.
    # Read off the ITERATES rather than off what the walk reported, so a bug in
    # the reporting cannot make this pass.
    disp = []
    for i in range(n):
        lengths = dict(zip(result.finger_names, result.at_iterate(i).tendon_lengths(0)))
        disp.append({name: open_lengths[name] - float(lengths[name][drive])
                     for name in fingers})
    span = {name: disp[-1][name] - disp[0][name] for name in fingers}
    worst_sync, worst_track = 0.0, 0.0
    for i, row in enumerate(disp):
        progress = [(row[name] - disp[0][name]) / span[name] for name in fingers]
        worst_sync = max(worst_sync, max(progress) - min(progress))
        # ...and against the schedule the walk was supposed to follow: pose i of
        # n-1 is i/(n-1) of the way through.
        want = i / float(n - 1)
        worst_track = max(worst_track,
                          max(abs(p - want) * span[name]
                              for p, name in zip(progress, fingers)))

    synced = worst_sync <= _CLOSE_SYNC_TOL
    tracked = worst_track <= _CLOSE_TRACK_TOL_M
    closed = n > 1 and all(v > 0.0 for v in span.values())
    ok = synced and tracked and closed
    print(f"  [   close] poses={n} "
          f"travel={', '.join(f'{k} {v * 1e3:+.1f}' for k, v in span.items())} mm "
          f"[{'ok' if ok else 'BAD'}]")
    print(f"           - in sync to {worst_sync * 100:.2f}% "
          f"(allow {_CLOSE_SYNC_TOL * 100:.0f}%) "
          f"[{'ok' if synced else 'BAD'}]")
    print(f"           - tracked the ramp to {worst_track * 1e3:.2f} mm "
          f"(allow {_CLOSE_TRACK_TOL_M * 1e3:.1f} mm) "
          f"[{'ok' if tracked else 'BAD'}]")
    for note in notes:
        print(f"           - {note}")
    return ok


def _smoke_lift():
    """Check phase 5: that a lift raises the whole hand, rigidly, to where it
    was sent.

    Same reasoning as :func:`_smoke_close`: the phase rests on a property of the
    binding (the FK warm start carrying the hand across each step), not of this
    file, so it is measured rather than trusted.
    """
    print("Smoke-testing the phase-5 wrist lift...")
    hand = get_hand()
    lift_height = hand.motion.lift_height_m
    lift_steps = hand.motion.lift_steps

    params = apply_phase_preset(HandSolveParams(), "phase5")
    # Off a CLOSED hand, since that is what the button lifts in practice, and a
    # curled rod is the harder thing to translate rigidly than a straight one.
    params.flexor_tensions = [GRASP_FLEXOR_TENSION] * len(hand.digit_names)
    solver = HandFKSolver(params, hand)

    z0 = float(params.wrist_pose[2, 3])
    result, notes = lift_wrist(solver)
    n = result.num_iterates()

    def tips(view):
        """Fingertip positions at a recorded pose -- the node the renderer draws
        the contact sphere on."""
        return {name: np.asarray(
            view.frames[0][name].marginals.rod.states[-1].pose.mean, float)[:3, 3]
            for name in view.finger_names}

    start_tips = tips(result.at_iterate(0))
    worst_arrive, worst_rigid = 0.0, 0.0
    for i in range(n):
        view = result.at_iterate(i)
        want = z0 + i * (lift_height / lift_steps)
        got = solved_wrist_pose(solver.configs, view.frames[0])
        worst_arrive = max(worst_arrive, abs(float(got[2, 3]) - want))
        # Every tip should sit exactly where it started, plus the rise so far.
        rise = np.array([0.0, 0.0, float(got[2, 3]) - z0])
        worst_rigid = max(worst_rigid,
                          max(float(np.linalg.norm(p - (start_tips[name] + rise)))
                              for name, p in tips(view).items()))

    stepped = n == lift_steps + 1
    arrived = worst_arrive <= _LIFT_ARRIVE_TOL_M
    rigid = worst_rigid <= _LIFT_RIGID_TOL_M
    ok = stepped and arrived and rigid
    print(f"  [    lift] poses={n} (expect {lift_steps + 1}) "
          f"rise={(float(solver.params.wrist_pose[2, 3]) - z0) * 1e3:+.1f} mm "
          f"[{'ok' if ok else 'BAD'}]")
    print(f"           - arrived within {worst_arrive * 1e3:.3f} mm of every "
          f"commanded height (allow {_LIFT_ARRIVE_TOL_M * 1e3:.1f} mm) "
          f"[{'ok' if arrived else 'BAD'}]")
    print(f"           - fingertips translated rigidly to {worst_rigid * 1e3:.3f} "
          f"mm (allow {_LIFT_RIGID_TOL_M * 1e3:.1f} mm) "
          f"[{'ok' if rigid else 'BAD'}]")
    for note in notes:
        print(f"           - {note}")
    return ok


def _smoke_calibration():
    """Check the closed-form landmark placement the Calibration folder is built on.

    The whole feature is one number -- how far the landmark ends up from where it
    was sent -- so that number is what this measures, with no viser and no
    hardware. It also tests the PREMISE separately: the placement is exact only
    because a metacarpal disc is rigid to the wrist, and if that ever stopped
    being true the residual check alone would not say why.
    """
    print("Smoke-testing the calibration landmark placement...")
    ok = True
    hand = get_hand()
    finger, disc = hand.digit_names[0], CAL_DEFAULT_DISC

    params = HandSolveParams()
    params.table_burial = 0.0
    solver = HandFKSolver(params)

    # -- the premise: which discs move when the tendons pull --
    def transforms(tension):
        params.flexor_tensions = [tension] * len(hand.digit_names)
        frame = solver.solve().frames[0]
        return {name: [wrist_to_disc(solver.configs, frame, name, d)
                       for d in (disc, disc + 1)]
                for name in hand.digit_names}

    slack, pulled = transforms(0.0), transforms(2.5)
    rigid_mm = max(np.linalg.norm(slack[n][0][:3, 3] - pulled[n][0][:3, 3])
                   for n in hand.digit_names) * 1e3
    moved_mm = min(np.linalg.norm(slack[n][1][:3, 3] - pulled[n][1][:3, 3])
                   for n in hand.digit_names) * 1e3
    premise = (rigid_mm < _CAL_RIGID_TOL_MM and moved_mm > _CAL_ARTICULATED_MIN_MM)
    ok = ok and premise
    print(f"  [ rigidity] disc {disc} moves {rigid_mm * 1e3:.1f} um, disc "
          f"{disc + 1} moves {moved_mm:.2f} mm over 0-2.5 N "
          f"[{'ok' if premise else 'BAD'}]")

    # -- the placement itself --
    params.flexor_tensions = list(HandSolveParams().flexor_tensions)
    frame = solver.solve().frames[0]
    # A target well away from where the landmark already is, and rotated, so a
    # placement that silently did nothing could not pass.
    target = disc_pose(frame, finger, disc).copy()
    target[:3, 3] += np.array([0.05, -0.05, 0.05])
    target[:3, :3] = euler_to_R(0.0, 0.0, np.deg2rad(15.0)) @ target[:3, :3]

    for _ in range(CAL_REFINE_PASSES):
        params.wrist_pose = wrist_pose_for_disc_target(
            solver.configs, frame, finger, disc, target)
        frame = solver.solve().frames[0]

    pos_mm, rot_deg = disc_frame_error(disc_pose(frame, finger, disc), target)
    landed = pos_mm < _CAL_SMOKE_POS_MM and rot_deg < _CAL_SMOKE_ROT_DEG
    ok = ok and landed
    print(f"  [    align] residual {pos_mm:.5f} mm / {rot_deg:.5f} deg after "
          f"{CAL_REFINE_PASSES} passes [{'ok' if landed else 'BAD'}]")
    return ok


def _smoke_robot_plan():
    """Check the robot-plan export the way the Robot folder uses it, headlessly.

    This is the half of the ROS integration that can be tested with no ROS, no
    hardware and no browser, and it is the half that decides which way the
    fingers move -- so it is worth running every time the solver changes, not
    only when someone opens the app.
    """
    print("Smoke-testing the robot plan export...")
    ok = True

    params = HandSolveParams()
    open_lengths = robot_plan.open_tendon_lengths(params)
    notes, sign_ok = robot_plan.check_open_lengths(open_lengths, params)
    ok = ok and sign_ok
    print(f"  [    open] {', '.join(f'{k} {v * 1e3:.1f}' for k, v in open_lengths.items())} mm "
          f"[{'ok' if sign_ok else 'BAD'}]")
    for note in notes:
        print(f"           - {note}")

    if not capabilities()["ik_stepping"]:
        print("  [    plan] skipped -- binding cannot step an IK solve")
        return ok

    # A short stepped solve, so the plan has real AL iterates to walk.
    stepper = HandIKStepper(HandSolveParams())
    last = {}
    stepper.run(max_steps=3, on_step=lambda r, s: last.update(res=r))
    result = last.get("res")
    spec, center, _rot, _pose = resolve_scene(stepper.params)
    corner = table_corner(
        resolve_table_origin(stepper.params, spec, center),
        np.asarray(stepper.params.plane_normal, float))

    plan = robot_plan.build_plan(result, stepper.configs, corner, open_lengths)

    # THE WHOLE PATH, one waypoint per recorded iterate. Checked rather than
    # assumed because the failure is silent and was live for a while: build_plan
    # used to take the convergence scrubber's index, the scrubber opens on the
    # LAST iterate, and the "recorded path" therefore collapsed to a single
    # waypoint -- one hop to the final pose with the trajectory dropped. Nothing
    # downstream can tell a one-waypoint plan from a legitimate one.
    n_iterates = result.num_iterates()
    whole = len(plan.waypoints) == n_iterates
    if not whole:
        print(f"  [    plan] BAD -- history gave {len(plan.waypoints)} waypoint(s) "
              f"for {n_iterates} recorded iterates; the path is being truncated")
    ok = ok and whole

    plan, clamp_notes = robot_plan.clamp_to_travel(plan)
    # The approach segment the bridge prepends at play time: pretend the robot is
    # at the hand-open pose, which is the worst case for the first segment.
    plan = robot_plan.prepend_current(
        plan, plan.waypoints[0].wrist_pose, {n: 0.0 for n in plan.finger_names})
    samples = robot_plan.interpolate(plan, hz=100.0)

    # Every sample must be finite and the last must land ON the final waypoint,
    # or the robot would be commanded somewhere the solve never asked for.
    final = plan.waypoints[-1]
    landed = np.allclose(samples[-1].wrist_pose, final.wrist_pose, atol=1e-9)
    finite = all(np.all(np.isfinite(s.wrist_pose)) for s in samples)
    status = "ok" if landed and finite and whole and len(samples) > 1 else "BAD"
    ok = ok and status == "ok"
    print(f"  [    plan] waypoints={len(plan.waypoints)} "
          f"({n_iterates} iterates + 1 approach) samples={len(samples)} "
          f"duration={samples[-1].t:.2f}s [{status}] | {robot_plan.summarize(plan)}")
    for note in clamp_notes:
        print(f"           - {note}")
    return ok


# ---------------------------------------------------------------------------
# Headless smoke test -- validates the solver classes independently of viser.
# ---------------------------------------------------------------------------

def _smoke():
    print(f"Smoke-testing the hand solver classes "
          f"({HandSolveParams().primitive}, defaults)...")
    ok = True
    caps = capabilities()

    # FK: one frame, no contact.
    res = HandFKSolver(HandSolveParams()).solve()
    status = "ok" if len(res.frames) == 1 else "BAD"
    ok = ok and status == "ok"
    print(f"  [{'FK':>8}] frames={len(res.frames)} (expect 1) [{status}] | "
          f"iters={res.meta.iterations} err={res.meta.error:.3g}")

    # The stepper, in the three contact configurations the split toggles exist
    # to bisect. A handful of steps is enough to prove the loop runs and carries;
    # convergence is what the GUI's Auto solve is for.
    if not caps["ik_stepping"]:
        print("  [IK-step] skipped -- binding has no HandSolver.reset_al_duals")
    else:
        cases = [("IK", False, False)]
        if caps["table"]:
            cases += [("IK-table", True, False), ("IK-both", True, True)]
        else:
            print("  [IK-table] skipped -- binding has no support-plane env fields")
        for label, table, obj in cases:
            params = HandSolveParams()
            if table:
                params.table = True
                params.table_contact = True
                params.object_contact = obj
            # The last stepped result, captured the way the GUI does -- run()
            # returns only the status, and the gaps live on the result.
            last = {}
            st = HandIKStepper(params).run(
                # noqa: B023 -- `last` is rebound each iteration and this lambda
                # is consumed synchronously by run() before the next one, so it
                # never outlives the binding it captures.
                max_steps=5, on_step=lambda r, s: last.update(res=r))  # noqa: B023
            res = last.get("res")
            # One snapshot per step plus the initial guess.
            n = res.num_iterates() if res is not None else 0
            status = "ok" if st.steps > 0 and n == st.steps + 1 else "BAD"
            if status == "BAD":
                ok = False
            extra = ""
            if res is not None:
                if params.object_contact:
                    extra += f" | worst object gap {res.worst_gap(0):+.5f} m"
                if res.table_contact_names():
                    extra += (f" | worst table gap "
                              f"{res.worst_table_gap(params, 0):+.5f} m")
            print(f"  [{label:>8}] steps={st.steps} state={st.state} "
                  f"snapshots={n} [{status}] | violation={st.violation:.3e} "
                  f"cost={st.cost:.4g}{extra}")
    ok = _smoke_close() and ok
    ok = _smoke_lift() and ok
    ok = _smoke_calibration() and ok
    ok = _smoke_robot_plan() and ok
    print("Smoke test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1
