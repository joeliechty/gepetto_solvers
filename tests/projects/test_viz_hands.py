"""The workbench, built for each registered hand.

``test_mixin_surface.py`` checks the app's attribute surface for the default
hand. This checks the thing that surface has to survive: being built for a hand
that has none of the tendon machinery, with whole panels absent rather than
present and dead.

The trap these guard against is specific and was live: the app built its FK
solver without passing its hand, so ``--hand allegro`` posed the DEFAULT hand
while every panel described the other one. Nothing raised -- it rendered a
perfectly good picture of the wrong robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import hands
from gepetto_solvers.core.solvers import phase_presets

viser = pytest.importorskip("viser", reason="the workbench needs the web extra")

pytestmark = pytest.mark.slow


@pytest.fixture(params=sorted(hands.registered_hands()))
def app(request):
    """One built workbench per registered hand, on its own port."""
    from gepetto_solvers.projects.viz.viz_interactive.app import HandVizApp

    hand = hands.get_hand(request.param)
    server = viser.ViserServer(port=0)
    try:
        yield HandVizApp(server, hand=hand)
    finally:
        server.stop()


def test_the_workbench_builds(app):
    assert app.digit_names == list(app.hand.digit_names)
    assert app.result is not None


def test_it_poses_the_hand_it_was_given(app):
    """THE TRAP. The app's solvers must be built with its hand; without that it
    silently draws the default one, which looks like a working workbench."""
    assert app.fk_solver.hand is app.hand
    assert app.result.finger_names == list(app.hand.digit_names)


def test_it_opens_on_the_hand_s_own_start_pose(app):
    """A shared default aims one of the two hands nowhere near the object."""
    wrist, _ = app.hand.default_pose()
    np.testing.assert_allclose(app.params.wrist_pose, wrist, atol=1e-9)
    np.testing.assert_allclose(
        [app.g_tx.value, app.g_ty.value, app.g_tz.value], wrist[:3, 3], atol=1e-3)


#: How far a grasp digit may sit from the object at the opening pose.
#:
#: A loose sanity bound, not a target -- the two hands legitimately open
#: differently. The tendon hand HOVERS: its digits sit 23-54 mm clear, which is
#: what a pre-grasp approach looks like. Allegro opens closer, 13-23 mm inside
#: the surface, because its default posture was measured to put the grasp
#: centroid on the object. Both are fine; a digit across the room is not, which
#: is the only thing this catches.
_OPENING_POSE_TOL_M = 0.08


def test_the_mount_button_poses_the_hand_at_its_own_mount(app):
    """THE SHARED-CONSTANT TRAP, the mount-transform version of the one above.

    "Pose at measured robot mount" used to drive the sliders from one global
    constant -- the tendon hand's Onshape fit -- whatever hand was loaded, so
    with ``--hand allegro`` it hung the Allegro hand off the arm at another
    robot's transform. Like the FK-solver trap, nothing raised: it drew a
    perfectly good picture of the wrong mounting.

    Also guards the reachability half. The button drives the same wrist sliders
    the opening pose seeds, and the tendon hand's mount sits at z = 134.7 mm --
    outside the +-100 mm those sliders used to be fixed at, so pressing it
    raised rather than posing anything.
    """
    expected = app.hand.mount_pose()
    app._pose_at_mount()

    np.testing.assert_allclose(
        [app.g_tx.value, app.g_ty.value, app.g_tz.value], expected[:3, 3],
        atol=1e-3)                              # sliders quantize to 1 mm
    np.testing.assert_allclose(app.params.wrist_pose[:3, 3], expected[:3, 3],
                               atol=1e-3)
    np.testing.assert_allclose(app.params.wrist_pose[:3, :3], expected[:3, :3],
                               atol=1e-2)       # ...and to 0.01 rad
    assert app.g_show_mount.value


def test_the_grasp_digits_start_near_the_object(app):
    """Not a solve, just a sane opening pose."""
    gaps = app.result.surface_gaps(0)
    for name in app.hand.default_contact_digits:
        assert abs(gaps[name]) < _OPENING_POSE_TOL_M, (
            app.hand.name, name, gaps[name])


# ---------------------------------------------------------------------------
# The panels a hand does or does not get.
# ---------------------------------------------------------------------------

def test_the_actuation_panel_matches_the_hand(app):
    """Exactly one of the two shapes, and it is the one the hand's actuation
    calls for: a scalar pull per digit, or a joint vector per digit."""
    if app.has("single_drive"):
        assert app.g_flexors and not app.g_joints
        assert len(app.g_flexors) == len(app.digit_names)
    else:
        assert app.g_joints and not app.g_flexors
        assert len(app.g_joints) == len(app.digit_names)
        assert all(len(row) == app.hand.actuation.n for row in app.g_joints)


def test_the_ramp_buttons_follow_the_feature(app):
    """Phases 4 and 5 walk MEASURED travel, so a hand without those
    measurements gets no buttons rather than buttons that cannot run."""
    present = app.g_close is not None
    assert present == app.has("close_ramp")
    assert (app.g_lift is not None) == app.has("close_ramp")


def test_every_input_handle_is_real(app):
    """Reset walks these. A hand missing a panel must leave no None behind."""
    handles = app._input_handles()
    assert handles and all(h is not None for h in handles)


def test_joint_sliders_are_bounded_by_the_urdf(app):
    """Sliders past a joint's stop command a posture the hand cannot reach, and
    nothing downstream flags it -- the solve does not enforce limits."""
    if not app.g_joints:
        pytest.skip("this hand has no joint sliders")
    limits = app.hand.joint_limits()
    for d, row in enumerate(app.g_joints):
        for j, slider in enumerate(row):
            lo, hi = limits[d][j]
            assert slider.min == pytest.approx(lo)
            assert slider.max == pytest.approx(hi)
            assert lo <= slider.value <= hi


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

def test_every_overlay_draws(app):
    """Every overlay this hand HAS, all on at once. The tendon-only ones are
    absent on a hand with no routing rather than present and empty, so a None
    here is the panel being right -- what must not happen is one of them
    raising."""
    for handle in (app.g_show_discs, app.g_show_disc_frames,
                   app.g_show_link_frames,
                   app.g_show_collision, app.g_show_contact,
                   app.g_show_gaps, app.g_show_finger_planes):
        if handle is not None:
            handle.value = True
    app._fk_solve()
    assert len(app.scene._dynamic) > 0


# ---------------------------------------------------------------------------
# Panels a hand has no use for are ABSENT, and the constraint they controlled
# is held at the value that hand's formulation fixes it to -- not left to a
# widget that is not there.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("feature", "handles"),
    [
        ("planar_bending", ("g_planar_bend", "g_planar_bend_sigma",
                            "g_planar_twist_sigma")),
        ("pregrasp", ("g_half_space", "g_half_sides", "g_half_margin",
                      "g_pregrasp_center", "g_h_clear", "g_pregrasp_centroid",
                      "g_axis_align")),
        ("normal_row_choice", ("g_drop_normal_row",)),
        ("pinch_table", ("g_obj_contact_plane", "g_show_finger_planes",
                         "g_show_planar_gap")),
        # The close and lift RUNNERS -- not the phase boxes that used to be
        # listed here. `phase4` names the tendon hand's commanded close, which
        # needs this feature, and the Allegro hand's grasp-wrench alignment,
        # which is an ordinary solve and needs nothing; keying the box to the
        # feature would have hidden a phase the hand can perfectly well run.
        # Which phases each hand offers is its own question, asserted below.
        ("close_ramp", ("g_close", "g_lift", "g_close_frac", "g_lift_height")),
        ("tendons", ("g_show_discs", "g_show_disc_frames")),
    ],
)
def test_a_panel_is_present_exactly_when_its_feature_is(app, feature, handles):
    want = app.has(feature)
    for name in handles:
        assert (getattr(app, name) is not None) == want, (
            f"{name} should be {'built' if want else 'absent'} on "
            f"{app.hand.name}, which {'has' if want else 'lacks'} {feature!r}")


def test_the_phase_boxes_are_this_hands_own_phases(app):
    """The Presets folder offers exactly the phases this hand's preset set names
    -- minus any whose runner it cannot walk.

    The panel is built by looping that set, so this is really a check that the
    hand-scoped presets reach the GUI: a hand with five phases gets five boxes,
    and a phase numbered 4 on both hands can be a commanded ramp on one and a
    solved constraint on the other without either panel mislabelling it."""
    expected = {name for name, preset in phase_presets(app.hand.name).items()
                if not preset.requires_feature or app.has(preset.requires_feature)}
    assert set(app.g_phases) == expected

    for name in expected:
        box = app.g_phases[name]
        assert box is not None
        # The label and help come off the preset, so the panel cannot describe
        # a phase differently from what applying it does.
        assert box.label == phase_presets(app.hand.name)[name].label
        # ...and the named alias still resolves, which Reset and the mixin
        # surface both rely on.
        assert getattr(app, f"g_{name}") is box


def test_absent_constraints_are_held_off(app):
    """A constraint whose control was never built must be OFF in the params,
    not carrying whatever the dataclass defaulted to."""
    app._sync_params()
    p = app.params
    if not app.has("pregrasp"):
        assert not p.half_space
        assert not p.pregrasp_center
        assert not p.pregrasp_axis_align
        assert not p.pregrasp_centroid
    if not app.has("planar_bending"):
        assert not p.planar_bending
    if not app.has("pinch_table"):
        assert not p.object_contact_in_plane


def test_the_contact_form_is_fixed_without_the_choice(app):
    """No 'drop contact normal row' box means the 4-row [c_R, c_O, c_T1, c_T2]
    witness form is what the formulation DEFINES for sphere-only collision
    geometry -- the tangential rows already force the radius vector collinear
    with the surface normal -- so the flag is derived, not left at a default.

    Derived from the FORM, though, never imposed on it: the flag is a row layout
    of the witness-point contact factor, so it rides on whether a witness point
    is what this scene builds. Hardcoding True instead made every `ycb:` object
    unsolvable on a hand with no box to untick (see
    test_a_ycb_object_solves_with_no_row_choice_to_untick)."""
    app._sync_params()
    if app.has("normal_row_choice"):
        pytest.skip("this hand offers the choice, so the box decides")
    witness = app._object_contact_form()[0] == "witness"
    assert app.params.contact_drop_normal_row is witness


def test_a_ycb_object_solves_with_no_row_choice_to_untick(app):
    """THE BUG: object contact on a `ycb:` object -- an ellipsoid SET, which has
    only the center-direct equality Eq 1.13 -- raised out of the HandSolver
    constructor, because the panel asked for the witness form's row layout
    alongside it and HandModel refuses the pair rather than ignoring it.

    Unsolvable rather than merely wrong: the flag was hardcoded True for a hand
    that offers no box, so nothing on screen could clear it, and every YCB object
    failed the same way ('solve contact with the table and the banana')."""
    from gepetto_solvers.core.solvers.stepper import HandIKStepper

    # Through the panel, not around it: the flag is derived in _sync_params and
    # cleared by the gate, so a test that set params by hand would prove nothing
    # about either -- and _sync_params re-reads the object off the dropdown, so
    # it would not even hold. This is the dropdown and the checkbox.
    app.g_object.value = "ycb:011_banana"
    app.g_obj_contact.value = True
    app._sync_params()
    assert app.params.primitive == "ycb:011_banana"
    assert app.params.object_contact, "the contact under test has to be ON"
    assert app.params.contact_drop_normal_row is False
    HandIKStepper(app.params, app.hand)


def test_the_row_choice_is_greyed_where_it_would_be_refused(app):
    """The box follows _refresh_planar_contact_gate's rule: a ticked box the next
    solve would REJECT is a lie about what is in the graph, so an object with no
    witness form greys it and clears it."""
    if not app.has("normal_row_choice") or not app.caps["drop_normal_row"]:
        pytest.skip("this hand has no box to grey")
    app.g_obj_contact.value = True
    app.g_drop_normal_row.value = True
    app._refresh_normal_row_gate()
    assert app.g_drop_normal_row.value is True   # an ellipsoid honors it
    assert not app.g_drop_normal_row.disabled

    app.g_object.value = "ycb:011_banana"
    assert app.g_drop_normal_row.disabled
    assert app.g_drop_normal_row.value is False
    app._sync_params()
    assert app.params.contact_drop_normal_row is False


def test_the_frame_overlay_matches_what_the_digit_is_made_of(app):
    """Routing discs or rigid links -- a hand has one kind of node to put
    triads on, so it gets one checkbox, and that checkbox draws."""
    handle = (app.g_show_disc_frames if app.has("tendons")
              else app.g_show_link_frames)
    assert handle is not None
    handle.value = True
    app._fk_solve()
    kind = "disc_frame" if app.has("tendons") else "link_frame"
    drawn = [n for n in app.scene._dynamic if kind in n]
    assert drawn, f"no /{kind}/ nodes drawn for {app.hand.name}"


def test_hand_frames_cover_the_palm_and_every_link(app):
    """'hand frames' is every rigid link: the palm, then each digit's links out
    to the tip. A triad short is a link whose frame cannot be checked."""
    if app.g_show_link_frames is None:
        pytest.skip("this hand draws disc frames instead")
    app.g_show_link_frames.value = True
    app._fk_solve()
    assert "/hand/link_frame/palm" in app.scene._dynamic
    for name in app.digit_names:
        sites = app.result.frames[0][name].marginals.sites
        drawn = [n for n in app.scene._dynamic
                 if n.startswith(f"/hand/{name}/link_frame/")]
        assert len(drawn) == len(sites)


def test_the_digit_is_drawn_the_way_it_is_built(app):
    """A continuum rod is a smooth curve through its nodes; a rigid linkage is
    straight segments between joint frames. Drawing a spline through the latter
    shows a bend the hardware does not have."""
    app._fk_solve()
    handle = app.scene._dynamic[f"/hand/{app.digit_names[0]}/rod"]
    kind = type(handle).__name__
    if app.has("tendons"):
        assert "Spline" in kind, kind
    else:
        assert "LineSegments" in kind, kind


def test_a_rebuilt_stepper_adopts_the_solve(app):
    """A rebuild re-imposes the actuation prior at whatever the panel commands,
    so it has to move the panel onto the solve first -- otherwise the next step
    hauls the hand back to a posture the last solve had already left.

    The regression: this reached straight into ``g_flexors[0]`` for the slider
    range, and a joint-space hand has no flexor sliders. Auto solve raised
    IndexError before taking a single step.
    """
    if not app.caps["ik_stepping"]:
        pytest.skip("this binding cannot step")
    app._fk_solve()
    app._invalidate_stepper()
    app._ensure_stepper()

    solved = {name: np.asarray(app.result.frames[0][name].actuation(), float)
              for name in app.digit_names}
    if app.g_joints:
        adopted = [[h.value for h in row] for row in app.g_joints]
        assert app.params.joint_targets == adopted
        for name, row in zip(app.digit_names, adopted):
            assert row == pytest.approx(list(solved[name]))
    else:
        assert app.params.flexor_tensions == [h.value for h in app.g_flexors]
        for name, handle in zip(app.digit_names, app.g_flexors):
            assert handle.value == pytest.approx(
                solved[name][app._drive_index()])


def test_the_adopted_posture_stays_inside_the_sliders(app):
    """The solve does NOT enforce joint limits, so an adopted value has to be
    clamped to the URDF range the sliders carry -- a prior commanded past a
    stop asks for a posture the real hand cannot reach."""
    if not app.caps["ik_stepping"] or not app.g_joints:
        pytest.skip("this hand has no joint sliders")
    app._fk_solve()
    app._invalidate_stepper()
    app._ensure_stepper()
    for row in app.g_joints:
        for handle in row:
            assert handle.min <= handle.value <= handle.max


def test_the_commanded_state_is_reported(app):
    """The joint states and the wrist, or the tendon pulls and lengths --
    whichever this hand is commanded in."""
    app._fk_solve()
    if app.has("displacement"):
        assert app.g_tendon_lengths.content.strip()
    else:
        content = app.g_actuation_report.content
        assert "joint states" in content
        assert "wrist" in content
        for name in app.digit_names:
            assert name in content


# ---------------------------------------------------------------------------
# The trajectory panel, which is sized by the hand
# ---------------------------------------------------------------------------

def test_the_panel_has_one_row_per_driven_actuator_plus_the_wrist(app):
    """5 + 6 = 11 rows on the tendon hand, 16 + 6 = 22 on the Allegro.

    Sized off the HAND rather than off a constant, which is what the old
    ``N_CHANNELS = 11`` could not be: a wider hand's rows would have silently
    slid the wrist channels along and drawn them on a digit plot.
    """
    width = len(app.hand.actuation.drive_indices)
    digits = len(app.digit_names)
    assert app.traj.n_digit_channels == digits * width
    assert app.traj.N_CHANNELS == digits * width + 6
    assert len(app.traj.plots) == app.traj.N_CHANNELS


def test_a_solved_row_fills_every_channel_in_the_panel_s_units(app):
    """``_traj_row`` is what the panel is fed, so its width has to be the
    panel's and its digit half has to be in the units the plan commands: tendon
    LENGTHS in mm, or joint positions in rad."""
    app._fk_solve()
    row = app._traj_row(app.result)
    assert len(row) == app.traj.N_CHANNELS
    assert np.all(np.isfinite(row))

    digits = app._digit_row(app.result)
    assert len(digits) == app.traj.n_digit_channels
    if app.has("displacement"):
        # Millimetres of tendon length: this hand's are 100-170 mm, and a value
        # in the 0.1 range would mean metres leaked through.
        assert min(digits) > 1.0
    else:
        # Radians. The opening posture is a pre-grasp, so nothing is near a
        # revolution; a degrees-vs-radians slip would show up here.
        assert max(abs(v) for v in digits) < 3.2


def test_the_measured_row_lines_up_with_the_solved_one(app):
    """The overlay is only meaningful if the two are the same quantity in the
    same order, so ``_robot_traj_row`` is checked against ``_traj_row``'s width
    -- and a digit the robot did not report must come back NaN rather than zero,
    which would draw as a fully open finger."""
    from gepetto_solvers.core import robot_plan

    app._fk_solve()
    width = len(app.hand.actuation.drive_indices)

    class _State:
        wrist_pose = np.eye(4)
        digit_cmd = {app.digit_names[0]: np.zeros(width)}
        age = 0.0
        source = "test"

    row = app._robot_traj_row(_State())
    assert len(row) == app.traj.N_CHANNELS
    # The one reported digit is finite; the rest of the digit block is NaN.
    per_digit = [row[i * width:(i + 1) * width] for i in range(len(app.digit_names))]
    assert np.all(np.isfinite(per_digit[0]))
    for block in per_digit[1:]:
        assert np.all(np.isnan(block))
    # The wrist block is always real.
    assert np.all(np.isfinite(row[app.traj.n_digit_channels:]))
    assert robot_plan.command_kind(app.hand)


def test_a_plan_can_be_exported_for_this_hand(app):
    """``_build_robot_plan``'s inputs, without ROS: the export must produce a
    plan in this hand's own units, of this hand's own width, and survive the
    flat encoding it crosses the wire in."""
    from gepetto_solvers.core import robot_plan

    app._fk_solve()
    open_lengths = app._open_lengths() if app.has("displacement") else None
    plan = robot_plan.build_plan(
        app.result, app.fk_solver.configs, app._corner_viz(), open_lengths,
        source="final", hand=app.hand)
    assert plan.digit_names == list(app.hand.digit_names)
    assert plan.dof_per_digit == len(app.hand.actuation.drive_indices)
    assert plan.command_kind == robot_plan.command_kind(app.hand)

    plan, _notes = robot_plan.clamp_to_travel(plan, hand=app.hand)
    restored = robot_plan.unflatten_plan(robot_plan.flatten_plan(plan))
    for before, after in zip(plan.waypoints, restored.waypoints, strict=True):
        for name in plan.digit_names:
            np.testing.assert_array_equal(after.digit_cmd[name],
                                          before.digit_cmd[name])
