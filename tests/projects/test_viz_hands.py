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
        ("close_ramp", ("g_phase4", "g_phase5")),
        ("tendons", ("g_show_discs", "g_show_disc_frames")),
    ],
)
def test_a_panel_is_present_exactly_when_its_feature_is(app, feature, handles):
    want = app.has(feature)
    for name in handles:
        assert (getattr(app, name) is not None) == want, (
            f"{name} should be {'built' if want else 'absent'} on "
            f"{app.hand.name}, which {'has' if want else 'lacks'} {feature!r}")


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
    with the surface normal -- so the flag is True, not left at a default."""
    app._sync_params()
    if app.has("normal_row_choice"):
        pytest.skip("this hand offers the choice, so the box decides")
    assert app.params.contact_drop_normal_row is True


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
