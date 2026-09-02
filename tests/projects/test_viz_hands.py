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
    """Including the tendon-only ones. On a hand with no routing they must draw
    NOTHING rather than raise -- the state says so by carrying no extras."""
    for handle in (app.g_show_discs, app.g_show_disc_frames,
                   app.g_show_collision, app.g_show_contact,
                   app.g_show_gaps, app.g_show_finger_planes):
        handle.value = True
    app._fk_solve()
    assert len(app.scene._dynamic) > 0


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
