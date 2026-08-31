"""The two mixin-composed classes expose everything their originals did.

``HandVizApp`` and ``ViserHandScene`` were each split out of a single large class
into a set of mixins. A method-by-method extraction is easy to get *almost* right:
methods are obvious and get carried across, while anything else in the class body
is silently left behind.

That is exactly what happened. Eight class-level attributes -- ``TENDON_IDLE``,
``UNFITTED_SUFFIX``, ``WRIST_PRIOR_GAUGE_LIMIT``, ``FINGERTIP_SHELL_M``,
``GRASPABLE_MAX_M``, ``_WRIST_RANGE_MARGIN``, ``_TENSION_BISECT_STEPS``,
``_TENSION_BISECT_TOL_M`` -- were dropped from ``HandVizApp``, because the
extraction walked the class body for ``FunctionDef`` nodes and nothing else. None
of the existing tests caught it: the smoke checks never build the GUI, so the
first symptom was an ``AttributeError`` on ``self.TENDON_IDLE`` when a person
actually opened the app.

These tests pin the composed surface so a future re-split cannot lose a member
the same way. They are cheap -- no viser, no solver, just attribute lookup.
"""

from __future__ import annotations

import pytest

# Every name the original single-class HandVizApp defined in its body that is NOT
# a method: the category the split dropped. Listed explicitly rather than derived,
# so the test states what it protects instead of restating the implementation.
HAND_VIZ_APP_CLASS_ATTRS = {
    "UNFITTED_SUFFIX": str,
    "WRIST_PRIOR_GAUGE_LIMIT": float,
    "FINGERTIP_SHELL_M": float,
    "GRASPABLE_MAX_M": float,
    "TENDON_IDLE": str,
    "_WRIST_RANGE_MARGIN": float,
    "_TENSION_BISECT_STEPS": int,
    "_TENSION_BISECT_TOL_M": float,
}

# The non-dunder members each original single class defined in its body, measured
# against that class at the commit before its split. Floors rather than
# equalities, so adding a method later does not fail the test.
MIN_HAND_VIZ_APP_MEMBERS = 126
MIN_VISER_HAND_SCENE_MEMBERS = 36


def _hand_viz_app():
    from gepetto_solvers.projects.viz.viz_interactive import HandVizApp

    return HandVizApp


def _viser_hand_scene():
    from gepetto_solvers.core.plotting.viser_hand import ViserHandScene

    return ViserHandScene


@pytest.mark.parametrize(("name", "kind"), sorted(HAND_VIZ_APP_CLASS_ATTRS.items()))
def test_hand_viz_app_keeps_its_class_attributes(name, kind):
    """A class attribute must survive the mixin split, and keep its type.

    These reach the composed class through whichever mixin owns them, so this
    also checks the MRO actually resolves them."""
    cls = _hand_viz_app()
    assert hasattr(cls, name), (
        f"HandVizApp lost the class attribute {name!r}. It lives on one of the "
        f"mixins in projects/viz/viz_interactive/; a split that moves methods "
        f"without moving the class body drops exactly this."
    )
    assert isinstance(getattr(cls, name), kind)


def test_hand_viz_app_surface_is_complete():
    """The composed class exposes at least as much as the original 4284-line one."""
    members = [m for m in dir(_hand_viz_app()) if not m.startswith("__")]
    assert len(members) >= MIN_HAND_VIZ_APP_MEMBERS, (
        f"HandVizApp exposes {len(members)} members, fewer than the "
        f"{MIN_HAND_VIZ_APP_MEMBERS} the original class defined -- something was "
        f"dropped in a split."
    )


def test_viser_hand_scene_surface_is_complete():
    members = [m for m in dir(_viser_hand_scene()) if not m.startswith("__")]
    assert len(members) >= MIN_VISER_HAND_SCENE_MEMBERS, (
        f"ViserHandScene exposes {len(members)} members, fewer than the "
        f"{MIN_VISER_HAND_SCENE_MEMBERS} the original class defined."
    )


def test_every_gui_attribute_reference_resolves():
    """Walk the GUI builder for `self.X` reads of ALL-CAPS names and check each one.

    `_build_gui` is 912 lines and runs only when a browser connects, so a missing
    constant there surfaces as an AttributeError in front of a person rather than
    in CI. This reads the source instead of executing it, which needs no viser.
    """
    import ast
    import inspect

    from gepetto_solvers.projects.viz.viz_interactive import _gui

    cls = _hand_viz_app()
    tree = ast.parse(inspect.getsource(_gui))
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr.isupper()
    }
    assert referenced, "expected the GUI builder to read some constants"

    missing = sorted(n for n in referenced if not hasattr(cls, n))
    assert not missing, (
        f"_build_gui reads {missing} off self, but HandVizApp does not define "
        f"them. This is the failure that reached a user: the GUI builder is not "
        f"exercised by any smoke test."
    )


@pytest.mark.slow
def test_the_gui_actually_builds():
    """Construct HandVizApp against a real viser server and build the whole GUI.

    This is the test that was missing. The five `--smoke` routines drive the
    SOLVER half of the app and never touch viser, so `_build_gui` -- 912 lines,
    the single largest function in the codebase -- ran for the first time only
    when a person opened the page. Two bugs reached a user that way: a class
    attribute dropped in the mixin split, and a function-local relative import
    left pointing one package too shallow.

    Costs a few seconds and a port. Worth it for the only coverage `_build_gui`,
    `_render_frame` and the object panel have.
    """
    viser = pytest.importorskip("viser", reason="needs the `web` extra")

    from gepetto_solvers.projects.viz.viz_interactive import HandVizApp

    server = viser.ViserServer(port=8129)
    try:
        app = HandVizApp(server)          # __init__ builds the GUI and solves FK
        assert app._input_handles(), "the GUI produced no input handles"
        assert not [k for k, v in app.caps.items() if not v]

        # The paths that reach the constants and the witness readouts.
        app._render_frame()
        app._report()
        app._refresh_object()
    finally:
        server.stop()
