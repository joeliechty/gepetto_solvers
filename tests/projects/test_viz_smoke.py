"""The interactive visualizer's own headless self-checks, run as tests.

``viz_interactive.py`` already ships five ``--smoke`` routines that exercise the
solver-driving half of the app with no viser, no browser and no hardware. They are the
only realistic coverage of ``HandVizApp`` -- a single 4300-line class -- so they are
wired in here rather than left behind a CLI flag nobody runs.

Each sub-check returns a bool and prints its own diagnostics; the tests below assert
that bool per phase, so a failure names the phase instead of just saying "smoke
failed". Run with ``-s`` to see the printed measurements.

These matter most when ``viz_interactive.py`` is split into panels: they are what
catches a collaborator object that lost a reference to the app's state.
"""

from __future__ import annotations

import pytest

from _pkg import viz_interactive

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def viz():
    return viz_interactive()


def test_smoke_close_stays_synchronized(viz):
    """Phase 4: a synchronized close must actually close in sync. Rests on the FK
    solver warm-starting across small upward tension steps -- a property of the
    binding, not of the Python, so it is measured rather than trusted."""
    assert viz._smoke_close() is True


def test_smoke_lift_is_rigid(viz):
    """Phase 5: a lift must raise the whole hand rigidly, to where it was sent."""
    assert viz._smoke_lift() is True


def test_smoke_calibration_places_the_landmark(viz):
    """The closed-form landmark placement, plus its premise -- that a metacarpal
    disc is rigid to the wrist, which is what makes the placement exact."""
    assert viz._smoke_calibration() is True


def test_smoke_robot_plan_exports(viz):
    """The half of the ROS integration testable with no ROS and no hardware -- and
    the half that decides which way the fingers move."""
    assert viz._smoke_robot_plan() is True


def test_smoke_suite_exits_clean(viz):
    """The composite the CLI runs: `python -m ... viz_interactive --smoke`.

    Redundant with the four above by construction, but it is the entry point a
    developer actually invokes, so a regression in its composition (a sub-check
    dropped from the `and` chain) would otherwise go unnoticed."""
    assert viz._smoke() == 0
