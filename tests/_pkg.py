"""The one place the test suite names the application layer's import path.

Routing all of the suite's imports through this module means a package move
updates one file instead of every test -- which is what it was for when the
application layer moved into ``gepetto_solvers/core``, and again when the hand
was separated from the solvers.

Tests should say::

    from _pkg import scene, solvers

and then ``scene.primitive_surface_gap(...)`` -- never a direct
``from gepetto_solvers.core.geometry.scene import ...``.

``config`` is the FIVE-DIGIT TENDON HAND's morphology package. It is named
``config`` because that is what the suite has always called it; what it now
points at is one hand among (eventually) several, so a test asserting a measured
number is asserting something about THAT hand. Use the ``pinned_hand`` fixture
where the number matters.
"""

from __future__ import annotations

from gepetto_solvers.core import hands, robot_plan, solvers
from gepetto_solvers.core.geometry import scene
from gepetto_solvers.core.hands import tendon_5f as config
from gepetto_solvers.core.hands.tendon_5f import finger_config

__all__ = ["config", "finger_config", "hands", "robot_plan", "scene", "solvers"]


def viz_interactive():
    """The visualizer module, imported lazily.

    Kept out of the module-level imports because it pulls in the plotting stack and
    is only needed by the smoke tests. ``viser`` itself is imported lazily *inside*
    the module, so this works headless.
    """
    from gepetto_solvers.projects.viz import viz_interactive as mod

    return mod
