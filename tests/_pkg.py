"""The one place the test suite names the application layer's import path.

The refactor moves every module in here (``python/tests/tendon_hand/*`` becomes
``gepetto_solvers/core/*``). Routing all of the suite's imports through this module
means that move updates one file instead of every test.

Tests should say::

    from _pkg import scene, solvers

and then ``scene.primitive_surface_gap(...)`` -- never a direct
``from python.tests.tendon_hand.scene import ...``.
"""

from __future__ import annotations

try:  # post-refactor layout
    from gepetto_solvers.core import robot_plan, solvers
    from gepetto_solvers.core.geometry import scene  # type: ignore[import-not-found]
    from gepetto_solvers.core.hand import config, finger_config
except ImportError:  # current layout
    from python.tests.tendon_hand import (
        config,
        finger_config,
        robot_plan,
        scene,
        solvers,
    )

__all__ = ["config", "finger_config", "robot_plan", "scene", "solvers"]


def viz_interactive():
    """The visualizer module, imported lazily.

    Kept out of the module-level imports because it pulls in the plotting stack and
    is only needed by the smoke tests. ``viser`` itself is imported lazily *inside*
    the module, so this works headless.
    """
    try:  # post-refactor layout
        from gepetto_solvers.projects.viz import (  # type: ignore[import-not-found]
            viz_interactive as mod,
        )
    except ImportError:  # current layout
        from python.tests.tendon_hand import viz_interactive as mod
    return mod
