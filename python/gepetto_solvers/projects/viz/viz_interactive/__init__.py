"""Interactive viser workbench.

Poses whichever hand ``--hand`` names (default: the registry default).

Pose the hand with FK, then step the IK solve one Augmented Lagrangian outer
iteration at a time and scrub the result; drive the staged pre-grasp pipeline;
and, with hardware attached, play a plan on the robot.

Run it::

    python scripts/viz_interactive.py            # opens http://localhost:8080
    python scripts/viz_interactive.py --smoke    # headless self-check, no viser

Layout, split out of what used to be one 5186-line module whose ``HandVizApp``
class alone was 4284 lines across 122 methods:

=================  ====================================================
:mod:`app`         :class:`HandVizApp` -- construction and composition
:mod:`constants`   tunables, labels, tolerances
:mod:`estop`       the emergency stop and the robot-access gate
:mod:`smoke`       the five headless self-checks
``_objects``       choosing and placing the object, the YCB catalogue
``_render``        camera aim, the hand frame, the table readout
``_params``        GUI controls <-> HandSolveParams
``_notes``         the status line and per-constraint readouts
``_stepping``      FK, the AL stepper, the warm-start carry
``_motion``        closing the hand and lifting it
``_phases``        the staged pipeline's presets
``_trajectory``    the trajectory panel and the robot trace
``_robot``         hardware playback, state readback, e-stop wiring
``_calibration``   landmark placement and wrist alignment
``_gui``           building the viser control panel
=================  ====================================================

The underscore-prefixed modules are mixins of :class:`HandVizApp`, not
standalone parts: they use the attributes ``app.__init__`` sets up.
"""

import argparse
import sys
import time

from gepetto_solvers.core.geometry.scene import TABLE_SPAN, TABLE_THICKNESS

from .app import HandVizApp
from .constants import binding_path
from .estop import EStop, Refused

# The five smoke routines are private by name but public in use: the test suite
# calls each one individually so a failure names the phase, and `main` runs the
# composite. Re-exported at the module path both have always used.
from .smoke import (
    _smoke as _smoke,
)
from .smoke import (
    _smoke_calibration as _smoke_calibration,
)
from .smoke import (
    _smoke_close as _smoke_close,
)
from .smoke import (
    _smoke_lift as _smoke_lift,
)
from .smoke import (
    _smoke_robot_plan as _smoke_robot_plan,
)

__all__ = [
    "EStop",
    "HandVizApp",
    "Refused",
    "binding_path",
    "main",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Headless self-check of the solver classes (no viser).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--hand", default="allegro",
        help="Which hand to pose, by registry name (default: the registry's "
             "default). Every panel is sized and named off it.")
    args = parser.parse_args()

    from gepetto_solvers.core.hands import get_hand, registered_hands

    try:
        hand = get_hand(args.hand)
    except KeyError as exc:
        parser.error(f"{exc} Available: {', '.join(registered_hands())}.")

    # Resolved BEFORE the --smoke branch, so `--smoke --hand allegro` checks the
    # hand it names. It used to exit above this, which silently smoke-tested the
    # default hand whatever --hand said.
    if args.smoke:
        sys.exit(_smoke(hand))

    import viser

    server = viser.ViserServer(port=args.port)
    app = HandVizApp(server, hand=hand)
    # Which binding got loaded, and what it can do. Printed unconditionally: a
    # capability-gated control that is silently disabled is indistinguishable
    # from one that does not work, and the usual cause is a stale .so shadowing
    # the installed build (see binding_path()).
    print(f"gepetto_solvers: {binding_path()}")
    print(f"hand: {hand.name} ({hand.kinematics} kinematics, "
          f"{len(hand.digit_names)} digits: {', '.join(hand.digit_names)})")
    missing = [k for k, v in app.caps.items() if not v]
    if missing:
        print(f"  capabilities MISSING from this build: {', '.join(missing)}")
    # The landmark's dimensions, printed so they can be copied into a real-robot
    # setup without opening the browser. The corner frame's position is scene
    # state (it follows the object-seated plane), so that one lives in the GUI.
    print(f"table square: {TABLE_SPAN:.3f} x {TABLE_SPAN:.3f} m, "
          f"{TABLE_THICKNESS * 1e3:.0f} mm thick -- top face is the constraint "
          f"plane, frame on its -X/-Y corner")
    print(f"viser hand visualizer running -- open http://localhost:{args.port}")
    while True:
        time.sleep(1.0)
