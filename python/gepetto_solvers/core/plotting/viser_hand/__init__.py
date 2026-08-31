"""viser scene renderer for the tendon hand -- the web-viewer analogue of the
PyVista ``TendonHandPlotter``.

Pure rendering: given a viser server and one solved hand *frame* (the
``{finger_name: solution}`` shim the solvers produce), it draws each finger's
Cosserat backbone, its tendons, and -- optionally -- the routing discs, the
per-fingertip contact spheres, the disc-node collision spheres, the grasp
object, the support-plane "table", and the per-finger pinch planes. No solving
and no GUI live here; the interactive app (``projects/viz/viz_interactive.py``)
owns those.

The world-frame geometry reproduces exactly what the PyVista mesh managers
compute (``core/plotting/tendon_hand_plotter.py``):

* backbone  -- node translations ``rod.states[n].pose.mean[:3, 3]``.
* tendons   -- ``R @ hole + t`` for each active hole, with the disc pose
  ``rod.states[tendon_config.disc_pose_idx[disc]]``.
* discs / disc frames / collision spheres -- one per disc node
  (``disc_pose_idx``); the frame is the body frame its hole locations are in.

Nodes are addressed by stable scene-tree names; re-adding a name upserts it, and
handles for a frame's dynamic geometry are tracked so toggled-off / vanished
geometry is removed on the next update.

Layout, split out of what used to be one 1100-line module whose class alone was
968 lines across 37 methods:

===============  ======================================================
:mod:`palette`   scene colours and the thresholds that pick between them
:mod:`scene`     :class:`ViserHandScene` -- construction, camera, update
``_object``      the grasp object's shell and scanned mesh
``_support``     the support plane, its grid, the opposition half-space
``_frames``      coordinate-frame triads
``_overlays``    per-frame tendons, gaps, discs, constraint witnesses
===============  ======================================================

The four underscore-prefixed modules are mixins of :class:`ViserHandScene`,
not standalone parts -- they use ``self.scene`` and ``self._dynamic``, which the
composed class owns.
"""

# _FINGER_PLANE_RGB and _wxyz_from_R are private but shared: traj_panel draws its
# swatches in the finger-plane colour, and viz_interactive builds viser poses with
# the quaternion helper. Re-exported at the paths those callers already use.
from ._geometry import (
    _wxyz_from_R as _wxyz_from_R,
)
from .palette import (
    _FINGER_PLANE_RGB as _FINGER_PLANE_RGB,
)
from .palette import (
    ANGLE_GREEN_MAX_DEG,
    GAP_GREEN_MAX_M,
)
from .scene import ViserHandScene

__all__ = [
    "ANGLE_GREEN_MAX_DEG",
    "GAP_GREEN_MAX_M",
    "ViserHandScene",
]
