"""The ``attach_*`` family: what constraints an environment carries.

Every function here mutates the per-finger configs in place and returns them for
chaining. Which fields get set decides which factors the C++ layer builds, so
this package is where a solve's *problem* is defined, as distinct from
:mod:`gepetto_solvers.core.hands`, which defines the *robot*.

===============  ====================================================
:mod:`contact`   drive fingertips onto the object surface
:mod:`collision` the sphere set, and the inequalities built on it
:mod:`support`   the support plane and the opposition half-space
:mod:`pregrasp`  position the wrist before anything closes
:mod:`grasp`     make the contacts surround the object, not just touch it
===============  ====================================================

Nothing here knows what KIND of hand it is attaching to. Each function writes
node indices and radii onto a per-digit ``EnvironmentConfig``, and the C++ graph
builder resolves those through the hand's kinematics -- so the same environment
attaches to a tendon hand and to a mechanism that is not built from digits at
all. The two hand-shaped decisions a caller must still make, which digits
participate and which one opposes the rest, are passed in.
"""

from .collision import attach_collision
from .contact import attach_contact
from .grasp import attach_grasp_alignment
from .pregrasp import (
    attach_pregrasp_axis_alignment,
    attach_pregrasp_center,
    attach_pregrasp_centroid,
)
from .support import (
    attach_half_space,
    attach_table,
    opposition_axis_from_object,
    opposition_directions,
)

__all__ = [
    "attach_collision",
    "attach_contact",
    "attach_grasp_alignment",
    "attach_half_space",
    "attach_pregrasp_axis_alignment",
    "attach_pregrasp_center",
    "attach_pregrasp_centroid",
    "attach_table",
    "opposition_axis_from_object",
    "opposition_directions",
]
