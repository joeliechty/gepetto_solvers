"""The ``attach_*`` family: what constraints an environment carries.

Every function here mutates the per-finger configs in place and returns them for
chaining. Which fields get set decides which factors the C++ layer builds, so
this package is where a solve's *problem* is defined, as distinct from
:mod:`gepetto_solvers.core.hand`, which defines the *robot*.

===============  ====================================================
:mod:`contact`   drive fingertips onto the object surface
:mod:`collision` the sphere set, and the inequalities built on it
:mod:`support`   the support plane and the opposition half-space
:mod:`pregrasp`  position the wrist before anything closes
===============  ====================================================

All of these used to live in ``hand/config.py``, which is why
``gepetto_solvers.core.hand.config`` still re-exports them: an environment
builder is not hand configuration, but every existing caller imports it from
there.
"""

from .collision import attach_collision
from .contact import attach_contact
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
    "attach_half_space",
    "attach_pregrasp_axis_alignment",
    "attach_pregrasp_center",
    "attach_pregrasp_centroid",
    "attach_table",
    "opposition_axis_from_object",
    "opposition_directions",
]
