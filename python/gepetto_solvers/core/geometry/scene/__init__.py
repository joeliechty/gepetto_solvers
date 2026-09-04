"""Shared scene geometry and grasp constants for the tendon-hand scripts.

These object primitives, placement constants, and analytic surface-distance
helpers were previously defined inside the runnable demo scripts and imported
across siblings. They live here so the demos import shared code instead of
importing each other. Nothing in this package is runnable -- it holds only data
and pure geometry.

Note: this is distinct from ``core/objects/``, which holds the baked SDF ``.vdb``
level-set files and the ``ycb/`` fits. The specs here must stay consistent with
the parameters the bakers in ``scripts/objects/`` were run with.

Layout, split out of what used to be one 1195-line module:

=================  =====================================================
:mod:`constants`   object placement, grasp tension, tendon names, goals
:mod:`polyhedra`   exact polyhedron geometry (the megaminx dodecahedron)
:mod:`primitives`  the registry: which objects exist and what they are
:mod:`surface`     analytic signed distance and the closest surface point
:mod:`extents`     support half-widths, silhouettes, the principal axis
:mod:`ellipsoids`  decompositions, grasp subsets, env configuration
:mod:`table`       the support plane: where it sits and how to draw it
=================  =====================================================

Every public name is re-exported here, so ``from ...geometry.scene import X``
and ``from ...geometry import scene`` both behave exactly as they did when this
was a single module.
"""

from .constants import (
    ELLIPSOID_SET_BETA,
    GRASP_FLEXOR_TENSION,
    GRASP_GOALS,
    GRASP_SPHERE_CENTER,
    OBJECT_CENTER,
    TENDON_NAMES,
    YCB_FITS_DIR,
)
from .ellipsoids import (
    attach_ellipsoid_set,
    configure_object_proxy_and_exact,
    configure_object_surface,
    ellipsoid_members,
    grasp_subset_indices,
    plane_ellipse_section,
    subset_spec,
)
from .extents import (
    INPLANE_DEGENERACY_RATIO,
    inplane_basis,
    object_extent_along,
    object_inplane_widths,
    object_principal_inplane_axis,
    proxy_semi_axes,
)
from .polyhedra import (
    DODECAHEDRON_CIRCUM_OVER_INRADIUS,
    MEGAMINX_FACE_TO_FACE,
    Rx,
    dodecahedron_vertices,
)
from .primitives import get_primitive_specs, ycb_primitive_specs
from .surface import (
    primitive_surface_gap,
    primitive_surface_normal,
    primitive_surface_witness,
)
from .table import (
    TABLE_ANCHOR,
    TABLE_NORMAL,
    TABLE_SPAN,
    TABLE_THICKNESS,
    table_corner,
    table_plot_spec,
    table_slab_center,
)

__all__ = [
    # constants
    "ELLIPSOID_SET_BETA",
    "GRASP_FLEXOR_TENSION",
    "GRASP_GOALS",
    "GRASP_SPHERE_CENTER",
    "OBJECT_CENTER",
    "TENDON_NAMES",
    "YCB_FITS_DIR",
    # polyhedra
    "DODECAHEDRON_CIRCUM_OVER_INRADIUS",
    "MEGAMINX_FACE_TO_FACE",
    "Rx",
    "dodecahedron_vertices",
    # primitives
    "get_primitive_specs",
    "ycb_primitive_specs",
    # surface
    "primitive_surface_gap",
    "primitive_surface_normal",
    "primitive_surface_witness",
    # extents
    "INPLANE_DEGENERACY_RATIO",
    "inplane_basis",
    "object_extent_along",
    "object_inplane_widths",
    "object_principal_inplane_axis",
    "proxy_semi_axes",
    # ellipsoids
    "attach_ellipsoid_set",
    "configure_object_proxy_and_exact",
    "configure_object_surface",
    "ellipsoid_members",
    "grasp_subset_indices",
    "plane_ellipse_section",
    "subset_spec",
    # table
    "TABLE_NORMAL",
    "TABLE_ANCHOR",
    "TABLE_SPAN",
    "TABLE_THICKNESS",
    "table_corner",
    "table_plot_spec",
    "table_slab_center",
]
