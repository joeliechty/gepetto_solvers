"""YCB object meshes and their ellipsoid-set decompositions.

Two halves, deliberately separate:

``data``
    Download / cache / load textured meshes from the YCB Object and Model Set.
    Reaches out to a public S3 bucket on first use and caches under ``ycb_data/``
    (gitignored -- ~0.6 GB for the whole set, and reproducible on demand).

``ellipsoids``
    Approximate a mesh with a small set of ellipsoids: sample, cluster, fit a
    minimum-volume enclosing ellipsoid per cluster, refine. The output is what
    the hand's factor graph consumes -- ``gepetto_solvers``'s
    ``EllipsoidSetCollisionGapFactor`` evaluates the union of these as a smooth
    min over per-member distances (Section 1.2, Eq 1.10-1.13), which is how a
    real object gets a contact/collision surface that a single hyper-ellipsoid
    cannot represent.

``browser`` is the authoring GUI (``python scripts/objects/ycb_browser.py``); the
fits it exports land in ``fits/`` and are committed, because a fit is a stochastic
sweep taking tens of seconds, not something to re-derive per run.

``tendon_hand/scene.py``'s ``ycb_primitive_specs()`` is the consumer: it turns each
committed fit into an ``ellipsoid_set`` object primitive keyed ``ycb:<name>``.
"""

from .data import (
    CATALOG_PATH,
    DEFAULT_CACHE,
    FITS_DIR,
    Catalog,
    ObjectInfo,
    YcbCache,
    describe,
    ground_and_center,
    ground_offset,
    prefetch,
)
from .ellipsoids import (
    BACKENDS,
    Ellipsoid,
    EllipsoidFit,
    FitMetrics,
    auto_fit,
    export_json,
    fit,
    load_cached,
    save_cached,
    support_hull,
)

__all__ = [
    "CATALOG_PATH", "DEFAULT_CACHE", "FITS_DIR",
    "Catalog", "ObjectInfo", "YcbCache",
    "describe", "ground_and_center", "ground_offset", "prefetch",
    "BACKENDS", "Ellipsoid", "EllipsoidFit", "FitMetrics",
    "auto_fit", "export_json", "fit", "load_cached", "save_cached",
    "support_hull",
]
