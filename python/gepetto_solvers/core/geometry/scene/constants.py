"""Scene constants shared across the grasp scripts: object placement, the grasp
tension, tendon names and goals, and the ellipsoid-set sharpness.

Data only -- no geometry, no imports beyond numpy, so anything may depend on it.
"""

import numpy as np

from gepetto_solvers.core.objects import YCB_FITS_DIR as _FITS_DIR

#: Where ``ycb/browser.py`` exports the committed decompositions.
YCB_FITS_DIR = _FITS_DIR

# Default LogSumExp sharpness for ellipsoid-set objects, mirroring the C++
# EnvironmentConfig::ellipsoid_set_beta default. Distances are in metres, so the
# smooth min understates by up to ln(K)/beta -- 1.4 mm at K=4 here. Kept in the
# spec (not just on the env) because primitive_surface_gap has to reproduce the
# solver's residual, and it can only do that if it uses the same beta.
ELLIPSOID_SET_BETA = 1000.0


# Object center, shared by all primitives: the p2p goal position used in the
# single-finger planner, mirrored across x=0 (X negated). The 6-tendon routing
# was rotated 180 deg about the finger axis to match the gepetto_core CAD
# convention, which flips the flexor curl from world +X to -X; the object moves
# with it. The SDF lives at the VDB local origin (see the _objects/make_*.py
# generators); we place it in the world by translating the object pose to this center.
OBJECT_CENTER = np.array([-6.02088876e-02, 3.77734425e-02, 0.0])


# --- Full five-finger grasp scene (the "big_sphere" grasp target) ---

# Flexor tension (N) at which the anatomical fingertips land exactly on the big
# grasp sphere with an identity wrist. The big sphere was sized/placed for this
# flexion; the static and trajectory grasp scripts share it so results compare.
GRASP_FLEXOR_TENSION = 0.6


# Center of the big grasp sphere: the flexed-fingertip locus at
# GRASP_FLEXOR_TENSION. Used by the five-finger grasp/collision scripts; the
# other (single-finger-scale) primitives stay at OBJECT_CENTER.
GRASP_SPHERE_CENTER = np.array([-0.0221, 0.0885, -0.0160])


# Anatomical 6-tendon finger routing (index 5 = flexor), config order.
TENDON_NAMES = ["Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor"]


# Per-finger world-frame tip-position goals (order = config order: index, middle,
# ring, pinky, thumb). These are the *collision-free* terminal fingertip positions
# from the collision+contact grasp solve on the big sphere: the hand wraps the
# sphere with every backbone node held outside it, so unlike a free-space flexor
# curl (whose main fingers spear straight through the sphere) these points ARE
# reachable with the whole finger collision-free.
GRASP_GOALS = np.array([
    [+0.01058010, +0.10938996, +0.02336805],  # index
    [+0.01125694, +0.12307751, +0.01202090],  # middle
    [+0.01993645, +0.12410185, -0.01172549],  # ring
    [+0.02291420, +0.11488003, -0.03186456],  # pinky
    [+0.01562034, +0.08011573, +0.02589826],  # thumb
])
