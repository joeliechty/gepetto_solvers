"""What the Allegro hand IS, in the terms the rigid kinematics takes.

The mechanism itself lives in a URDF (see ``urdf/NOTICE.md``); this module is the
small amount that the URDF does not say -- which joints and frames make up a
digit, in which order, and where the sites a task constraint can attach to are.

Everything here is a property of THIS hand, the same way
``hands/tendon_5f/dimensions.py`` is a property of that one. The kinematics that
consumes it (``"rigid_urdf"``) is generic: any set of serial 1-DOF chains hanging
off a common palm is the same mechanism to it, so a second URDF hand is another
table like this one rather than more C++.
"""

from __future__ import annotations

from pathlib import Path

import gepetto_solvers

#: The vendored Drake-corrected URDF. Kinematics only -- the meshes it names are
#: deliberately not vendored, and ``buildModel`` does not read them.
URDF_PATH = Path(__file__).parent / "urdf" / "allegro_hand_right.urdf"

#: Digit order. The thumb is LAST, matching the convention every per-digit list
#: in the solvers uses (and what ``tendon_5f`` does), so an index into one list
#: means the same digit in all of them.
DIGIT_NAMES = ["index", "middle", "ring", "thumb"]

#: The opposing digit, for the pre-grasp constraints.
OPPOSING_DIGIT = "thumb"

#: Joint numbers per digit, base to tip. Allegro numbers its 16 joints
#: 0-3 index, 4-7 middle, 8-11 ring, 12-15 thumb -- note the thumb is NOT last in
#: the URDF's numbering even though it is last in ours, which is exactly why this
#: table is written out rather than derived from an arithmetic pattern.
_JOINT_NUMBERS = {
    "index": [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "ring": [8, 9, 10, 11],
    "thumb": [12, 13, 14, 15],
}

#: Frames for sites 1..N of each digit, base to tip.
#:
#: Site 0 is NOT here: it is the digit's fixed mount on the palm, which the
#: rigid kinematics resolves to the wrist variable itself rather than a frame.
#:
#: EVERY MOVING LINK IS LISTED, then the fingertip. All four are needed, and
#: leaving one out is not merely a coarser picture -- it silently merges two
#: joints. Omitting `link_3` (the distal link) made joint_2 and joint_3 each
#: move exactly one drawn point, the tip, so the two sliders looked like they
#: drove the same thing; and it drew the last segment as one 65 mm bar where the
#: hand really has 38 mm + 27 mm about a joint between them.
#:
#: The last entry is the fingertip, and is what `contact_node` addresses. Allegro
#: names those `link_*_tip` -- fixed frames past the distal joint, which is where
#: a fingertip contact sphere belongs.
_SITE_FRAMES = {
    "index": ["link_0", "link_1", "link_2", "link_3", "link_3_tip"],
    "middle": ["link_4", "link_5", "link_6", "link_7", "link_7_tip"],
    "ring": ["link_8", "link_9", "link_10", "link_11", "link_11_tip"],
    "thumb": ["link_12", "link_13", "link_14", "link_15", "link_15_tip"],
}

#: Joints per digit, and therefore the dimension of each digit's actuation
#: variable q^d.
DOF_PER_DIGIT = 4

#: Sites per digit INCLUDING the fixed mount at index 0.
SITES_PER_DIGIT = len(_SITE_FRAMES["index"]) + 1

#: How many sites from the base are rigidly co-mounted enough that a pair of them
#: is not worth a self-collision constraint. Site 0 is the mount (skipped anyway
#: as a root site) and site 1 is the first link, which barely moves relative to
#: the palm.
NUM_PROXIMAL_SITES = 2


def digit_specs():
    """One ``RigidDigitSpec`` per digit, in :data:`DIGIT_NAMES` order."""
    specs = []
    for name in DIGIT_NAMES:
        spec = gepetto_solvers.RigidDigitSpec()
        spec.name = name
        spec.joints = [f"joint_{n}" for n in _JOINT_NUMBERS[name]]
        spec.site_frames = list(_SITE_FRAMES[name])
        specs.append(spec)
    return specs


def kinematics_config(q_init=None, sigma_fk=None, site_sigma_fk=None):
    """A ``RigidHandKinematicsConfig`` for the Allegro hand.

    ``q_init`` is the seed configuration, one list of :data:`DOF_PER_DIGIT`
    values per digit. Seed it at the SAME posture the joint prior is centred on
    (q_S) wherever possible: the solve then starts at zero kinematics residual
    and converges in a single iteration, where a seed a few tenths of a radian
    away costs tens of iterations for the same answer.

    ``sigma_fk`` overrides the kinematic relaxation Sigma_fk, and
    ``site_sigma_fk`` overrides it per site -- the formulation defines it per
    frame, so a caller wanting the fingertip pinned harder than the proximal
    links says so there.
    """
    config = gepetto_solvers.RigidHandKinematicsConfig()
    config.urdf_path = str(URDF_PATH)
    config.digits = digit_specs()
    config.q_init = ([list(q) for q in q_init] if q_init is not None
                     else [[0.0] * DOF_PER_DIGIT for _ in DIGIT_NAMES])
    if sigma_fk is not None:
        config.sigma_fk = sigma_fk
    if site_sigma_fk is not None:
        config.site_sigma_fk = site_sigma_fk
    return config


def contact_site():
    """Site index a task constraint contacts with: the fingertip, i.e. the last.

    Negative, so it stays correct if a digit ever gains a site -- the same
    from-the-tip addressing ``EnvironmentConfig`` already allows."""
    return -1


def collision_sites():
    """``(site indices, is_proximal flags)`` for the collision spheres.

    Site 0 is excluded: it is the fixed mount, which the graph builder skips as a
    root site anyway (its pose variable IS the wrist, so a sphere there would
    report the wrist origin rather than the mount).
    """
    sites = list(range(1, SITES_PER_DIGIT))
    flags = [s < NUM_PROXIMAL_SITES for s in sites]
    return sites, flags
