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

#: The vendored description: Wonik Robotics' Allegro Hand **V5**, right hand,
#: **type B**. Verbatim from the manufacturer's ROS 2 package; see
#: ``urdf/NOTICE.md`` for provenance and for what type B means.
#:
#: Kinematics only -- ``buildModel`` reads the joint tree and never opens the
#: meshes the URDF names.
URDF_PATH = Path(__file__).parent / "urdf" / "allegro_hand_v5_right_B.urdf"

#: Digit order. The thumb is LAST, matching the convention every per-digit list
#: in the solvers uses (and what ``tendon_5f`` does), so an index into one list
#: means the same digit in all of them.
#:
#: Wonik's own documentation calls the third finger the PINKY (the V5 hand has
#: four digits, not five, so there is no separate ring). It is called ``ring``
#: here because that is what the rest of this repository calls the digit in that
#: position, and a hand-specific rename would break every caller that indexes
#: digits by name.
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

#: The trailing index on every V5 joint and link name (``joint_0_0``,
#: ``link_3_0_tip``). It is the HAND number in Wonik's multi-hand ROS setup, not
#: a digit or a link index -- one workspace running two hands publishes
#: ``allegroHand_0`` and ``allegroHand_1``, and the description is generated per
#: hand. It is a constant for us because we describe one hand, but it is written
#: out here rather than inlined so that the V4 names (``joint_0``, ``link_3_tip``)
#: and these differ in one visible place.
_HAND_INDEX = 0


def _joint(number: int) -> str:
    """The URDF's name for joint ``number``."""
    return f"joint_{number}_{_HAND_INDEX}"


def _link(number: int, tip: bool = False) -> str:
    """The URDF's name for the link driven by joint ``number``."""
    return f"link_{number}_{_HAND_INDEX}" + ("_tip" if tip else "")


#: Frames for sites 1..N of each digit, base to tip.
#:
#: Site 0 is NOT here: it is the digit's fixed mount on the palm, which the
#: rigid kinematics resolves to the wrist variable itself rather than a frame.
#:
#: EVERY MOVING LINK IS LISTED, then the fingertip. All four are needed, and
#: leaving one out is not merely a coarser picture -- it silently merges two
#: joints. Omitting the distal link made joint_2 and joint_3 each move exactly
#: one drawn point, the tip, so the two sliders looked like they drove the same
#: thing; and it drew the last segment as one bar where the hand really has two
#: links about a joint between them.
#:
#: The last entry is the fingertip, and is what `contact_node` addresses. Allegro
#: names those `link_*_tip` -- fixed frames past the distal joint, which is where
#: a fingertip contact sphere belongs.
_SITE_FRAMES = {
    name: [_link(n) for n in numbers] + [_link(numbers[-1], tip=True)]
    for name, numbers in _JOINT_NUMBERS.items()
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
        spec.joints = [_joint(n) for n in _JOINT_NUMBERS[name]]
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
