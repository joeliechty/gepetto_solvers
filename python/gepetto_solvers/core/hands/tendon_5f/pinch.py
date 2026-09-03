"""The measured pinch table: where each digit combination meets, and what closes it.

A lookup table, not a model. It exists because that geometry is cheap to measure
and expensive to solve for -- see ``scripts/fk_pinch_centroids.py``, which
generated it.

Two things callers get wrong: :func:`pinch_pose` returns None for any set without
the thumb (those digits are all on one side of the palm, so their closest
approach is a fist curl rather than a pinch) and for fewer than two digits; and
``gap > 0`` means the combination NEVER closes, so :meth:`PinchPose.touches` is
the check, not ``centroid is not None``. 7 of the 15 do not close.
"""

from dataclasses import dataclass

from .dimensions import FINGER_NAMES
from .discs import _resolve_contact_mask

# ---------------------------------------------------------------------------
# Measured pinch geometry for THIS hand
# ---------------------------------------------------------------------------
#
# Where each combination of digits actually meets when it closes, measured
# offline by core/hand/fk_pinch_centroids.py (--q-max 4.5) against the
# hand get_default_hand_configs() builds above.
#
# THESE NUMBERS BELONG TO THIS MORPHOLOGY. They are a property of the bone
# lengths, palm origins and base angles in DEFAULT_HAND_DIMENSIONS /
# epfl_hand_core, and of finger_base_offset()'s mounting convention -- change any
# of those and every entry here is silently wrong, because nothing in the code
# can detect the mismatch. That is exactly why they live in this module, next
# to the dimensions they were derived from, rather than in a solver or a demo
# script. Regenerate with:
#
#     python scripts/fk_pinch_centroids.py --q-max 4.5
#
# The centroid is in the WRIST / HAND-BASE frame, which is what makes it usable
# as a constraint: PreGraspCentroidFactor pushes it through the wrist pose to
# get a world point. Measured with the wrist pinned at identity (the solved
# wrist held to 7e-16, so the tip poses were already base-frame).

# The digit order get_default_hand_configs() returns, thumb last. FINGER_NAMES
# above is the four non-thumb fingers only, so it cannot be reused here.
DIGIT_ORDER = FINGER_NAMES + ["thumb"]


@dataclass(frozen=True)
class PinchPose:
    """Where one combination of digits meets, and what closes it.

    ``centroid``   (x, y, z) in meters, WRIST/HAND-BASE frame: the centroid of
                   the combination's fingertip contact spheres at their closest
                   approach.
    ``tensions``   ``{finger: flexor tension N}`` that produces that pose --
                   what to command to actually close this pinch. Note some
                   exceed the interactive viewer's 3 N slider maximum (the
                   pinky needs up to 3.55 N).
    ``gap``        meters, the closest tip-sphere pair's SURFACE separation
                   there. <= 0 means the spheres genuinely touch; positive
                   means this combination never closes and ``centroid`` is a
                   closest-approach point rather than a contact point. Quoted
                   to the source log's 0.1 mm precision.
    """
    centroid: tuple[float, float, float]
    tensions: dict[str, float]
    gap: float

    def touches(self, tol=2e-4):
        """Whether these digits actually reach each other (vs merely getting
        as close as the hand allows). 7 of the 15 combinations do not."""
        return self.gap <= tol


def _pinch_key(finger_names):
    """Canonical lookup key: the given digits in ``DIGIT_ORDER``, deduplicated.

    Order-insensitive so a caller can pass a contact mask, a set, or whatever
    order the GUI checkboxes happen to be read in and still hit the same entry.
    """
    wanted = set(finger_names)
    return tuple(n for n in DIGIT_ORDER if n in wanted)


HAND_PINCH_POSES: dict[tuple[str, ...], PinchPose] = {
    ("index", "thumb"): PinchPose(
        (-0.07212, 0.07190, 0.00335), {"thumb": 1.25, "index": 2.70}, -0.0002),
    ("middle", "thumb"): PinchPose(
        (-0.07260, 0.07270, -0.00795), {"thumb": 1.35, "middle": 2.20}, 0.0013),
    ("ring", "thumb"): PinchPose(
        (-0.06388, 0.06684, -0.02104), {"thumb": 1.50, "ring": 2.75}, 0.0009),
    ("pinky", "thumb"): PinchPose(
        (-0.05512, 0.06067, -0.02966), {"thumb": 1.60, "pinky": 3.55}, 0.0000),
    ("index", "middle", "thumb"): PinchPose(
        (-0.07223, 0.06981, -0.00553),
        {"thumb": 1.35, "index": 2.70, "middle": 2.25}, -0.0030),
    ("index", "ring", "thumb"): PinchPose(
        (-0.06792, 0.06592, -0.01107),
        {"thumb": 1.40, "index": 2.80, "ring": 2.80}, 0.0024),
    ("index", "pinky", "thumb"): PinchPose(
        (-0.06178, 0.06110, -0.01766),
        {"thumb": 1.50, "index": 2.95, "pinky": 3.50}, 0.0050),
    ("middle", "ring", "thumb"): PinchPose(
        (-0.06911, 0.06667, -0.01631),
        {"thumb": 1.45, "middle": 2.30, "ring": 2.75}, 0.0012),
    ("middle", "pinky", "thumb"): PinchPose(
        (-0.06443, 0.06283, -0.02112),
        {"thumb": 1.50, "middle": 2.40, "pinky": 3.40}, 0.0041),
    ("ring", "pinky", "thumb"): PinchPose(
        (-0.06135, 0.06281, -0.02616),
        {"thumb": 1.55, "ring": 2.85, "pinky": 3.40}, 0.0009),
    ("index", "middle", "ring", "thumb"): PinchPose(
        (-0.07047, 0.06670, -0.01122),
        {"thumb": 1.40, "index": 2.75, "middle": 2.30, "ring": 2.75}, 0.0008),
    ("index", "middle", "pinky", "thumb"): PinchPose(
        (-0.06446, 0.05954, -0.01608),
        {"thumb": 1.50, "index": 2.95, "middle": 2.45, "pinky": 3.50}, 0.0001),
    ("index", "ring", "pinky", "thumb"): PinchPose(
        (-0.06282, 0.05976, -0.01868),
        {"thumb": 1.50, "index": 2.95, "ring": 2.95, "pinky": 3.50}, 0.0024),
    ("middle", "ring", "pinky", "thumb"): PinchPose(
        (-0.06508, 0.06159, -0.02133),
        {"thumb": 1.50, "middle": 2.40, "ring": 2.90, "pinky": 3.40}, 0.0017),
    ("index", "middle", "ring", "pinky", "thumb"): PinchPose(
        (-0.06628, 0.06096, -0.01636),
        {"thumb": 1.45, "index": 2.90, "middle": 2.40, "ring": 2.90,
         "pinky": 3.45}, 0.0004),
}


def pinch_pose(finger_names) -> PinchPose | None:
    """The measured :class:`PinchPose` for a set of digits, or None.

    None means the combination was never measured, which is the honest answer
    for anything the scan did not cover: fewer than two digits, or any set
    WITHOUT the thumb. Non-thumb sets are excluded on purpose -- those fingers
    are all on the same side of the palm, so their "closest approach" is a
    fist curl rather than a pinch, and calling that a grasp centroid would be
    wrong. Callers must handle None rather than substituting a default.
    """
    return HAND_PINCH_POSES.get(_pinch_key(finger_names))


def pinch_pose_for_mask(configs, contact_fingers) -> PinchPose | None:
    """:func:`pinch_pose` driven by a per-finger bool mask in ``configs``
    order -- the form the solver params and the GUI checkboxes carry."""
    mask = _resolve_contact_mask(configs, contact_fingers)
    return pinch_pose([name for (name, _), on in zip(configs, mask) if on])
