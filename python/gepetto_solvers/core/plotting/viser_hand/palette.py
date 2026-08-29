"""Scene colours and the thresholds that pick between them.

RGB 0-255, loosely matching the PyVista plotter so the two viewers read the same.
"""

# Tendon colours (RGB 0-255), cycled per tendon; loosely matches the PyVista
# TENDON_COLORS palette.
_TENDON_RGB = [
    (220, 20, 60), (34, 139, 34), (65, 105, 225), (186, 85, 211),
    (218, 165, 32), (255, 20, 147),
]


_ROD_RGB = (40, 90, 200)


_OBJECT_RGB = (218, 165, 32)


# The real scanned mesh behind an ellipsoid-set approximation. Deliberately a
# different hue from _OBJECT_RGB: the two are drawn together so the fit can be
# judged, and in one colour the shells would be indistinguishable from the mesh
# they are approximating.
_OBJECT_MESH_RGB = (120, 150, 190)


# A shell of an ellipsoid set that contact is NOT allowed to target (it is
# outside the object's grasp subset). Grey and fainter than _OBJECT_RGB, so it
# reads as present-but-not-a-target: it is still collision geometry, and drawing
# it as absent would suggest the hand may pass through it.
_OBJECT_EXCLUDED_RGB = (140, 140, 145)


_OBJECT_EXCLUDED_OPACITY = 0.18


_CONTACT_RGB = (80, 200, 120)


_COLLISION_RGB = (230, 120, 60)


_DISC_RGB = (100, 149, 237)


_TABLE_RGB = (150, 150, 160)


_HALF_SPACE_RGB = (255, 140, 0)


_CENTER_TARGET_RGB = (180, 60, 220)


_MOUNT_RGB = (240, 240, 240)


# Darker than the slab it is drawn on, so the grid reads as ruled lines on the
# table rather than as another translucent overlay floating above it.
_TABLE_GRID_RGB = (90, 90, 100)


# Per-finger pinch-plane patches, in finger_names order (index, middle, ring,
# pinky, thumb). One colour per finger rather than one for the overlay: the five
# planes all pass through the same pinch point, so a single colour would draw
# five sheets fanned about one line with no way to tell whose is whose.
_FINGER_PLANE_RGB = [
    (231, 76, 60), (241, 196, 15), (46, 204, 113), (52, 152, 219),
    (155, 89, 182),
]


# Fingertip-to-object gap overlay: green within GAP_GREEN_MAX_M of the surface
# (including interpenetration, which is simply "not far"), red beyond it.
GAP_GREEN_MAX_M = 0.015


_GAP_NEAR_RGB = (0, 190, 60)


_GAP_FAR_RGB = (220, 40, 40)


# Opposition half-space margin overlay: green when the constraint is
# SATISFIED (margin >= 0, i.e. the finger is on its designated side), red when
# violated -- a sign-based rule rather than GAP_GREEN_MAX_M's magnitude-based
# one, since what matters here is which side of the plane, not how far.
_MARGIN_OK_RGB = _GAP_NEAR_RGB


_MARGIN_VIOLATED_RGB = _GAP_FAR_RGB


# Pre-grasp short-axis alignment overlay: green within ANGLE_GREEN_MAX_DEG of
# the target axis (either direction), red beyond it.
ANGLE_GREEN_MAX_DEG = 10.0
