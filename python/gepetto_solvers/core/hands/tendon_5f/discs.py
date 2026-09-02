"""Disc indexing: which rod nodes carry a disc, and which of those are proximal.

The disc set is also the collision-sphere set. Finger-finger collision skips a
pair iff BOTH spheres are proximal, so marking the metacarpal discs keeps the
rigidly-attached bases from being checked against each other.
"""

# ---------------------------------------------------------------------------
# Collision avoidance (Section 1.5)
# ---------------------------------------------------------------------------


def disc_node_indices(config):
    """Rod node index of each disc, proximal (0) -> distal (tip)."""
    return [i * (config.num_between_nodes + 1) for i in range(config.num_discs)]


def proximal_disc_flags(config, num_proximal_discs=2):
    """Parallel to :func:`disc_node_indices`: 1 for disc nodes on the rigidly-
    attached proximal (metacarpal) bone, else 0.

    The metacarpal is the first bone in the ``bone_joint`` spec and spans discs
    0 and 1, so ``num_proximal_discs`` defaults to 2. Finger-finger collision in
    the hand skips a sphere pair iff *both* spheres are proximal, so marking the
    metacarpal discs proximal keeps the rigidly-attached bases from being checked
    against each other (they cannot move relative to one another).
    """
    return [1 if d < num_proximal_discs else 0 for d in range(config.num_discs)]


def _resolve_contact_mask(configs, contact_fingers):
    """Validate a per-finger contact mask against ``configs`` and normalize it to
    a list of bools (``None`` => every finger contacts, the legacy behavior)."""
    if contact_fingers is None:
        return [True] * len(configs)
    mask = [bool(f) for f in contact_fingers]
    if len(mask) != len(configs):
        raise ValueError(
            f"contact_fingers has {len(mask)} entries but there are "
            f"{len(configs)} fingers; pass one flag per finger.")
    return mask
