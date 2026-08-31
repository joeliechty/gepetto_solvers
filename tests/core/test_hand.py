"""Hand morphology: the bone/joint spec, the digit set, disc indexing, and the
measured pinch table.

All hermetic: every test that touches hand dimensions goes through the ``pinned_dims``
fixture, so nothing here depends on whether ``gepetto_core`` happens to be installed.
See the fixture's docstring for why that matters (the two sources disagree).
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import config

# ---------------------------------------------------------------------------
# bone_joint_spec_from_bones -- the 4-bone / 3-joint interleave
# ---------------------------------------------------------------------------


def test_spec_is_seven_alternating_segments():
    spec = config.bone_joint_spec_from_bones([40.0, 30.0, 20.0, 15.0])

    assert len(spec) == 7
    assert [kind for kind, _ in spec] == [
        "bone",
        "joint",
        "bone",
        "joint",
        "bone",
        "joint",
        "bone",
    ]
    assert all(length > 0 for _, length in spec)


def test_total_digit_length_is_preserved():
    """The documented contract: half of each joint's length is carved off its
    neighbouring bone ends, so rigid length moves into flexible length while the
    total stays put. That invariant is what keeps the hand the right size."""
    bones_mm = [40.0, 30.0, 20.0, 15.0]
    joints_mm = [9.0, 8.0, 7.0, 6.0]  # jd[1:4] is what the function reads

    spec = config.bone_joint_spec_from_bones(bones_mm, joints_mm)
    total_m = sum(length for _, length in spec)

    assert total_m == pytest.approx(sum(bones_mm) / 1000.0)


def test_joint_segments_come_from_the_cad_diameters():
    joints_mm = [99.0, 8.0, 7.0, 6.0]  # element 0 is deliberately ignored
    spec = config.bone_joint_spec_from_bones([40.0, 30.0, 20.0, 15.0], joints_mm)

    joint_lengths_mm = [length * 1000.0 for kind, length in spec if kind == "joint"]
    np.testing.assert_allclose(joint_lengths_mm, [8.0, 7.0, 6.0])


def test_standard_joints_used_when_no_cad_dimensions_given():
    a = config.bone_joint_spec_from_bones([40.0, 30.0, 20.0, 15.0])
    b = config.bone_joint_spec_from_bones([40.0, 30.0, 20.0, 15.0], [1.0, 2.0])
    # Fewer than 3 usable entries falls back to the standard joints.
    assert a == b


def test_wrong_bone_count_raises():
    with pytest.raises(ValueError, match="expected 4 bone lengths"):
        config.bone_joint_spec_from_bones([40.0, 30.0, 20.0])


# ---------------------------------------------------------------------------
# The default hand
# ---------------------------------------------------------------------------


def test_default_hand_is_five_digits_thumb_last(hand_configs):
    """Thumb-last is load-bearing, not cosmetic: the pre-grasp factors identify the
    thumb by name and collect the other digits as 'the fingers'."""
    names = [name for name, _ in hand_configs]
    assert names == ["index", "middle", "ring", "pinky", "thumb"]


def test_tip_radii_are_positive_and_per_digit(pinned_dims):
    radii = config.default_hand_tip_radii(pinned_dims)
    assert len(radii) == 5
    assert all(r > 0 for r in radii)
    # Derived from each digit's distal tip width, so they are not all identical.
    assert len(set(np.round(radii, 6))) > 1


def test_tip_node_index_is_the_last_rod_node(hand_configs):
    for _name, cfg in hand_configs:
        n_nodes = cfg.num_discs + (cfg.num_discs - 1) * cfg.num_between_nodes
        assert config.tip_node_index(cfg) == n_nodes - 1


# ---------------------------------------------------------------------------
# Disc indexing -- the collision sphere set
# ---------------------------------------------------------------------------


def test_disc_node_indices_are_evenly_spaced_and_start_at_zero(hand_configs):
    for _name, cfg in hand_configs:
        idx = config.disc_node_indices(cfg)
        assert len(idx) == cfg.num_discs
        assert idx[0] == 0
        # Discs sit every (num_between_nodes + 1) rod nodes.
        assert idx == [i * (cfg.num_between_nodes + 1) for i in range(cfg.num_discs)]
        assert idx == sorted(idx)


def test_last_disc_is_at_or_before_the_tip(hand_configs):
    for _name, cfg in hand_configs:
        assert config.disc_node_indices(cfg)[-1] <= config.tip_node_index(cfg)


def test_proximal_flags_mark_the_metacarpal_discs(hand_configs):
    for _name, cfg in hand_configs:
        flags = config.proximal_disc_flags(cfg)
        assert len(flags) == cfg.num_discs
        # Default: discs 0 and 1 span the metacarpal.
        assert flags[:2] == [1, 1]
        assert set(flags[2:]) <= {0}


# ---------------------------------------------------------------------------
# The measured pinch table
# ---------------------------------------------------------------------------


def test_pinch_lookup_is_order_insensitive():
    assert config.pinch_pose(["index", "thumb"]) is config.pinch_pose(
        ["thumb", "index"]
    )


def test_no_pinch_pose_without_the_thumb():
    """Documented and easy to get wrong: those digits are all on one side of the
    palm, so their closest approach is a fist curl, not a pinch."""
    assert config.pinch_pose(["index", "middle"]) is None
    assert config.pinch_pose(["index", "middle", "ring"]) is None


def test_no_pinch_pose_for_fewer_than_two_digits():
    assert config.pinch_pose(["thumb"]) is None
    assert config.pinch_pose([]) is None


def test_touches_is_the_check_not_centroid_is_not_none():
    """7 of the 15 combinations never close. They still carry a centroid -- a
    closest-approach point -- so `centroid is not None` is not the contact test."""
    table = config.HAND_PINCH_POSES
    non_touching = [k for k, v in table.items() if not v.touches()]

    assert non_touching, "expected some combinations that never close"
    for key in non_touching:
        pose = table[key]
        assert pose.centroid is not None  # present but NOT a contact point
        assert pose.gap > 0


def test_every_pinch_entry_includes_the_thumb_and_is_well_formed():
    for key, pose in config.HAND_PINCH_POSES.items():
        assert "thumb" in key, key
        assert len(key) >= 2, key
        assert len(pose.centroid) == 3, key
        # A tension for every digit in the combination.
        assert set(pose.tensions) == set(key), key
        assert all(t > 0 for t in pose.tensions.values()), key


def test_pinch_pose_for_mask_matches_the_named_lookup(hand_configs):
    # index + thumb, in configs order (index, middle, ring, pinky, thumb).
    mask = [True, False, False, False, True]
    assert config.pinch_pose_for_mask(hand_configs, mask) is config.pinch_pose(
        ["index", "thumb"]
    )


def test_pinch_pose_for_mask_none_means_all_fingers(hand_configs):
    assert config.pinch_pose_for_mask(hand_configs, None) is config.pinch_pose(
        ["index", "middle", "ring", "pinky", "thumb"]
    )


def test_wrong_length_mask_raises(hand_configs):
    with pytest.raises(ValueError, match="one flag per finger"):
        config.pinch_pose_for_mask(hand_configs, [True, False])
