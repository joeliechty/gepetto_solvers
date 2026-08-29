"""Pure helpers from ``solvers.py``: pose conventions, residual readouts, and the
opposition-sign resolution.

None of these build a factor graph, so they run in milliseconds and can be the first
thing to fail when ``solvers.py`` is split apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import solvers

# ---------------------------------------------------------------------------
# The ZYX euler convention -- shared by the sliders, the CLIs and these tests
# ---------------------------------------------------------------------------


def test_euler_to_R_returns_a_rotation():
    R = solvers.euler_to_R(0.3, -0.7, 1.1)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_identity_angles_give_identity():
    np.testing.assert_allclose(solvers.euler_to_R(0.0, 0.0, 0.0), np.eye(3), atol=1e-15)


def test_single_axis_rotations_are_about_the_axis_they_name():
    # yaw about +Z
    np.testing.assert_allclose(
        solvers.euler_to_R(0.0, 0.0, np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12
    )
    # pitch about +Y
    np.testing.assert_allclose(
        solvers.euler_to_R(0.0, np.pi / 2, 0.0) @ [0, 0, 1], [1, 0, 0], atol=1e-12
    )
    # roll about +X
    np.testing.assert_allclose(
        solvers.euler_to_R(np.pi / 2, 0.0, 0.0) @ [0, 1, 0], [0, 0, 1], atol=1e-12
    )


def test_composition_order_is_Rz_Ry_Rx():
    r, p, y = 0.3, -0.4, 0.9
    Rx = solvers.euler_to_R(r, 0, 0)
    Ry = solvers.euler_to_R(0, p, 0)
    Rz = solvers.euler_to_R(0, 0, y)
    np.testing.assert_allclose(solvers.euler_to_R(r, p, y), Rz @ Ry @ Rx, atol=1e-12)


@pytest.mark.parametrize(
    "rpy",
    [
        (0.0, 0.0, 0.0),
        (0.3, -0.4, 0.9),
        (-1.2, 0.8, -2.5),
        (0.0, -1.22, 0.0),  # the default wrist pitch
    ],
)
def test_euler_round_trip(rpy):
    R = solvers.euler_to_R(*rpy)
    np.testing.assert_allclose(solvers.R_to_euler(R), rpy, atol=1e-9)


def test_R_to_euler_survives_gimbal_lock():
    """pitch = -90 deg makes roll and yaw degenerate. The documented behavior is to
    fold roll into yaw rather than return nan, so a slider readout stays finite."""
    R = solvers.euler_to_R(0.0, np.pi / 2, 0.0)
    roll, pitch, yaw = solvers.R_to_euler(R)

    assert np.isfinite([roll, pitch, yaw]).all()
    assert pitch == pytest.approx(-np.pi / 2) or pitch == pytest.approx(np.pi / 2)
    assert roll == 0.0
    # It still describes the same rotation.
    np.testing.assert_allclose(solvers.euler_to_R(roll, pitch, yaw), R, atol=1e-9)


def test_R_to_euler_clips_rather_than_returning_nan():
    # A matrix whose [2,0] is a hair outside [-1, 1] from round-off.
    R = solvers.euler_to_R(0.0, np.pi / 2, 0.0)
    R[2, 0] = -1.0 - 1e-15
    assert np.isfinite(solvers.R_to_euler(R)).all()


# ---------------------------------------------------------------------------
# Pose assembly
# ---------------------------------------------------------------------------


def test_wrist_pose_from_xyzrpy_is_a_homogeneous_transform():
    T = solvers.wrist_pose_from_xyzrpy((0.1, -0.2, 0.3), (0.4, 0.5, -0.6))

    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
    np.testing.assert_allclose(T[:3, 3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(T[:3, :3], solvers.euler_to_R(0.4, 0.5, -0.6))


def test_default_wrist_pose_is_a_fresh_array_each_call():
    """It backs a dataclass field default and callers mutate poses in place, so a
    shared array would leak one solve's wrist into the next."""
    a = solvers.default_wrist_pose()
    b = solvers.default_wrist_pose()
    assert a is not b
    a[0, 3] = 99.0
    assert b[0, 3] != 99.0


def test_default_wrist_pose_matches_the_documented_constants():
    T = solvers.default_wrist_pose()
    np.testing.assert_allclose(T[:3, 3], solvers.DEFAULT_WRIST_XYZ)
    np.testing.assert_allclose(
        solvers.R_to_euler(T[:3, :3]), solvers.DEFAULT_WRIST_RPY, atol=1e-12
    )


# ---------------------------------------------------------------------------
# disc_frame_error -- the calibration readout
# ---------------------------------------------------------------------------


def test_identical_frames_have_zero_error():
    T = solvers.wrist_pose_from_xyzrpy((0.1, 0.2, 0.3), (0.4, -0.5, 0.6))
    pos_mm, rot_deg = solvers.disc_frame_error(T, T)
    assert pos_mm == pytest.approx(0.0)
    assert rot_deg == pytest.approx(0.0, abs=1e-6)


def test_position_error_is_reported_in_millimetres():
    A = np.eye(4)
    B = np.eye(4)
    B[:3, 3] = [0.003, 0.004, 0.0]  # 5 mm
    pos_mm, _ = solvers.disc_frame_error(A, B)
    assert pos_mm == pytest.approx(5.0)


def test_rotation_error_is_the_geodesic_angle_in_degrees():
    A = np.eye(4)
    B = np.eye(4)
    B[:3, :3] = solvers.euler_to_R(0.0, 0.0, np.deg2rad(30.0))
    _, rot_deg = solvers.disc_frame_error(A, B)
    assert rot_deg == pytest.approx(30.0)


def test_rotation_error_never_returns_nan_at_perfect_alignment():
    """The clip exists because a trace a hair over 3 from round-off makes arccos
    nan, which would read as a broken solve rather than a perfect one."""
    A = np.eye(4)
    B = np.eye(4)
    B[:3, :3] = np.eye(3) * (1.0 + 1e-15)
    _, rot_deg = solvers.disc_frame_error(A, B)
    assert np.isfinite(rot_deg)


# ---------------------------------------------------------------------------
# tip_gap_matrix -- finger-to-finger separation
# ---------------------------------------------------------------------------


def test_gap_matrix_is_surface_separation_not_centre_distance():
    tips = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    radii = np.array([0.01, 0.015])
    gaps = solvers.tip_gap_matrix(tips, radii)

    # 50 mm apart, radii summing to 25 mm -> 25 mm of surface gap.
    assert gaps[0, 1] == pytest.approx(0.025)
    assert gaps[1, 0] == pytest.approx(0.025)


def test_touching_spheres_read_zero_and_overlapping_read_negative():
    radii = np.array([0.01, 0.01])
    touching = solvers.tip_gap_matrix([[0, 0, 0], [0.02, 0, 0]], radii)
    overlapping = solvers.tip_gap_matrix([[0, 0, 0], [0.015, 0, 0]], radii)

    assert touching[0, 1] == pytest.approx(0.0)
    assert overlapping[0, 1] == pytest.approx(-0.005)


def test_diagonal_is_inf_so_min_finds_the_closest_distinct_pair():
    tips = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.5, 0, 0]])
    radii = np.array([0.01, 0.01, 0.01])
    gaps = solvers.tip_gap_matrix(tips, radii)

    assert np.isinf(np.diag(gaps)).all()
    assert gaps.min() == pytest.approx(0.03)  # the 0-1 pair, not a self-pair


def test_gap_matrix_is_symmetric_and_square():
    rng = np.random.default_rng(0)
    tips = rng.normal(size=(5, 3)) * 0.05
    radii = np.full(5, 0.008)
    gaps = solvers.tip_gap_matrix(tips, radii)

    assert gaps.shape == (5, 5)
    np.testing.assert_allclose(gaps, gaps.T)


# ---------------------------------------------------------------------------
# orient_opposition_axis -- the side assignment
# ---------------------------------------------------------------------------


def test_axis_is_signed_to_point_from_the_fingers_toward_the_thumb():
    axis = np.array([1.0, 0.0, 0.0])
    thumb = np.array([0.1, 0.0, 0.0])
    fingers = np.array([[-0.1, 0.0, 0.0], [-0.1, 0.02, 0.0]])

    oriented, flipped = solvers.orient_opposition_axis(axis, thumb, fingers)
    assert not flipped
    np.testing.assert_allclose(oriented, [1.0, 0.0, 0.0])
    # Points toward the thumb.
    assert (thumb - fingers.mean(axis=0)) @ oriented > 0


def test_axis_is_inverted_when_it_points_away_from_the_thumb():
    axis = np.array([-1.0, 0.0, 0.0])  # derived the wrong way up
    thumb = np.array([0.1, 0.0, 0.0])
    fingers = np.array([[-0.1, 0.0, 0.0]])

    oriented, flipped = solvers.orient_opposition_axis(axis, thumb, fingers)
    assert flipped
    np.testing.assert_allclose(oriented, [1.0, 0.0, 0.0])


def test_flip_override_bypasses_the_measurement():
    axis = np.array([1.0, 0.0, 0.0])
    thumb = np.array([0.1, 0.0, 0.0])
    fingers = np.array([[-0.1, 0.0, 0.0]])

    kept, flipped = solvers.orient_opposition_axis(axis, thumb, fingers, flip=False)
    np.testing.assert_allclose(kept, axis)
    assert flipped is False

    inverted, flipped = solvers.orient_opposition_axis(axis, thumb, fingers, flip=True)
    np.testing.assert_allclose(inverted, -axis)
    assert flipped is True


def test_axis_is_normalized():
    oriented, _ = solvers.orient_opposition_axis(
        [0.0, 7.0, 0.0], [0.0, 0.1, 0.0], [[0.0, -0.1, 0.0]]
    )
    assert np.linalg.norm(oriented) == pytest.approx(1.0)


def test_no_finger_points_leaves_the_axis_alone():
    axis = np.array([0.0, 0.0, 1.0])
    oriented, flipped = solvers.orient_opposition_axis(
        axis, [0.0, 0.0, 0.1], np.empty((0, 3))
    )
    np.testing.assert_allclose(oriented, axis)
    assert not flipped


# ---------------------------------------------------------------------------
# capabilities() -- the stale-.so guard the visualizers gate controls on
# ---------------------------------------------------------------------------


def test_capabilities_reports_every_documented_key():
    caps = solvers.capabilities()
    expected = {
        "ellipsoid",
        "table",
        "collision_cull",
        "k_touch",
        "solve_iterates",
        "ik_stepping",
        "solver_seed",
        "dual_transfer",
        "self_collision",
        "drop_normal_row",
        "opposition",
        "half_space_margin",
        "half_space_standalone",
        "pregrasp_center",
        "pregrasp_axis_align",
        "pregrasp_centroid",
    }
    missing = expected - set(caps)
    assert not missing, f"capabilities() lost keys: {sorted(missing)}"
    assert all(isinstance(v, bool) for v in caps.values())


def test_capabilities_all_true_against_a_current_build():
    """A False here means the installed extension is older than the Python layer --
    usually a stale in-tree .so shadowing the installed one. Worth failing loudly:
    the symptom otherwise is a GUI control that silently does nothing."""
    caps = solvers.capabilities()
    stale = sorted(k for k, v in caps.items() if not v)
    assert not stale, (
        f"extension is missing {stale}; rebuild with `rm -rf build && pip install .`"
    )
