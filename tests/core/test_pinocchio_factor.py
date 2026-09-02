"""The kinematics likelihood factor, and above all its Jacobians.

The factor implements

    p(T_i | T_w, q) ~ exp( -1/2 || T_i (-) f_fk,i(T_w, q) ||^2_{Sigma_fk,i} )

with ``f_fk,i(T_w, q) = T_w * T_fk,i(q)``, ``T_fk,i`` Pinocchio's placement of
frame i, and (-) the SE(3) on-manifold difference.

**Why this file is mostly derivative checks.** The one thing here that is easy to
get wrong and impossible to notice is the row swap: Pinocchio stacks a spatial
velocity ``[v; w]`` and GTSAM's ``Pose3`` tangent is ``[w; v]``, so every
Jacobian from Pinocchio has its top and bottom three rows exchanged before it
reaches GTSAM. A wrong swap does not raise and does not produce nonsense poses --
the error function is still correct, so the solve still converges, just slowly
and from a smaller basin. Nothing downstream would report it as anything but a
badly conditioned problem.

So the analytic blocks are checked against numerical differentiation, using
GTSAM's OWN retraction for the perturbation (``pose3_retract``). Writing an
exponential map here instead would compare the factor against a second
implementation of the very convention under test.

The model is a two-joint arm written inline, per the suite's hermeticity rule --
no asset on disk, and no dependency on the Allegro URDF landing first.
"""

from __future__ import annotations

import numpy as np
import pytest

import gepetto_solvers

pin = pytest.importorskip(
    "pinocchio",
    reason="pinocchio is a conda C++ dependency; see conda_setup_*.sh")


# j1 rotates about +z at the base, j2 about +y a fifth of a metre out, so the
# two columns of the Jacobian are independent and neither is axis-aligned with
# the other's effect.
TOY_URDF = """<?xml version="1.0"?>
<robot name="toy">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
</robot>
"""

SIGMA = np.full(6, 1e-3)


@pytest.fixture(scope="module")
def chain():
    return gepetto_solvers.RigidChainModel.from_urdf_xml(TOY_URDF)


@pytest.fixture(scope="module")
def factor(chain):
    qi, vi = zip(*(chain.joint_indices(n) for n in ("j1", "j2")))
    return gepetto_solvers.PinocchioFKFactor(
        0, 1, 2, chain, chain.frame_id("l2"), list(qi), list(vi), SIGMA)


def _pose(rpy=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, 0.0)):
    """A 4x4 from ZYX euler angles and a translation."""
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


#: A pose, configuration and site that are all generic -- no zeros, no axis
#: alignment, nothing that could make a wrong Jacobian look right by symmetry.
WRIST = _pose(rpy=(0.21, -0.35, 0.47), xyz=(0.03, -0.02, 0.11))
Q = np.array([0.37, -0.52])
SITE = _pose(rpy=(-0.13, 0.28, 0.09), xyz=(0.19, 0.07, 0.16))


# ---------------------------------------------------------------------------
# Numerical derivatives, on the manifold GTSAM actually uses.
# ---------------------------------------------------------------------------

def _numeric_pose_jacobian(f, which, wrist, q, site, eps=1e-6):
    """d(error)/d(pose), by central differences retracting on SE(3)."""
    cols = []
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        if which == "wrist":
            plus = f.error(gepetto_solvers.pose3_retract(wrist, d), q, site)
            minus = f.error(gepetto_solvers.pose3_retract(wrist, -d), q, site)
        else:
            plus = f.error(wrist, q, gepetto_solvers.pose3_retract(site, d))
            minus = f.error(wrist, q, gepetto_solvers.pose3_retract(site, -d))
        cols.append((plus - minus) / (2 * eps))
    return np.column_stack(cols)


def _numeric_q_jacobian(f, wrist, q, site, eps=1e-6):
    cols = []
    for i in range(len(q)):
        d = np.zeros(len(q))
        d[i] = eps
        plus = f.error(wrist, q + d, site)
        minus = f.error(wrist, q - d, site)
        cols.append((plus - minus) / (2 * eps))
    return np.column_stack(cols)


def test_h_q_matches_the_numerical_derivative(factor):
    """THE ROW-SWAP GATE.

    H_q = Hl * Hc * SWAP * J_pin. If the swap were dropped or applied twice,
    this is the only assertion in the repository that would notice.
    """
    _, _, H_q, _ = factor.error_and_jacobians(WRIST, Q, SITE)
    np.testing.assert_allclose(
        H_q, _numeric_q_jacobian(factor, WRIST, Q, SITE), rtol=1e-5, atol=1e-7)


def test_h_wrist_matches_the_numerical_derivative(factor):
    _, H_w, _, _ = factor.error_and_jacobians(WRIST, Q, SITE)
    np.testing.assert_allclose(
        H_w, _numeric_pose_jacobian(factor, "wrist", WRIST, Q, SITE),
        rtol=1e-5, atol=1e-7)


def test_h_site_matches_the_numerical_derivative(factor):
    """H_site is GTSAM's exact ``-Jr^-1(e)``, not the ``-I`` approximation. The
    two agree only as the error goes to zero, so this is evaluated at a pose
    well away from the solution where they visibly differ."""
    _, _, _, H_s = factor.error_and_jacobians(WRIST, Q, SITE)
    np.testing.assert_allclose(
        H_s, _numeric_pose_jacobian(factor, "site", WRIST, Q, SITE),
        rtol=1e-5, atol=1e-7)


def test_h_site_is_not_merely_negative_identity(factor):
    """Guards the choice above: if H_site were hardcoded to -I, the derivative
    test would still pass near the solution and fail far from it. Assert that
    the exact form is actually in use."""
    _, _, _, H_s = factor.error_and_jacobians(WRIST, Q, SITE)
    assert not np.allclose(H_s, -np.eye(6), atol=1e-3)


# ---------------------------------------------------------------------------
# The error function itself.
# ---------------------------------------------------------------------------

def test_the_error_vanishes_at_the_prediction(factor):
    """Zero when the site sits exactly where the kinematics puts it -- the
    definition of f_fk,i."""
    pred = factor.predict(WRIST, Q)
    np.testing.assert_allclose(factor.error(WRIST, Q, pred), np.zeros(6),
                               atol=1e-12)


def test_the_prediction_composes_the_wrist_with_the_fk(factor):
    """f_fk,i(T_w, q) = T_w * T_fk,i(q): moving the wrist rigidly moves the
    prediction with it, which is what makes the factor ternary rather than a
    binary factor over a clamped base."""
    at_identity = factor.predict(np.eye(4), Q)
    np.testing.assert_allclose(factor.predict(WRIST, Q), WRIST @ at_identity,
                               atol=1e-12)


def test_the_error_is_the_on_manifold_difference(factor):
    """e = T_i (-) f_fk,i, i.e. GTSAM's Local(f, T_i). Checked against the bound
    operator so the convention cannot drift from the one the math specifies."""
    pred = factor.predict(WRIST, Q)
    np.testing.assert_allclose(factor.error(WRIST, Q, SITE),
                               gepetto_solvers.pose3_local(pred, SITE),
                               atol=1e-12)


def test_the_error_grows_with_a_displaced_site(factor):
    pred = factor.predict(WRIST, Q)
    near = gepetto_solvers.pose3_retract(pred, np.r_[np.zeros(3), 1e-3, 0, 0])
    far = gepetto_solvers.pose3_retract(pred, np.r_[np.zeros(3), 1e-2, 0, 0])
    assert (np.linalg.norm(factor.error(WRIST, Q, near))
            < np.linalg.norm(factor.error(WRIST, Q, far)))


# ---------------------------------------------------------------------------
# It is usable as a solver would use it.
# ---------------------------------------------------------------------------

def test_gauss_newton_on_the_analytic_jacobian_recovers_q(factor):
    """A Jacobian can be self-consistent and still useless. Solve for q from a
    commanded site pose using nothing but the factor's own error and H_q -- if
    the block were wrong, this would stall or diverge rather than converge."""
    q_true = np.array([0.44, -0.61])
    target = factor.predict(WRIST, q_true)

    q = np.zeros(2)
    for _ in range(40):
        e, _, H_q, _ = factor.error_and_jacobians(WRIST, q, target)
        step, *_ = np.linalg.lstsq(H_q, -e, rcond=None)
        q = q + step
        if np.linalg.norm(step) < 1e-12:
            break

    np.testing.assert_allclose(q, q_true, atol=1e-8)


# ---------------------------------------------------------------------------
# Construction is checked rather than trusted.
# ---------------------------------------------------------------------------

def test_a_missing_frame_is_named(chain):
    with pytest.raises(ValueError, match="no frame named"):
        chain.frame_id("no_such_link")


def test_a_missing_joint_is_named(chain):
    with pytest.raises(ValueError, match="no joint named"):
        chain.joint_indices("no_such_joint")


def test_mismatched_index_lists_are_rejected(chain):
    with pytest.raises(ValueError, match="index the same joints"):
        gepetto_solvers.PinocchioFKFactor(
            0, 1, 2, chain, chain.frame_id("l2"), [0, 1], [0], SIGMA)


def test_an_out_of_range_index_is_rejected(chain):
    with pytest.raises(ValueError, match="out of range"):
        gepetto_solvers.PinocchioFKFactor(
            0, 1, 2, chain, chain.frame_id("l2"), [0, 99], [0, 1], SIGMA)


def test_a_wrongly_sized_q_is_rejected(factor):
    with pytest.raises(ValueError, match="this factor owns"):
        factor.error(WRIST, np.array([0.1, 0.2, 0.3]), SITE)


def test_urdf_position_limits_are_readable(chain):
    """Nothing enforces them yet, but a caller seeding or clamping a
    configuration needs them, and a URDF that lost its limits should be visible
    here rather than as a hand that hyperextends."""
    np.testing.assert_allclose(chain.lower_position_limits, [-1.5, -1.5])
    np.testing.assert_allclose(chain.upper_position_limits, [1.5, 1.5])
