"""SO(3) and se(3) on bare numpy.

Hand-rolled rather than taken from scipy so this half of ``robot_plan`` needs
nothing but numpy -- the point of it being pure is that the ROS node and the
headless smoke test can both import it.

TWIST ORDERING IS ``[v(3), w(3)]``, linear first, everywhere in this package and
in ``servo_drivers``. Stated in every docstring below because the opposite
convention is equally common and mixing the two is silent: the result is still a
6-vector, still finite, and simply moves the arm wrongly.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Timing: waypoints -> a fixed-rate sample stream.
# ---------------------------------------------------------------------------

def _rotation_error(R_to, R_from):
    """The rotation vector (axis * angle, rad) taking ``R_from`` to ``R_to``.

    Hand-rolled rather than pulled from scipy so this module has no dependency
    beyond numpy: the whole point of it being pure is that the ROS node and the
    headless smoke test can both import it.

    The angle comes from `atan2(|skew part|, cos)` rather than `arccos` alone.
    Both are correct on paper; only the first is usable near zero. `arccos` takes
    its argument to 1.0 as the rotation vanishes, which is exactly where its
    derivative is infinite, so a trace correct to machine epsilon yields an angle
    correct to about `sqrt(eps)` -- 1e-8 absolute, which swamps the microradian
    rotations between consecutive iterates of a converged solve. `atan2` is
    well-conditioned across the whole range and needs no clip to stay finite.
    """
    R = np.asarray(R_to, float) @ np.asarray(R_from, float).T
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sin_term = 0.5 * float(np.linalg.norm(axis))         # |sin(angle)|
    cos_term = 0.5 * (float(np.trace(R)) - 1.0)          # cos(angle)
    angle = float(np.arctan2(sin_term, cos_term))
    if angle < 1e-12:
        return np.zeros(3)
    if sin_term > 1e-7:
        return axis * (angle / (2.0 * sin_term))
    if angle < 1.0:
        # Small angle: axis/2 IS the rotation vector to first order, and the
        # scaling above is an ill-conditioned 0/0 here.
        return 0.5 * axis
    # Near pi: sin(angle) has vanished but the rotation has not, so the skew part
    # carries no usable direction. At exactly pi, R = 2kk' - I, so R + I = 2kk' --
    # every column is a multiple of the axis, and the one with the largest
    # diagonal is the best conditioned. Taking the column (rather than the
    # sqrt of the diagonal) keeps the RELATIVE signs between components, which
    # a per-component sqrt throws away.
    M = R + np.eye(3)
    k = M[:, int(np.argmax(np.diag(M)))]
    norm = float(np.linalg.norm(k))
    if norm < 1e-12:
        return np.zeros(3)
    k = k / norm
    # k and -k describe the same rotation at exactly pi, but just short of it the
    # skew part still resolves the sign; below that it is genuinely ambiguous.
    if float(np.dot(k, axis)) < 0.0:
        k = -k
    return k * angle


def _rotation_from_vector(rotvec):
    """Rodrigues: a rotation vector back to a 3x3. Inverse of _rotation_error."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(rotvec, float) / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------
# se(3). The reference a segment is walked along is T_k @ se3_exp(V * t) for a
# CONSTANT body twist V, which is what makes the feed-forward handed to the
# controller the exact derivative of the reference at every instant rather than
# only at the segment edges. Hand-rolled on numpy for the same reason the SO(3)
# pair above is -- see _rotation_error.
#
# TWIST ORDERING IS [v(3), w(3)], linear first, everywhere in this module and in
# `servo_drivers`. It is stated in every docstring below because the opposite
# convention is equally common and mixing the two is silent: the result is still
# a 6-vector, still finite, and simply moves the arm wrongly.
# ---------------------------------------------------------------------------

#: Below this rotation angle (rad) the se(3) Jacobian coefficients are evaluated
#: by series rather than closed form. Set where the two agree to ~1e-11: high
#: enough that the closed form is never used in its cancelling regime, low enough
#: that the two-term series is still exact to well past double precision's needs.
_SMALL_ANGLE = 1e-4


def _skew(v):
    """The 3x3 skew-symmetric matrix with ``_skew(a) @ b == np.cross(a, b)``."""
    x, y, z = np.asarray(v, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def se3_log(T):
    """``T`` (4x4) to its body twist ``xi = [v(3), w(3)]``, with ``se3_exp(xi) == T``.

    The translational half is NOT the raw position column: it carries the inverse
    of the left Jacobian, so that travelling along a CONSTANT twist for unit time
    lands exactly on ``T``. Using ``T[:3, 3]`` in its place is the classic error --
    it agrees only when the rotation is zero, and elsewhere it bends the path into
    the wrong helix.
    """
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    w = _rotation_error(R, np.eye(3))               # log(R), the SO(3) half
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    # Coefficient on W@W in the inverse of the left Jacobian V built by se3_exp.
    # The closed form is a 0/0 as theta vanishes -- numerator and denominator both
    # go to zero as theta**2 -- so it is evaluated by series below _SMALL_ANGLE,
    # where the series is good to ~1e-11 relative and the closed form has already
    # lost half its digits to cancellation.
    if theta < _SMALL_ANGLE:
        coefficient = (1.0 / 12.0) + (theta**2) / 720.0
    else:
        coefficient = (1.0 - (theta * np.sin(theta))
                       / (2.0 * (1.0 - np.cos(theta)))) / theta**2
    V_inv = np.eye(3) - 0.5 * W + coefficient * (W @ W)
    return np.concatenate([V_inv @ p, w])


def se3_exp(xi):
    """Body twist ``xi = [v(3), w(3)]`` to the 4x4 it generates. Inverse of se3_log."""
    xi = np.asarray(xi, float).reshape(6)
    v, w = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    # The left Jacobian V, so that the pair round-trips with se3_log above. Both
    # coefficients are 0/0 at theta = 0 and cancel badly just above it, so they
    # get the same series treatment as se3_log's.
    if theta < _SMALL_ANGLE:
        a = 0.5 - (theta**2) / 24.0
        b = (1.0 / 6.0) - (theta**2) / 120.0
    else:
        a = (1.0 - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**3
    V = np.eye(3) + a * W + b * (W @ W)
    T = np.eye(4)
    T[:3, :3] = _rotation_from_vector(w)
    T[:3, 3] = V @ v
    return T


def se3_adjoint(T):
    """The 6x6 Adjoint of ``T``, in the ``[v, w]`` ordering: ``[[R, skew(p)R], [0, R]]``.

    Maps a twist expressed in ``T``'s frame into the frame ``T`` is expressed in.
    Used to pull the segment's feed-forward twist -- which is defined relative to
    the REFERENCE pose -- back onto the frame the arm is ACTUALLY at, so the
    feed-forward stays exact while there is tracking error rather than only when
    the two coincide.
    """
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = _skew(p) @ R
    Ad[3:, 3:] = R
    return Ad
