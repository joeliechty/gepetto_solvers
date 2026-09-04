"""Analytic signed distance to a primitive's surface, and the closest point on it.

These mirror the SDFs the ``make_*.py`` bakers wrote, so a script can report the
achieved contact gap INDEPENDENTLY of the solver -- which is what makes a stall
diagnosable.

The ellipsoid gap mirrors the C++ ``EllipsoidDistance``, so the reported number
agrees with what the solver drives to zero -- including its ``taubin`` switch,
which every function here takes as a keyword with the same default
(``EnvironmentConfig::ellipsoid_taubin``). Exact by default: the true orthogonal
distance to the surface. ``taubin=True`` is the first-order algebraic
approximation, for reading a solve that was run with it.

Mirrors rather than calls: these must answer for a spec dict with no solver, no
graph and no compiled extension in play, which is what makes them usable to
diagnose a stall and to bake an SDF. The pair is kept honest by
``tests/core/test_geometry.py``.
"""

import numpy as np

from .constants import ELLIPSOID_SET_BETA

# A component of |x| below this fraction of its own semi-axis is nudged up to it.
# The exact value and the reasoning are the C++ EllipsoidDistance's
# kMinAxisFraction; the two must agree, since a difference here is a difference
# between the reported gap and the solved residual.
_AXIS_FLOOR = 1e-8


def _nudge_off_axes(x, semi_axes):
    """``x`` with any vanishing component moved just off its principal plane.

    That case is degenerate for the Eberly parameterization -- a zero component
    removes the pole that brackets the root, and makes
    ``foot_i = a_i^2 x_i / (t + a_i^2)`` a 0/0 -- so every caller works on the
    nudged point. Crucially the SAME nudged point is used to solve for the foot
    and to measure the distance to it: the displacement then cancels out of the
    difference instead of landing in it, which is what keeps a point on the
    surface reading 0 rather than reading the nudge."""
    floor = _AXIS_FLOOR * np.asarray(semi_axes, dtype=float)
    return np.where(np.abs(x) < floor, floor, x)


def _taubin_gap(x, semi_axes):
    """Taubin's first-order distance from ``x`` to ``sum((x_i/a_i)^2) = 1``."""
    a = np.asarray(semi_axes, dtype=float)
    m_diag = 1.0 / (a * a)
    x = np.asarray(x, dtype=float)
    Mx = m_diag * x
    g = float(np.linalg.norm(Mx))
    if g < 1e-9:
        g = 1e-9
    return float((x @ Mx - 1.0) / (2.0 * g))


def _ellipsoid_closest_point(x, semi_axes):
    """Exact closest point on the ellipsoid ``sum((x_i/a_i)^2) = 1`` to ``x``.

    Stationarity gives ``foot_i = a_i^2 x_i / (t + a_i^2)`` for the Lagrange
    multiplier ``t``, which is the unique root of the decreasing function
    ``f(t) = sum((a_i x_i / (t + a_i^2))^2) - 1`` on ``t > -min(a_i^2)``. Bisected
    rather than Newton-solved: unconditionally convergent, and 5 points per frame
    makes the cost irrelevant.

    Exact to machine precision except for a point *inside* the ellipsoid that lies
    exactly on a principal plane (some ``x_i == 0.0``), where the closest point is a
    tie broken by the epsilon below: the foot can then be off by up to ~0.4 mm on the
    flattest primitive here (``credit_card``). That needs a fingertip buried inside
    the object at an exact coordinate zero, so it does not arise in practice."""
    a2 = np.asarray(semi_axes, dtype=float) ** 2
    x = np.asarray(x, dtype=float).reshape(3)

    if np.linalg.norm(x) < 1e-12:
        # Dead center: every direction ties, so pick the nearest surface point --
        # the pole of the shortest semi-axis.
        i = int(np.argmin(a2))
        foot = np.zeros(3)
        foot[i] = np.sqrt(a2[i])
        return foot

    x = _nudge_off_axes(x, np.sqrt(a2))

    def f(t):
        return np.sum(a2 * x * x / (t + a2) ** 2) - 1.0

    lo = -a2.min() + 1e-15
    hi = lo + 1.0
    while f(hi) > 0.0:
        hi = lo + 2.0 * (hi - lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return a2 * x / (0.5 * (lo + hi) + a2)


def _exact_gap(x, semi_axes):
    """Exact orthogonal signed distance from ``x`` to ``sum((x_i/a_i)^2) = 1``.

    The NumPy mirror of ``EllipsoidDistance``'s default metric: the length of the
    segment to the closest surface point, signed by which side ``x`` is on. The
    sign comes from the algebraic test, the same one the C++ uses, so the two
    cannot disagree about inside-vs-outside on a point they both round to the
    surface."""
    a = np.asarray(semi_axes, dtype=float)
    x = np.asarray(x, dtype=float).reshape(3)
    sign = 1.0 if float(np.sum((x / a) ** 2)) > 1.0 else -1.0
    foot = _ellipsoid_closest_point(x, a)
    return float(sign * np.linalg.norm(_nudge_off_axes(x, a) - foot))


def _ellipsoid_gap(x, semi_axes, taubin):
    """One member's signed distance, under whichever metric is in force."""
    return _taubin_gap(x, semi_axes) if taubin else _exact_gap(x, semi_axes)


def _to_member_frame(p_local, member):
    """Object-local point into one set member's own frame: ``R_k^T (p - t_k)``."""
    return np.asarray(member["rotation"], float).T @ (
        np.asarray(p_local, float) - np.asarray(member["center"], float))


def primitive_surface_gap(p_local, spec, *, taubin=False):
    """Analytic signed distance from a point (in the object's local frame) to
    the primitive surface. Mirrors the SDFs in the _objects/make_*.py scripts so
    we can report the achieved contact gap independently of the solver.

    ``taubin`` selects the ellipsoid metric, mirroring
    ``EnvironmentConfig::ellipsoid_taubin`` with the same default -- pass the
    solve's own flag when the number is being compared against a residual. Inert
    for every non-ellipsoid type, whose distances are exact SDFs either way."""
    ptype = spec["type"]
    if ptype == "sphere":
        return float(np.linalg.norm(p_local) - spec["radius"])
    if ptype == "cylinder":
        # Axis along Y, rims filleted by edge_radius (shrink bounds, offset out).
        er = spec.get("edge_radius", 0.0)
        r = spec["radius"] - er
        half_h = spec["height"] / 2.0 - er
        dist_xz = np.hypot(p_local[0], p_local[2])
        dx = dist_xz - r
        dy = abs(p_local[1]) - half_h
        out_dist = np.hypot(max(dx, 0.0), max(dy, 0.0))
        in_dist = min(max(dx, dy), 0.0)
        return float(out_dist + in_dist - er)
    if ptype == "capsule":
        # Distance to the Y-axis segment [-half_h, half_h] minus the radius.
        r = spec["radius"]
        half_h = spec["height"] / 2.0
        dy = p_local[1] - np.clip(p_local[1], -half_h, half_h)
        dist = np.sqrt(p_local[0] ** 2 + dy ** 2 + p_local[2] ** 2)
        return float(dist - r)
    if ptype == "cube":
        # Edges/corners filleted by edge_radius (shrink bounds, offset out).
        er = spec.get("edge_radius", 0.0)
        hx, hy, hz = spec["half_extents"]
        d = np.abs(p_local) - (np.array([hx, hy, hz]) - er)
        out_dist = np.linalg.norm(np.maximum(d, 0.0))
        in_dist = min(max(d[0], max(d[1], d[2])), 0.0)
        return float(out_dist + in_dist - er)
    if ptype == "ellipsoid":
        # Signed distance to x^T M x = 1 (Section 1.6.3, Eq 1.91), M =
        # diag(a^-2, b^-2, c^-2), under the same metric the C++
        # EllipsoidCollisionGapFactor uses -- so the reported gap agrees with
        # what the solver drives to zero.
        return _ellipsoid_gap(p_local, spec["semi_axes"], taubin)
    if ptype == "ellipsoid_set":
        # LogSumExp smooth min over the members (Section 1.2, Eq 1.11), which is
        # what EllipsoidSetCollisionGapFactor evaluates. It must be the smooth min
        # and not a hard one: the two differ by up to ln(K)/beta (1.4 mm at K=4,
        # beta=1000), and this number is compared against a solver residual that
        # carries exactly that bias.
        beta = float(spec.get("beta", ELLIPSOID_SET_BETA))
        d = np.array([_ellipsoid_gap(_to_member_frame(p_local, m),
                                     m["semi_axes"], taubin)
                      for m in spec["members"]])
        d_min = d.min()
        # Shift by d_min so no exponent is positive -- same guard as the C++.
        return float(d_min - np.log(np.exp(-beta * (d - d_min)).sum()) / beta)
    raise ValueError(f"Unknown primitive type: {ptype!r}")


def _primitive_surface_gradient(p_local, spec, h):
    grad = np.empty(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = h
        grad[i] = (primitive_surface_gap(p_local + step, spec)
                   - primitive_surface_gap(p_local - step, spec)) / (2.0 * h)
    return grad


def _primitive_surface_normal(p_local, spec, h=1e-6):
    """Outward unit normal at ``p_local``: the central-difference gradient of
    ``primitive_surface_gap``, which is a unit SDF gradient for every primitive here.

    The gradient vanishes on the medial axis (e.g. the exact center of the box, where
    opposite faces tie), so retry just off it before giving up: an equal nudge on all
    three axes breaks the symmetry and picks the genuinely nearest face, which keeps
    the projected foot point on the surface."""
    grad = _primitive_surface_gradient(p_local, spec, h)
    if np.linalg.norm(grad) < 1e-9:
        grad = _primitive_surface_gradient(p_local + h, spec, h)
    norm = np.linalg.norm(grad)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return grad / norm




def primitive_surface_witness(p_local, spec, *, h=1e-6):
    """``(signed distance, closest surface point, outward unit normal there)`` for a
    point in the object's local frame -- ``primitive_surface_gap`` plus the *where*,
    so a viewer can draw the gap as a line that lands on the surface.

    Sphere / cylinder / capsule / cube: ``primitive_surface_gap`` is an exact SDF, so
    the foot point is one step along its gradient.

    Ellipsoid: solved for the exact closest point, because the *where* is wanted
    and not only the length -- and because this has to be a real distance whatever
    metric a solve was run under. Its length therefore equals
    ``primitive_surface_gap`` exactly at the default (both are the orthogonal
    distance), and diverges from it in the far field under ``taubin=True``, which
    is first-order: a true 15 mm gap from the ``coin`` reads ~8 mm there.
    Reporting/rendering only: the solver's own witness points come from the C++
    contact factors."""
    x = np.asarray(p_local, dtype=float).reshape(3)

    if spec["type"] == "ellipsoid":
        a = np.asarray(spec["semi_axes"], dtype=float)
        foot = _ellipsoid_closest_point(x, a)
        n = foot / (a * a)
        n = n / (np.linalg.norm(n) or 1.0)
        # The LENGTH comes from _exact_gap rather than being recomputed here, so
        # the number drawn as a line and the number printed in a table are one
        # definition and cannot drift apart at the last decimal.
        return _exact_gap(x, a), foot, n

    if spec["type"] == "ellipsoid_set":
        # Nearest point on the nearest MEMBER, each solved exactly and mapped back
        # into the object frame. Deliberately a hard min, unlike
        # primitive_surface_gap's smooth one: this answers "where on the object is
        # the fingertip closest to", and a blended witness would sit off every
        # actual surface -- there is no point in between two ellipsoids to draw a
        # gap line to. The reported LENGTH still comes from this exact solve, so
        # near a seam it can differ from the solver's smooth-min residual by up to
        # ln(K)/beta; that is the standoff the smooth min buys, not an error.
        best = None
        for member in spec["members"]:
            a = np.asarray(member["semi_axes"], dtype=float)
            R = np.asarray(member["rotation"], dtype=float)
            t = np.asarray(member["center"], dtype=float)
            x_k = R.T @ (x - t)
            foot_k = _ellipsoid_closest_point(x_k, a)
            sign = 1.0 if np.sum((x_k / a) ** 2) > 1.0 else -1.0
            d_k = float(sign * np.linalg.norm(x_k - foot_k))
            if best is None or d_k < best[0]:
                n_k = foot_k / (a * a)
                n_k = n_k / (np.linalg.norm(n_k) or 1.0)
                best = (d_k, R @ foot_k + t, R @ n_k)
        return best

    d = primitive_surface_gap(x, spec)
    n = _primitive_surface_normal(x, spec, h)
    return float(d), x - d * n, n
