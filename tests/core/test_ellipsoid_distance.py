"""The signed distance every ellipsoid factor measures with.

Two metrics share this one entry point (``EnvironmentConfig::ellipsoid_taubin``),
and the flag is only safe to flip because of what is asserted here:

* they agree EXACTLY on the surface, so the zero set the constraints pin to --
  the contact equality and the collision inequality alike -- does not move when
  the flag does;
* the exact form really is the orthogonal distance, checked against a brute-force
  search over the surface rather than against a committed number;
* both gradients are the analytic derivative of their own distance, checked by
  central differences;
* the exact gradient has norm 1 at every eccentricity and the Taubin one does
  not, which is the entire reason the exact form is the default.

Hermetic: three semi-axis triples and a fixed-seed point cloud, no object files
and no solve.
"""

from __future__ import annotations

import numpy as np
import pytest

import gepetto_solvers

pytestmark = pytest.mark.skipif(
    not hasattr(gepetto_solvers, "ellipsoid_distance"),
    reason="binding predates ellipsoid_distance; rebuild (pip install -e .)",
)

# A ball, a coin and a rod: the three regimes the metric has to hold up in. The
# coin is the case the exact form exists for -- 150:1 between its longest and
# shortest axis is where the Taubin gradient's drift stops being cosmetic.
SHAPES = {
    "ball": np.array([0.050, 0.050, 0.050]),
    "coin": np.array([0.060, 0.060, 0.0004]),
    "rod": np.array([0.004, 0.004, 0.090]),
    "generic": np.array([0.080, 0.030, 0.012]),
}
METRICS = [False, True]


def _cloud(semi_axes, n=60, seed=0):
    """Points inside, on and outside, at the object's own scale."""
    rng = np.random.default_rng(seed)
    scale = float(np.max(semi_axes))
    pts = rng.normal(size=(n, 3)) * scale
    # Drop anything within a micron of the centre: the medial axis is where the
    # closest surface point is not unique, and no distance field has a gradient
    # there (a baked SDF has the same hole).
    return pts[np.linalg.norm(pts, axis=1) > 1e-4]


def _surface_points(semi_axes, n=40, seed=1):
    """Points exactly on x^T M x = 1, by scaling random directions onto it."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.linalg.norm(dirs / semi_axes, axis=1, keepdims=True)


def _brute_force_distance(point, semi_axes):
    """The distance to the surface by search, so the assertion is against the
    geometry rather than against the implementation being tested."""
    from scipy.optimize import minimize

    def surface(t):
        u, v = t
        return semi_axes * [np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)]

    u = np.linspace(0.0, np.pi, 120)
    v = np.linspace(0.0, 2.0 * np.pi, 240)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    pts = np.stack(
        [
            semi_axes[0] * np.sin(grid_u) * np.cos(grid_v),
            semi_axes[1] * np.sin(grid_u) * np.sin(grid_v),
            semi_axes[2] * np.cos(grid_u),
        ],
        axis=-1,
    )
    start = np.unravel_index(
        np.argmin(np.linalg.norm(pts - point, axis=-1)), grid_u.shape
    )
    best = minimize(
        lambda t: float(np.linalg.norm(surface(t) - point)),
        [grid_u[start], grid_v[start]],
        method="Nelder-Mead",
        options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 20000},
    )
    return float(best.fun)


# ---------------------------------------------------------------------------
# The zero set: the one thing the flag must NOT change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("taubin", METRICS)
def test_surface_is_the_zero_set_for_both_metrics(name, taubin):
    """Both metrics report 0 on x^T M x = 1. The AL constraints are equalities and
    inequalities against this level set, so a metric that moved it would move the
    contact surface itself rather than just the field around it."""
    semi_axes = SHAPES[name]
    for point in _surface_points(semi_axes):
        d = gepetto_solvers.ellipsoid_distance(point, semi_axes, taubin)["distance"]
        assert abs(d) < 1e-12, f"{name} taubin={taubin}: d={d:g} at {point}"


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("taubin", METRICS)
def test_sign_is_negative_inside_and_positive_outside(name, taubin):
    """Negative inside, the convention the penetration residual r - d is built on.
    Both metrics take the sign from the same algebraic test, so they cannot
    disagree about which side of the surface a point is on."""
    semi_axes = SHAPES[name]
    for point in _cloud(semi_axes):
        d = gepetto_solvers.ellipsoid_distance(point, semi_axes, taubin)["distance"]
        inside = float(np.sum((point / semi_axes) ** 2)) < 1.0
        assert (d < 0.0) == inside, f"{name} taubin={taubin}: d={d:g} at {point}"


# ---------------------------------------------------------------------------
# The exact metric is the orthogonal distance
# ---------------------------------------------------------------------------


def test_exact_distance_on_a_sphere_is_the_radial_one():
    """The closed form that needs no search: on a ball of radius R the distance is
    ||x|| - R, inside and out."""
    semi_axes = SHAPES["ball"]
    radius = float(semi_axes[0])
    for point in _cloud(semi_axes):
        d = gepetto_solvers.ellipsoid_distance(point, semi_axes, False)["distance"]
        assert d == pytest.approx(float(np.linalg.norm(point)) - radius, abs=1e-12)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_exact_distance_matches_a_brute_force_search(name):
    """Against a search over the surface, on every shape -- the assertion the
    Newton solve is actually here to satisfy."""
    pytest.importorskip("scipy")
    semi_axes = SHAPES[name]
    for point in _cloud(semi_axes, n=12, seed=3):
        d = gepetto_solvers.ellipsoid_distance(point, semi_axes, False)["distance"]
        assert abs(d) == pytest.approx(
            _brute_force_distance(point, semi_axes), abs=1e-9
        )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_taubin_understates_the_distance_from_outside(name):
    """Outside the surface, 0 <= Taubin <= exact on every shape: the first-order
    model measures to the tangent plane at the query point, and a convex body
    lies entirely on the far side of it. That understatement is what makes the
    Taubin form conservative for a collision inequality -- it keeps the sphere
    further out than asked -- and a standoff for a contact equality."""
    semi_axes = SHAPES[name]
    for point in _cloud(semi_axes):
        if float(np.sum((point / semi_axes) ** 2)) < 1.0:
            continue
        exact = gepetto_solvers.ellipsoid_distance(point, semi_axes, False)["distance"]
        taubin = gepetto_solvers.ellipsoid_distance(point, semi_axes, True)["distance"]
        assert 0.0 <= taubin <= exact + 1e-12, f"{name}: {taubin} vs {exact}"


def test_taubin_overstates_penetration_deep_inside():
    """INSIDE the surface the inequality reverses, and hard. On a 50 mm ball a
    point 5 mm from the centre is 45 mm inside; Taubin calls it 247 mm, because
    ||M x|| -> 0 as the query point approaches the centre and it is the
    denominator. The exact form is right by construction.

    This is what a penetration residual r - d feeds the AL solver on the pass
    where a finger starts inside the object, and it is the second half of the
    conditioning argument -- the first being the gradient drift below."""
    ball = SHAPES["ball"]
    point = np.array([0.005, 0.0, 0.0])
    exact = gepetto_solvers.ellipsoid_distance(point, ball, False)["distance"]
    taubin = gepetto_solvers.ellipsoid_distance(point, ball, True)["distance"]
    assert exact == pytest.approx(-0.045, abs=1e-12)
    assert taubin == pytest.approx(-0.2475, abs=1e-9)
    assert taubin < 5.0 * exact   # both negative: Taubin is 5.5x too deep


# ---------------------------------------------------------------------------
# Gradients: the reason the exact form is the default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("taubin", METRICS)
def test_gradient_matches_central_differences(name, taubin):
    """The analytic gradient is the derivative of the distance the same call
    returns. Both metrics, because both feed a factor Jacobian."""
    semi_axes = SHAPES[name]
    step = 1e-7 * float(np.max(semi_axes))
    for point in _cloud(semi_axes, n=25, seed=5):
        grad = np.asarray(
            gepetto_solvers.ellipsoid_distance(point, semi_axes, taubin)["gradient"]
        )
        fd = np.array(
            [
                (
                    gepetto_solvers.ellipsoid_distance(
                        point + step * np.eye(3)[i], semi_axes, taubin
                    )["distance"]
                    - gepetto_solvers.ellipsoid_distance(
                        point - step * np.eye(3)[i], semi_axes, taubin
                    )["distance"]
                )
                / (2.0 * step)
                for i in range(3)
            ]
        )
        # Loose against the finite difference, not against the gradient: the step
        # divides the distance's last few bits by 1e-8, and near the interior
        # medial axis the closest point moves fast enough to amplify that.
        np.testing.assert_allclose(grad, fd, atol=1e-5, err_msg=f"{name} at {point}")


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_exact_gradient_is_a_unit_vector(name):
    """||grad d|| == 1 everywhere, on every shape. This IS the conditioning claim:
    the c_O row of a contact residual is then scaled exactly like the Euclidean
    c_R row beside it, whatever the object's eccentricity."""
    semi_axes = SHAPES[name]
    for point in _cloud(semi_axes):
        grad = np.asarray(
            gepetto_solvers.ellipsoid_distance(point, semi_axes, False)["gradient"]
        )
        assert np.linalg.norm(grad) == pytest.approx(1.0, abs=1e-12)


def test_taubin_gradient_drifts_with_eccentricity_and_exact_does_not():
    """The failure the exact form was adopted to remove, pinned as a number.

    One millimetre off the coin's flat face: the true gap is 1 mm and the true
    normal is a unit vector, and the exact metric reports both. Taubin's gradient
    is off by more than a factor of two there -- so under a shared noise model
    that row is silently down-weighted against every other row in the graph, and
    the AL solve stalls short of contact rather than converging to it."""
    coin = SHAPES["coin"]
    point = np.array([0.0, 0.0, coin[2] + 0.001])

    exact = gepetto_solvers.ellipsoid_distance(point, coin, False)
    assert exact["distance"] == pytest.approx(0.001, abs=1e-12)
    assert np.linalg.norm(exact["gradient"]) == pytest.approx(1.0, abs=1e-12)

    taubin = gepetto_solvers.ellipsoid_distance(point, coin, True)
    assert np.linalg.norm(taubin["gradient"]) == pytest.approx(0.5408, abs=1e-3)
    assert taubin["distance"] == pytest.approx(0.000643, abs=1e-6)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("taubin", METRICS)
def test_points_on_the_axes_are_handled(taubin):
    """Zero components are the degenerate case of the Eberly parameterization (a
    vanishing component removes the pole that brackets the root), so the axes are
    exactly where a naive implementation returns NaN."""
    semi_axes = SHAPES["generic"]
    for i in range(3):
        for reach in (0.3, 1.0, 2.5):
            point = np.zeros(3)
            point[i] = reach * semi_axes[i]
            out = gepetto_solvers.ellipsoid_distance(point, semi_axes, taubin)
            d, grad = out["distance"], np.asarray(out["gradient"])
            assert np.isfinite(d) and np.isfinite(grad).all()
            # OUTSIDE, on an axis, the closest surface point is that axis's
            # own vertex, by symmetry -- so the distance is known in closed
            # form. Not so inside: from a point on the LONG axis the nearest
            # surface is off to the side, which is the interior medial axis and
            # the reason this only asserts the exterior reach.
            if not taubin and reach > 1.0:
                assert d == pytest.approx((reach - 1.0) * semi_axes[i], abs=1e-12)
                assert grad[i] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("taubin", METRICS)
def test_a_non_positive_semi_axis_is_rejected(taubin):
    """A flat ellipsoid is a division by zero that would otherwise surface as a
    NaN residual several frames later. Fail where the mistake was made."""
    with pytest.raises(Exception, match="semi-axis"):
        gepetto_solvers.ellipsoid_distance(
            np.array([0.1, 0.0, 0.0]), np.array([0.05, 0.0, 0.05]), taubin
        )


# ---------------------------------------------------------------------------
# The C++ metric and its NumPy mirror
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("taubin", METRICS)
def test_python_mirror_agrees_with_the_cxx_metric(name, taubin):
    """``scene.primitive_surface_gap`` is a hand-written NumPy copy of the C++
    metric -- it has to be, since it must answer with no solver in play -- and a
    copy is only useful while it still agrees.

    This is the assertion that keeps the reported gap and the solved residual the
    same quantity. Without it the two drift silently, and what that looks like
    from the outside is a solve that converged against a diagnostic insisting it
    did not."""
    from _pkg import scene

    semi_axes = SHAPES[name]
    spec = {"type": "ellipsoid", "semi_axes": list(semi_axes)}
    for point in _cloud(semi_axes, n=30, seed=11):
        cxx = gepetto_solvers.ellipsoid_distance(point, semi_axes, taubin)["distance"]
        py = scene.primitive_surface_gap(point, spec, taubin=taubin)
        # 1e-8 because both nudge a vanishing component off zero by that fraction
        # of its semi-axis (kMinAxisFraction), and they solve for the root either
        # side of it differently -- Newton here, bisection there. Sub-nanometre on
        # every shape in this table; anything larger is a real divergence.
        assert py == pytest.approx(cxx, abs=1e-8), f"{name} taubin={taubin} at {point}"
