"""Analytic object geometry: signed distances, witnesses, proxies and extents.

Every function here is a pure function of a spec dict and a point, so none of these
need the solver. That is what makes them the safety net for splitting ``scene.py``:
if a split moves a helper into the wrong module or drops a branch of a type switch,
these fail immediately and cheaply.

Assertions are against closed-form geometry, not against committed magic numbers,
so they document what the code is *supposed* to compute.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import scene

# ---------------------------------------------------------------------------
# primitive_surface_gap -- the analytic SDF the demos report gaps with
# ---------------------------------------------------------------------------


def test_sphere_gap_is_radial_distance():
    spec = {"type": "sphere", "radius": 0.035}
    assert scene.primitive_surface_gap([0.0, 0.0, 0.0], spec) == pytest.approx(-0.035)
    assert scene.primitive_surface_gap([0.035, 0.0, 0.0], spec) == pytest.approx(0.0)
    assert scene.primitive_surface_gap([0.05, 0.0, 0.0], spec) == pytest.approx(0.015)
    # Sign convention: negative inside, positive outside.
    assert scene.primitive_surface_gap([0.01, 0.0, 0.0], spec) < 0.0


def test_sphere_gap_is_isotropic():
    spec = {"type": "sphere", "radius": 0.04}
    r = 0.06
    for direction in np.eye(3):
        assert scene.primitive_surface_gap(r * direction, spec) == pytest.approx(
            r - 0.04
        )


def test_capsule_gap_is_distance_to_its_axis_segment():
    # Capsule is modeled about LOCAL Y, per the make_*.py generators.
    spec = {"type": "capsule", "radius": 0.02, "height": 0.10}
    # Beside the cylindrical body: distance is purely radial.
    assert scene.primitive_surface_gap([0.05, 0.0, 0.0], spec) == pytest.approx(0.03)
    # Past the cap: distance is measured from the segment end at y = +0.05.
    assert scene.primitive_surface_gap([0.0, 0.09, 0.0], spec) == pytest.approx(0.02)
    # On the axis, inside: -radius everywhere along the segment.
    assert scene.primitive_surface_gap([0.0, 0.03, 0.0], spec) == pytest.approx(-0.02)


def test_cube_gap_outside_face_and_corner():
    spec = {"type": "cube", "half_extents": [0.03, 0.03, 0.03]}
    # Straight out from a face.
    assert scene.primitive_surface_gap([0.05, 0.0, 0.0], spec) == pytest.approx(0.02)
    # Out from a corner: Euclidean distance to the corner point.
    p = [0.06, 0.06, 0.06]
    assert scene.primitive_surface_gap(p, spec) == pytest.approx(
        np.linalg.norm(np.array(p) - 0.03)
    )
    # Dead centre is -half_extent.
    assert scene.primitive_surface_gap([0, 0, 0], spec) == pytest.approx(-0.03)


def test_ellipsoid_gap_vanishes_on_the_surface():
    # Taubin distance shares its zero set with the algebraic form, so a point
    # placed exactly on x^T M x = 1 must read 0 whatever the axis ratios.
    spec = {"type": "ellipsoid", "semi_axes": [0.05, 0.03, 0.01]}
    for axis, a in enumerate(spec["semi_axes"]):
        p = np.zeros(3)
        p[axis] = a
        assert scene.primitive_surface_gap(p, spec) == pytest.approx(0.0, abs=1e-12)
    # An off-axis surface point too.
    u = np.array([1.0, 1.0, 1.0])
    u /= np.linalg.norm(u / np.asarray(spec["semi_axes"]))
    assert scene.primitive_surface_gap(u, spec) == pytest.approx(0.0, abs=1e-12)


def test_ellipsoid_gap_is_signed():
    spec = {"type": "ellipsoid", "semi_axes": [0.05, 0.03, 0.01]}
    assert scene.primitive_surface_gap([0.0, 0.0, 0.0], spec) < 0.0
    assert scene.primitive_surface_gap([0.2, 0.0, 0.0], spec) > 0.0


def test_sphere_as_ellipsoid_agrees_with_the_sphere_branch_at_contact():
    """Taubin is a *first-order* distance, so it equals the exact SDF only on the
    surface -- and that is the only place the contact equality drives it to.

    For a sphere of radius r the closed form is available, so assert it exactly
    rather than asserting a tolerance and hoping: at a point r + d along an axis,
    Taubin returns ``d(2r + d) / (2(r + d))``, which is d at d = 0 and falls
    increasingly short of d further out (1.4% low at d = 1 mm on a 35 mm sphere).
    This is the same first-order/exact split that
    ``test_ellipsoid_witness_is_the_exact_closest_point_not_taubin`` covers from
    the other side.
    """
    r = 0.035
    sph = {"type": "sphere", "radius": r}
    ell = {"type": "ellipsoid", "semi_axes": [r, r, r]}

    # On the surface the two agree exactly -- the shared zero set.
    p0 = [r, 0.0, 0.0]
    assert scene.primitive_surface_gap(p0, ell) == pytest.approx(
        scene.primitive_surface_gap(p0, sph), abs=1e-12
    )

    for d in (0.001, 0.002, 0.005):
        p = [r + d, 0.0, 0.0]
        expected = d * (2 * r + d) / (2 * (r + d))
        assert scene.primitive_surface_gap(p, ell) == pytest.approx(expected, rel=1e-12)
        # Always an underestimate of the true distance, never an overestimate:
        # the contact constraint must not think it has arrived early.
        assert scene.primitive_surface_gap(p, ell) < d


def test_unknown_primitive_type_raises():
    with pytest.raises(ValueError, match="Unknown primitive type"):
        scene.primitive_surface_gap([0, 0, 0], {"type": "dodecahedron"})


# ---------------------------------------------------------------------------
# primitive_surface_witness -- gap plus the closest point and normal
# ---------------------------------------------------------------------------


def test_sphere_witness_lands_on_the_surface():
    spec = {"type": "sphere", "radius": 0.035}
    gap, point, normal = scene.primitive_surface_witness([0.1, 0.0, 0.0], spec)

    assert gap == pytest.approx(0.065)
    np.testing.assert_allclose(point, [0.035, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-9)


def test_witness_point_lies_on_the_surface_for_every_analytic_type():
    specs = [
        {"type": "sphere", "radius": 0.03},
        {"type": "capsule", "radius": 0.02, "height": 0.08},
        {"type": "cube", "half_extents": [0.02, 0.03, 0.04]},
        {"type": "ellipsoid", "semi_axes": [0.05, 0.03, 0.02]},
    ]
    probe = np.array([0.07, 0.06, 0.05])
    for spec in specs:
        _gap, point, normal = scene.primitive_surface_witness(probe, spec)
        # The reported foot point must itself be a zero of the surface function.
        assert scene.primitive_surface_gap(point, spec) == pytest.approx(
            0.0, abs=1e-4
        ), spec["type"]
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6), spec["type"]


def test_ellipsoid_witness_is_the_exact_closest_point_not_taubin():
    # Documented behavior: in the FAR field the witness solves for the true closest
    # point while primitive_surface_gap stays first-order, so the two disagree --
    # a true 15 mm gap from the coin reads ~8 mm through Taubin.
    spec = {"type": "ellipsoid", "semi_axes": [0.019, 0.019, 0.00125]}  # coin-like
    probe = [0.0, 0.0, 0.01625]  # 15 mm off the flat face
    gap, point, _normal = scene.primitive_surface_witness(probe, spec)
    taubin = scene.primitive_surface_gap(probe, spec)

    assert gap == pytest.approx(0.015, abs=1e-4)
    assert taubin < gap  # Taubin is the pessimistic one out here
    np.testing.assert_allclose(point, [0.0, 0.0, 0.00125], atol=1e-6)


# ---------------------------------------------------------------------------
# proxy_semi_axes -- the Section 1.7 bounding ellipsoid
# ---------------------------------------------------------------------------


def test_cube_proxy_encloses_every_corner():
    half = np.array([0.02, 0.03, 0.04])
    axes = scene.proxy_semi_axes({"type": "cube", "half_extents": half.tolist()})
    np.testing.assert_allclose(axes, np.sqrt(3.0) * half)

    # The documented property: each corner satisfies x^T M x = 1 exactly.
    for signs in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3):
        corner = signs * half
        assert float(np.sum((corner / axes) ** 2)) == pytest.approx(1.0)


def test_sphere_proxy_is_the_sphere():
    np.testing.assert_allclose(
        scene.proxy_semi_axes({"type": "sphere", "radius": 0.035}),
        [0.035, 0.035, 0.035],
    )


def test_capsule_proxy_extends_by_one_radius_along_local_y():
    axes = scene.proxy_semi_axes({"type": "capsule", "radius": 0.02, "height": 0.10})
    np.testing.assert_allclose(axes, [0.02, 0.05 + 0.02, 0.02])


def test_ellipsoid_is_its_own_proxy():
    np.testing.assert_allclose(
        scene.proxy_semi_axes({"type": "ellipsoid", "semi_axes": [0.05, 0.03, 0.01]}),
        [0.05, 0.03, 0.01],
    )


def test_proxy_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="no proxy ellipsoid"):
        scene.proxy_semi_axes({"type": "teapot"})


# ---------------------------------------------------------------------------
# object_extent_along -- what seats the support plane
# ---------------------------------------------------------------------------


def test_sphere_extent_is_the_radius_in_every_direction():
    spec = {"type": "sphere", "radius": 0.035}
    for n in (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.ones(3) / np.sqrt(3)):
        assert scene.object_extent_along(spec, n) == pytest.approx(0.035)


def test_extent_is_insensitive_to_normal_scaling():
    # The half-width along a direction cannot depend on how long the vector is.
    spec = {"type": "cube", "half_extents": [0.02, 0.03, 0.04]}
    n = np.array([0.0, 0.0, 1.0])
    assert scene.object_extent_along(spec, n) == pytest.approx(
        scene.object_extent_along(spec, 7.5 * n)
    )


def test_cube_extent_along_each_axis_is_its_half_extent():
    half = [0.02, 0.03, 0.04]
    spec = {"type": "cube", "half_extents": half}
    for axis in range(3):
        n = np.zeros(3)
        n[axis] = 1.0
        assert scene.object_extent_along(spec, n) == pytest.approx(half[axis])


# ---------------------------------------------------------------------------
# The primitive registry
# ---------------------------------------------------------------------------


def test_analytic_primitives_need_no_vdb_grid():
    """The committed suite depends on this: these specs must carry their geometry
    inline, so a checkout with no baked .vdb grids can still run solve tests.

    Every spec NAMES a grid now, since every object has an exact form as well as
    an approximate one -- so what has to be checked is that the analytic ones can
    still be configured without that file being present, not that they decline to
    mention it. The two questions were the same before objects carried both forms
    and are not the same now: the first is about this machine, the second about
    the object."""
    specs = scene.get_primitive_specs()
    analytic = [
        "coin",
        "credit_card",
        "pen",
        "big_sphere_ellipsoid",
        "mid_sphere_ellipsoid",
        "small_sphere_ellipsoid",
        "megaminx",
    ]
    for name in analytic:
        assert name in specs, f"{name} missing from the registry"
        spec = specs[name]
        assert spec["type"] in ("ellipsoid", "ellipsoid_set"), name
        # The geometry is inline: available with no file read at all.
        assert scene.ellipsoid_members(spec) is not None, name
        # ...and configuring one takes the analytic branch, so it works whether
        # or not this checkout has ever run the SDF baker. Pointed at a directory
        # that is definitely empty, so a grid sitting in OBJECTS_DIR cannot mask
        # a regression here.
        import gepetto_solvers

        env = gepetto_solvers.EnvironmentConfig()
        scene.configure_object_surface(env, spec, "/nonexistent", name)
        assert env.ellipsoid_semi_axes.any() or env.ellipsoid_set, name


def test_every_registry_spec_has_a_type():
    for name, spec in scene.get_primitive_specs().items():
        assert "type" in spec, name


def test_dodecahedron_has_twenty_vertices_on_a_common_sphere():
    verts = scene.dodecahedron_vertices(0.070)
    assert verts.shape == (20, 3)
    radii = np.linalg.norm(verts, axis=1)
    # All 20 lie on the circumsphere.
    np.testing.assert_allclose(radii, radii[0], rtol=1e-9)
