"""h_grasp: the contacts must SURROUND the object, not merely touch it.

Two halves, because the constraint has two independent ways of being wrong and
they fail at different places:

* the RESIDUAL -- does the 6-vector vanish exactly when the contacts oppose one
  another? Checked against cases whose answer geometry fixes, on a grid built in
  the test rather than on one of the gitignored bakes.
* the WIRING -- does asking for it actually put a factor in the graph, and does
  asking for it wrongly say so? A constraint that quietly does not get built is
  invisible in the resulting pose, which is exactly what makes this worth a test.

The readout under test is :func:`solvers.grasp_wrench_witness`, which recomputes
the residual from the solved contact points. That is not merely convenient: the
violation the AL reports is WHITENED, so it falls as the sigmas are loosened
whether or not a fingertip moves, and it cannot be used to judge this constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import hands, solvers

pytest.importorskip("openvdb", reason="h_grasp reads the object's SDF")

RADIUS = 0.05


@pytest.fixture(scope="module")
def sphere_grid():
    """A 50 mm level-set sphere, built here rather than loaded: the baked grids
    are gitignored, and this suite has to run without them."""
    from gepetto_solvers.core.objects.sdf import require_openvdb

    vdb = require_openvdb()
    return vdb.createLevelSetSphere(radius=RADIUS, center=(0.0, 0.0, 0.0),
                                    voxelSize=0.001, halfWidth=30.0)


def _wrench(grid, points, origin=None, h=5e-4):
    """h_grasp at ``points``, from the grid's own gradient.

    Written out rather than called through the solver, so this measures the MATH
    the C++ factor implements rather than agreeing with it by construction."""
    origin = np.zeros(3) if origin is None else np.asarray(origin, float)
    accessor = grid.getConstAccessor()
    transform = grid.transform

    def phi(p):
        c = transform.worldToIndex(tuple(float(x) for x in p))
        return accessor.getValue(tuple(int(round(v)) for v in c))

    force = np.zeros(3)
    torque = np.zeros(3)
    for p in points:
        g = np.array([(phi(p + e) - phi(p - e)) / (2 * h)
                      for e in (np.array([h, 0, 0]), np.array([0, h, 0]),
                                np.array([0, 0, h]))])
        n = g / np.linalg.norm(g)
        force += -n
        torque += -np.cross(np.asarray(p) - origin, n)
    return force, torque


def test_antipodal_contacts_balance(sphere_grid):
    """The defining case: two contacts on opposite poles push against each other
    exactly, so both halves of the residual vanish."""
    points = [np.array([RADIUS, 0, 0]), np.array([-RADIUS, 0, 0])]
    force, torque = _wrench(sphere_grid, points)
    assert np.linalg.norm(force) < 1e-2
    assert np.linalg.norm(torque) < 1e-3


def test_same_side_contacts_do_not(sphere_grid):
    """Two contacts a quarter turn apart both push inward-ish the same way. This
    is the posture every per-contact witness factor is equally happy with, and
    the one h_grasp exists to reject: |(-x) + (-y)| = sqrt(2)."""
    points = [np.array([RADIUS, 0, 0]), np.array([0, RADIUS, 0])]
    force, _torque = _wrench(sphere_grid, points)
    assert np.linalg.norm(force) == pytest.approx(np.sqrt(2.0), abs=1e-2)


def test_an_equatorial_tripod_balances(sphere_grid):
    """Three contacts at 120 degrees -- the three-finger solution, and what a
    phase-4 solve on a sphere is being asked to find."""
    points = [RADIUS * np.array([np.cos(a), np.sin(a), 0.0])
              for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    force, torque = _wrench(sphere_grid, points)
    assert np.linalg.norm(force) < 1e-2
    assert np.linalg.norm(torque) < 1e-3


def test_the_torque_rows_are_degenerate_on_a_sphere(sphere_grid):
    """Worth pinning, because it explains a tuning result that otherwise looks
    like a bug: on a sphere the moment arm is parallel to the surface normal, so
    the cross product is identically zero and the three torque rows carry no
    information at all. ``sigma_grasp_torque`` does nothing on a round object."""
    for points in ([np.array([RADIUS, 0, 0]), np.array([0, RADIUS, 0])],
                   [np.array([0, 0, RADIUS])]):
        _force, torque = _wrench(sphere_grid, points)
        assert np.linalg.norm(torque) < 1e-3


# ---------------------------------------------------------------------------
# Wiring: the constraint reaches the graph, and a mis-request says so.
# ---------------------------------------------------------------------------

def _allegro_params(**overrides):
    hand = hands.get_hand("allegro")
    wrist, means = hand.default_pose()
    params = solvers.HandSolveParams(hand="allegro")
    params.wrist_pose = wrist
    params.joint_targets = [list(m) for m in means]
    for key, value in overrides.items():
        setattr(params, key, value)
    return hand, params


def test_the_capability_is_reported():
    """The panel greys its control off this, so a binding without the field must
    say so rather than offering a box that builds nothing."""
    assert "grasp_alignment" in solvers.capabilities()
    assert "contact_exact" in solvers.capabilities()


@pytest.mark.slow
def test_it_reaches_the_graph_alongside_the_proxy_collision():
    """THE SPLIT. In an exact-contact phase the contact reads the baked SDF while
    the collision inequalities keep reading the ellipsoid -- both families in one
    graph, on different surfaces. Asserted through the constraint TAGS, which are
    what the solver actually built."""
    from gepetto_solvers.core.geometry.scene import get_primitive_specs
    from gepetto_solvers.core.objects import has_exact_form

    primitive = "mid_sphere_ellipsoid"
    if not has_exact_form(get_primitive_specs()[primitive]):
        pytest.skip("grid not baked (python scripts/objects/setup_objects.py)")

    hand, params = _allegro_params(primitive=primitive)
    solvers.apply_phase_preset(params, "phase4")
    stepper = solvers.HandIKStepper(params, hand)
    stepper.step()

    duals = stepper.al_duals()
    families = {tag.split("|")[0]
                for tag in (list(duals.tags_equality) + list(duals.tags_inequality))}
    assert "grasp.align" in families, "h_grasp was not built"
    assert "obj.witness" in families, "the contact is not on the exact surface"
    assert "col.obj" in families, "object collision is not being built at all"


@pytest.mark.slow
def test_grasp_alignment_without_the_exact_contact_is_refused():
    """h_grasp keys off witness points, and the centre-direct contact form has
    none. Rather than skipping a constraint the caller asked for -- invisible in
    the pose -- the C++ layer raises."""
    hand, params = _allegro_params()
    solvers.apply_phase_preset(params, "phase2")     # proxy contact: no witness
    params.grasp_alignment = True
    with pytest.raises(Exception, match="grasp_alignment_enabled"):
        solvers.HandIKStepper(params, hand).step()


def test_one_contact_is_refused():
    """A single unit force cannot sum to zero, so the constraint would be
    infeasible rather than merely tight. Caught in Python, where the message can
    name the digit selection that caused it."""
    hand, params = _allegro_params()
    solvers.apply_phase_preset(params, "phase4")
    params.contact_fingers = [True, False, False, False]
    with pytest.raises(ValueError, match="two or more contacts"):
        solvers.HandIKStepper(params, hand)


@pytest.mark.slow
def test_the_witness_reports_the_raw_residual_not_the_whitened_one():
    """The readout must be independent of the sigmas, because that is the whole
    reason it exists: the AL's own violation is whitened and shrinks as they are
    loosened, with no fingertip having moved."""
    from gepetto_solvers.core.geometry.scene import get_primitive_specs
    from gepetto_solvers.core.objects import has_exact_form

    primitive = "mid_sphere_ellipsoid"
    if not has_exact_form(get_primitive_specs()[primitive]):
        pytest.skip("grid not baked (python scripts/objects/setup_objects.py)")

    norms = []
    for sigma in (100.0, 10000.0):
        hand, params = _allegro_params(primitive=primitive)
        solvers.apply_phase_preset(params, "phase4")
        params.sigma_grasp_force = sigma
        params.sigma_grasp_torque = sigma / 10.0
        stepper = solvers.HandIKStepper(params, hand)
        result = stepper.step()
        witness = solvers.grasp_wrench_witness(result)
        # Only the CONTACT digits, never the idle ones: including the Allegro's
        # uncommanded ring finger would add a whole unit vector to a residual
        # whose satisfied value is zero.
        assert witness.digits == result.contact_names()
        norms.append(witness.norm)

    # A hundredfold change in whitening moves the reported residual by nothing
    # like a hundredfold -- it is measuring the geometry, not the noise model.
    assert norms[1] == pytest.approx(norms[0], rel=1.0)
