"""The Augmented Lagrangian constraint tags, and the dual transfer they enable.

This is the guard for anything that touches ``TendonHandModel::build_graph``.

``WarmALState`` indexes multipliers by a constraint's POSITION in
``ConstrainedOptProblem::eConstraints()``, which is graph insertion order. The C++
emits a semantic tag for every hard constraint at its insertion site, and
``remap_al_state()`` re-seats a carried state onto a rebuilt problem by matching
tag and dimension. So the tag list and the constraint enumeration have to stay in
lockstep: reorder the graph builder, or add a constraint through anything other
than ``add_eq``/``add_ineq``, and every tag after that point silently names the
wrong multiplier.

The failure is silent by construction -- the solve still runs, the numbers are
merely wrong -- which is exactly why it needs a test rather than a review. The
signal the architecture doc names is **0 matched against a non-empty carry**.

All marked ``slow``: each test builds a real constrained graph.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import solvers

pytestmark = pytest.mark.slow


def _params():
    return solvers.HandSolveParams(
        primitive="mid_sphere_ellipsoid", wrist_pose=np.eye(4)
    )


@pytest.fixture(scope="module")
def stepped(pinned_dims_module):
    """One stepped IK solve, and the tagged duals it produced."""
    stepper = solvers.HandIKStepper(_params())
    stepper.step()
    return stepper, stepper.al_duals()


@pytest.fixture(scope="module")
def pinned_dims_module():
    """Module-scoped twin of the `pinned_dims` fixture in conftest.

    Needed because the solves here are shared across tests via a module-scoped
    fixture, and pytest will not let a module-scoped fixture depend on a
    function-scoped one.
    """
    from _pkg import config, solvers as s

    original_cfg = config.load_hand_dimensions
    original_slv = getattr(s, "load_hand_dimensions", None)

    def _fallback():
        return config.DEFAULT_HAND_DIMENSIONS

    config.load_hand_dimensions = _fallback
    if original_slv is not None:
        s.load_hand_dimensions = _fallback
    yield config.DEFAULT_HAND_DIMENSIONS
    config.load_hand_dimensions = original_cfg
    if original_slv is not None:
        s.load_hand_dimensions = original_slv


def test_duals_are_tagged(stepped):
    """An untagged carry cannot be transferred across a rebuild at all."""
    _stepper, duals = stepped
    assert duals.tagged, (
        "multipliers arrived without constraint identities; a transfer across a "
        "rebuilt graph has nothing to match on"
    )


def test_every_constraint_has_a_tag(stepped):
    """One tag per constraint, in both families. A constraint added without going
    through add_eq/add_ineq shows up here as a length mismatch."""
    _stepper, duals = stepped

    assert len(duals.tags_equality) == duals.num_equality
    assert len(duals.tags_inequality) == duals.num_inequality
    assert duals.num_equality > 0
    assert duals.num_inequality > 0


def test_tags_are_well_formed(stepped):
    """Tags are `family.kind|scope` strings from a known vocabulary. A blank or
    duplicated tag means an insertion site forgot to name itself."""
    _stepper, duals = stepped
    tags = list(duals.tags_equality) + list(duals.tags_inequality)

    assert all(t and t.strip() for t in tags), "a constraint was inserted untagged"

    known = (
        "obj.center", "obj.witness", "obj.sphwit", "obj.sphere",
        "col.obj", "col.ff", "col.plane",
        "tbl.contact", "sup.contact", "half",
        "pregrasp.center", "pregrasp.align", "pregrasp.centroid",
    )
    for tag in tags:
        family = tag.split("|", 1)[0]
        assert family in known, f"unknown constraint tag family: {tag!r}"


def test_contact_tags_cover_the_contact_fingers(stepped):
    """The five fingertips are driven onto an ellipsoid, which the model builds in
    its witness-free center-direct form -- so there is one object equality per
    finger, and they are the equalities."""
    _stepper, duals = stepped
    obj = [t for t in duals.tags_equality if t.startswith("obj.")]

    assert len(obj) == 5, f"expected one object equality per finger, got {obj}"
    assert {t.split("|")[1] for t in obj} == {f"f{i}" for i in range(5)}


def test_self_collision_tags_name_both_spheres(stepped):
    """`col.ff|fAnI|fBnJ` -- a finger-finger pair names a node on each side. If the
    two halves ever collapse to one, distinct pairs share a tag and the transfer
    matches the wrong multiplier."""
    _stepper, duals = stepped
    ff = [t for t in duals.tags_inequality if t.startswith("col.ff|")]

    assert ff, "expected finger-finger collision constraints"
    for tag in ff:
        parts = tag.split("|")
        assert len(parts) == 3, tag
        assert parts[1] != parts[2], f"a sphere paired with itself: {tag}"
    assert len(set(ff)) == len(ff), "duplicate finger-finger tags"


def test_duals_transfer_across_a_rebuild(pinned_dims_module):
    """The end-to-end property all of the above exists to protect.

    Carry a converged solve's multipliers into a NEW solver posing the same
    constrained problem. The architecture doc names the regression signal
    exactly: 0 matched against a non-empty carry means a tag drifted, not that
    the problem genuinely changed.
    """
    first = solvers.HandIKStepper(_params())
    first.step()
    carried = first.al_duals()
    assert carried.num_equality + carried.num_inequality > 0

    params = _params()
    params.initial_duals = carried
    second = solvers.HandIKStepper(params)
    second.step()

    report = second.dual_transfer()
    assert report is not None, "no transfer report -- the carry was not consumed"
    assert report.total > 0
    assert report.matched > 0, (
        f"0 of {report.total} constraints matched against a non-empty carry. "
        "That is a drifted tag, not a changed problem -- check that every "
        "constraint still goes in through add_eq/add_ineq and that build_graph's "
        "insertion order is unchanged."
    )
    # The same problem posed twice should match essentially all of it.
    assert report.matched == report.total, (
        f"only {report.matched}/{report.total} matched for an identical problem"
    )
