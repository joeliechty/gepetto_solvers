"""The Allegro pipeline, walked end to end.

Each phase warm-starts from the last -- posture, wrist and AL multipliers -- which
is what makes it one continuous move rather than five unrelated solves, and is
the part most likely to break silently: a rebuilt solver that cold-starts still
produces a picture, just one that drifts off constraints the previous phase had
already satisfied.

What is asserted is the CONSTRAINT SET each phase builds, read back from the
solver's own tags. Convergence is not: phase 4 in particular is a genuinely hard
ask, and pinning a violation here would turn a research result into a broken
test. See ``test_grasp_alignment.py`` for what is checked about that constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import hands, solvers
from gepetto_solvers.core.geometry.scene import get_primitive_specs
from gepetto_solvers.core.objects import has_exact_form

pytestmark = pytest.mark.slow

#: Analytic, so phases 0-2 run on any checkout. Phases 3-4 additionally need its
#: grid, and skip when it has not been baked.
PRIMITIVE = "mid_sphere_ellipsoid"

#: What each phase must put in the graph, by constraint-tag family. The
#: pipeline's shape, in the terms the solver reports.
EXPECTED = {
    "phase0": {"col.obj", "col.plane", "col.ff"},
    "phase1": {"col.obj", "col.plane", "col.ff", "tbl.contact"},
    "phase2": {"col.obj", "col.plane", "col.ff", "tbl.contact", "obj.center"},
    "phase3": {"col.obj", "col.plane", "col.ff", "tbl.contact", "obj.witness"},
    "phase4": {"col.obj", "col.plane", "col.ff", "obj.witness", "grasp.align"},
}


def _families(stepper):
    duals = stepper.al_duals()
    return {tag.split("|")[0]
            for tag in list(duals.tags_equality) + list(duals.tags_inequality)}


@pytest.fixture(scope="module")
def walk():
    """Run the pipeline once, keeping each phase's stepper and result."""
    if not has_exact_form(get_primitive_specs()[PRIMITIVE]):
        pytest.skip(f"{PRIMITIVE} has no baked grid "
                    "(python scripts/objects/setup_objects.py)")
    hand = hands.get_hand("allegro")
    wrist, means = hand.default_pose()

    out = {}
    carry = {}
    for phase in EXPECTED:
        params = solvers.HandSolveParams(hand="allegro")
        params.primitive = PRIMITIVE
        params.wrist_pose = np.asarray(wrist, float)
        params.joint_targets = [list(m) for m in means]
        solvers.apply_phase_preset(params, phase)
        for key, value in carry.items():
            setattr(params, key, value)

        stepper = solvers.HandIKStepper(params, hand)
        status = stepper.run(max_steps=25)
        result = stepper.step()
        out[phase] = (stepper, status, result)

        carry = dict(
            initial_state=result.state(0),
            initial_duals=result.duals,
            wrist_pose=np.asarray(result.wrist_pose(0), float),
            joint_targets=[list(np.asarray(result.frames[0][n].actuation(), float))
                           for n in hand.digit_names])
    return out


@pytest.mark.parametrize("phase", list(EXPECTED))
def test_each_phase_builds_its_own_constraint_set(walk, phase):
    stepper, _status, _result = walk[phase]
    assert _families(stepper) == EXPECTED[phase]


def test_the_exact_phases_contact_the_grid_while_collision_keeps_the_proxy(walk):
    """THE SPLIT, stated as a transition rather than a snapshot: phase 2 contacts
    the ellipsoid centre-direct, phase 3 contacts the grid through a witness, and
    ``col.obj`` -- the ellipsoid inequality -- is there in both.

    Without the split, moving the contact to the exact geometry would have to
    move collision there too, and the free spheres would go back to being steered
    by a baked grid's flat faces and sharp edges."""
    assert "obj.center" in _families(walk["phase2"][0])
    assert "obj.witness" not in _families(walk["phase2"][0])
    assert "obj.witness" in _families(walk["phase3"][0])
    assert "obj.center" not in _families(walk["phase3"][0])
    for phase in ("phase2", "phase3", "phase4"):
        assert "col.obj" in _families(walk[phase][0]), phase


def test_the_support_equality_is_released_for_the_grasp(walk):
    """Phase 4 lets go of the table: the hand is holding the object by then, and
    pinning the grasp digits to the plane would fight the arrangement the phase
    is solving for. Avoidance stays -- and now covers every sphere, since no
    digit carries the table contact node any more."""
    assert "tbl.contact" in _families(walk["phase3"][0])
    assert "tbl.contact" not in _families(walk["phase4"][0])
    assert "col.plane" in _families(walk["phase4"][0])


def test_the_approach_phases_close_on_their_constraints(walk):
    """Phases 0-2 are expected to actually work; they are the part of the
    pipeline this hand and object are comfortably capable of."""
    for phase in ("phase0", "phase1", "phase2"):
        _stepper, status, _result = walk[phase]
        assert status.violation < 5e-2, (phase, status)


def test_the_warm_start_carries_the_posture_across_a_phase(walk):
    """Each phase changes the constraint set, which forces a rebuilt solver --
    and a rebuild cold-starts unless the posture is handed across. Measured as
    the hand not jumping between the end of one phase and the start of the next."""
    for before, after in (("phase1", "phase2"), ("phase2", "phase3")):
        end = walk[before][2].wrist_pose(0)
        start = walk[after][2].wrist_pose(0)
        moved = float(np.linalg.norm(np.asarray(end)[:3, 3]
                                     - np.asarray(start)[:3, 3]))
        assert moved < 0.05, (before, after, moved)
