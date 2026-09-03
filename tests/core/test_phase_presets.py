"""The phase presets, checked against the hand each set belongs to.

These are data, so what can go wrong with them is data-shaped: a per-digit list
of the wrong length, a field that no longer exists, two mutually exclusive
options set at once. None of those raises when the preset is written -- they
raise, or worse do not, when somebody applies it to a hand.

The five-element mask is the specific trap. Every preset carries a positional
``contact_fingers``, and the tendon hand's is ``[True, True, False, False,
True]``; handed to a four-digit hand that is not a smaller grasp, it is a
different one, silently.
"""

from __future__ import annotations

import pytest

from _pkg import hands, solvers

HAND_NAMES = sorted(hands.registered_hands())


def _preset_ids():
    for hand_name in HAND_NAMES:
        for phase in solvers.phase_presets(hand_name):
            yield hand_name, phase


ALL_PRESETS = list(_preset_ids())


@pytest.mark.parametrize(("hand_name", "phase"), ALL_PRESETS,
                         ids=[f"{h}-{p}" for h, p in ALL_PRESETS])
def test_a_preset_applies_to_its_own_hand(hand_name, phase):
    """No unknown fields: ``apply_phase_preset`` raises on one, which is the
    point of it doing so rather than silently no-opping."""
    params = solvers.HandSolveParams(hand=hand_name)
    solvers.apply_phase_preset(params, phase)


@pytest.mark.parametrize(("hand_name", "phase"), ALL_PRESETS,
                         ids=[f"{h}-{p}" for h, p in ALL_PRESETS])
def test_a_preset_masks_match_its_hands_digit_count(hand_name, phase):
    """THE TRAP. A positional per-digit list sized for another hand does not
    fail -- it selects different fingers."""
    hand = hands.get_hand(hand_name)
    params = solvers.HandSolveParams(hand=hand_name)
    solvers.apply_phase_preset(params, phase)
    assert len(params.contact_fingers) == len(hand.digit_names), (
        f"{hand_name}/{phase} names {len(params.contact_fingers)} digits; the "
        f"hand has {len(hand.digit_names)}")


@pytest.mark.parametrize(("hand_name", "phase"), ALL_PRESETS,
                         ids=[f"{h}-{p}" for h, p in ALL_PRESETS])
def test_a_preset_names_one_object_contact_form(hand_name, phase):
    """There is one object contact, in one of three forms. ``object_contact_form``
    raises on a combination naming two, so calling it IS the assertion."""
    params = solvers.HandSolveParams(hand=hand_name)
    solvers.apply_phase_preset(params, phase)
    assert solvers.object_contact_form(params) in solvers.OBJECT_CONTACT_FORMS


@pytest.mark.parametrize("hand_name", HAND_NAMES)
def test_every_hand_has_the_default_phase(hand_name):
    """The workbench opens on ``DEFAULT_PHASE`` and applies it for real. A hand
    whose set does not name it would open showing a phase it cannot enter."""
    from gepetto_solvers.projects.viz.viz_interactive.constants import DEFAULT_PHASE

    assert DEFAULT_PHASE in solvers.phase_presets(hand_name)


def test_the_hands_have_their_own_sets():
    """Not one shared table. The Allegro pipeline's phase 4 is the grasp-wrench
    alignment and the tendon hand's is a commanded close -- the same key naming
    two different things is exactly why the sets are per hand."""
    tendon = solvers.phase_presets("tendon_5f")
    allegro = solvers.phase_presets("allegro")
    assert tendon is not allegro
    assert allegro["phase4"].overrides["grasp_alignment"] is True
    assert tendon["phase4"].overrides["object_contact"] is False
    assert tendon["phase4"].requires_feature == "close_ramp"
    assert allegro["phase4"].requires_feature is None


def test_an_unknown_hand_falls_back_rather_than_raising():
    """A newly registered hand with no set of its own gets the default one, so
    it is usable before anyone has written its pipeline down."""
    assert solvers.phase_presets("no_such_hand") is solvers.PHASE_PRESETS
    assert solvers.phase_presets(None) is solvers.PHASE_PRESETS


def test_the_allegro_phases_are_the_formulations():
    """The constraint table, transcribed -- so a preset edited by hand cannot
    quietly stop matching the pipeline it implements.

    Note what is NOT asserted: which spheres each inequality covers. The paper
    writes h_pen over ``I`` in some phases and ``I \\ C`` in others, and no
    preset says so, because the C++ layer exempts a sphere that carries the
    corresponding contact node. The masks follow from the contact set."""
    want = {
        #            contact form   table   grasp
        "phase0": ("none", False, False),
        "phase1": ("none", True, False),
        "phase2": ("proxy", True, False),
        "phase3": ("exact", True, False),
        "phase4": ("exact", False, True),
    }
    presets = solvers.phase_presets("allegro")
    assert set(presets) == set(want)
    for phase, (form, table, grasp) in want.items():
        params = solvers.HandSolveParams(hand="allegro")
        solvers.apply_phase_preset(params, phase)
        assert solvers.object_contact_form(params) == form, phase
        assert params.table_contact is table, phase
        assert params.grasp_alignment is grasp, phase
        # Avoidance is on in EVERY phase -- the departure the tendon presets
        # make (turning the plane's off where fingers are driven onto it) is
        # unnecessary, because the contact sphere is already exempt.
        assert params.collision and params.table and params.plane_avoidance, phase
        # And no pre-grasp constraint anywhere: this formulation puts the
        # pre-grasp in the priors, not in a constraint.
        assert not (params.pregrasp_center or params.pregrasp_axis_align
                    or params.pregrasp_centroid or params.half_space), phase
