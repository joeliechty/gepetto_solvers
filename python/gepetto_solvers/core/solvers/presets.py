"""Named :class:`HandSolveParams` override groups for the staged pipeline.

A preset touches ONLY the fields it lists -- wrist pose, flexor tensions, AL
sliders and table height stay wherever the caller left them, because those are
solver knobs rather than part of what defines a phase. :func:`apply_phase_preset`
raises on an override naming a field ``HandSolveParams`` does not have, so a typo
fails loudly instead of silently no-opping.

**The presets are PER HAND** (:data:`HAND_PHASE_PRESETS`, :func:`phase_presets`).
They were not always, and the reason they had to become so is that a phase is a
statement about a specific mechanism grasping in a specific way, not a universal:

  * the per-digit masks are POSITIONAL, so a five-element ``contact_fingers``
    written for the tendon hand is meaningless -- and silently wrong -- on a
    four-digit one;
  * whole constraints are unavailable on one hand and central on another. The
    tendon presets lean on the measured pinch table (``pregrasp_centroid``, the
    Eq 13 in-plane contact form) that a hand without one simply does not have;
  * even the NUMBERING differs. ``phase4`` is the tendon hand's commanded
    synchronized close; for the Allegro hand it is the grasp-wrench alignment.
    Both are "phase 4" in their own formulation, and renumbering either to make
    one flat table work would leave the panel disagreeing with the paper it is
    implementing.

:data:`PHASE_PRESETS` remains the tendon hand's set and keeps its name, so every
existing caller and every existing reference to it is unchanged.
"""

from dataclasses import dataclass

from .params import HandSolveParams

# ---------------------------------------------------------------------------
# Phase presets.
# ---------------------------------------------------------------------------

@dataclass
class PhasePreset:
    """A named group of ``HandSolveParams`` overrides for one phase of the
    §1.8-style pipeline (0: pre-grasp positioning, 1: support contact, 2:
    object approach, 3: on-object servoing -- 3 is not populated yet). Phase 4,
    the synchronized close, is the odd one out: it is not an AL solve at all but
    a commanded tendon ramp (:func:`synchronized_close`), and its overrides say
    so by switching every constraint off. Only the fields listed in
    ``overrides`` are touched when applied --
    wrist pose, per-finger flexor tensions, AL/collision tuning sliders, table
    height offset etc. are left at whatever the caller already has, since
    those are generic solver knobs rather than part of what DEFINES a phase."""
    label: str
    overrides: dict[str, object]
    #: One paragraph of panel help: what this phase enforces and what to press
    #: next. Lives on the preset rather than in the GUI because it describes the
    #: PHASE, and the phases are per hand -- a panel that held its own copy would
    #: have to carry one hand's explanation of another hand's pipeline.
    hint: str = ""
    #: A ``hands.base.FEATURES`` name this phase needs the hand to declare, or
    #: None. Only the two commanded RAMPS use it: they are not solves at all, and
    #: their runners exist only for a hand carrying the measured travel to walk,
    #: so offering the preset on a hand that cannot run it would be a control
    #: that does nothing. Stated here rather than as a rule about phase NUMBERS
    #: in the panel, because the number means different things per hand -- on the
    #: Allegro hand phase 4 is the grasp alignment and needs no feature at all.
    requires_feature: str | None = None


PHASE_PRESETS: dict[str, PhasePreset] = {
    "phase0": PhasePreset(
        label="Phase 0: pre-grasp positioning",
        hint=(
        "Apply the phase-0 preset: no object/table contact yet, "
        "collision avoidance on, pinch-centroid centering + short- "
        "axis alignment on (the opposition half-space and fingertip- "
        "midpoint centering stay OFF -- the pinch centroid already "
        "positions the hand and the other two fight it), and a loose "
        "wrist prior (this is a big repositioning move). Writes "
        "straight onto the Constraints/Wrist controls -- check this, "
        "then press Auto solve. Your finger selection is left alone, "
        "as it is by every preset. Unchecking is a no-op."),
        overrides=dict(
            object_contact=False,
            table_contact=False,
            collision=True,
            table=True,
            plane_avoidance=True,
            # Centering is done by the PINCH CENTROID, not by the achieved
            # fingertip midpoint: the measured hand-frame pinch point is a
            # constraint on the wrist alone, so it positions the hand for a
            # grasp whatever the fingers are doing now, while pregrasp_center
            # only says something once they are nearly closed. The two impose
            # different targets, so exactly one of them runs.
            pregrasp_center=False,
            pregrasp_centroid=True,
            # Off with it: the opposition half-space keeps the thumb and the
            # opposing fingers apart around the split, which the pinch centroid
            # already implies (it places the hand so closing those digits closes
            # them ON the object) -- and it is the constraint most prone to
            # stalling the solve on a bad side assignment.
            half_space=False,
            # The one term here that actually rotates the wrist: the centroid
            # constraint is satisfiable by translation alone.
            pregrasp_axis_align=True,
            # Standoff above the object centroid for the pinch point.
            h_clear=0.07,
            contact_drop_normal_row=False,
            contact_fingers=[True, True, False, False, True],  # index, middle, thumb
            # Loose wrist prior: phase 0 is a big repositioning move, so the
            # wrist must be free to get there rather than held near its start.
            sigma_wrist_pos=1.0,
            sigma_wrist_rot=1.0,
            # Explicit even though it equals the field's own default -- states
            # plainly that phase 0 uses the standard flexor looseness rather
            # than leaving it at "whatever the slider happened to be at."
            flexor_tension_sigma=0.1 ** 0.5,
        ),
    ),
    "phase1": PhasePreset(
        label="Phase 1: support contact",
        hint=(
        "Apply the phase-1 preset: table contact ON (object contact "
        "stays off), table COLLISION avoidance OFF -- a deliberate "
        "departure from the paper, since this phase drives the "
        "fingers onto the plane the avoidance half-space would push "
        "them off (the table itself stays on; object/self collision "
        "are untouched) -- the three pre-grasp constraints "
        "(opposition half-space, centering, short-axis alignment) "
        "turned back OFF now that they've done their job, and a "
        "tighter wrist prior than phase 0 (held closer to where it "
        "ended up, not free to roam). Writes straight onto the "
        "Constraints/Wrist controls -- check this, then press Auto "
        "solve; whichever fingers phase 0 was solved with carry over "
        "untouched. Unchecking is a no-op."),
        overrides=dict(
            object_contact=False,
            table_contact=True,
            collision=True,
            table=True,
            # Table COLLISION off (a deliberate departure from the paper, which
            # keeps it on): phase 1 drives the fingers deliberately onto the
            # plane, so the avoidance half-space is pushing against the very
            # contact this phase exists to make. `table` stays on -- the plane
            # itself is still needed, table_contact is built against it.
            plane_avoidance=False,
            # The three pre-grasp-only constraints did their job getting the
            # hand into position in phase 0; phase 1 slides the fingers onto
            # the table and doesn't need them anymore.
            half_space=False,
            pregrasp_center=False,
            pregrasp_axis_align=False,
            pregrasp_centroid=False,
            contact_drop_normal_row=False,
            contact_fingers=[True, True, False, False, True],  # index, middle, thumb
            # Tighter than phase 0's 1.0: the big repositioning move is done,
            # so the wrist is held closer to where phase 0 left it -- but not
            # fully rigid (phase 0's own sigma_wrist_pos/rot default is much
            # smaller still), since settling into contact needs some give.
            sigma_wrist_pos=0.01,
            sigma_wrist_rot=0.01,
            flexor_tension_sigma=0.1 ** 0.5,
            # h_clear intentionally omitted -- pregrasp_center is off here, so
            # a clearance value would be inert and misleading to state.
        ),
    ),
    "phase2": PhasePreset(
        label="Phase 2: object approach",
        hint=(
        "Apply the phase-2 preset: object contact turned back ON and "
        "table contact turned OFF -- the fingers are handed off from "
        "the plane they settled on to the object itself, in the Eq 13 "
        "IN-PLANE form (measured inside each finger's pulling plane, "
        "so the solve is not asked for torsion the tendons cannot "
        "produce; falls back to the 3D form on a scene that cannot "
        "build it). Table collision avoidance still OFF as in phase 1 "
        "(the fingers arrive still lying on the plane, so the half- "
        "space would be violated from the first step; object and self "
        "collision stay on), pre-grasp constraints still off, and the "
        "wrist prior kept TIGHT at phase 1's level -- with nothing "
        "else holding the hand, a loose wrist rides the whole hand "
        "onto the object instead of closing the fingers around it. "
        "Tendon sigmas set to the standard loose-flexor/tight-passive "
        "pair. Writes straight onto the Constraints/Wrist/Tensions "
        "controls -- check this, then press Auto solve; the finger "
        "selection carries over from phase 1. Unchecking is a no-op."),
        overrides=dict(
            # The change from phase 1: the object becomes the contact target
            # and the table stops being one. Phase 1 put the fingers ON the
            # plane; phase 2 hands them off to the object, so keeping the
            # table equalities would pin the fingertips to the plane while
            # the object constraint tries to lift them onto its surface.
            object_contact=True,
            table_contact=False,
            # Eq 13: measure the fingertip-onto-object equality inside each
            # finger's pulling plane rather than in full 3D. Same factor
            # count and the same zero set -- but the solve is not asked for
            # out-of-plane torsion the tendons cannot produce, which is what
            # the approach actually has to execute. Needs an ellipsoid/ycb
            # object and a digit set including the thumb (both hold for the
            # contact_fingers below); see the gate in viz_interactive.
            object_contact_in_plane=True,
            collision=True,
            self_collision=True,
            table=True,
            # Off as in phase 1. Table contact is no longer requested here,
            # but the fingers arrive at the object still lying on the plane
            # they slid in on, so the avoidance half-space would be violated
            # from the first step. Object collision (`collision`) stays on --
            # only the PLANE's avoidance is dropped.
            plane_avoidance=False,
            half_space=False,
            pregrasp_center=False,
            pregrasp_axis_align=False,
            pregrasp_centroid=False,
            contact_drop_normal_row=False,
            contact_fingers=[True, True, False, False, True],  # index, middle, thumb
            # Tight, as in phase 1 rather than phase 0's 1.0: the big
            # repositioning move belongs to phase 0. With the pre-grasp terms
            # off, a loose wrist lets the object equality drag the whole hand
            # onto the object instead of closing the fingers around it, so
            # the wrist is held near where phase 1 left it and the FINGERS
            # make the approach.
            sigma_wrist_pos=0.01,
            sigma_wrist_rot=0.01,
            flexor_tension_sigma=0.1 ** 0.5,
            # Stated for the same reason as the flexor sigma above: the
            # passive tendons stay at their tight default rather than
            # inheriting whatever the slider was last dragged to. Do not go
            # much below this against the flexor's far looser scale -- see
            # the IndeterminantLinearSystem note on the field itself.
            passive_tension_sigma=1e-3,
            # h_clear intentionally omitted, as in phase1 -- pregrasp_center
            # is off, so a clearance value would be inert and misleading.
        ),
    ),
    # phase3 lands here later, same shape.
    "phase4": PhasePreset(
        requires_feature="close_ramp",
        label="Phase 4: synchronized close",
        hint=(
        "Apply the phase-4 preset: every constraint OFF -- object and "
        "table contact, collision avoidance, the opposition half- "
        "space and all three pre-grasp terms -- because this phase "
        "does not SOLVE for anything. It shuts the grasping fingers "
        "on a commanded schedule and whatever they meet on the way, "
        "they meet. The runner is **Close**, up in the Solver folder, "
        "NOT Auto solve: check this, then press Close. The fingers it "
        "shuts are the ones checked below -- the same set phases 0-2 "
        "positioned, since no preset touches that mask -- and the "
        "wrist prior is left tight (the close does not move the wrist "
        "at all). Unchecking is a no-op."),
        overrides=dict(
            # Nothing is ENFORCED in phase 4. The close is a commanded tendon
            # ramp run through the FK solver (:func:`synchronized_close`), not
            # an AL solve: the grasping fingers are pulled shut on a schedule
            # and whatever they meet on the way, they meet. Every constraint
            # therefore goes OFF, so the panel cannot advertise a goal the
            # phase is not pursuing -- the object and table equalities, the
            # opposition half-space, and all three pre-grasp terms.
            object_contact=False,
            table_contact=False,
            half_space=False,
            pregrasp_center=False,
            pregrasp_axis_align=False,
            pregrasp_centroid=False,
            contact_drop_normal_row=False,
            # Collision avoidance goes with them, and for a blunter reason than
            # phase 1 had for dropping the plane's: FK does not read these flags
            # AT ALL (HandFKSolver never attaches the environment), so leaving
            # them ticked would draw avoidance spheres around a solve that is
            # not avoiding anything and claim a guarantee the close cannot make.
            # This is the phase where the fingers are allowed to hit things.
            collision=False,
            self_collision=False,
            # The plane itself stays. It is what seats the table in the scene,
            # and the table square's corner is the registration the robot plan
            # is built against -- dropping it would move every pose the Robot
            # folder sends. Only its avoidance half-space is gone.
            table=True,
            plane_avoidance=False,
            # The grasping set: the digits the close actually drives, and the
            # same three-finger pinch phases 0-2 positioned, so the close shuts
            # the hand that the pre-grasp aimed. Fingers left out of this mask
            # are HELD at whatever tension they are already carrying.
            contact_fingers=[True, True, False, False, True],  # index, middle, thumb
            # Kept at phase 1/2's tight values. The wrist is not a variable to
            # be traded here -- FK re-commands it every substep and the close
            # does not move it by a millimetre -- but the numbers are written
            # anyway so the panel does not sit showing phase 0's loose prior
            # next to a phase that holds the wrist still.
            sigma_wrist_pos=0.01,
            sigma_wrist_rot=0.01,
        ),
    ),
    "phase5": PhasePreset(
        requires_feature="close_ramp",
        label="Phase 5: lift",
        hint=(
        "Apply the phase-5 preset: every constraint OFF, for phase "
        "4's reason -- this phase does not solve for anything either. "
        "It raises the wrist on a commanded ramp and the hand goes up "
        "holding whatever the close left it holding; nothing in the "
        "model holds the OBJECT, so the object stays where it is. The "
        "runner is **Lift**, up in the Solver folder, NOT Auto solve: "
        "check this, then press Lift. The finger checkboxes are left "
        "alone, as by every preset -- a lift follows a close, and the "
        "grasping set is whatever that close shut. Unchecking is a "
        "no-op."),
        overrides=dict(
            # Phase 5 enforces exactly as much as phase 4 does: nothing. The
            # lift is a commanded wrist ramp run through the FK solver
            # (:func:`lift_wrist`), so every constraint goes off for phase 4's
            # reasons -- FK never attaches the environment, and a ticked box
            # here would claim a guarantee the lift cannot make. Worth saying
            # out loud for this phase in particular: NOTHING in the model holds
            # the object. The hand rises carrying whatever tension the close
            # left it pulling with, and whether that is enough to take the
            # object with it is a question this phase does not ask.
            object_contact=False,
            table_contact=False,
            half_space=False,
            pregrasp_center=False,
            pregrasp_axis_align=False,
            pregrasp_centroid=False,
            contact_drop_normal_row=False,
            collision=False,
            self_collision=False,
            # The plane stays for phase 4's reason -- it is the registration the
            # robot plan is built against, not a constraint.
            table=True,
            plane_avoidance=False,
            # contact_fingers is DELIBERATELY absent. A lift follows a close,
            # and the grasping set is what the close just shut; re-writing it
            # here would let ticking this box quietly change which digits the
            # panel says are holding on. `_apply_phase_preset` only writes the
            # keys a preset names, so omitting it leaves the set alone.
            #
            # The wrist prior, unlike phase 4's, is doing real work: it is the
            # only thing pulling the hand up the ramp, and the lift checks that
            # the solve tracked it (HandFKSolver._WRIST_TRACKING_TOL_M).
            sigma_wrist_pos=0.01,
            sigma_wrist_rot=0.01,
        ),
    ),
}


# ---------------------------------------------------------------------------
# The Allegro hand's phases: the formulation, transcribed.
# ---------------------------------------------------------------------------

#: The three-digit precision grasp, in ``allegro.DIGIT_NAMES`` order
#: (index, middle, ring, thumb -- four digits, thumb last). Written once and
#: shared, because every preset names it and a length mismatch against the hand
#: is the single most likely way to get this file wrong.
_ALLEGRO_PINCH = [True, True, False, True]

#: Loose enough for phase 0's repositioning move, tight enough for everything
#: after it. Same two values the tendon presets use, for the same reasons.
_FREE_WRIST = dict(sigma_wrist_pos=1.0, sigma_wrist_rot=1.0)
_HELD_WRIST = dict(sigma_wrist_pos=0.01, sigma_wrist_rot=0.01)

#: What every Allegro phase shares. Collected rather than repeated five times,
#: because these are exactly the settings that do NOT vary across the pipeline,
#: and spelling them out per phase invites one of them to drift.
#:
#: The three avoidance switches are ON THROUGHOUT. That looks like a departure
#: from the tendon presets, which turn ``plane_avoidance`` off for the phases
#: that drive fingers onto the plane -- but it is not a policy difference, it is
#: the same policy stated correctly. The formulation asks for h_pen over
#: ``I \ C``, and the C++ layer builds precisely that: a sphere carrying the
#: contact node for a surface is EXEMPT from that surface's inequality
#: (``HandGraph.cpp``, the ``is_contact`` and ``table_contact_node`` skips). So
#: the avoidance never fights the contact it is paired with, and the mask that
#: results changes on its own as each phase changes which contacts it sets:
#: a phase with no table contact gets ``forall i in I``, one with it gets
#: ``forall i in I \ C``. Nothing here has to say which.
#:
#: ``object_proxy_and_exact`` is likewise on throughout, so h_pen is measured
#: against ``E_obj`` in every phase -- including 3 and 4, where the CONTACT has
#: moved to the exact geometry. That split is the whole point of the field.
_ALLEGRO_COMMON = dict(
    collision=True,
    self_collision=True,
    table=True,
    plane_avoidance=True,
    object_proxy_and_exact=True,
    contact_fingers=_ALLEGRO_PINCH,
    # Every pre-grasp term is off in every phase. Not because this hand lacks
    # the measured pinch pose two of them need (it does), but because the
    # formulation has no such constraint: its phase 0 encodes the pre-grasp in
    # the PRIORS p(T_w) and p(q) -- i.e. in `wrist_pose` and `joint_targets` --
    # and constrains only non-penetration. Stated in the shared block rather
    # than per phase so it reads as the standing fact it is.
    pregrasp_center=False,
    pregrasp_axis_align=False,
    pregrasp_centroid=False,
    half_space=False,
    # The tendon hand's Eq 13 accommodation: measure the object contact inside
    # the plane a tendon can pull along. A joint-space hand has no such plane and
    # the formulation asks for the plain 3D distance, so this stays off and the
    # contact FORM is chosen between the other two below.
    object_contact_in_plane=False,
)


def _allegro_phase(label, hint="", **overrides):
    """One Allegro phase: the shared settings, with this phase's on top."""
    return PhasePreset(label=label, hint=hint,
                       overrides={**_ALLEGRO_COMMON, **overrides})


#: The formulation's five phases for the Allegro hand.
#:
#: The object CONTACT is one thing in one of three forms, never two things:
#: ``object_contact`` (3D, against the ellipsoid ``E_obj``),
#: ``object_contact_in_plane`` (unused here) and ``object_contact_exact`` (the
#: witness contact against the baked SDF). Phases 3 and 4 therefore turn
#: ``object_contact`` OFF while contacting the object harder than ever -- the
#: contact has not gone away, it has changed surface.
ALLEGRO_PHASE_PRESETS: dict[str, PhasePreset] = {
    "phase0": _allegro_phase(
        "Phase 0: pre-grasp positioning",
        hint=(
        "Apply the phase-0 preset: nothing is CONTACTED yet, only "
        "avoided -- the fingers and the wrist are kept out of the "
        "object and off the table while the hand moves to its pre- "
        "grasp. There is no pre-grasp CONSTRAINT here on purpose: the "
        "formulation puts that posture in the priors, so it is the "
        "wrist sliders and the joint sliders that say where the hand is "
        "going, and this phase only guarantees the way there is clear. "
        "The wrist prior is loose, because this is the one big "
        "repositioning move in the pipeline. Check this, then press "
        "Auto solve."),
        # Nothing is CONTACTED in phase 0. The pre-grasp posture is carried
        # entirely by the priors -- the wrist pose and the commanded joint
        # targets -- so all this phase enforces is that getting there does not
        # pass through the object or the table.
        object_contact=False,
        object_contact_exact=False,
        table_contact=False,
        grasp_alignment=False,
        contact_drop_normal_row=False,
        # Loose, because this is the one big repositioning move in the pipeline
        # and the wrist has to be free to make it. Every later phase holds the
        # wrist near where the previous one left it.
        **_FREE_WRIST,
    ),
    "phase1": _allegro_phase(
        "Phase 1: support contact",
        hint=(
        "Apply the phase-1 preset: the checked fingertips are driven "
        "down onto the SUPPORT SURFACE. The object is still only an "
        "obstacle. Avoidance stays ON, unlike the tendon hand's phase 1 "
        "-- it does not fight the contact, because a sphere carrying "
        "the table contact is exempt from the table inequality, so what "
        "gets built is avoidance for every OTHER sphere, which is what "
        "the formulation asks for. The wrist prior tightens here and "
        "stays tight for the rest of the pipeline. Check this, then "
        "press Auto solve."),
        object_contact=False,
        object_contact_exact=False,
        # The change from phase 0: the grasp digits are driven down onto the
        # support surface. The object is still only an obstacle.
        table_contact=True,
        grasp_alignment=False,
        contact_drop_normal_row=False,
        **_HELD_WRIST,
    ),
    "phase2": _allegro_phase(
        "Phase 2: object approach",
        hint=(
        "Apply the phase-2 preset: slide along the support surface into "
        "contact with the object's APPROXIMATION -- the smooth "
        "ellipsoid form, whose analytic gradients let a fingertip slide "
        "across the surface instead of catching on it. Table contact "
        "stays ON, so the hand slides along the plane rather than "
        "lifting off it. This is an approach, not a grasp: the contacts "
        "are asked to touch, not yet to oppose. Check this, then press "
        "Auto solve."),
        # Slide along the support surface into contact with the object's
        # APPROXIMATION -- the smooth ellipsoid form, whose analytic Jacobians
        # let the solve slide across the surface instead of catching on it.
        # Table contact stays on: the hand slides along the plane rather than
        # lifting off it, which is what makes this a slide and not a reach.
        object_contact=True,
        object_contact_exact=False,
        table_contact=True,
        grasp_alignment=False,
        contact_drop_normal_row=False,
        **_HELD_WRIST,
    ),
    "phase3": _allegro_phase(
        "Phase 3: exact object contact",
        hint=(
        "Apply the phase-3 preset: the same contact, on the EXACT "
        "geometry. A witness point per finger is driven onto the "
        "object's baked signed-distance field, in the 4-row form (the "
        "normal-alignment row is redundant with sphere-only collision "
        "geometry). Collision does NOT follow it there -- the free "
        "spheres keep being steered by the smooth ellipsoid, which is "
        "the split this phase exists to make. Needs an object whose "
        "exact form has been baked; the box greys out when it has not. "
        "Check this, then press Auto solve."),
        # The handoff: same contact, exact surface. A witness point per digit is
        # driven onto the true geometry (h_rad, h_sdf, h_tan1, h_tan2) while the
        # free spheres keep being steered by the ellipsoid, which is why
        # `collision` above stays on and means E_obj here.
        object_contact=False,
        object_contact_exact=True,
        # The 4-row form. With the collision geometry modeled exclusively as
        # spheres, the two tangential-slip rows already force the sphere's radius
        # vector collinear with the surface normal, so the explicit normal-
        # alignment row c_N is redundant.
        contact_drop_normal_row=True,
        table_contact=True,
        grasp_alignment=False,
        **_HELD_WRIST,
    ),
    "phase4": _allegro_phase(
        "Phase 4: grasp alignment",
        hint=(
        "Apply the phase-4 preset: the same exact contact, plus the "
        "constraint that makes the posture a GRASP rather than several "
        "independent touches -- the contacts' virtual forces and "
        "torques must cancel, so the digits arrange themselves AROUND "
        "the object. The support equality is released here, since the "
        "hand is holding the object and pinning it to the plane would "
        "fight that. Judge the result with the grasp-wrench readout, "
        "not with the reported violation: that number is whitened and "
        "falls as the sigmas are loosened whether or not a fingertip "
        "moves. Check this, then press Auto solve."),
        # Same exact contact as phase 3, plus the constraint that makes the
        # posture a GRASP: the contacts' virtual forces and torques must cancel,
        # so the digits arrange themselves around the object rather than
        # clustering wherever they happened to arrive.
        object_contact=False,
        object_contact_exact=True,
        contact_drop_normal_row=True,
        grasp_alignment=True,
        # The support equality is released here, and only here. The hand is
        # holding the object now, so pinning the grasp digits to the plane would
        # fight the arrangement this phase is solving for. Avoidance stays on --
        # and because no digit carries the table contact node any more, it
        # applies to every sphere again (`forall i in I`), which is exactly what
        # the formulation asks for in this phase.
        table_contact=False,
        **_HELD_WRIST,
    ),
}


# ---------------------------------------------------------------------------
# Which set belongs to which hand.
# ---------------------------------------------------------------------------

#: Phase presets by hand name. A hand with no entry falls back to
#: :data:`PHASE_PRESETS`, which keeps every pre-existing caller working and gives
#: a newly registered hand something rather than a KeyError -- though a hand that
#: means to be driven through a staged pipeline should have its own set here, for
#: the reasons in this module's header.
HAND_PHASE_PRESETS: dict[str, dict[str, PhasePreset]] = {
    "tendon_5f": PHASE_PRESETS,
    "allegro": ALLEGRO_PHASE_PRESETS,
}


def phase_presets(hand_name: str | None = None) -> dict[str, PhasePreset]:
    """The phase presets for a hand, by registry name.

    ``None`` or an unregistered name gives :data:`PHASE_PRESETS` -- see
    :data:`HAND_PHASE_PRESETS` on why that fallback is a convenience and not a
    recommendation."""
    if hand_name is None:
        return PHASE_PRESETS
    return HAND_PHASE_PRESETS.get(hand_name, PHASE_PRESETS)


def apply_phase_preset(params: HandSolveParams, name: str,
                       hand: str | None = None) -> HandSolveParams:
    """Apply a phase preset's overrides onto ``params`` IN PLACE (``setattr`` per
    field), returning it for chaining. An override naming a field that doesn't
    exist on ``HandSolveParams`` raises -- a typo in a preset should fail loudly,
    not silently no-op.

    Which SET the preset comes from is decided by ``hand``, defaulting to
    ``params.hand``. So the two-argument call every existing caller makes keeps
    working and now picks up the right set for whichever hand the params name."""
    preset = phase_presets(hand if hand is not None else params.hand)[name]
    for field_name, value in preset.overrides.items():
        if not hasattr(params, field_name):
            raise AttributeError(
                f"phase preset {name!r} sets unknown HandSolveParams field "
                f"{field_name!r}")
        setattr(params, field_name, value)
    return params
