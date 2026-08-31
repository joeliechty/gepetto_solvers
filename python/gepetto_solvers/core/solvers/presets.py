"""Named :class:`HandSolveParams` override groups for the staged pipeline.

A preset touches ONLY the fields it lists -- wrist pose, flexor tensions, AL
sliders and table height stay wherever the caller left them, because those are
solver knobs rather than part of what defines a phase. :func:`apply_phase_preset`
raises on an override naming a field ``HandSolveParams`` does not have, so a typo
fails loudly instead of silently no-opping.
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


PHASE_PRESETS: dict[str, PhasePreset] = {
    "phase0": PhasePreset(
        label="Phase 0: pre-grasp positioning",
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
        label="Phase 4: synchronized close",
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
        label="Phase 5: lift",
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


def apply_phase_preset(params: HandSolveParams, name: str) -> HandSolveParams:
    """Apply ``PHASE_PRESETS[name]``'s overrides onto ``params`` IN PLACE
    (``setattr`` per field), returning it for chaining. An override naming a
    field that doesn't exist on ``HandSolveParams`` raises -- a typo in a
    preset should fail loudly, not silently no-op."""
    preset = PHASE_PRESETS[name]
    for field_name, value in preset.overrides.items():
        if not hasattr(params, field_name):
            raise AttributeError(
                f"phase preset {name!r} sets unknown HandSolveParams field "
                f"{field_name!r}")
        setattr(params, field_name, value)
    return params
