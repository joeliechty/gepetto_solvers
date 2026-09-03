""":class:`HandSolveParams` -- every knob the three solvers expose.

Shared by FK / IK / planner; each solver reads only the fields it needs. The
interactive visualizer mutates one instance of this from its GUI controls.

The defaults are the demo-script defaults, and most of them are load-bearing in a
way the field comments record: the wrist hover pose, the asymmetric planar-bend
sigmas, and the deliberately soft flexor tension prior each cost something
specific when changed.
"""

from dataclasses import dataclass, field

import numpy as np

from ..geometry.scene import GRASP_FLEXOR_TENSION, TABLE_NORMAL
from .frames import default_wrist_pose

# ---------------------------------------------------------------------------
# Params / results.
# ---------------------------------------------------------------------------

def _default_digit_count():
    """How many digits the DEFAULT hand has.

    The per-digit lists below (``flexor_tensions``, ``contact_fingers``) are
    positional, so their length has to match the hand being posed. Resolved
    lazily, per instance, rather than captured at import: importing the hands
    package from module scope here would close a cycle (a hand imports the
    compiled bindings, which the solvers also pull in), and a caller who
    registers a hand after this module is imported would get a stale count.

    A caller posing a hand with a different digit count must set these two
    fields to match it -- there is no way to infer which digits they meant.

    Cached: building a hand is not free (the tendon one parses a CAD geometry
    table and announces it), and every HandSolveParams() would otherwise pay for
    it -- including one built to pose a DIFFERENT hand entirely.
    """
    global _DEFAULT_DIGIT_COUNT
    if _DEFAULT_DIGIT_COUNT is None:
        from ..hands import get_hand
        _DEFAULT_DIGIT_COUNT = len(get_hand().digit_names)
    return _DEFAULT_DIGIT_COUNT


_DEFAULT_DIGIT_COUNT: int | None = None


#: The object contact FORMS, in the order a pipeline reaches for them: no object
#: contact, the 3D distance to the ellipsoid ``E_obj``, that distance measured
#: inside a tendon's pulling plane, and the witness contact against the baked SDF.
#:
#: Exactly one is in force at a time -- see :func:`object_contact_form`.
OBJECT_CONTACT_FORMS = ("none", "proxy", "proxy_in_plane", "exact")


def object_contact_form(params) -> str:
    """Which of :data:`OBJECT_CONTACT_FORMS` ``params`` selects.

    THE one definition of that question, because the three forms are not spelled
    symmetrically and never can be. ``object_contact_in_plane`` is a different
    METRIC on the same surface, so it rides on ``object_contact`` as a modifier;
    ``object_contact_exact`` is a different SURFACE, so it stands alone and turns
    ``object_contact`` off. Reading the fields directly, three call sites will
    eventually disagree about what "contacting the object" means -- which is the
    kind of disagreement that shows up as a solve quietly enforcing a different
    constraint than the panel claims.

    Raises on a combination that names two forms at once. There is only ever one
    contact; a caller asking for two has not asked for something exotic, it has
    made a mistake, and the C++ layer would reject the same combination further
    down where the message is about fields rather than about intent.
    """
    exact = bool(getattr(params, "object_contact_exact", False))
    plain = bool(params.object_contact)
    in_plane = bool(params.object_contact_in_plane)
    if exact and (plain or in_plane):
        raise ValueError(
            "object_contact_exact contacts the baked SDF and object_contact"
            f"{'_in_plane' if in_plane else ''} contacts the ellipsoid proxy: "
            "these are two FORMS of the one object contact, not two contacts. "
            "Set at most one.")
    if exact:
        return "exact"
    if not plain:
        return "none"
    return "proxy_in_plane" if in_plane else "proxy"


@dataclass
class HandSolveParams:
    """Every knob the three solvers expose, with the demo-script defaults.

    Shared by FK / IK / planner; each solver reads only the fields it needs. The
    interactive visualizer mutates one instance of this from its GUI controls.
    """
    # --- The hand ---
    # Which hand to pose, by registry name (gepetto_solvers.core.hands). Every
    # hand-specific fact the solve needs -- the digit list, the actuation
    # layout, which digit opposes the rest, the measured pinch table -- is read
    # off the hand this names, so the per-digit fields below (flexor_tensions,
    # contact_fingers) must be sized to ITS digit count.
    #
    # A caller holding a Hand object passes it to the solver directly
    # (HandSolverBase(params, hand=...)) and this field is not consulted; it
    # exists so a hand choice can ride along in a params dataclass that is
    # serialized, preset, or driven from a GUI.
    hand: str = "tendon_5f"

    # Commanded joint positions for a JOINT-SPACE hand: one vector per digit,
    # one entry per joint of that digit. This is q_S, the mean of p(q).
    #
    # Separate from `flexor_tensions` because that is one SCALAR per digit and
    # cannot say where four independent joints should go. Each hand reads
    # whichever it needs through `Hand.actuation_means`, so a caller only fills
    # the one its hand uses. None means the neutral configuration -- the open
    # hand -- for a hand that reads this at all, and is ignored entirely by one
    # that does not.
    joint_targets: list | None = None

    # --- Scene / object ---
    # The Section 1.8 default scene: a 35 mm-radius analytic sphere, half-buried
    # in the support plane (see table_burial). Resting ON the table its crown
    # would sit 70 mm up, outside the ~50 mm the fingertips can reach off their
    # ~55 mm shell; half-buried the exposed dome is 35 mm, which is both
    # reachable and the low-profile-object case Section 1.8 is about.
    primitive: str = "mid_sphere_ellipsoid"
    object_center: np.ndarray | None = None      # None => derive from primitive
    object_rotation: np.ndarray | None = None     # None => primitive's rotation
    # LogSumExp sharpness for an `ellipsoid_set` (ycb:) object; None keeps the
    # spec's own value. Only the smooth-min standoff moves with it -- the
    # constraint surface sits up to ln(K)/beta outside the true union. Inert for
    # every other primitive type.
    ellipsoid_set_beta: float | None = None
    # Send the fingertips only to the shells a `ycb:` fit names as grasp targets
    # (its `grasp_subset`), rather than to the nearest point of the whole union.
    # A decomposition is not all handles: on the power drill 5 of 6 shells are
    # the housing, and a contact equality against the union is as happy landing
    # on those as on the grip.
    #
    # Default True because a subset that was authored is a statement about the
    # object, and ignoring it by default would make the curated objects behave
    # like the uncurated ones. Inert for every object without an authored subset,
    # which is every non-`ycb:` primitive and most of the YCB set.
    #
    # CONTACT ONLY -- collision keeps the whole union either way, so the excluded
    # shells still keep the fingers out. See `config.attach_contact`.
    use_grasp_subset: bool = True

    # --- Wrist start pose + prior ---
    # A palm-down HOVER above the object (see DEFAULT_WRIST_XYZ /
    # DEFAULT_WRIST_RPY), not the identity pose. Identity puts the hand at the
    # grasp locus itself, which for the bigger primitives means starting INSIDE
    # them -- on big_sphere the collision spheres begin ~31 mm through the
    # surface, the merit function starts at ~3e6 and the AL solve stalls at
    # iters=1 with nothing able to move. From the hover pose the same scene
    # starts clear of everything at a cost of ~58.
    #
    # What it costs, measured: this is a start to APPROACH from, so a single-shot
    # IK with the default (tight, sigma 1e-4) wrist prior cannot close it -- the
    # base is pinned ~0.11 m off big_sphere and the contact violation freezes.
    # Loosen sigma_wrist_pos/rot, or let something that is allowed to move the
    # base do the positioning (the planner's GP-linked wrist states).
    #
    # Callers that derive their own start overwrite this and are unaffected.
    wrist_pose: np.ndarray = field(default_factory=default_wrist_pose)
    sigma_wrist_pos: float = 1e-4
    sigma_wrist_rot: float = 1e-3

    # --- Rod physics: planar bending ---
    #
    # The discs are keyed to the backbone, so a finger bends about its local +y
    # axis and neither deflects sideways nor twists. The Cosserat rod does not
    # know that, and spends the free out-of-plane DOFs to satisfy contact,
    # collision and the four passive tendons routed at +/-90 deg -- which is the
    # sideways splay visible in a phase-2 approach. PlanarBendFactor states the
    # constraint; see its header for the residual.
    #
    # On by default: this is a property of the hardware, not an experiment. The
    # sigmas are curvatures (rad/m), directly comparable to sigma_twist_rot
    # (1e-2 in the finger configs).
    #
    # Asymmetric on purpose -- soft bend, tight twist. Twist is the CAUSE (the
    # spiral-routed lateral tendons inject it, and it rotates the material frame
    # so the next segment's flexion lands out of plane); out-of-plane bend is the
    # symptom. Constraining torsion alone collapses the splay 8-200x across
    # big_sphere / mid_sphere / knife / power_drill while COSTING no reach -- the
    # finger curls further instead of splaying. Making the bend row equally tight
    # buys nothing extra and costs ~10 mm of reach, and stalls the AL outright on
    # the power drill. See TendonFingerSolverConfig for the measured table.
    planar_bending: bool = True
    sigma_planar_bend: float = 1e-2
    sigma_planar_twist: float = 1e-4

    # --- Tensions (per-finger flexor + shared passive background) ---
    passive_tension: float = 0.5
    flexor_tensions: list[float] = field(
        default_factory=lambda: [GRASP_FLEXOR_TENSION] * _default_digit_count())
    tip_wrench_sigma: float = 1e-3
    # How loose the ACTUATED (flexor) tendon's tension prior is once contact
    # is expected to move it away from its commanded value -- squared into
    # the tension covariance's flexor entry by _flexor_tension_cov(), the
    # same sigma-squared-into-covariance pattern tip_wrench_sigma uses above.
    # Default reproduces the historical hardcoded flexor variance (1e-1)
    # exactly.
    flexor_tension_sigma: float = 0.1 ** 0.5
    # Same, for the five PASSIVE tendons -- their physics is a spring holding
    # roughly constant tension, so this is normally left tight. Default
    # reproduces the historical hardcoded passive variance (1e-6) exactly.
    # Dropping much below that (mixed with a much looser flexor scale) risks
    # an IndeterminantLinearSystem, so treat 1e-3 as close to a floor.
    passive_tension_sigma: float = 1e-3

    # --- Which fingertips are solved for contact (IK / planner; FK ignores it) ---
    # One flag per finger, in ``configs`` order. A False finger contributes no
    # contact constraint -- to any surface -- but keeps its collision spheres and
    # plane avoidance, so it is still kept out of the object and (wherever
    # avoidance is active) off the table. All-True is the legacy behavior.
    # WHICH surfaces the True fingers are driven onto is object_contact /
    # table_contact below.
    contact_fingers: list[bool] = field(
        default_factory=lambda: [True] * _default_digit_count())

    # --- WHICH SURFACE those fingers are driven onto (IK / planner) ---
    # Orthogonal to contact_fingers, which stays the FINGER selection: the
    # effective per-surface mask is (contact_fingers AND the flag below). So a
    # solve can chase the object only (the legacy behavior), the table only, or
    # both -- which is what makes a stalled grasp bisectable, since the two
    # constraint families can be switched on one at a time.
    #
    # table_contact additionally needs `table` on; without a configured plane
    # there is nothing to touch and it is silently inert.
    object_contact: bool = True
    table_contact: bool = False
    # Eq 13: measure the OBJECT contact equality inside each finger's pulling
    # plane (Eq 11) instead of in 3D. A different distance METRIC for the same
    # constraint, not an extra constraint -- same center-direct form, same zero
    # set (d = tip radius), one factor per contact finger either way. So it is
    # meaningful only with object_contact on, and the two are mutually exclusive
    # as contact FORMS: the visualizer enforces that with its checkboxes, and a
    # script setting both gets the in-plane form -- this flag selects the FORM,
    # object_contact selects whether there is a contact at all.
    #
    # Needs an ellipsoid-form object and a measured pinch pose for the checked
    # digits; attach_contact RAISES rather than falling back if either is missing.
    object_contact_in_plane: bool = False
    # Attach BOTH object representations to every env: the ellipsoid form E_obj
    # AND the baked SDF. The two constraint families then read different
    # geometry, which is what the staged pipeline's later phases ask for --
    # collision resolves its surface by the C++ precedence and so always takes
    # the ellipsoid, while contact takes whichever `object_contact_exact` below
    # selects.
    #
    # False (the default) keeps the single-surface behavior every existing caller
    # relies on: one representation, shared by contact and collision alike.
    object_proxy_and_exact: bool = False
    # The THIRD object contact form, alongside the 3D and in-plane ones above:
    # the witness-point contact against the baked SDF -- the exact geometry
    # rather than the approximation the approach slid along. This is the phase
    # 3-4 form, and it pairs with contact_drop_normal_row for the 4-row
    # [c_R, c_O, c_T1, c_T2] residual.
    #
    # The three are ONE contact in one of three forms, never two contacts, so at
    # most one is set at a time -- the visualizer enforces that with its
    # checkboxes and the C++ layer raises on the combination. Unlike the in-plane
    # form, this one does not ride on `object_contact`: it names a different
    # SURFACE, not just a different metric, so it stands on its own field and
    # `_object_contact_mask` reads the pair.
    #
    # Needs `object_proxy_and_exact` (the collision inequalities still want the
    # proxy) and an object with a baked grid; attach_contact RAISES rather than
    # falling back if either is missing.
    object_contact_exact: bool = False
    # h_grasp: the contacts must SURROUND the object, not merely touch it. One
    # Vector6 equality over every contacting digit's witness point, driving the
    # net virtual wrench -- unit inward forces along the surface normals, plus
    # their torques about the object origin -- to zero.
    #
    # Needs a WITNESS-point contact to key off, so it goes with
    # object_contact_exact; a center-direct contact has no witness variable and
    # the C++ layer raises rather than skipping. Needs two or more contact
    # digits: one unit force cannot cancel.
    grasp_alignment: bool = False
    # The two halves of h_grasp's residual, scaled separately because they do not
    # share units: the force rows are a sum of unit normals (DIMENSIONLESS), the
    # torque rows a sum of moment arms (METRES).
    #
    # The defaults put both on the same whitened scale as everything else in the
    # graph. Every other constraint row here is a distance in metres whitened at
    # sigma 1, i.e. ~1e-2 whitened; a force row is O(1) raw, so it wants ~1e2, and
    # a torque row is O(object radius) raw, so it wants ~1e1. Left at 1.0 the
    # force rows carry ~100x every other constraint in the problem and the inner
    # LM spends its whole budget on them -- the same graph-scaling failure the
    # ctrl_al_iters note below describes, and it shows up the same way: the AL
    # stalls after two outer iterations with nothing having moved.
    #
    # WHAT THESE CANNOT DO is make a grasp better, and the reported violation
    # will lie to you about that. It is WHITENED, so it falls as 1/sigma whether
    # or not a fingertip moves. Measured on the Allegro hand reaching for the
    # 35 mm sphere (index/middle/thumb), sweeping sigma_grasp_force alone and
    # reading the raw residual back with solvers.grasp_wrench_witness:
    #
    #   sigma_force  |   1    |   10   |   100  |  1000  |  1e4
    #   AL violation | 7.3e-1 | 1.3e-1 | 1.4e-2 | 1.7e-3 | 1.9e-4
    #   raw |wrench| |  2.33  |  2.14  |  1.11  |  1.67  |  1.75
    #
    # Four orders of magnitude off the reported number; none off the grasp. 100
    # is where the raw residual is actually lowest, which is the same place the
    # scaling argument above puts it. Judge this constraint with the witness.
    sigma_grasp_force: float = 100.0
    sigma_grasp_torque: float = 10.0
    # Eq 2.12-2.15: use the 4-row [c_R, c_O, c_T1, c_T2] SDF witness contact
    # form (c_N dropped) instead of the default 5-row form. Only affects a
    # non-ellipsoid (SDF) object's witness contact -- inert for the analytic
    # ellipsoid's center-direct form, which has no normal row.
    contact_drop_normal_row: bool = False
    # Eq 2.18-2.19: center the midpoint of the thumb's and the opposing
    # fingers' contact points over the object, raised by h_clear along
    # plane_normal. Needs the thumb AND at least one other finger checked in
    # contact_fingers, or the C++ layer silently skips it.
    pregrasp_center: bool = False
    # Companion to Eq 2.16-2.17: align the vector between the thumb's and the
    # opposing fingers' contact centroids with the SAME axis the opposition
    # half-space split uses (perpendicular to the object's longest in-plane
    # axis), direction-agnostically. Independent of half_space/pregrasp_center
    # -- it computes its own copy of that axis via default_half_space_axis()
    # regardless of whether the opposition constraint itself is on. Needs the
    # thumb AND at least one other finger checked in contact_fingers.
    pregrasp_axis_align: bool = False
    # Pre-grasp PINCH-CENTROID centering: drive the point where the checked
    # digits are MEASURED to meet (config.HAND_PINCH_POSES, in the wrist
    # frame) onto the object centroid, raised by h_clear along plane_normal.
    #
    # The hardcoded-point sibling of pregrasp_center above. That one averages
    # the fingertips' achieved positions, so it only says something once the
    # fingers are already near the grasp; this one constrains the WRIST alone,
    # so it positions the hand such that closing those digits would close them
    # on the object -- true whatever the fingers are currently doing.
    #
    # Silently inert for a digit set with no measured pose (fewer than two
    # digits, or any set without the thumb) -- see config.pinch_pose.
    pregrasp_centroid: bool = False

    # --- Augmented Lagrangian (IK / planner) ---
    al_mu: float = 1.0
    al_rate: float = 2.0
    al_iters: int = 40
    # How many leading stepper steps run with the flexor prior PINNED as tightly
    # as the passives (see HandIKStepper.step). Settles the cold start before the
    # flexor is released to flexor_tension_sigma; 0 restores the old behaviour.
    ik_settle_steps: int = 1

    # --- Planner-only ---
    K: int = 10
    dt: float = 0.1
    gp_wrist: float = 1e-2
    gp_tense: float = 1.0
    gp_len: float = 0.0
    start_flexor: float = 0.5
    al_inner_tol: float = 1e-2
    al_abs_cost_tol: float = 1e12

    # Warm-start posture for the IK stepper: the ``marginals`` of any solve on
    # the same finger configs (``HandResult.state(k)``), or None for the
    # straight-rod, zero-tension cold start. Carries a converged grasp across a
    # rebuild -- which is the only way to change the CONSTRAINT SET and continue
    # from where the solve got to, since a new constraint set needs a new solver
    # and a new solver otherwise cold-starts. Needs a binding with
    # ``HandSolverConfig.initial_state`` (capabilities()["solver_seed"]).
    initial_state: object | None = None

    # The other half of a warm start: the Augmented Lagrangian multipliers and
    # penalty weight of a previous solve (``HandResult.duals``), matched onto
    # this solve's constraints by identity. ``initial_state`` carries where the
    # hand IS; this carries how hard each constraint was being held there. Both
    # are needed to change a constraint and continue -- with the posture alone
    # the rebuilt solve restarts at mu = al_mu with every multiplier at zero and
    # visibly drifts off the constraints it had already satisfied before being
    # dragged back. Needs capabilities()["dual_transfer"].
    initial_duals: object | None = None

    # Ceiling on the penalty weight a transfer may carry in. mu is global, so a
    # rebuilt problem inherits it for constraints it has never seen: too high and
    # the new constraint is pinned as rigidly as the old ones and cannot recruit
    # any motion, too low and the old ones are held only weakly.
    al_transfer_mu_max: float = 1e4

    # --- Diagnostics (opt-in; off by default so normal solves are unchanged) ---
    # When True the C++ side records the per-outer-iteration AL trace
    # (al_iteration_mus / _costs / _violations on the result meta) plus
    # step-by-step Values snapshots. Used by debug_al_trace.py; left off for the
    # visualizer since it adds per-iteration bookkeeping.
    record_iterations: bool = False

    # --- Collision avoidance (Section 1.5, opt-in; IK / planner) ---
    # OBJECT collision: keep every non-contact sphere out of the object surface.
    # Independent of the table's own avoidance (plane_avoidance below) and of
    # finger-finger (self_collision): the three families share one set of
    # collision spheres but each is gated on its own field, so any combination
    # of them is available. The sphere set is attached whenever any of the three
    # wants it (_attach_environment).
    collision: bool = False
    # FINGER-FINGER collision: keep the fingers out of each other. Default True,
    # unlike the other two -- self-intersection is never wanted, and it needs no
    # object and no table. Costs the most factors of the three (every
    # cross-finger sphere pair), which is what cull_margin exists to trim.
    self_collision: bool = True
    collision_radius: float = 0.003
    collision_sigma: float = 1e-4
    num_proximal_discs: int = 2
    cull_margin: float | None = None

    # --- Support plane / "table" (Section 1.6, opt-in; IK / planner) ---
    table: bool = False
    plane_origin: np.ndarray | None = None       # None => seat from the scene
    plane_normal: np.ndarray = field(
        default_factory=lambda: np.array(TABLE_NORMAL, float))
    # Signed height of the CONSTRAINT plane above the table surface, along
    # plane_normal. The table is the physical bench -- the robot, the workspace
    # table URDF and the viser/`lbr_workspace_table_link` registration are all
    # expressed against it, so it must not move to satisfy a planning tweak. This
    # is the knob that moves instead: it raises or lowers the plane the support
    # equality seats fingertips on and the avoidance half-space starts at,
    # leaving every table-frame transform alone. 0.0 keeps the two coincident,
    # which is the geometry every script solved before the split.
    constraint_plane_height: float = 0.0
    # TABLE collision: keep every non-contact sphere out of the half-space. Needs
    # only `table`, not `collision` -- the solvers attach the collision sphere set
    # whenever any of the three avoidance consumers wants it.
    plane_avoidance: bool = True
    k_touch: int | None = None                    # planner slide-grasp schedule
    # Fraction of the object's FULL along-normal extent sitting BELOW the plane.
    # 0.0 = tangent to the underside, i.e. the object rests on the table (the
    # Section 1.6 slide-and-grasp geometry); 0.5 = plane through the centroid,
    # i.e. half-buried. Consumed by auto_table_origin(); ignored entirely when
    # plane_origin is set explicitly.
    #
    # Default 0.5 because a whole object resting on the table puts its crown out
    # of the hand's reach envelope (see the `primitive` note above), and because
    # a half-buried proxy is how a genuinely low-profile object presents itself:
    # a shallow dome above the surface with no undercut to reach around.
    #
    # NOTE this does NOT feed h_clear, which stays measured from the object
    # CENTROID. Half-buried, the centroid lies on the
    # plane, so the hand hovers pregrasp_margin + half_extent above the table
    # over a half_extent dome -- a larger effective gap than pregrasp_margin
    # nominally promises. That is known and accepted, not an oversight.
    table_burial: float = 0.5

    # --- Phased controller (Section 1.8, Controller mode only) ---
    # Which constraint set is active: 0 = pre-grasp positioning, 1 = support
    # contact, 2 = object approach, 3 = on-object servoing. The controller never
    # advances itself — the policy stays here (or in the GUI) so it can be
    # iterated on without a rebuild.
    phase: int = 1
    # What anchors a tick to the measured state: "tension" (Eq 1.95, the
    # simulation default), "length" (the Eq 1.13 analogue, hardware-faithful) or
    # "both" (diagnostic; over-constrains a real tick).
    step_anchor: str = "tension"
    # Step-prior covariances (Eq 1.94/1.95 and the length analogue). These are the
    # per-tick trust region: how far the hand base, the tendon tensions and the
    # tendon lengths may move in one control step.
    #
    # The base sigmas are LOOSE on purpose. A trust region tight enough to pin
    # the base (the old 1e-3 / 1e-2) makes the prior stiffer than anything that
    # can push against it -- the recorded AL penalty ceiling is mu ~ 8e3, versus
    # 1/sigma^2 = 1e6 at sigma = 1e-3 -- so neither phase 1's support equality
    # nor phase 0's pre-grasp target can move the hand at all, and the controller
    # silently solves with a frozen base. Measured: phase 1 descends only above
    # sigma_pos ~ 1.1e-2, exactly where 1/sigma^2 drops below that mu ceiling.
    #
    # Loosened again to 1e-1 / 1.0: the wrist is arm-mounted, so a control tick
    # may legitimately command a macro repositioning, and 1e-2 was still an order
    # of magnitude short of letting phase 1 use it. Measured over 40 phase-1 ticks
    # (small sphere, index+thumb contact, plane through the object midpoint):
    #
    #   sigma_pos / sigma_rot | base travel | support violation
    #   1e-2  / 1e-1          |     1.9 mm  | 0.085 -> 0.083 m
    #   3.2e-2 / 3.2e-1       |     7.3 mm  | 0.085 -> 0.075 m
    #   1e-1  / 1.0           |    32.2 mm  | 0.085 -> 0.048 m
    #
    # NOTE what this does NOT buy: the base still barely ROTATES (~3 deg at every
    # setting above, since a loose prior only removes resistance -- it supplies no
    # torque), so a phase whose residual needs the palm tilted is not fixed by
    # loosening these. That is an al_mu / ctrl_al_iters question; see the AL
    # penalty budget below.
    #
    # And what it COSTS: freeing the base also frees it to push the fingers into
    # things, because the collision inequality resisting that is only as strong as
    # the same weak per-tick penalty. On ctrl_5f_phases (which already failed its
    # penetration check at the old default, in phase 1) the worst finger-object
    # clearance goes -6.3 mm -> -6.5 mm in phase 1 and +3.5 mm -> -9.4 mm in phase
    # 2. Raising al_mu to ~1e2 more than recovers it (+39 mm / +46 mm) but freezes
    # the servo -- the inner LM then reports iters=1 from the second tick. There
    # is no setting of this pair that does both, which points at the real problem
    # being graph SCALING rather than the trust region: the passive-tension step
    # priors (1e-6 variance, 150 of them) carry ~3.1e6 of error against ~3e-2 in
    # the constraints, i.e. 99.9% of the graph, and that is what the inner LM is
    # actually solving.
    #
    # Both are per-tick trust regions, so a caller that wants a slower hand should
    # rate-limit the COMMAND (as phase 0 does with pregrasp_slew_*) rather than
    # tighten these -- see the frozen-base note above.
    sigma_wrist_pos_step: float = 1e-1
    sigma_wrist_rot_step: float = 1.0

    # --- Tendon step priors: ACTIVE and PASSIVE are different machines -------
    # The controller has no BetweenFactor GP (that is the trajectory planner);
    # p_step(Q | Q_curr) and p_step(L | L_curr) ARE its entire step-to-step
    # regularization, so how they are split across the six tendons is what
    # decides which parts of the hand can move in a tick.
    #
    # Only the DRIVEN tendon (hand.actuation.drive_indices) is actuated. The
    # others are spring-backed,
    # and the two facts that follow are not symmetric:
    #
    #   TENSION  A spring holds roughly CONSTANT tension as it takes up slack,
    #            so the passive tensions are pinned hard (1e-6) and stay pinned
    #            under BOTH anchors -- that is their physics, not a modelling
    #            convenience, and it does not depend on what the motor is doing.
    #            Only the flexor's tension is free, at sigma_q_step.
    #   LENGTH   The motor commands the ACTIVE tendon's length, so that is the
    #            real measurement to anchor on (tight). A passive tendon's length
    #            changes freely as the finger moves -- pinning it would be
    #            pinning the joint angles, freezing the hand.
    #
    # sigma_l_step_active is a per-tick trust region on commanded tendon travel:
    # 1e-3 allows ~1 mm of 1-sigma motion. It has to stay loose enough that the
    # hand can actually get from one state to the next; if ticks stall with the
    # flexor barely moving, this is the first thing to check against the real
    # flexor excursion between an open and a closed hand.
    sigma_q_step: float = 1e-1
    sigma_l_step_passive: float = 1e-1
    sigma_l_step_active: float = 1e-3
    # Opposition half-space (Eq 2.16-2.17 / Eq 1.92), read by
    # HandSolverBase._attach_opposition(): the splitting point (None => the
    # object centroid) and the in-plane axis the split runs along (None =>
    # solvers.default_half_space_axis, derived from the object's own longest
    # in-plane axis so the split runs along an elongated object's length
    # rather than a fixed world direction). Needs table_contact fingers to act
    # on. Default False (this field used to be read only by the deleted §1.8
    # controller, which gated it by phase rather than this flag -- defaulting
    # it True here, now that it is live, would silently add a constraint to
    # every existing caller of HandSolveParams() that never touches this
    # field).
    half_space: bool = False
    half_space_split: np.ndarray | None = None
    half_space_axis: np.ndarray | None = None
    # Which SIDE of the split the thumb is asked to stay on. The derived axis
    # only fixes the split LINE; its sign is an arbitrary object-frame
    # convention, and getting it backwards asks the thumb and fingers to trade
    # sides (see orient_opposition_axis -- it stalls the solve outright).
    # None (default) = orient by the hand's current posture, False = keep the
    # derived sign, True = invert it. Ignored when half_space_axis is given
    # explicitly, which is taken as already oriented.
    half_space_flip: bool | None = None
    # Minimum standoff (m) each contact finger must keep from the splitting
    # line, along its own m_hat: HalfSpaceGapFactor's d_min, so the constraint
    # is -(c - p_split) . m_hat + half_space_margin <= 0. 0.0 (the default) is
    # the original "anywhere on my own side" form, which a fingertip sitting
    # exactly ON the split already satisfies -- so the thumb and the opposing
    # fingers can be driven arbitrarily close together while both are "legal".
    # A positive value holds them 2 * margin apart, which is what makes this
    # useful as a PRE-grasp opening. Needs a binding carrying
    # EnvironmentConfig.half_space_margin (capabilities()["half_space_margin"]).
    half_space_margin: float = 0.0
    # Optional per-finger phase-3 witness targets (Eq 1.111); None entries mean
    # "contact anywhere on the surface" for that finger.
    witness_targets: list[np.ndarray | None] | None = None
    # A control tick's AL budget: outer iterations per tick. Small on purpose,
    # because with ctrl_al_warm_duals below the outer loop genuinely IS amortized
    # across ticks -- mu and the multipliers pick up where the last tick left off,
    # so a tick only has to advance the homotopy a little.
    #
    # HISTORICAL NOTE, kept because the conclusion inverted. This used to be
    # documented as amortized when it was not: SolverBase::optimize() built a
    # fresh AugmentedLagrangianOptimizer every call, so mu restarted at al_mu and
    # the duals at zero, capping a tick's penalty at
    # al_mu * al_rate^(ctrl_al_iters - 1) -- mu ~ 8 at the defaults, against the
    # mu ~ 8e3 an offline solve reaches. Measured on phase 1 then (small sphere,
    # index+thumb, 30 ticks), raising the budget bought nothing:
    #
    #   iters/tick |  mu cap  | support viol | base rotation | AL iters RUN
    #        4     |      8   |   0.0529 m   |    2.9 deg    |     2
    #       20     |   5.2e5  |   0.0478 m   |    2.9 deg    |    1-5
    #       40     |   5.5e11 |   0.0480 m   |    2.9 deg    |    1-5
    #
    # The outer loop never spent the budget it had: it exits on the stagnation
    # test (|d violation| < al_rel_violation_tol && |d cost| < al_rel_cost_tol),
    # which fires as soon as the inner LM rejects every step and both deltas are
    # exactly zero. That is still true -- a bigger budget WITHIN a tick still
    # buys mostly no-op outer iterations. What changed is that the progress a
    # tick does make now survives into the next one, so the ladder is climbed
    # across ticks instead of being rebuilt and abandoned on each.
    ctrl_al_iters: int = 4
    # Carry mu and the Lagrange multipliers from tick to tick (see above). This
    # is what makes the phased controller an Augmented Lagrangian method rather
    # than a weak penalty method restarted 30 times.
    ctrl_al_warm_duals: bool = True
    # Ceiling on the carried mu. mu compounds across ticks by design, and this
    # is what stops it running away -- but the value is NOT just a safety guard,
    # it is the balance point between the two constraint families and it is
    # sharp. Measured on ctrl_5f_phases (mid sphere, half-buried, all five
    # fingers), sweeping only this:
    #
    #   mu_max |  2   4   8  | 16
    #   result | PASS PASS PASS | FAIL (phase 1 penetrates)
    #
    # Above ~8 the support EQUALITY out-muscles the plane-avoidance
    # INEQUALITIES: the contact tips are driven onto the plane hard enough to
    # rotate the finger until a proximal sphere dips through it. Both families
    # carry multipliers, but the equality is always active while an inequality
    # only accumulates once violated, so a big shared mu favours the equality.
    #
    # Keeping mu small is the right shape for AL anyway -- lambda is supposed to
    # do the feasibility work, and mu only has to be large enough to keep the
    # subproblem convex. This is what a penalty method gets wrong, and why the
    # fix here was carrying lambda rather than raising mu.
    ctrl_al_mu_max: float = 8.0
    # Skip the Marginals factorization (a tick only consumes the means).
    ctrl_skip_marginals: bool = True

    # --- Phase 0: pre-grasp positioning (Section 1.8, Eq 1.92-1.98) ---
    # Explicit 4x4 target T_base,pre. None => derive it from the hand's own
    # forward kinematics via :func:`pregrasp_wrist_pose`.
    pregrasp_wrist_pose: np.ndarray | None = None
    # Hover height of the CONTACT-SPHERE CENTROID above the object centroid along
    # the support normal. None => object_extent_along(spec, n_hat) +
    # pregrasp_margin, i.e. scaled to the object rather than an absolute number
    # (the capsule and cylinder stand their long axis along +Z, so a value tuned
    # on a sphere would not clear them).
    #
    # Also read live by HandSolverBase._attach_pregrasp_center() (Eq 2.19's
    # h_clear) when pregrasp_center is on -- same physical quantity, a
    # clearance offset along the support normal. None there falls back to a
    # flat 0.02 m rather than the object_extent_along derivation above (that
    # helper belonged to the deleted §1.8 phase-0 code).
    h_clear: float | None = None
    pregrasp_margin: float = 0.04
    # Eq 1.92: Q_pre = [c]*5 + [c + pregrasp_flexor_offset], the "slightly curled"
    # pre-grasp posture. pregrasp_flexor_absolute overrides the offset form.
    # NOTE the default puts the flexor at 0.75 N, MORE curled than
    # GRASP_FLEXOR_TENSION (0.6), so phase 1 extends the fingers to reach the
    # table rather than curling further.
    pregrasp_flexor_offset: float = 0.25
    pregrasp_flexor_absolute: float | None = None
    # Per-tick cap on how far the phase-0 TARGET may advance toward T_base,pre
    # (m and rad). This -- not the sigma ratio -- is the real rate limiter, and
    # it is what makes phase 0 work at all.
    #
    # The pre-grasp pose is roughly a 172 deg rotation away from an identity base
    # pose, and handing a stiff Pose3 prior a target that far off drives the merit
    # function to ~3e6 and the inner LM rejects every step (iters=1, nothing
    # moves, forever). Slewing a waypoint toward the target keeps the commanded
    # pose close to the achieved one, so the prior stays well-scaled and the
    # linearization stays valid the whole way. Expressed in m/tick and rad/tick
    # because that is what a caller actually wants to reason about.
    pregrasp_slew_pos: float = 0.02
    pregrasp_slew_rot: float = 0.25
    # Eq 1.94 Sigma_pre,base -- how hard the hand tracks the slewed waypoint. The
    # tracking lag is the sigma RATIO against sigma_wrist_*_step: the target and
    # step priors multiply, so a fraction
    #   rho = sigma_pre^2 / (sigma_pre^2 + sigma_step^2)
    # of the remaining error survives each tick.
    #
    # But the ABSOLUTE stiffness matters more than the ratio, and in the opposite
    # direction to what you might expect. A prior tight enough to whiten its own
    # residual to ~80 (e.g. sigma_rot = 3e-3 against a 0.25 rad waypoint) leaves
    # the linear system too badly scaled against the rod-physics factors for the
    # inner LM to take any step at all: it quits at iters=1 and the hand never
    # moves. Measured, sigma_rot >= ~1e-2 is navigable and reproduces the
    # predicted rho exactly; below that the servo is dead. Keep these loose and
    # let pregrasp_slew_* set the speed.
    sigma_pregrasp_pos: float = 3e-3
    sigma_pregrasp_rot: float = 3e-2
    # Eq 1.95's SECOND tension prior (on top of the step prior). Off by default
    # because it is inert in simulation: _tension_priors' mean is the COMMANDED
    # tension, not a measurement, so phase 0 simply commands Q_pre directly and a
    # second Gaussian at the same target adds nothing. Turn it on to exercise the
    # spec-faithful two-prior form, which is what hardware (a genuinely measured
    # Q_curr) will need.
    pregrasp_tension_prior: bool = False
    # Split passive/active for consistency with every other tendon prior, but
    # equal by default: unlike the step priors, Q_pre names a target for all six
    # tendons and there is no reason to pull on them with different authority.
    sigma_pregrasp_q_passive: float = 1e-1
    sigma_pregrasp_q_active: float = 1e-1
    # SUPERSEDED / unused: this used to gate deriving the Eq 1.92 half-space
    # axis from the object's longest in-plane axis as an opt-in, off by
    # default (world +X as m_hat). That derivation (m_hat = n_hat x e_long) is now
    # UNCONDITIONAL whenever half_space_axis is None -- see
    # HandSolverBase._attach_opposition() / default_half_space_axis() -- since
    # world +X turned out to be actively wrong for elongated objects (it
    # bisects a pen across its short axis instead of splitting along its
    # length). Kept only so an old caller that set this doesn't hit an
    # AttributeError; it is read nowhere.
    derive_half_space_axis: bool = False
    # Close the Eq 1.93 Theta_curr loop: after each tick, write the SOLVED base
    # pose back so the step prior is anchored to the achieved state. Without this
    # the prior mean stays at the construction-time pose forever and the base is
    # effectively pinned there — phase 0 cannot servo at all, and phases 1-3
    # cannot reposition the hand to reach the object.
    wrist_feedback: bool = True
