""":class:`HandSolverBase` -- the shared build/solve/extract skeleton.

Builds the anatomical hand from the loaded dimensions, resolves the scene, and
attaches the environment. Subclasses differ almost entirely in how many models
they build and which constraints they switch on.
"""

import os

import numpy as np

import gepetto_solvers

from ..environment import (
    attach_collision,
    attach_contact,
    attach_half_space,
    attach_pregrasp_axis_alignment,
    attach_pregrasp_center,
    attach_pregrasp_centroid,
    attach_table,
    opposition_directions,
)
from ..geometry.scene import grasp_subset_indices
from ..hands import get_hand
from .capabilities import _OBJECTS_DIR, _set_if
from .params import HandSolveParams
from .result import HandResult
from .scene_resolve import (
    default_half_space_axis,
    orient_opposition_axis,
    resolve_constraint_plane_origin,
    resolve_scene,
)

# ---------------------------------------------------------------------------
# Solver base + the three flavours.
# ---------------------------------------------------------------------------

class HandSolverBase:
    """Shared setup for the hand solvers: holds the hand being posed and the
    resolved scene. Subclasses implement :meth:`solve`.

    The HAND is a parameter, not a constant. ``hand`` takes a
    :class:`~gepetto_solvers.core.hands.base.Hand` directly; otherwise
    ``params.hand`` names one in the registry. Everything hand-specific --
    the digit list, the actuation layout, which digit opposes the rest, the
    measured pinch table -- is read off it, so the solver itself contains no
    statement about what kind of mechanism it is posing.
    """

    def __init__(self, params: HandSolveParams | None = None, hand=None):
        self.params = params or HandSolveParams()
        self.hand = hand if hand is not None else get_hand(self.params.hand)
        # Fresh configs per solver: the attach_* family mutates them in place,
        # so two solvers sharing one list would see each other's constraints.
        self.configs = self.hand.digit_configs()
        self.tip_radii = self.hand.tip_radii
        self.finger_names = list(self.hand.digit_names)
        self._attach_planar_bending()
        self.spec, self.object_center, self.object_rotation, self.object_pose = \
            resolve_scene(self.params)
        # Which object shells contact may target (None = all). Resolved for real
        # in _attach_contact, but seeded here so a solver that never attaches
        # contact -- FK -- still answers the question its result is asked.
        self.contact_subset = grasp_subset_indices(
            self.spec, self.params.use_grasp_subset)

    def _attach_planar_bending(self):
        """Rod physics, so it rides on the per-finger config rather than the
        environment. Attached from ``__init__`` rather than
        ``_attach_environment`` because it is not environment-dependent and
        because ``HandFKSolver`` builds its ``HandSolver`` in its own
        ``__init__`` -- a later attach would miss FK entirely."""
        for _, cfg in self.configs:
            _set_if(cfg, "planar_bending", self.params.planar_bending)
            _set_if(cfg, "sigma_planar_bend", self.params.sigma_planar_bend)
            _set_if(cfg, "sigma_planar_twist", self.params.sigma_planar_twist)

    # -- contact masks --
    #
    # One finger selection (params.contact_fingers) times one per-surface switch.
    # Both surfaces read the same finger list, so "index and thumb, on the table
    # only" is a two-flag change rather than two masks to keep in sync.

    def _object_contact_mask(self):
        """Fingers driven onto the OBJECT surface."""
        return [bool(b) and self.params.object_contact
                for b in self.params.contact_fingers]

    def _table_contact_mask(self):
        """Fingers driven onto the SUPPORT PLANE. Empty without ``params.table``:
        there is no plane configured to touch."""
        on = self.params.table_contact and self.params.table
        return [bool(b) and on for b in self.params.contact_fingers]

    # -- environment attachment (mutates self.configs in place) --

    def _attach_contact(self):
        """Per-finger contact env: shared object surface + this finger's tip node
        as the terminal contact (``ik_5f_contact.py`` block). Fingers masked off
        get a collision-only env instead -- which is also what every finger gets
        with ``params.object_contact`` off, leaving the object present as
        collision geometry but with nothing driven onto it.

        ``object_contact_in_plane`` selects the contact FORM (Eq 13's in-plane
        distance in place of the 3D one), so it needs the pinch centroid of the
        digits actually being CONTACTED -- keyed off the same mask the contact
        nodes come from, not the raw finger selection, so unchecking a finger
        moves the plane's third point exactly as it moves the constraint set.

        ``contact_subset`` narrows which SHELLS of the object may be touched, as
        the mask narrows which FINGERS do the touching -- two independent halves
        of "what is this grasp". It is resolved here and stashed so the result
        can report gaps against the shells the graph actually targeted."""
        mask = self._object_contact_mask()
        in_plane = self.params.object_contact and self.params.object_contact_in_plane
        pinch = self.hand.pinch_pose(mask) if in_plane else None
        self.contact_subset = grasp_subset_indices(
            self.spec, self.params.use_grasp_subset)
        attach_contact(self.configs, self.spec, _OBJECTS_DIR,
                       self.params.primitive, self.object_pose,
                       tip_radii=self.tip_radii,
                       contact_nodes=self._contact_nodes(),
                       contact_fingers=mask,
                       drop_normal_row=self.params.contact_drop_normal_row,
                       ellipsoid_set_beta=self.params.ellipsoid_set_beta,
                       in_plane=in_plane,
                       pinch_centroid=(pinch.centroid if pinch is not None else None),
                       contact_subset=self.contact_subset)

    def _hand_spec(self):
        """The C++ ``HandSpec`` for this solve, built from the configs AFTER the
        environment has been attached to them -- the spec's task half is made of
        exactly those envs.

        ``params`` goes with them because a hand whose kinematics is seeded from
        the commanded posture needs it: a joint-space hand seeded at the same
        configuration its prior is centred on converges in one iteration."""
        return self.hand.build_spec(self.configs, self.params)

    # -- where the constraints attach --
    #
    # The hand names its own sites. A rod derives them from its disc spacing and
    # a URDF hand from its link frames, so neither the environment layer nor
    # this class can derive them -- they are asked for.

    def _contact_nodes(self):
        return [self.hand.contact_node(i) for i in range(len(self.configs))]

    def _collision_sites(self):
        sites = [self.hand.collision_sites(i) for i in range(len(self.configs))]
        return [s for s, _ in sites], [f for _, f in sites]

    def _attach_collision(self, avoidance=True):
        """Add Section 1.5 collision spheres onto each finger's (already attached)
        env. Reuses the contact env, so it works for SDF and ellipsoid objects
        alike (the vdb path is only used if a finger has no env yet).

        ``avoidance`` selects whether the finger-OBJECT inequalities are built
        and ``params.self_collision`` whether the finger-finger ones are; the
        spheres themselves are declared either way, because the support plane
        builds its own inequalities on the same set."""
        vdb = (None if self.spec["type"] in ("ellipsoid", "ellipsoid_set")
               else os.path.normpath(os.path.join(_OBJECTS_DIR, self.spec["vdb"])))
        nodes, proximal = self._collision_sites()
        attach_collision(self.configs, vdb, self.object_pose,
                         collision_nodes=nodes,
                         collision_proximal=proximal,
                         radius=self.params.collision_radius,
                         sigma=self.params.collision_sigma,
                         num_proximal_discs=self.params.num_proximal_discs,
                         cull_margin=self.params.cull_margin,
                         avoidance=avoidance,
                         self_collision=self.params.self_collision)

    def _attach_table(self):
        """Attach the Section 1.6 support plane to every finger's env.

        The CONSTRAINT plane, not the table surface -- they coincide until a
        caller raises ``constraint_plane_height`` (see
        :func:`resolve_constraint_plane_origin`)."""
        origin = resolve_constraint_plane_origin(self.params, self.spec,
                                                 self.object_center)
        attach_table(self.configs, origin, self.params.plane_normal,
                     avoidance=self.params.plane_avoidance,
                     tip_radii=self.tip_radii,
                     contact_nodes=self._contact_nodes(),
                     contact_fingers=self._table_contact_mask())

    def _attach_opposition(self):
        """Attach the Eq 2.16-2.17 opposition half-space to every finger's env.

        Masked by the shared ``contact_fingers``, like every other constraint
        in the set: the C++ layer builds this one off its own
        ``half_space_node``, so it no longer needs -- or silently waits for --
        table contact on the same finger.
        The thumb (identified by name, the hand-wide convention) gets ``+axis``
        and every other checked finger gets ``-axis``
        (:func:`opposition_directions`). ``half_space_split`` defaults to the
        object center; ``half_space_axis`` defaults to
        :func:`default_half_space_axis` -- derived from the object's own
        longest in-plane axis, so the split runs along an elongated object's
        length (e.g. a pen) rather than a fixed world direction that is only
        right by coincidence.

        That derived axis fixes the split LINE only. Its SIGN -- which half the
        thumb is sent to -- is oriented against the hand's current posture by
        :func:`orient_opposition_axis`, because the object-frame sign is
        arbitrary and the wrong one asks the hand to turn itself inside out.
        The resolved axis is written back onto ``params.half_space_axis`` so the
        witness overlay and any later rebuild describe the constraint that was
        actually built, rather than re-deriving and disagreeing with it."""
        explicit = self.params.half_space_axis is not None
        axis = (self.params.half_space_axis if explicit
               else default_half_space_axis(self.spec, self.object_rotation,
                                            self.params.plane_normal))
        if not explicit:
            thumb, others = self._opposition_tips()
            axis, _flipped = orient_opposition_axis(
                axis, thumb, others, flip=self.params.half_space_flip)
            self.params.half_space_axis = np.asarray(axis, float)
        directions = opposition_directions(
            self.configs, thumb_index=self.hand.opposing_index, axis=axis)
        split = (self.params.half_space_split if self.params.half_space_split is not None
                else self.object_center)
        attach_half_space(self.configs, split, directions,
                          contact_fingers=self.params.contact_fingers,
                          contact_nodes=self._contact_nodes(),
                          margin=self.params.half_space_margin)

    def _opposition_tips(self):
        """``(thumb_tip, [other checked fingertips])`` at the posture this solve
        starts from, for orienting the opposition axis.

        Measured with a throwaway FK solve (~180 ms, cached for the life of this
        solver) rather than read off the configs: the fingertips are where the
        TENSIONS put them, and the thumb-vs-fingers direction swings by more
        than a right angle across the flexor range -- the finger BASES, which
        are free to read, sit only ~5 mm apart along the opposition axis and get
        the sign wrong for 3 of 7 sampled wrist poses. Only the sign of one dot
        product is taken from this, so the ~100 mm the tips move between the FK
        pose and a warm-started posture cannot change the answer."""
        if getattr(self, "_fk_probe_tips", None) is None:
            # Deferred: fk imports base, so a module-level import here would be
            # circular. This is the only reference either way.
            from .fk import HandFKSolver

            frame = HandFKSolver(self.params).solve().frames[0]
            self._fk_probe_tips = {
                name: np.asarray(frame[name].marginals.sites[-1].pose.mean,
                                 float)[:3, 3]
                for name in self.finger_names}
        tips = self._fk_probe_tips
        mask = self.params.contact_fingers
        opposing = self.hand.opposing_digit
        others = [tips[n] for n, on in zip(self.finger_names, mask)
                  if on and n != opposing]
        return (tips.get(opposing) if opposing is not None else None), others

    def _attach_pregrasp_center(self):
        """Attach the Eq 2.18-2.19 pre-grasp hand-centering constraint, using
        the shared ``contact_fingers`` mask to pick which fingers (thumb +
        opposing set) participate, and ``plane_normal`` as the clearance axis."""
        h_clear = self.params.h_clear if self.params.h_clear is not None else 0.02
        attach_pregrasp_center(self.configs, clearance_height=h_clear,
                               clearance_normal=self.params.plane_normal,
                               contact_nodes=self._contact_nodes(),
                               contact_fingers=self.params.contact_fingers)

    def _attach_pregrasp_axis_alignment(self):
        """Attach the pre-grasp short-axis alignment constraint (companion to
        Eq 2.16-2.17), using the shared ``contact_fingers`` mask for the thumb
        + opposing set. Computes its own copy of the opposition axis via
        :func:`default_half_space_axis` -- independent of whether
        ``_attach_opposition()`` itself runs, so this stays toggleable on its
        own."""
        axis = default_half_space_axis(self.spec, self.object_rotation,
                                       self.params.plane_normal)
        attach_pregrasp_axis_alignment(self.configs, axis,
                                       contact_nodes=self._contact_nodes(),
                                       contact_fingers=self.params.contact_fingers)

    def _attach_pregrasp_centroid(self):
        """Attach the pre-grasp pinch-centroid constraint for the CHECKED
        digits, and report whether it went on.

        Returns the :class:`config.PinchPose` used, or None when the checked
        set has no measured pose. The return value exists so a caller can say
        so out loud: the C++ layer skips an unconfigured constraint silently,
        and a constraint that quietly does nothing is the trap this whole
        family of toggles keeps setting.
        """
        pose = self.hand.pinch_pose(self.params.contact_fingers)
        if pose is None:
            return None
        h_clear = self.params.h_clear if self.params.h_clear is not None else 0.02
        attach_pregrasp_centroid(self.configs, pose.centroid,
                                 clearance_height=h_clear,
                                 clearance_normal=self.params.plane_normal)
        return pose

    def _attach_environment(self):
        """The whole constraint environment for one solve, per the independent
        toggles (object contact, table contact, object collision, table
        collision, opposition half-space, pre-grasp centering, pre-grasp
        short-axis alignment).

        Every constraint family is gated on its own toggle alone -- checking one
        builds it, full stop. The collision sphere SET is shared, so it is
        attached whenever ANY of its three consumers (object, finger-finger,
        plane) wants it, and each family's own field then decides what gets
        built on it.

        Shared by the IK solver, the IK stepper and the planner so the three
        cannot drift into building different environments from the same params."""
        self._attach_contact()
        if (self.params.collision or self.params.self_collision
                or (self.params.table and self.params.plane_avoidance)):
            self._attach_collision(avoidance=self.params.collision)
        if self.params.table:
            self._attach_table()
        if self.params.half_space:
            self._attach_opposition()
        if self.params.pregrasp_center:
            self._attach_pregrasp_center()
        if self.params.pregrasp_axis_align:
            self._attach_pregrasp_axis_alignment()
        if self.params.pregrasp_centroid:
            self._attach_pregrasp_centroid()

    # -- prior builders --

    def _tension_priors(self, cov, means=None):
        """One ``VectorXGaussian`` per digit: passive actuators at the background
        hold, driven ones at that digit's commanded value.

        Which entries are driven comes from ``hand.actuation``, so a hand with a
        different actuator count or more than one driven actuator per digit needs
        no change here.

        ``means`` overrides the per-digit mean vectors wholesale -- the Section
        1.8 phase-0 pre-grasp posture commands ``Q_pre`` that way.
        """
        if means is None:
            means = self.hand.actuation_means(self.params)
        return [gepetto_solvers.VectorXGaussian(np.asarray(m, float), cov)
                for m in means]

    def _length_priors(self, means, cov):
        """One ``VectorXGaussian`` per finger pinning that finger's tendon lengths
        near ``means[i]`` — the Eq 1.13 / Eq 1.95 length step prior the Section 1.8
        controller uses to anchor a tick to the measured motor positions."""
        return [gepetto_solvers.VectorXGaussian(np.asarray(m, float), cov)
                for m in means]

    def _tip_wrenches(self):
        cov = self.params.tip_wrench_sigma ** 2 * np.eye(6)
        return [gepetto_solvers.Vector6Gaussian(np.zeros(6), cov) for _ in self.configs]

    def _flexor_tension_cov(self):
        """The "tight-passive / loose-driven" actuation-prior covariance used
        outside the leading settle steps: passives at
        ``params.passive_tension_sigma ** 2`` (their physics -- a spring holds
        roughly constant tension, so this is normally left tight), and the
        DRIVEN actuators at ``params.flexor_tension_sigma ** 2`` so contact can
        push them away from their commanded value. Read live every call, like
        ``_tip_wrenches()``, so a mid-solve slider drag takes effect on the next
        step with no stepper rebuild."""
        return self.hand.actuation.prior_cov(
            self.params.passive_tension_sigma, self.params.flexor_tension_sigma)

    def _result(self, frames, meta, contact_fingers=None, states=None,
                iterates=None, iterate_states=None, iterate_notes=None,
                table_contact_fingers=None, duals=None, dual_transfer=None):
        # The table mask defaults to the one this solve actually built, so every
        # result carries it without each call site restating it; a caller that
        # means "no table" (the controller) passes an explicit all-False list.
        if table_contact_fingers is None:
            table_contact_fingers = self._table_contact_mask()
        return HandResult(frames, meta, self.spec, self.object_center,
                          self.object_rotation, self.finger_names, self.tip_radii,
                          contact_fingers, states, iterates, iterate_states,
                          iterate_notes, table_contact_fingers, duals,
                          dual_transfer, contact_subset=self.contact_subset,
                          opposing_digit=self.hand.opposing_digit)

    def solve(self) -> HandResult:  # pragma: no cover - abstract
        raise NotImplementedError
