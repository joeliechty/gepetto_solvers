""":class:`HandResult` -- the uniform result all three solvers return.

``frames`` has length 1 for FK/IK and K+1 for the planner, so a step scrubber
indexes them identically. The gap and witness accessors report against the
ANALYTIC surface, independent of the solver.
"""

from dataclasses import dataclass, replace

import numpy as np

from ..geometry.scene import (
    primitive_surface_witness,
    subset_spec,
)
from .witness import plane_witness


class _FingerSol:
    """One digit of one solved frame, as the renderers and witnesses consume it.

    ``.marginals`` is a ``DigitState`` and ``.meta`` the solve metadata. The
    accessors below are the point: they are what let a reader ask for "the tip
    pose" or "the collision-sphere sites" without knowing whether the digit is a
    Cosserat rod or a chain of revolute joints. Reach through to ``.marginals``
    only for something genuinely mechanism-specific -- and then via
    :meth:`tendon`, which returns None on a hand that has no tendons rather than
    raising.

    The demo scripts build this shim inline (e.g. ``ik_5f_contact.py``), so it
    has to stay cheap to construct."""

    __slots__ = ("marginals", "meta")

    def __init__(self, marginals, meta):
        self.marginals = marginals
        self.meta = meta

    # -- geometry, the mechanism-neutral half --

    def num_sites(self):
        """How many places this digit exposes, base to tip."""
        return len(self.marginals.sites)

    def site_pose(self, i):
        """The 4x4 world pose of site ``i``. Negative indices count from the tip,
        matching the node addressing ``EnvironmentConfig`` uses."""
        return np.asarray(self.marginals.sites[i].pose.mean, float)

    def site_point(self, i):
        """The world position of site ``i``."""
        return self.site_pose(i)[:3, 3]

    def tip_pose(self):
        """The 4x4 world pose of the tip -- the site every contact and gap
        witness measures from."""
        return self.site_pose(-1)

    def tip_point(self):
        return self.tip_pose()[:3, 3]

    def sphere_sites(self):
        """Site indices carrying a collision sphere.

        Read off the STATE rather than off a config, deliberately: an overlay can
        then never mark a sphere the solve did not actually carry."""
        return list(self.marginals.collision_sites)

    # -- actuation --

    def actuation(self):
        """What drives this digit: tendon tensions, or joint positions."""
        return np.asarray(self.marginals.actuation.mean, float)

    def displacement(self):
        """The digit's displacement readout (tendon lengths), or an empty array
        on a hand whose actuation IS its position."""
        return np.asarray(self.marginals.displacement, float)

    # -- the mechanism-specific half --

    def tendon(self):
        """This digit's ``TendonDigitExtras``, or None on a hand without tendons.

        The routing, the per-disc external wrenches and the tension Jacobian all
        live here. A caller that needs them is by definition tendon-specific and
        should gate on this being non-None."""
        return getattr(self.marginals, "extras", None)


def _make_frame(finger_names, hand_marginals, meta):
    """One render frame: ``{finger_name: _FingerSol}`` for a single hand state."""
    return {name: _FingerSol(fm, meta)
            for name, fm in zip(finger_names, hand_marginals.digits)}


def _tip_points(frame):
    """Every fingertip of one render frame, as an ``(n, 3)`` array in frame
    order -- the same point the contact and gap witnesses measure from."""
    return np.array([fs.tip_point() for fs in frame.values()])


@dataclass
class HandResult:
    """Uniform result for all three solvers. ``frames`` has length 1 for FK/IK and
    K+1 for the planner, so a step-scrubber can index it the same way regardless."""
    frames: list[dict]
    meta: object
    spec: dict
    object_center: np.ndarray
    object_rotation: np.ndarray
    finger_names: list[str]
    tip_radii: list[float]
    # Which fingers are DESIGNATED for contact (None = all). The mask, not the
    # set a solve happened to constrain: FK constrains none of them but still
    # carries it, so the §1.8 goal overlays -- p_bar, the opposition split, the
    # support-plane equalities -- describe the same finger set in the FK posing
    # state that the controller will enforce once a phase is picked.
    contact_fingers: list[bool] | None = None
    # The raw ``HandState`` behind each frame, same indexing as
    # ``frames``. ``frames`` splits a solve up per finger for rendering, which
    # loses the bundle the C++ side wants back: this is the form
    # ``HandSolveParams.initial_state`` takes to warm-start a solver from a
    # posture instead of a straight hand.
    states: list[object] | None = None
    # Solver-convergence snapshots: one entry per recorded iteration, each a
    # full ``frames``-shaped list (so an entry is indexed by trajectory step
    # exactly like ``frames`` is). Populated only when the solve ran with
    # ``HandSolveParams.record_iterations`` on a binding that exposes the
    # snapshots; None otherwise. ``iterate_states`` is the raw-marginals
    # parallel, the same relationship ``states`` has to ``frames``.
    iterates: list[list[dict]] | None = None
    iterate_states: list[list[object]] | None = None
    # One short markdown line per iterate, supplied by whoever produced the
    # snapshots. A stepped solve knows the cost/violation/mu behind each of its
    # entries directly; a one-shot recorded solve leaves this None and the
    # caller falls back to indexing ``meta``'s AL trace.
    iterate_notes: list[str] | None = None
    # Which fingers were driven onto the SUPPORT PLANE, the table counterpart of
    # ``contact_fingers`` (which stays the object set). None = none of them, i.e.
    # the object-only solves every caller ran before the two were separable.
    # Appended last on purpose: several call sites build a result positionally.
    table_contact_fingers: list[bool] | None = None
    # The solve's Augmented Lagrangian state (``gepetto_solvers.ALDuals``): the
    # multipliers and penalty weight, tagged with the identity of the constraint
    # each belongs to. Feed to ``HandSolveParams.initial_duals`` to continue this
    # solve after a rebuild. Only the stepper fills it; None everywhere else.
    duals: object | None = None
    # ``gepetto_solvers.ALTransferReport`` for the transfer INTO this solve, i.e.
    # how many of its constraints inherited a multiplier. None when nothing was
    # carried in.
    dual_transfer: object | None = None
    # Which of an ``ellipsoid_set`` object's members contact was allowed to
    # target (``scene.grasp_subset_indices``); None = all of them, which is every
    # object that is not a curated ``ycb:`` set.
    #
    # Carried on the RESULT, not just used at build time, because the gap
    # readouts have to measure against the same shells the graph did. Reporting a
    # fingertip's distance to the drill housing, when the constraint drove it to
    # the grip, describes a solve that never ran.
    contact_subset: list[int] | None = None
    # Name of the digit that opposes the others on the hand this was solved for
    # (the thumb on an anatomical hand), or None where the hand declares none.
    #
    # Carried on the RESULT because the witness readouts describe constraints
    # that are DEFINED by that opposition -- the half-space split, the pre-grasp
    # centering and axis alignment. They used to match the literal string
    # "thumb", which silently measured nothing on any hand that does not have
    # one. Appended last: several call sites build a result positionally.
    opposing_digit: str | None = None

    def state(self, k=0):
        """The solved hand state at frame ``k``, for seeding another solver.
        None on a result built before this field existed."""
        return None if self.states is None else self.states[k]

    def num_iterates(self):
        """How many solver-convergence snapshots this result carries (0 when the
        solve did not record any)."""
        return 0 if self.iterates is None else len(self.iterates)

    def at_iterate(self, i):
        """This result as it stood at recorded iteration ``i`` -- the same object
        with ``frames``/``states`` swapped for that snapshot.

        Everything downstream (the gap readouts, ``worst_gap``, the renderer)
        works off ``frames``, so a swapped-frames view makes all of it describe
        the intermediate state with no further plumbing. The view drops its own
        ``iterates`` so it cannot be re-scrubbed recursively."""
        if self.iterates is None:
            raise ValueError(
                "this result carries no iterates -- the solve ran without "
                "record_iterations, so there is nothing to scrub")
        return replace(self, frames=self.iterates[i],
                       states=None if self.iterate_states is None
                       else self.iterate_states[i],
                       iterates=None, iterate_states=None, iterate_notes=None)

    def contact_names(self):
        """The fingers designated to touch the object -- everything the gap
        readouts should be judged on. All of them when unmasked."""
        if self.contact_fingers is None:
            return list(self.finger_names)
        return [name for name, on in zip(self.finger_names, self.contact_fingers)
                if on]

    def table_contact_names(self):
        """The fingers designated to touch the SUPPORT PLANE. Empty unless the
        solve targeted the table (unlike :meth:`contact_names`, whose None case
        means "all of them" -- there the mask is an optional restriction, here it
        is the whole opt-in)."""
        if self.table_contact_fingers is None:
            return []
        return [name for name, on in zip(self.finger_names,
                                         self.table_contact_fingers) if on]

    def contact_witness(self, k=0):
        """Per-finger ``{name: (sphere_surface_pt, object_surface_pt, gap_m)}`` in
        world coordinates at frame ``k``: the shortest segment from each fingertip
        contact sphere to the object surface, and its signed length (~0 at contact,
        negative if the sphere interpenetrates).

        Uses the analytic ``primitive_surface_witness``, so for the baked-SDF
        primitives this measures against the analytic look-alike rather than the
        .vdb grid -- the same approximation :meth:`surface_gaps` has always made,
        differing only within the ``edge_radius`` fillets.

        Measured against the CONTACT surface -- narrowed to ``contact_subset``
        when the solve was -- because this is the number the contact equality was
        driven to zero, and it is what ``worst_gap`` scores a grasp on. The
        excluded shells are still there as collision geometry; they are simply
        not what the fingertip was aiming at."""
        frame = self.frames[k]
        out = {}
        R = self.object_rotation
        spec = subset_spec(self.spec, self.contact_subset)
        for name, radius in zip(self.finger_names, self.tip_radii):
            # Same site the renderer draws the contact sphere on.
            tip = frame[name].tip_point()
            dist, foot_local, n_local = primitive_surface_witness(
                R.T @ (tip - self.object_center), spec)
            surface_pt = self.object_center + R @ foot_local
            sphere_pt = tip - radius * (R @ n_local)
            out[name] = (sphere_pt, surface_pt, dist - radius)
        return out

    def surface_gaps(self, k=0):
        """Per-finger fingertip surface gap (m, ~0 at contact) at frame ``k``,
        reusing the analytic surface distance the demos report with."""
        return {name: gap for name, (_, _, gap) in self.contact_witness(k).items()}

    def displacements(self, k=0):
        """Per-digit displacement at frame ``k``, in ``finger_names`` order --
        tendon lengths on the tendon hand, the L component of Theta_curr a
        Section 1.8 control tick anchors on. Empty per digit on a hand whose
        actuation is already its position."""
        frame = self.frames[k]
        return [frame[name].displacement() for name in self.finger_names]

    def actuations(self, k=0):
        """Per-digit actuation at frame ``k``, in ``finger_names`` order --
        tendon tensions on the tendon hand, joint positions on a rigid one."""
        frame = self.frames[k]
        return [frame[name].actuation() for name in self.finger_names]

    def wrist_pose(self, k=0):
        """The solved wrist as a 4x4 at frame ``k``.

        Straight off the state bundle: each kinematics answers it for itself, so
        no caller has to know that the tendon hand's node 0 is not a variable.
        Falls back to None on a result built before the bundle carried it."""
        state = self.state(k)
        return None if state is None else np.asarray(state.wrist_pose, float)

    def worst_gap(self, k=0):
        """Largest |gap| to the OBJECT over the fingers that were *asked* to touch
        it, so a masked subset grasp isn't scored on fingers left free."""
        gaps = self.surface_gaps(k)
        names = self.contact_names()
        return max((abs(gaps[n]) for n in names if n in gaps), default=0.0)

    def worst_table_gap(self, params, k=0):
        """The same score against the SUPPORT PLANE, over the fingers driven onto
        it. 0.0 when the solve targeted no table contact.

        Takes ``params`` because the plane is not part of a result: its origin is
        re-resolved from the scene the same way the solve resolved it."""
        names = self.table_contact_names()
        if not names:
            return 0.0
        gaps = plane_witness(params, self, k, names=names)
        return max((abs(g) for _p, _f, g in gaps.values()), default=0.0)
