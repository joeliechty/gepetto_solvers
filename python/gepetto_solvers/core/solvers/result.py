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
    """Duck-typed per-finger solution the viser/pyvista renderers consume:
    exposes ``.marginals`` (a ``TendonFingerMarginals``) and ``.meta``. Same shim
    the demo scripts build inline (e.g. ``ik_5f_contact.py``)."""

    __slots__ = ("marginals", "meta")

    def __init__(self, marginals, meta):
        self.marginals = marginals
        self.meta = meta


def _make_frame(finger_names, hand_marginals, meta):
    """One render frame: ``{finger_name: _FingerSol}`` for a single hand state."""
    return {name: _FingerSol(fm, meta)
            for name, fm in zip(finger_names, hand_marginals.fingers)}


def _tip_points(frame):
    """Every fingertip of one render frame, as an ``(n, 3)`` array in frame
    order -- the last rod node of each finger, which is the same point the
    contact and gap witnesses measure from."""
    return np.array([
        np.asarray(fs.marginals.rod.states[-1].pose.mean, float)[:3, 3]
        for fs in frame.values()])


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
    # The raw ``TendonHandMarginals`` behind each frame, same indexing as
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
            fm = frame[name].marginals
            # Same node the renderer draws the contact sphere on (tip_node_index).
            tip = np.asarray(fm.rod.states[-1].pose.mean)[:3, 3]
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

    def tendon_lengths(self, k=0):
        """Per-finger tendon lengths at frame ``k``, in ``finger_names`` order --
        the L component of Theta_curr a Section 1.8 control tick anchors on."""
        frame = self.frames[k]
        return [np.asarray(frame[name].marginals.tendon_lengths, float)
                for name in self.finger_names]

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
