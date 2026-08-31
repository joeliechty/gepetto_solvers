"""Where the solver's wrist frame sits on the *physical* hand.

The solver hangs every digit off one floating wrist variable, but nothing in the
repo says where that wrist is on the printed part -- ``DEFAULT_WRIST_XYZ/RPY`` in
``solvers.py`` is a demo pose picked so the hand hovers over a grasp object, not a
measurement. This module supplies the missing half: the fixed convention change
between the exported hand STL and the solver's wrist frame, so that a transform
measured in CAD (e.g. an Onshape assembly whose origin is the robot flange) can be
turned into a wrist pose the solver accepts.

The chain is::

    T_flange<-wrist  =  T_flange<-stl  @  T_stl<-wrist
                        ^ measured      ^ this module (pure convention)

Two facts fix ``T_stl<-wrist``:

**The origins coincide, so it is a pure rotation.** ``hand()`` in the upstream
``OpenSCAD/hand.scad`` places each digit at ``translate(o_finger[i])`` and draws
the palm with no translate() of its own, so ``o_finger``/``o_thumb`` are measured
from the SCAD origin. :func:`config.finger_base_offset` uses those *same numbers*
verbatim as wrist-frame translations (``palm[:3,3] = o_mm / 1000``). Both frames
are therefore anchored on the one point the digit origins are measured from --
``T_stl<-wrist`` has zero translation, and every millimetre of a measured
``T_flange<-wrist`` comes from where the STL was placed in the assembly.

**The rotation is two stacked flips.** ``parameters.scad`` exports through
``rotate([180,0,0]) hand()``, giving ``R_stl<-hand = Rx(180)``. And the CAD grows
digits along +X (``digits.scad`` chains bones by ``[bl+lig,0,0]``) while the solver
hand is "+Z-up, fingers extending +Y and curling toward -X (palmar)" -- with the
knuckle row along Z in *both* (the ``o_finger`` z entries 0/-12/-24/-34 survive
into the solver unchanged). x->y with z fixed is ``Rz(+90)``, so
``R_wrist<-hand = Rz(90)`` and ``R_wrist<-stl = Rz(90) @ Rx(180)``.

That second step is an inference about sign conventions, not something the code
states, so it is offered as :data:`R_WRIST_FROM_STL` *and* as one entry in
:data:`CANDIDATE_ROTATIONS`; ``mount_onshape_fit.py`` scores all the candidates
against real part geometry rather than trusting the derivation.

Run ``python scripts/mount.py`` (from ``crest-sparse/``) for the
self-check: it validates the rotations and prints the digit-base landmarks used to
verify a fit against Onshape's Measure tool.
"""

import numpy as np

from gepetto_solvers.core.hand.config import (
    FINGER_NAMES,
    _Rx,
    _Ry,
    _Rz,
    finger_base_offset,
    load_hand_dimensions,
)
from gepetto_solvers.core.solvers import R_to_euler

DIGIT_NAMES = FINGER_NAMES + ["thumb"]

# parameters.scad wraps the whole model in ``rotate([180,0,0]) hand()``, so the
# exported STL's frame is the hand() frame flipped about X.
R_STL_FROM_HAND = _Rx(np.pi)

# CAD digits grow along +X with the knuckle row along Z; solver digits grow along
# +Y with the knuckle row along Z. x->y, z->z is a +90 deg yaw.
R_WRIST_FROM_HAND = _Rz(np.pi / 2)

#: Rotation taking a vector in the exported-STL frame to the solver wrist frame.
#: Equals a 180 deg rotation about (1,1,0)/sqrt(2); it is its own inverse.
R_WRIST_FROM_STL = R_WRIST_FROM_HAND @ R_STL_FROM_HAND.T


# ---------------------------------------------------------------------------
# The measurement itself.
# ---------------------------------------------------------------------------
#
# Measured 2026-08-18 by mount_onshape_fit.py against the "gepetto_hand /
# Assembly 1" Onshape assembly (document f0ca0f8f..., element e0d176c1...), whose
# origin is the KUKA attach point. Source of truth is the "hand <1>" instance's
# occurrence transform; the convention used was the derived Rz(+90)@Rx(180), which
# won both discriminating tests outright:
#
#   * yaw sign      4.69 deg mean growth-axis disagreement, vs 154.92 deg flipped
#   * export flip   zero overhang, vs 16.91 mm of digits outside the part's own
#                   bounding box without it
#
# Re-run mount_onshape_fit.py after ANY change to the assembly, to
# finger_base_offset(), or to the hand morphology -- nothing here can detect a
# stale value.
MOUNT_WRIST_XYZ = (-0.009490, -0.010641, 0.134688)
MOUNT_WRIST_RPY = (1.570796, 0.174533, -1.570796)


def measured_mount_pose():
    """``T_flange<-wrist`` as a 4x4, from the measurement recorded above.

    Fresh array per call: callers assign it into ``HandSolveParams.wrist_pose``
    and mutate poses in place, exactly as ``solvers.default_wrist_pose`` does.
    """
    from gepetto_solvers.core.solvers import wrist_pose_from_xyzrpy
    return wrist_pose_from_xyzrpy(MOUNT_WRIST_XYZ, MOUNT_WRIST_RPY)


def _T(R=None, t=None):
    """4x4 from an optional 3x3 rotation and optional 3-vector translation."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, float)
    if t is not None:
        T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


#: Name of the candidate this module derives analytically.
DERIVED_CANDIDATE = "Rz(+90)@Rx(180)  [derived]"


class Candidate:
    """One axis convention, split into the two independently testable halves.

    Keeping them apart matters: the *yaw* (which way CAD growth +X maps onto solver
    growth +Y) is a statement about the solver's frame and is tested by comparing
    digit growth directions, while the *export flip* is a statement about the STL
    alone and is tested by containment in the part's bounding box. Scoring the
    composed rotation against one piece of evidence conflates them, and on this hand
    the two readings differ by only a couple of millimetres of bounding box --
    close enough that noise picks the winner.
    """

    def __init__(self, R_wrist_from_hand, R_hand_from_stl, note):
        self.R_wrist_from_hand = R_wrist_from_hand
        self.R_hand_from_stl = R_hand_from_stl
        self.R = R_wrist_from_hand @ R_hand_from_stl   # R_wrist<-stl
        self.note = note


#: Every axis convention worth testing.
#:
#: Only two things are genuinely uncertain: the sign of the yaw carrying CAD growth
#: +X onto solver growth +Y, and whether the ``rotate([180,0,0])`` export flip is
#: present (it is not, if the STL was regenerated without it or re-oriented on
#: import). Four combinations -- and they are *all* of them, because dropping the
#: export flip is the same operation as swapping which side is palmar:
#: ``diag(-1,1,-1) @ Rz(+-90) @ Rx(180) == Rz(+-90)``.
CANDIDATE_ROTATIONS = {
    DERIVED_CANDIDATE:
        Candidate(_Rz(np.pi / 2), _Rx(np.pi), "yaw +90, export flip present"),
    "Rz(-90)@Rx(180)  [yaw flipped]":
        Candidate(_Rz(-np.pi / 2), _Rx(np.pi), "yaw -90, export flip present"),
    "Rz(+90)  [no export flip]":
        Candidate(_Rz(np.pi / 2), np.eye(3), "yaw +90, palmar swapped"),
    "Rz(-90)  [no export flip, yaw flipped]":
        Candidate(_Rz(-np.pi / 2), np.eye(3), "yaw -90, palmar swapped"),
}


def as_rotation(candidate):
    """3x3 ``R_wrist<-stl`` from a :class:`Candidate`, a raw 3x3, or None."""
    if candidate is None:
        return R_WRIST_FROM_STL
    if isinstance(candidate, Candidate):
        return candidate.R
    return np.asarray(candidate, float)


def T_stl_from_wrist(R_wrist_from_stl=None):
    """4x4 taking wrist-frame points into the exported-STL frame.

    Pure rotation -- the origins coincide (see the module docstring). Pass a
    candidate from :data:`CANDIDATE_ROTATIONS` to test an alternative convention.
    """
    return _T(R=as_rotation(R_wrist_from_stl).T)


def compose_flange_from_wrist(T_flange_from_stl, R_wrist_from_stl=None):
    """``T_flange<-wrist`` from a measured ``T_flange<-stl`` (both 4x4).

    ``T_flange<-stl`` is what the CAD tool reports for the hand part: Onshape's
    occurrence transform, in metres, maps that part's own coordinates into
    assembly coordinates.
    """
    return np.asarray(T_flange_from_stl, float) @ T_stl_from_wrist(R_wrist_from_stl)


def as_xyz_rpy(T):
    """``(xyz, rpy)`` for a 4x4, in the repo's ZYX convention.

    Feeds straight into :func:`solvers.wrist_pose_from_xyzrpy`, so the numbers
    printed here mean the same rotation as the visualizer's sliders.
    """
    T = np.asarray(T, float)
    return tuple(T[:3, 3]), R_to_euler(T[:3, :3])


# ---------------------------------------------------------------------------
# Landmarks: the digit base origins, the only points both frames name.
# ---------------------------------------------------------------------------

def digit_base_points_wrist(dims=None):
    """``{digit: (3,) xyz in metres}`` -- digit bases as the *solver* places them.

    Read out of the same :func:`config.finger_base_offset` the live hand configs
    use, so these move if the mounting convention ever changes.
    """
    if dims is None:
        dims = load_hand_dimensions()
    pts = {}
    for i, name in enumerate(FINGER_NAMES):
        pts[name] = finger_base_offset(dims["o_finger"][i],
                                       dims["a_finger"][i])[:3, 3]
    pts["thumb"] = finger_base_offset(dims["o_thumb"][0], dims["a_thumb"][0])[:3, 3]
    return pts


def digit_base_points_cad(dims=None):
    """``{digit: (3,) xyz in metres}`` -- digit bases as the *CAD* places them,
    expressed in the wrist frame.

    Straight from ``o_finger``/``o_thumb`` (the SCAD hand() frame), pushed through
    ``R_wrist<-hand``. This is what :func:`digit_base_points_wrist` *would* return
    if the mounting code rotated its translations as well as its rotations.
    """
    if dims is None:
        dims = load_hand_dimensions()
    o = [np.asarray(dims["o_finger"][i], float) for i in range(len(FINGER_NAMES))]
    o.append(np.asarray(dims["o_thumb"][0], float))
    return {name: R_WRIST_FROM_HAND @ (v / 1000.0)
            for name, v in zip(DIGIT_NAMES, o)}


def mounting_discrepancy(dims=None):
    """``{digit: (delta_xyz_m, norm_m)}`` between the CAD and solver digit bases.

    ``config.finger_base_offset`` permutes its *rotations* from CAD axes to solver
    axes (``Rz(rz) @ Rx(-ry) @ Ry(rx)``) but uses ``o_mm`` as a translation
    unrotated. Under ``R_wrist<-hand = Rz(90)`` a CAD origin ``(ox,oy,oz)`` belongs
    at ``(-oy,ox,oz)``, so every digit base is displaced by this much: small for the
    fingers (``ox=0``, ``oy`` a few mm), ~9 mm for the thumb.

    That is a real modelling error in the hand, but correcting it would silently
    invalidate ``HAND_PINCH_POSES`` and every hand-frame constant measured against
    the current mounting (see the warning in ``config.py``). It is quantified here
    so it can be recognised in a fit residual instead of mistaken for a bad fit.
    """
    wrist = digit_base_points_wrist(dims)
    cad = digit_base_points_cad(dims)
    return {name: ((cad[name] - wrist[name]),
                   float(np.linalg.norm(cad[name] - wrist[name])))
            for name in DIGIT_NAMES}


def growth_axis_wrist(dims=None):
    """Mean digit growth direction in the wrist frame (unit vector).

    Mirrors :func:`config.hand_growth_axis` but without building solver configs,
    so it stays usable as a cheap orientation sanity check ("do the fingers point
    away from the flange?"). Points essentially along +Y.
    """
    if dims is None:
        dims = load_hand_dimensions()
    offs = [finger_base_offset(dims["o_finger"][i], dims["a_finger"][i])
            for i in range(len(FINGER_NAMES))]
    offs.append(finger_base_offset(dims["o_thumb"][0], dims["a_thumb"][0]))
    axes = [T[:3, :3] @ np.array([0.0, 0.0, 1.0]) for T in offs]
    g = np.mean(axes, axis=0)
    return g / np.linalg.norm(g)


# ---------------------------------------------------------------------------
# The CAD side: where hand.scad actually puts the digits.
# ---------------------------------------------------------------------------
#
# ``hand()`` places each digit as ``translate(o[i]) rotate(a[i])`` with the digit
# growing along +x for the sum of its bone lengths, and ``rotate([x,y,z])`` in
# OpenSCAD means ``Rz(z) Ry(y) Rx(x)``. Every reference digit specifies only ``ry``,
# so in practice the placement is a splay about the CAD +Y axis.
#
# ``a_print`` (45 deg) is deliberately NOT applied here. The measured STL bounding
# box settles it: the fingers reach ~169 mm along the part's x axis, which is the
# full straight digit length (the middle finger sums to 164 mm). Had a_print swung
# each digit 45 deg within the xy-plane, the same digits would reach only ~116 mm in
# x and ~-116 mm in y, and the real part spans just [-53, +55] in y. So whatever
# a_print does to the printed geometry, it does not rotate the digit axes out of
# the xz-plane -- and it is the digit *axes* these landmarks track.

def cad_digit_points_hand(dims=None):
    """``{label: xyz_m}`` -- digit base and straight-out tip in the CAD hand() frame.

    Independent of anything in the solver: this is read straight off the SCAD
    parameters, so it is the reference the solver's own mounting gets checked
    against.
    """
    if dims is None:
        dims = load_hand_dimensions()
    pts = {}
    for i, name in enumerate(FINGER_NAMES):
        pts.update(_cad_digit(name, dims["o_finger"][i], dims["a_finger"][i],
                              dims["bl_finger"][i]))
    pts.update(_cad_digit("thumb", dims["o_thumb"][0], dims["a_thumb"][0],
                          dims["bl_thumb"][0]))
    return pts


def _cad_digit(name, o_mm, a_deg, bones_mm):
    o = np.asarray(o_mm, float) / 1000.0
    rx, ry, rz = np.deg2rad(np.asarray(a_deg, float))
    R = _Rz(rz) @ _Ry(ry) @ _Rx(rx)          # OpenSCAD rotate([rx,ry,rz])
    reach = float(np.sum(bones_mm)) / 1000.0
    direction = R @ np.array([1.0, 0.0, 0.0])
    return {f"{name}_base": o, f"{name}_tip": o + reach * direction}


def cad_growth_axes_hand(dims=None):
    """``{digit: unit xyz}`` -- digit growth directions in the CAD hand() frame."""
    if dims is None:
        dims = load_hand_dimensions()
    axes = {}
    for i, name in enumerate(FINGER_NAMES):
        rx, ry, rz = np.deg2rad(np.asarray(dims["a_finger"][i], float))
        axes[name] = (_Rz(rz) @ _Ry(ry) @ _Rx(rx)) @ np.array([1.0, 0.0, 0.0])
    rx, ry, rz = np.deg2rad(np.asarray(dims["a_thumb"][0], float))
    axes["thumb"] = (_Rz(rz) @ _Ry(ry) @ _Rx(rx)) @ np.array([1.0, 0.0, 0.0])
    return axes


def solver_growth_axes_wrist(dims=None):
    """``{digit: unit xyz}`` -- digit growth directions as the solver mounts them."""
    if dims is None:
        dims = load_hand_dimensions()
    axes = {}
    for i, name in enumerate(FINGER_NAMES):
        T = finger_base_offset(dims["o_finger"][i], dims["a_finger"][i])
        axes[name] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    T = finger_base_offset(dims["o_thumb"][0], dims["a_thumb"][0])
    axes["thumb"] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return axes


def yaw_agreement_deg(candidate, dims=None, fingers_only=True):
    """Mean angle (deg) between CAD and solver digit growth axes under ``candidate``.

    Decides the yaw sign, offline and without any CAD measurement: the CAD says
    which way each digit points in the palm frame, the solver's ``hand_base_offset``
    says which way it points in the wrist frame, and only the correct yaw brings the
    two into agreement. Flipping the sign sends every finger's growth component to
    the opposite side, so the wrong yaw scores ~180 deg out.

    ``fingers_only`` excludes the thumb, whose 100 deg base angle passes through
    ``finger_base_offset``'s ``a_print`` conjugation and lands tens of degrees off in
    the solver regardless of yaw -- it is not evidence either way.
    """
    if dims is None:
        dims = load_hand_dimensions()
    cad = cad_growth_axes_hand(dims)
    solver = solver_growth_axes_wrist(dims)
    R = candidate.R_wrist_from_hand if isinstance(candidate, Candidate) else candidate
    names = list(FINGER_NAMES) if fingers_only else DIGIT_NAMES
    angles = []
    for name in names:
        a = R @ cad[name]
        b = solver[name]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        angles.append(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))
    return float(np.mean(angles))


def digit_reach_wrist(dims=None):
    """``{digit: reach_m}`` -- summed bone length per digit, straight from CAD.

    Used to predict where a fingertip lands so a candidate rotation can be checked
    against a part bounding box.
    """
    if dims is None:
        dims = load_hand_dimensions()
    reach = {name: float(np.sum(dims["bl_finger"][i])) / 1000.0
             for i, name in enumerate(FINGER_NAMES)}
    reach["thumb"] = float(np.sum(dims["bl_thumb"][0])) / 1000.0
    return reach


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _check_rotation(name, R):
    R = np.asarray(R, float)
    orth = float(np.max(np.abs(R.T @ R - np.eye(3))))
    det = float(np.linalg.det(R))
    ok = orth < 1e-12 and abs(det - 1.0) < 1e-12
    print(f"  {'ok ' if ok else 'BAD'} {name:34s} det={det:+.12f}  "
          f"orthonormality residual={orth:.2e}")
    return ok


def main():
    print("=" * 78)
    print("wrist <- STL convention (pure rotation; the origins coincide)")
    print("=" * 78)
    print("\nR_wrist<-stl = Rz(+90) @ Rx(180):")
    for row in R_WRIST_FROM_STL:
        print("   [" + "  ".join(f"{v:+.0f}" for v in row) + "]")
    axis_angle = "180 deg about (1,1,0)/sqrt(2)"
    print(f"   ({axis_angle}; self-inverse: "
          f"{np.allclose(R_WRIST_FROM_STL @ R_WRIST_FROM_STL, np.eye(3))})")

    print("\nrotation validity:")
    ok = _check_rotation("R_WRIST_FROM_STL", R_WRIST_FROM_STL)
    for name, cand in CANDIDATE_ROTATIONS.items():
        ok &= _check_rotation(name, cand.R)
    if not ok:
        raise SystemExit("a candidate rotation is not a proper rotation")

    # Duplicates would tie in the fitter's scoring and trip its "the two best
    # conventions are indistinguishable" warning for no reason.
    names = list(CANDIDATE_ROTATIONS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.allclose(CANDIDATE_ROTATIONS[a].R, CANDIDATE_ROTATIONS[b].R):
                raise SystemExit(f"candidates {a!r} and {b!r} are the same rotation")
    print(f"  ok  all {len(names)} candidates are distinct")

    dims = load_hand_dimensions()

    print("\nyaw sign, from CAD vs solver digit growth axes (fingers only, offline):")
    for name, cand in CANDIDATE_ROTATIONS.items():
        print(f"  {name:40s} mean axis disagreement "
              f"{yaw_agreement_deg(cand, dims):7.2f} deg")
    print("  The two yaw signs are ~180 deg apart, so this alone settles the sign;")
    print("  the export flip is settled separately, by part-geometry containment.")

    wrist = digit_base_points_wrist(dims)
    cad = digit_base_points_cad(dims)
    disc = mounting_discrepancy(dims)
    reach = digit_reach_wrist(dims)

    print("\n" + "=" * 78)
    print("digit base landmarks, wrist frame (mm) -- measure these in CAD to verify")
    print("=" * 78)
    print(f"{'digit':8s} {'solver mounting':>26s} {'CAD o_finger rotated':>26s} "
          f"{'delta':>8s} {'reach':>7s}")
    for name in DIGIT_NAMES:
        w = wrist[name] * 1000.0
        c = cad[name] * 1000.0
        print(f"{name:8s} "
              f"({w[0]:7.2f},{w[1]:7.2f},{w[2]:7.2f}) "
              f"({c[0]:7.2f},{c[1]:7.2f},{c[2]:7.2f}) "
              f"{disc[name][1] * 1000.0:7.2f} {reach[name] * 1000.0:6.1f}")

    worst = max(DIGIT_NAMES, key=lambda n: disc[n][1])
    print(f"\nknown mounting discrepancy: worst is {worst} at "
          f"{disc[worst][1] * 1000.0:.2f} mm -- config.finger_base_offset() rotates")
    print("its digit rotations from CAD axes into solver axes but leaves the digit")
    print("*translations* unrotated. Expect residuals of this size in any fit; they")
    print("are the hand model, not the measurement.")

    g = growth_axis_wrist(dims)
    print(f"\nmean growth axis (wrist frame): "
          f"({g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f})  -- should be ~+Y")

    print("\nSelf-check passed. Supply a measured T_flange<-stl to")
    print("compose_flange_from_wrist(), or run mount_onshape_fit.py.")


if __name__ == "__main__":
    main()
