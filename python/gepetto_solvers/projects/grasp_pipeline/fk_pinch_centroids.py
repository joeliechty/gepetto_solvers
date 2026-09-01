"""Brute-force the pinch centroid of every thumb-opposition finger combination.

For each combination of the thumb plus one or more fingers, this finds the
per-finger flexor tensions at which those fingertips come together, and reports
the centroid of their contact spheres in the WRIST / HAND-BASE frame.

No optimization, no contact constraints: it scans a grid of tensions and reads
off the geometry.

WHY PER-FINGER TENSIONS MATTER. An earlier version of this swept a single
SHARED flexor tension across all five digits, which forces every finger to the
same curl. The thumb and a finger reach each other at very different tensions
(index ~2.4 N against thumb ~1.25 N), so a shared sweep cannot find a pinch at
all -- it bottoms out with the tips still ~65 mm apart and makes the hand look
incapable of opposition. Each digit gets its own tension here.

WHAT MAKES THE BRUTE FORCE CHEAP. With no contact attached and the wrist prior
pinned, the fingers are kinematically INDEPENDENT: the only variable they share
is the wrist, and with nothing pulling against its prior the wrist does not
move (asserted below). So each finger's tip is a function of its own flexor
tension alone, one sweep of NUM_Q solves builds a per-finger lookup table, and
scanning every combination of tensions is pure array indexing with no further
solves. :func:`check_independence` verifies this rather than assuming it.

The wrist is pinned at the IDENTITY pose, so tip poses are already expressed in
the hand-base frame. Since the answers are wrist-frame, the choice of wrist pose
does not affect them.

Run (from the ``python/`` directory):
    python scripts/fk_pinch_centroids.py
"""

import itertools

import numpy as np

from gepetto_solvers.core.diagnostics import PlannerLogger
from gepetto_solvers.core.hands.tendon_5f import (
    default_hand_tip_radii,
)
from gepetto_solvers.core.solvers import (
    HandFKSolver,
    HandSolveParams,
    solved_wrist_pose,
    tip_gap_matrix,
)

FINGER_NAMES = ["index", "middle", "ring", "pinky", "thumb"]
THUMB = "thumb"

# The viewer's tension sliders: 0..3 N in 0.05 N steps. Matching them means a
# result here can be dialled straight into viz_interactive.py and looked at.
#
# The FK solve itself stays well-posed to about 4.5 N (past that it hits an
# IndeterminantLinearSystem), and the pinky in particular is still closing at
# 3 N -- so --q-max raises the ceiling past the slider range to find out
# whether a pinch that looks unreachable is genuinely unreachable or merely
# out of slider travel.
Q_MIN, Q_MAX, Q_STEP = 0.0, 3.0, 0.05
Q_GRID = np.round(np.arange(Q_MIN, Q_MAX + 1e-9, Q_STEP), 4)


def set_q_max(q_max):
    """Re-bind the tension grid to a new ceiling (see the note above)."""
    global Q_MAX, Q_GRID
    Q_MAX = float(q_max)
    Q_GRID = np.round(np.arange(Q_MIN, Q_MAX + 1e-9, Q_STEP), 4)
    return Q_GRID

# Spatial search grid for the common pinch point, in meters. 2 mm is well below
# the ~5-7 mm tip radii, and the winner is refined against the tension grid
# afterwards, so this only has to get close enough to pick the right basin.
C_GRID_STEP = 0.002
C_GRID_PAD = 0.010          # slack around the swept tip cloud

# Half-width (in tension-grid steps) of the local re-scan around the spatial
# stage's pick. +/-6 steps = +/-0.3 N, comfortably wider than a 2 mm spatial
# discretization can be wrong by.
REFINE_SWEEP = 6

# Tie-break weight (cost units per Newton of total tension). Several tension
# pairs can put two tips exactly in contact; without a tie-break the winner is
# whichever the argmin happens to hit, which can be a deeply curled pose that
# reaches the same contact on the way back round. Tiny enough never to override
# a real difference in fit, big enough to prefer the least-curled pinch.
TENSION_TIEBREAK = 1e-12


# ---------------------------------------------------------------------------
# Step 1: the per-finger tip lookup table.
# ---------------------------------------------------------------------------

def build_tip_table(verbose=True):
    """``{finger: (len(Q_GRID), 3)}`` tip sphere centers vs that finger's flexor
    tension, in the wrist frame.

    Swept with every finger at the same tension, which is legitimate ONLY
    because the fingers are independent -- :func:`check_independence` is what
    makes that safe to rely on. The solver instance is kept across the sweep so
    each step warm-starts from the last, matching how the interactive viewer
    drives FK (``viz_interactive._fk_solve`` reuses a cached HandFKSolver).
    """
    params = HandSolveParams(wrist_pose=np.eye(4))
    solver = HandFKSolver(params)

    tips = {name: [] for name in FINGER_NAMES}
    worst_wrist = 0.0

    for i, q in enumerate(Q_GRID):
        params.flexor_tensions = [float(q)] * len(FINGER_NAMES)
        frame = solver.solve().frames[0]
        for name in FINGER_NAMES:
            pose = np.asarray(frame[name].marginals.sites[-1].pose.mean, float)
            tips[name].append(pose[:3, 3])
        worst_wrist = max(worst_wrist, float(
            np.abs(solved_wrist_pose(solver.configs, frame) - np.eye(4)).max()))
        if verbose and i % 10 == 0:
            print(f"  [table] Q = {q:.2f} N ({i + 1}/{len(Q_GRID)})")

    # The whole "positions are already in the hand-base frame" claim rests on
    # this, so it is an assertion rather than a printed number.
    assert worst_wrist < 1e-6, (
        f"wrist moved off identity by {worst_wrist:.2e}; tip positions are no "
        f"longer hand-base-frame")
    if verbose:
        print(f"  [table] {len(Q_GRID)} tensions, wrist held to {worst_wrist:.1e}")

    return {name: np.array(v) for name, v in tips.items()}


def check_independence(table, tol_mm=0.1):
    """Assert each finger's tip depends only on ITS OWN flexor tension.

    Solves heterogeneous tension vectors (fingers at different tensions) and
    compares every tip against the shared-sweep table. This is the one
    assumption the lookup shortcut rests on; if it fails, the table is
    meaningless and everything downstream is wrong.

    The first case is the configuration the hand was observed pinching in
    (index 2.4 N, thumb 1.25 N, the rest at the 0.6 N default), so this doubles
    as a check against a known-good pose.
    """
    cases = [
        {"index": 2.40, "middle": 0.60, "ring": 0.60, "pinky": 0.60, "thumb": 1.25},
        {"index": 0.00, "middle": 1.50, "ring": 3.00, "pinky": 0.75, "thumb": 2.00},
        {"index": 2.95, "middle": 0.05, "ring": 1.10, "pinky": 2.50, "thumb": 0.35},
    ]

    worst = 0.0
    for case in cases:
        params = HandSolveParams(wrist_pose=np.eye(4))
        params.flexor_tensions = [case[n] for n in FINGER_NAMES]
        # A FRESH solver per case: a warm-started one would carry the previous
        # case's posture in and could mask a coupling by starting near the
        # answer.
        frame = HandFKSolver(params).solve().frames[0]
        for name in FINGER_NAMES:
            got = np.asarray(
                frame[name].marginals.sites[-1].pose.mean, float)[:3, 3]
            want = table[name][int(np.argmin(np.abs(Q_GRID - case[name])))]
            worst = max(worst, float(np.linalg.norm(got - want)) * 1000.0)

    print(f"  [independence] worst tip mismatch vs table: {worst:.3f} mm")
    if worst > tol_mm:
        raise AssertionError(
            f"fingers are NOT independent ({worst:.2f} mm > {tol_mm} mm): the "
            f"shared-tension lookup table is invalid. Sweep each finger "
            f"separately instead.")
    return worst


# ---------------------------------------------------------------------------
# Step 3: brute-force the pinch point.
# ---------------------------------------------------------------------------
#
# For a combination with tip radii r_i we want the tensions and the common
# point c at which every tip SPHERE touches c:
#
#     min over c, Q:   sum_i ( ||tip_i(Q_i) - c|| - r_i )^2
#
# For a FIXED c each finger picks its own Q_i independently, which is what
# makes this tractable: scanning c over a spatial grid costs
# O(|c grid| * n_fingers * |Q grid|), against 61^5 = 845M for a naive scan of
# the tension product on the five-digit combination.


def _spatial_fields(table, radii):
    """Per finger, over a shared spatial grid: the best achievable squared
    surface residual to each grid point, and the tension index achieving it.

    Returns ``(grid_points (M,3), {finger: (best_sq (M,), best_idx (M,))})``.
    """
    allpts = np.concatenate([table[n] for n in FINGER_NAMES], axis=0)
    lo, hi = allpts.min(axis=0) - C_GRID_PAD, allpts.max(axis=0) + C_GRID_PAD
    axes = [np.arange(lo[k], hi[k] + C_GRID_STEP, C_GRID_STEP) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    fields = {}
    for name in FINGER_NAMES:
        r = radii[name]
        best_sq = np.full(len(grid), np.inf)
        best_idx = np.zeros(len(grid), dtype=np.int32)
        for qi, tip in enumerate(table[name]):
            resid = np.linalg.norm(grid - tip, axis=1) - r
            sq = resid * resid
            better = sq < best_sq
            best_sq[better] = sq[better]
            best_idx[better] = qi
        fields[name] = (best_sq, best_idx)
    return grid, fields


def pinch_cost(tips, r, tensions):
    """The objective, for a batch of candidate poses.

    ``tips`` is ``(M, k, 3)``, ``r`` is ``(k,)``, ``tensions`` is ``(M, k)``.
    Cost is ``sum_i (||x_i - c|| - r_i)^2`` about each pose's own centroid
    ``c``: zero when every tip sphere touches that common point, i.e. when the
    spheres are in contact rather than merely as close as the fingers can drag
    them. Plus the :data:`TENSION_TIEBREAK` nudge toward the least-curled pose.
    """
    c = tips.mean(axis=1, keepdims=True)                 # (M, 1, 3)
    resid = np.linalg.norm(tips - c, axis=2) - r         # (M, k)
    return (resid * resid).sum(axis=1) + TENSION_TIEBREAK * tensions.sum(axis=1)


def _refine(table, radii, combo, start_idx, sweep=REFINE_SWEEP):
    """Exhaustive local re-scan of the tension grid around ``start_idx``.

    The spatial stage picks tensions against a DISCRETIZED point c; this
    re-optimizes them against the true objective (each pose's own exact
    centroid) over a small window at full tension resolution. Vectorized over
    the whole window at once -- the five-digit combination is 13^5 = 371k
    candidate poses, which is a fraction of a second as array work and minutes
    as a Python loop.
    """
    ranges = [np.arange(max(0, i0 - sweep), min(len(Q_GRID), i0 + sweep + 1))
              for i0 in start_idx]
    mesh = np.meshgrid(*ranges, indexing="ij")
    idx = np.stack([m.ravel() for m in mesh], axis=1)     # (M, k)

    tips = np.stack([table[n][idx[:, j]] for j, n in enumerate(combo)], axis=1)
    r = np.array([radii[n] for n in combo])
    cost = pinch_cost(tips, r, Q_GRID[idx])
    return idx[int(np.argmin(cost))]


def solve_combo(table, radii, combo, grid, fields):
    """Best pinch for one finger combination -> a result dict."""
    total = np.zeros(len(grid))
    for name in combo:
        total += fields[name][0]
    g = int(np.argmin(total))
    coarse_idx = [int(fields[name][1][g]) for name in combo]

    idx = _refine(table, radii, combo, coarse_idx)

    tips = np.stack([table[n][i] for n, i in zip(combo, idx)])
    r = np.array([radii[n] for n in combo])
    centroid = tips.mean(axis=0)
    gaps = tip_gap_matrix(tips, r)

    # A digit parked on the last grid point was still closing when the scan ran
    # out of tension, so its optimum is a boundary artifact rather than a found
    # minimum -- the real pinch may lie beyond Q_MAX.
    at_limit = [n for n, i in zip(combo, idx) if i == len(Q_GRID) - 1]

    return {
        "combo": combo,
        "tensions": {n: float(Q_GRID[i]) for n, i in zip(combo, idx)},
        "centroid": centroid,
        "pinch_point": grid[g],
        "tips": tips,
        "radii": r,
        "min_gap": float(gaps.min()),
        "max_gap": float(gaps[np.isfinite(gaps)].max()),
        "spread": float(np.linalg.norm(tips - centroid, axis=1).mean()),
        "at_limit": at_limit,
    }


# ---------------------------------------------------------------------------
# Verification helpers.
# ---------------------------------------------------------------------------

def exhaustive_pair(table, radii, combo):
    """Full 61x61 scan for a 2-digit combination -- the ground truth the
    spatial-grid search is validated against before it is trusted on the larger
    combinations.

    Scores candidates with :func:`pinch_cost`, the SAME objective the refiner
    uses, so a disagreement means the search missed the optimum rather than
    that the two stages wanted different things. (Note this is not the same as
    minimizing centre distance: for unequal radii the objective drives the tips
    to r_a + r_b apart -- touching -- not to maximum overlap.)
    """
    a, b = combo
    ia, ib = np.meshgrid(np.arange(len(Q_GRID)), np.arange(len(Q_GRID)),
                         indexing="ij")
    idx = np.stack([ia.ravel(), ib.ravel()], axis=1)
    tips = np.stack([table[a][idx[:, 0]], table[b][idx[:, 1]]], axis=1)
    r = np.array([radii[a], radii[b]])
    best = idx[int(np.argmin(pinch_cost(tips, r, Q_GRID[idx])))]
    return {a: float(Q_GRID[best[0]]), b: float(Q_GRID[best[1]])}


def roundtrip(table, result, tol_mm=0.5):
    """Re-solve FK at the reported tensions and confirm the centroid survives an
    actual solve rather than only a table lookup."""
    params = HandSolveParams(wrist_pose=np.eye(4))
    params.flexor_tensions = [result["tensions"].get(n, 0.6) for n in FINGER_NAMES]
    frame = HandFKSolver(params).solve().frames[0]
    tips = np.stack([
        np.asarray(frame[n].marginals.sites[-1].pose.mean, float)[:3, 3]
        for n in result["combo"]])
    return float(np.linalg.norm(tips.mean(axis=0) - result["centroid"])) * 1000.0


# ---------------------------------------------------------------------------

def main(q_max=None, log_name="fk_pinch_centroids"):
    if q_max is not None:
        set_q_max(q_max)
    with PlannerLogger(log_name, timestamp=False):
        radii_list = default_hand_tip_radii()
        radii = dict(zip(FINGER_NAMES, radii_list))
        print("tip contact-sphere radii (mm): "
              + ", ".join(f"{n}={radii[n]*1000:.2f}" for n in FINGER_NAMES))
        print(f"tension grid: {Q_MIN}..{Q_MAX} N step {Q_STEP} "
              f"({len(Q_GRID)} values/finger)\n")

        print("building tip lookup table...")
        table = build_tip_table()

        print("\nverifying finger independence...")
        check_independence(table)

        print("\nbuilding spatial fields...")
        grid, fields = _spatial_fields(table, radii)
        print(f"  [spatial] {len(grid):,} grid points at {C_GRID_STEP*1000:.0f} mm")

        others = [n for n in FINGER_NAMES if n != THUMB]
        combos = [(THUMB,) + c
                  for k in range(1, len(others) + 1)
                  for c in itertools.combinations(others, k)]

        print(f"\nsolving {len(combos)} thumb-opposition combinations...")
        results = [solve_combo(table, radii, c, grid, fields) for c in combos]

        # --- verification: pairs against an exhaustive scan ---
        print("\nmethod check (spatial grid vs exhaustive 61x61 on pairs):")
        for res in results:
            if len(res["combo"]) != 2:
                continue
            ex = exhaustive_pair(table, radii, res["combo"])
            got = res["tensions"]
            agree = all(abs(ex[n] - got[n]) <= Q_STEP + 1e-9 for n in ex)
            label = "+".join(res["combo"])
            print(f"  {label:22s} exhaustive {ex} vs found {got}  "
                  f"{'OK' if agree else 'MISMATCH'}")

        # --- verification: round-trip a few through a real solve ---
        print("\nround-trip check (re-solve at the reported tensions):")
        for res in results[:3]:
            err = roundtrip(table, res)
            print(f"  {'+'.join(res['combo']):22s} centroid moved {err:.3f} mm")

        # --- the answer ---
        print("\n" + "=" * 98)
        print("PINCH CENTROIDS -- centroid of the combination's fingertip contact")
        print("spheres at closest approach, in the WRIST / HAND-BASE frame (mm).")
        print("=" * 98)
        print(f"{'combination':30s} {'tensions (N)':36s} "
              f"{'centroid x, y, z':>26s} {'gap':>6s}")
        print("-" * 104)
        for res in results:
            label = "+".join(res["combo"])
            tens = " ".join(f"{n[:2]}={res['tensions'][n]:.2f}"
                            for n in res["combo"])
            c = res["centroid"] * 1000.0
            gap = res["min_gap"] * 1000.0
            mark = "touch" if gap <= 0.2 else ""
            flag = "*" if res["at_limit"] else ""
            print(f"{label:30s} {tens:36s} "
                  f"({c[0]:7.2f},{c[1]:7.2f},{c[2]:7.2f}) {gap:+6.1f} "
                  f"{mark:5s}{flag}")

        print("\n'gap' is the closest pair's SURFACE separation in mm: 0 = the two "
              "tip\nspheres just touch, negative = they overlap, positive = they "
              "never meet.")

        limited = [r for r in results if r["at_limit"]]
        if limited:
            print(f"\n* {len(limited)} combination(s) have a digit pinned at the "
                  f"{Q_MAX} N grid limit -- still closing when the scan ended, so "
                  f"the true pinch may need more tension than the sliders allow:")
            for res in limited:
                print(f"    {'+'.join(res['combo']):30s} at limit: "
                      f"{', '.join(res['at_limit'])}")

        far = [r for r in results if r["min_gap"] > 0.001]
        if far:
            print(f"\n{len(far)} combination(s) stay more than 1 mm apart even at "
                  f"their best pose:")
            for res in far:
                print(f"    {'+'.join(res['combo']):30s} closest surface gap "
                      f"{res['min_gap']*1000:5.1f} mm")

        return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q-max", type=float, default=None,
                    help="flexor tension ceiling in N (default 3.0, the viewer's "
                         "slider max; the FK solve stays well-posed to ~4.5)")
    ap.add_argument("--log-name", default="fk_pinch_centroids",
                    help="basename for the .log written next to this script")
    args = ap.parse_args()
    main(q_max=args.q_max, log_name=args.log_name)
