# Five-Finger Grasp — Why the Small Objects Don't Solve

Investigation notes for `five_finger_hand_grasp_test.py`. Summarizes why the
sphere grasp works but the cylinder/cube fail, based on reading the factor
wiring and running controlled experiments.

## Symptoms

- **`big_sphere`** — solves; every fingertip lands on the surface (gap ≈ 0).
- **`sphere` / `cylinder`** — fingers barely move, stall with ~0.11 m surface
  gaps; some fingers bend *backwards*.
- **`cube`** — bends backwards, does not solve.
- Loosening the wrist prior "didn't help" (see below — it was overridden and
  too tight).
- Earlier, `big_sphere` didn't just stall — it **crashed** with
  `IndeterminantLinearSystem (Symbol: Q0)`. That was a separate,
  now-fixed conditioning bug in the tension prior; see
  [The `IndeterminantLinearSystem` crash](#the-indeterminantlinearsystem-crash-separate-from-the-stalls-above).

## How contact is actually wired (what's true)

Each finger gets its **own** witness point and its own full 5-residual contact
constraint — they are *not* competing for one shared point:

- `TendonHandModel.cpp` adds one `SdfWitnessContactFactor` per finger, keyed to
  `witness_key(i)` (per finger). Only `object_key()` (the object pose) is
  shared and anchored once.
- The 5 residuals are `[c_R, c_O, c_N, c_T1, c_T2]` (`EnvironmentFactors.h`).
- The fingertip contact normal is **radial**: `N_i = (p_c - c_i)/‖·‖`. A sphere
  tip has no preferred orientation, so the normal-alignment residual
  `c_N = 1 + N_i·N_obj` is satisfiable for **any** finger bend direction. There
  is no factor that prefers a physical curl over hyperextension, and no joint
  limit forbidding backward bending.
- The wrist is a real optimization variable — the single shared `wrist_key()`
  = `Symbol('W', 0)`. Each finger's Cosserat chain roots at it: the finger's
  `root_base_key_` is pointed at the shared wrist via `set_hand_base()`
  (`TendonHandModel.cpp`), so node 0 = `T_wrist ∘ T_offset` feeds the first
  Cosserat factor and a fingertip contact *does* have a Jacobian path back to
  the wrist.

## The wrist prior idea — it works, but was mis-set

Hypothesis: a loose wrist prior lets the solver slide the whole hand to bring
the fingertips to the object. **Correct.** Two reasons it appeared not to:

1. The test **hard-coded** `sigma_wrist_pos/rot = 1e-1` and commented out the
   CLI args, so the `--sigma-wrist-*` flags did nothing.
2. `1e-1` is still too tight. The tips start ~11 cm from the object, so the
   wrist must travel ~10 cm; a 0.1 m-std prior only allows ~1.4 cm of drift
   before its cost dominates.

Sweep on the **sphere** (`|wrist|` = distance the hand slid; gaps = per-finger):

| sigma | wrist moved | gaps [m] | result |
|-------|-------------|----------|--------|
| 1e-3 (pinned) | 0.000 | ~0.12 all | fail |
| 1e-1 (old default) | 0.014 | ~0.11 all | fail (matches screenshots) |
| 1.0 | 0.097 | ~0.000 all | **full grasp** |
| 10  | 0.097 | 0.000 all | full grasp |

**Fix applied:** un-hardcode the wrist sigma and default it loose (`1e1`). The
sphere now closes. Keep rotation from being wildly loose if you don't want the
hand to tumble.

## Why the cylinder and cube still fail

Experiments (loose wrist = `1e1` unless noted):

| case | iters | fingers (idx/mid/ring/pinky) | thumb |
|------|-------|------------------------------|-------|
| single finger, any primitive | 32–33 | solves, gap **0.000** | — |
| 5-finger **sphere** | 28 | 0 / 0 / 0 / 0 | **0** ✅ |
| 5-finger **cylinder** | 14 | 0.02 / 0.03 / 0.03 / 0.06 | 0.11 ✗ |
| 5-finger **cube** (identity seed) | **3** | 0.10 all | 0.09 ✗ |
| 5-finger **cube** (wrist warm-started at object) | 3 | 0.045 all | 0.09 ✗ |
| 5-finger **cube** (drop thumb constraint) | 3 | unchanged 0.10 | — |

A **single finger solves all three primitives perfectly** (cube gap 0.000), so
the box SDF is numerically fine on its own. The failures are specific to the
coupled 5-finger solve, and there are **two independent root causes**:

### 1. Directional contact + opposition (geometric — the sphere is the lucky case)

The formulation asks every fingertip to touch the *nearest surface point* of one
small primitive, with an orientation-free radial contact normal.

- A **sphere** is contactable from every direction with a smooth radial normal,
  so all five splayed fingers — including the opposed thumb — each find a
  tangent point and the shared-wrist gradient is coherent → full solve.
- A **cylinder** (curved side) and a **cube** (flat faces) are *directional*.
  The four palmar fingers can approach, but the **opposed thumb cannot present
  its tip to the same small side**. The wrist is a *rigid* transform — it slides
  the whole hand but cannot make an opposed thumb and four fingers touch the
  same 2.5 cm side. Hence the thumb stays ~0.10 m off in every cylinder/cube
  run. This is geometrically correct: a real hand grasps these by **opposition**
  (thumb on the far face), which "all tips → nearest surface" does not encode.

### 2. Flat-face early stall (numerical — cube-specific)

From the identity seed the cube quits at **iters = 3** with nothing moving, and
dropping the thumb constraint changes nothing — so this is not the thumb. It is
the flat-face landscape: orientation-free sphere contact on a flat face has a
large null-space of satisfying configs, and the box's edge/corner normal
discontinuities give poor Jacobians, so the coupled AL takes a bad first step
and terminates. That is the "bends backwards" look — the hard contact yanks the
tips toward the mis-placed face, hyperextends them (no joint limit), and the
solver gives up there. Warm-starting the wrist at the object gets the four
fingers moving (0.10 → 0.045 m) but it still plateaus → local-min/conditioning
stall, not infeasibility for those four.

## Recommendations

1. **Sphere is the only primitive well-posed as written** ("all tips tangent to
   nearest surface point"). Keep it as the working grasp demo.
2. **To grasp the cylinder/cube, encode opposition and size/place the object for
   the hand** — mirror what `make_big_sphere.py` did (custom center at the
   flexed-hand locus). Concretely: build `big_cylinder` / `big_cube` assets
   placed at the grasp locus, sized so opposing faces fit the thumb–finger span,
   and aim the **thumb's contact at the opposite face** from the four fingers
   rather than the nearest surface point. No wrist prior can substitute for
   opposition.
3. **The cube's early stall additionally needs** a warm-started wrist (toward
   the object) and/or a **rounded box SDF** (small corner radius → smooth
   normals), which removes the edge/corner Jacobian discontinuities that kill
   the first AL step.
4. **Optional robustness:** only attach contact constraints to fingers that can
   plausibly reach the object (or make the un-reachable ones soft priors) so a
   single infeasible hard equality doesn't corrupt the shared-wrist gradient.
5. **Deeper fixes (formulation):** the backward-bending is fundamentally that a
   spherical-tip contact is rotationally unconstrained. Consider a joint-limit /
   bend-sign factor (no hyperextension), or a **pad-aware** tip contact
   (require a specific face of the tip pose to face the object, not the radial
   sphere normal). Also note the doc's collision/`f_run` term is unimplemented
   (`= 1`), so nothing keeps a large object out of the palm.

## The `IndeterminantLinearSystem` crash (separate from the stalls above)

Before any of the geometric stalls, `big_sphere` **crashed** with
`gtsam::IndeterminantLinearSystemException ... near variable ... (Symbol: Q0)`.
This is a distinct failure from the "stall with a gap" cases and has a distinct
cause: it is **not** contact, placement, the object shape, or the AL parameters
— it is the **tension prior itself being numerically ill-conditioned**.

The per-finger tension prior was `diag([1e-8]*5, 1e-1)`: the five passive
tendons pinned at variance `1e-8` and the flexor left loose at `1e-1`. That is
**seven orders of magnitude of variance inside one 6-dim `Q` variable**. When
GTSAM eliminates that block, the flexor's (relatively tiny) information is
subtracted against the passive rows' ~`1e8` weight and the stiff rod/stress
factors; the flexor's conditional information vanishes into floating-point
cancellation, the `R` diagonal collapses, and the variable is reported as
underconstrained.

Note on the key name: the tension key is `Symbol('Q', 1000*id)` where `id` is a
**global** finger counter across *all* solver instances in the process. So
`Q0` is the first finger of the first solver built; `Q5000` in a later run is
that run's finger 0, etc. — the number is not a finger index.

### Evidence (controlled sweep, `big_sphere`)

| case | result |
|------|--------|
| solo index / middle / ring / pinky (with contact) | **FAIL** near own `Q` |
| solo thumb | OK (gap 0) — happened to be geometry-marginal |
| all 5, **no contact at all** (free space) | **FAIL in 0.5 s** near `Q` |
| legacy mount / zero splay / loose wrist | all **FAIL** — placement irrelevant |
| `al_iters` 10 / 20 / 40, `al_rate` 1.5 | all **FAIL** — AL params irrelevant |
| passive var **`1e-6`**, flexor `1e-1` | **OK**, all gaps 0 |
| passive var **`1e-4`** | **OK**, all gaps 0 |
| flexor var `1e-2` | **OK** |
| all tendons loose (`1e-2` uniform) | **OK** |

The **no-contact free-space failure is the smoking gun**: it fails in 0.5 s with
no object in the graph, so contact and the AL optimizer are ruled out entirely.
It is purely the prior's variance spread. (This is the same reason
`kinematics_test.py` uses a *uniform* `1e-2` prior — it hit this first.)

### Fix (applied)

Loosen the passive variance so it no longer spans the cliff — keep the flexor
loose as before:

```python
tensions_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
```

`1e-6` variance is ~1 mN std — still effectively "pinned by springs," but 100×
more numerical headroom. The full five-finger `big_sphere` grasp then converges
to gap ≈ 0 on all five tips. Deeper fix (optional): don't model near-constants as
`~0`-variance Gaussians — fold the passive `Q·SpatialWrench` into the disc-factor
constant, or split `Q` into separate passive/active variable blocks so one block
never spans both scales.

Two adjacent fragile spots worth knowing if `Q` crashes recur despite a sane
prior: `SolverBase::optimize()` builds `Marginals` with GTSAM's default
**Cholesky** even when `linear_solver_type = MULTIFRONTAL_QR`, and on the AL path
it re-adds each constraint as `penaltyFactor(final_mu)` where `mu` can reach
`2^al_iters` if the contact never converged — both raise the effective
conditioning demand at the very end of the solve.

## Cold-start vs. warm-start (the "slow high-tension solves")

`kinematics_test.py` was slow because it **rebuilt the solver every frame** to
change the wrist pose (which was baked in at construction). A fresh solver
re-seeds its initial guess to a *straight* hand, so every frame cold-started and
had to drive the rod from straight to a deep curl — and the higher the flexor
tension, the farther that equilibrium sits from the straight-rod linearization
point, so it took progressively more iterations (58–100+ at high tension vs ~30
warm).

Key realization: `SolverBase::optimize()` **never resets `values_`** — it is
seeded once in the constructor and thereafter holds the previous solution. So
warm-starting is automatic *as long as you keep one solver instance*. The only
thing forcing the rebuild was the construction-time wrist pose.

**Change made:** added `TendonHandSolver::set_wrist_pose(Matrix4)` (pybind:
`solver.set_wrist_pose(T)`) that re-aims the shared wrist prior without
rebuilding. Only `wrist_pose_` feeds that prior; the finger offsets and the
root-reparameterization factors are anchored to the shared wrist *key*,
independent of where it points — so updating the mean is all `build_graph()`
needs. `kinematics_test.py` now builds the solver **once** and calls
`set_wrist_pose()` + `solve()` each frame, so only the first frame is a cold
start and every subsequent frame is a small nudge that converges in a handful of
iterations even at full tension. Caveat: `set_wrist_pose()` deliberately does
*not* re-seed `get_initial_values()`, so a large discontinuous jump in the
commanded wrist from a near-straight state can still be slow — fine for a smooth
sweep, worth knowing for scripted jumps.

## Reproduction

```
# from crest-sparse/python, in the crest_py11 env
python -m tests.tendon_hand.five_finger_hand_grasp_test sphere      # solves with loose wrist
python -m tests.tendon_hand.five_finger_hand_grasp_test cylinder    # 4 fingers close, thumb ~0.10 m
python -m tests.tendon_hand.five_finger_hand_grasp_test cube        # stalls (iters=3)

# single finger reaches every primitive (gap 0.000):
python -m tests.tendon_hand.sdf_3dof_contact_kinematics_test cube

# warm-started wrist sweep (build once, set_wrist_pose per frame) — iters= drops
# from the cold-start count to a handful after the first frame:
python -m tests.tendon_hand.kinematics_test
```
