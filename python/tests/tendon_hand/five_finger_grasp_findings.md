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
- The wrist is a real optimization variable (`root_base_key`); node 0 =
  `wrist ∘ offset` feeds the first Cosserat factor, so a fingertip contact
  *does* have a Jacobian path back to the wrist.

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

## Reproduction

```
# from crest-sparse/python, in the crest_py11 env
python -m tests.tendon_hand.five_finger_hand_grasp_test sphere      # solves with loose wrist
python -m tests.tendon_hand.five_finger_hand_grasp_test cylinder    # 4 fingers close, thumb ~0.10 m
python -m tests.tendon_hand.five_finger_hand_grasp_test cube        # stalls (iters=3)

# single finger reaches every primitive (gap 0.000):
python -m tests.tendon_hand.sdf_3dof_contact_kinematics_test cube
```
