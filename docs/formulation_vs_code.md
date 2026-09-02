# Where the code departs from the written formulation

Every entry here is a place the implementation does something other than what a
paper or plan says on the page. Each is deliberate, and each is written up with
the reason — because a reader who has the maths in front of them and finds the
code disagreeing needs to know, quickly, whether they have found a bug or a
decision.

Read this alongside [hand_solvers.md](hand_solvers.md), which explains what the
code *does*. This file only covers where it differs and why.

> **Numbering.** Most of the code cites the chapter-1 numbering (`§1.x`,
> `Eq 1.x`); the newest pre-grasp work cites a chapter-2 renumbering of the same
> material. Both point at the same equations; the source has not been swept.

---

## Contents

| # | Departure | Kind |
|---|---|---|
| [1](#1-the-fk-factor-is-ternary-not-binary) | The FK factor is ternary, not binary | correctness |
| [2](#2-h_t-is-the-exact--j_r-not--i) | `H_T` is the exact `−J_r⁻¹`, not `−I` | numerical |
| [3](#3-pinocchio-and-gtsam-order-a-twist-differently) | Pinocchio and GTSAM order a twist differently | convention |
| [4](#4-q-is-carried-as-one-block-per-digit) | `q` is carried as one block per digit | representation |
| [5](#5-sigma_fki-is-per-frame-and-its-tightness-is-a-tradeoff) | `Σ_fk,i` is per-frame, and its tightness is a tradeoff | tuning |
| [6](#6-joint-limits-are-read-but-not-enforced) | Joint limits are read but not enforced | **known gap** |
| [7](#7-a-digits-base-is-not-a-variable) | A digit's base is not a variable | representation |
| [8](#8-contact-is-witness-free-where-the-surface-allows-it) | Contact is witness-free where the surface allows it | representation |
| [9](#9-the-support-plane-uses-one-residual-not-five) | The support plane uses one residual, not five | representation |
| [10](#10-plane-avoidance-is-off-during-the-contact-phases) | Plane avoidance is off during the contact phases | scheduling |
| [11](#11-a-urdfs-visual-meshes-need-two-corrections-nothing-states) | A URDF's visual meshes need two corrections nothing states | asset convention |

---

## 1. The FK factor is ternary, not binary

**Written:** a `NoiseModelFactor2` over `(q, T_i)`, with the hand's base fixed by
a prior.

**Code:** [`PinocchioFKFactor`](../include/gepetto_solvers/digits/rigid/PinocchioFKFactor.h)
is ternary over `(T_w, q, T_i)`, and

    f_fk,i(T_w, q) = T_w · T_fk,i(q)

**Why.** The wrist is a *variable* here, not a clamp. It carries only a soft
prior — `p(T_w)`, with `sigma_wrist_pos` defaulting to 1e-4 — and that softness
is load-bearing: pressing the hand onto a table moves the base tens of
millimetres off the commanded pose, and the solve is meant to allow it.

A binary factor treating the base as fixed would evaluate `T_fk(q)` in a frame
that had already moved. Every FK residual would then carry the wrist's
displacement as error, and the two halves of the problem would fight: contact
pulling the wrist one way, the kinematics insisting the links sit where a
stationary base would put them.

**Guarded by** `test_the_prediction_composes_the_wrist_with_the_fk` in
[tests/core/test_pinocchio_factor.py](../tests/core/test_pinocchio_factor.py).

## 2. `H_T` is the exact `−J_r⁻¹`, not `−I`

**Written:** `H_T = −I₆`.

**Code:** the exact `−J_r⁻¹(e)`, obtained without deriving it.

**Why.** For `e = Log(·)`, the derivative with respect to the argument is the
inverse right Jacobian of SE(3), so `∂e/∂T_i = −J_r⁻¹(e)`. `−I₆` is its
first-order approximation, exact only as `e → 0`. The two agree near
convergence and diverge away from it — which is precisely where a Gauss-Newton
step needs to be right.

The exact form is free if the error is composed out of GTSAM's own operations
and Pinocchio is injected only where `q` enters:

```cpp
T_pred = T_w.compose(T_fk, Hc1, Hc2);          // ∂T_pred/∂T_w, ∂T_pred/∂T_fk
e      = T_pred.localCoordinates(T_i, Hl1, Hl2);  // ∂e/∂T_pred, ∂e/∂T_i

H_wrist = Hl1 * Hc1;
H_q     = Hl1 * Hc2 * SWAP * J_pin;
H_site  = Hl2;                                  // the exact −J_r⁻¹(e)
```

GTSAM owns every manifold derivative; the only hand-written piece is `SWAP`.

**Guarded by** `test_h_site_is_not_merely_negative_identity`, which asserts the
exact form is genuinely in use — evaluated well away from the solution, where
the two visibly differ.

## 3. Pinocchio and GTSAM order a twist differently

**Written:** correctly, in the plan — noted here because it is the single
easiest thing in the integration to get wrong.

**Code:** every Jacobian from Pinocchio has its top and bottom three rows
exchanged before it reaches GTSAM.

    Pinocchio:  [v; ω]        GTSAM Pose3 tangent:  [ω; v]

`pinocchio::LOCAL` (not `LOCAL_WORLD_ALIGNED`) is the right reference frame,
because `compose`'s second-argument Jacobian is in the body frame.

**Why this one gets its own section.** *A wrong swap does not raise, and does not
produce wrong poses.* The error function never touches the Jacobian — it is pure
forward kinematics — so the solve still converges to the correct answer. It just
takes a worse path: more iterations, a smaller basin of attraction. It reads as a
badly conditioned problem, not as a bug, and is the kind of thing that costs a
day of tuning `Σ_fk` to chase.

So it is proved rather than inspected, at two levels:

* [tests/core/test_pinocchio_env.py](../tests/core/test_pinocchio_env.py) pins
  the *convention* — that Pinocchio really does return `[v; ω]` — independently
  of any code that relies on it;
* [tests/core/test_pinocchio_factor.py](../tests/core/test_pinocchio_factor.py)
  checks all three Jacobian blocks against numerical differentiation taken
  through **GTSAM's own retraction** (`pose3_retract` is bound for exactly this:
  a hand-written exponential map would compare the factor against a second
  implementation of the convention under test).

Both were verified to **fail** with the swap removed — and to fail *nothing
else*, which is the measured demonstration that the bug is otherwise invisible.

## 4. `q` is carried as one block per digit

**Written:** `q ∈ ℝ^N` for the hand's `N` degrees of freedom — one vector.

**Code:** one `Vector4` variable per digit, four for the Allegro hand.

**Why.** The posterior is identical: splitting is *exact* as long as no factor
couples joints across digits, and none does — each digit's FK touches only its
own joints, `p(q)` is built block-diagonal, the GP `Qc` is isotropic, and the
task constraints key off site poses rather than `q` at all. Allegro is fully
actuated with no mechanical coupling.

Three things follow from the split:

* **Sparsity.** A shared `q` is adjacent to every site pose, so eliminating it
  forms one large clique. Per-digit, each block touches only its own sites.
* **A silent 4× tightening, avoided.** `add_temporal_gp`,
  `add_actuation_priors` and `add_displacement_priors` are per-digit loops. With
  one shared key they would each emit four factors on the *same* variable —
  and independent Gaussians on one variable multiply, so the prior would come
  out four times tighter than requested, with nothing reporting it.
* **Reversibility.** Coupling can be added later as a factor spanning several
  `q^d` keys, which is how GTSAM expresses coupling anyway. Splitting a shared
  vector afterwards would mean rewriting the factor, the priors, the GP chain
  and the state bundle.

## 5. `Σ_fk,i` is per-frame, and its tightness is a tradeoff

**Written:** `Σ_fk,i` is the kinematic relaxation for frame `i`, and as
`Σ_fk,i → 0` the likelihood approaches a hard constraint.

**Code:** implemented literally — it is the factor's own noise model, so it is
per frame, with a per-site override on
[`RigidHandKinematicsConfig`](../include/gepetto_solvers/hand/kinematics/rigid/RigidHandKinematics.h).

**What the maths does not say** is what it costs. Tightening never *fails* —
every setting below reaches machine-zero residual — it costs iterations, because
a stiffer likelihood means a larger step in the site poses for the same step in
`q`. Measured on the Allegro hand, seeded a full 0.4 rad per joint from the prior
mean:

| σ rot / pos | iterations |
|---|---|
| 1e-2 / 1e-3 | 4 |
| 1e-3 / 1e-4 | 5 |
| 3e-4 / 3e-5 | 9 |
| **1e-4 / 1e-5** | **18 (default)** |
| 3e-5 / 3e-6 | 44 |
| 1e-5 / 1e-6 | 103 |

1e-5 m is ten microns — two orders below any contact tolerance in this
repository — so the kinematics is already exact for every purpose it is put to.
Seeded *at* the prior mean, which is the normal case since `q_init` and `q_S`
are the same posture, it converges in one iteration at any of these.

## 6. Joint limits are read but not enforced

**This is a known gap, not a decision.**

The URDF carries limits, `RigidChainModel` exposes them, and the workbench's
sliders are bounded by them — but **nothing constrains the solve**, so IK can
hyperextend into a configuration the real hand cannot reach. Only `p(q)` keeps
`q` plausible.

This was demonstrated while tuning Allegro's default posture: an unconstrained
search proposed a thumb abduction of 1.35 rad against a ±0.47 rad stop.

The fix is AL inequalities emitted through the `ConstraintTagger` that
`HandKinematics::add_kinematics_factors` is already handed. It additionally needs
`HandModel` to learn that a hand can *have* hard kinematics constraints:
[HandModel.cpp](../src/hand/HandModel.cpp) currently decides the Augmented
Lagrangian path from the task environment alone, so a limit-constrained hand
would build the constraints and never enforce them — the worst of both.

## 7. A digit's base is not a variable

**Written:** the wrist and the digit bases are jointly distributed.

**Code:** one Gaussian on the wrist, times a *deterministic* SE(3) composition
per digit:

    T_0^d = T_w · T_offset^d

**Why.** Expressing the base as its own variable tied to the wrist by a stiff
prior leaves a soft-rigid null space for the optimizer to wander in. Composing it
away removes the freedom instead of penalising it.

Both mechanisms honour this, by different means: the tendon hand's rod uses a
root reparameterization (Eq 1.43) so node 0 is not a variable, and
`RigidHandKinematics` makes site 0 *alias the wrist key* rather than carry a
variable of its own. Either way `site_is_root` reports true for it and the
collision passes skip it — a sphere there would read the wrist origin.

The invariant this buys is that the wrist is recoverable from a frame alone, by
inverting digit 0's offset — which `solved_wrist_pose` does. `HandState` also
carries the wrist directly, and a test asserts the two routes agree.

## 8. Contact is witness-free where the surface allows it

**Written:** object contact introduces a witness point `p_c,obj` on the surface,
with a 5-row residual (Eq 1.35–1.39).

**Code:** for an analytic ellipsoid or an ellipsoid set, contact is the
*center-direct* equality (Eq 1.101 / Eq 1.13) — one residual on the tip sphere's
centre, no witness variable at all. The witness form is kept for a baked SDF,
which has no closed-form distance, and for the two settings that are meaningless
without a witness variable to attach to (`contact_drop_normal_row`,
`witness_target`).

**Why.** Where the surface has a closed-form distance, four of the witness
form's five rows exist only to pin the gauge of the free point it introduced.
Dropping it removes three variables and four residual rows per digit.

`HandModel::uses_center_direct_contact` is the single source of truth for the
choice — `build_graph` picks the factor with it and `get_initial_values` decides
whether to seed a witness with it, and a disagreement between those two would
leave either an orphan variable or an unseeded key.

## 9. The support plane uses one residual, not five

**Written:** table contact as a 5-residual witness form (Eq 1.60–1.64).

**Code:** one residual on the contact sphere's centre,
`Dist_plane(c) = 0`, via `PlaneCollisionGapFactor` wrapped as an equality.

**Why.** The same argument as §8, and §1.8 already makes it for
`support_contact_node`: for a *plane* a single scalar on the centre leaves no
rotational freedom to brick the solver, and the witness's other four rows only
pinned the gauge of the point it added. The signed form is used rather than the
paper's `|·|`, which has a kink exactly where the solver operates.

Consequently there is deliberately **no table witness key** — the support plane
introduces no variable of its own at any step.

## 10. Plane avoidance is off during the contact phases

**Written:** the table collision inequality is kept on throughout.

**Code:** `phase1` and `phase2` both set `plane_avoidance = False`.

**Why.** Phase 1 drives the fingers *onto* the plane, so the avoidance
half-space would be fighting the contact that phase exists to make; phase 2
inherits fingers still resting on it, so re-arming avoidance would start that
phase already in violation. The plane itself stays configured, and
object/finger–finger collision are untouched.

## 11. A URDF's visual meshes need two corrections nothing states

Not a departure from the maths — the meshes never touch the graph — but two
traps that cost real time on the Allegro hand and will cost it again on the next
URDF one, because in both cases *nothing in the data says the correction is
needed*.

**glTF is Y-up; URDF and ROS are Z-up.** The conversion is implicit in the
format. These files carry an identity node transform, so a loader that just
reads vertices produces a hand lying on its side, and no tool reports anything
wrong. Drake applies the rotation internally; we apply `GLTF_TO_URDF`, an
`R_x(+90°)`, in
[`hands/allegro/meshes.py`](../python/gepetto_solvers/core/hands/allegro/meshes.py).

**A mesh belongs to its LINK, not to the frame you attach it to.** The palm mesh
rides on the wrist, but it is authored in `palm_link`, which the URDF places
95 mm up the root's +Z through the fixed `root_to_base` joint. Drawn at the
wrist it lands a whole palm-height low, hanging below the finger bases. The
link's own fixed placement composes in ahead of the axis correction.

**Both were found the same way, and both are guarded that way**: against the
URDF's own `<collision>` boxes and origins, which are written in the link frame
and so are independent ground truth. Extents alone cannot pin a rotation — they
are symmetric under ±90° — so the mesh CENTRES are what fix the sign: the right
rotation puts them a mean 3.2 mm from the collision origins, the wrong one
36.8 mm. `tests/core/test_allegro_hand.py` checks the axis, the sign and the
palm's placement separately, and each was verified to fail on its own bug.

The transform is carried **per mesh**, as the third element of
`Hand.visual_meshes()`, rather than assumed by the renderer: it is a property of
the asset, so a hand shipping Z-up STL or OBJ returns the identity.
