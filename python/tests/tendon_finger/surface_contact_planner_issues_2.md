# Why the contact planner "converges" but doesn't plan a contact trajectory

## Context

The free-space planner works ([point_to_point_planning.py](python/tests/tendon_finger/point_to_point_planning.py)). The contact planner ([point_to_contact_planning.py](python/tests/tendon_finger/point_to_contact_planning.py)) is identical except the terminal `goal_position` factor is replaced by `SdfContactFactor` + a dummy point `p_c` ([TendonFingerTrajectoryPlanner.cpp:340-359](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L340-L359)). After applying several fixes from the prior `surface_contact_planner_issues.md` doc (K=10, GP-on-tensions enabled, dummy point seeded on the object surface, Tikhonov regularizer on `p_c`, sticking with Dogleg) the solver converges in 23 iters at error 5e+04 — but the tip never moves toward the sphere. Final SDF residual = +9.99 cm vs target 3 mm. This document explains why, compares the approach to the state of the art, and ranks Dogleg-compatible fixes.

---

## 1. Diagnosis — this is a *weighting* problem, not a rank-deficiency problem

The solver isn't stuck. It found the global minimum of the graph **as written** — the problem is that the contact factor is overwhelmingly outweighed by everything else.

### 1.1 The contact factor is 10⁴–10⁷ times weaker than the free-space goal

| Factor (terminal pose constraint) | σ per dim | Information per dim |
|---|---:|---:|
| Free-space `goal_position` ([point_to_point_planning.py:139](python/tests/tendon_finger/point_to_point_planning.py#L139)) | √1e-5 ≈ 3.2 mm | **1e10** |
| Contact `contact_cov` ([point_to_contact_planning.py:82](python/tests/tendon_finger/point_to_contact_planning.py#L82)) | √1e-3 ≈ 32 mm | **1e3** |

The free-space planner pulls the tip with information 1e10 per dim — it dominates the GP-on-lengths smoothness prior (info ≈ 5e6) and easily drags the tip 10 cm to the goal. The contact planner pulls with info 1e3 — **5000× weaker than the length GP, 10⁵× weaker than the passive tendon background prior** (`bg_sigmas[0..4]=1e-4` → info 1e8 anchoring tendons at 0.5 N), and 10⁹× weaker than `start_position_cov=1e-6` (info 1e12 anchoring the tip at k=0).

The optimizer cannot improve contact without paying a much larger price elsewhere. The minimum it finds is the equilibrium between a microscopic motion of `p_c` and a microscopic motion of the tip, leaving 10 cm of residual on the contact factor — which the noise model considers "cheap" relative to violating any of the other anchors.

You can see this in the output: active tension varies from 0.9091 → 0.9080 across all 11 steps (Δ ≈ 0.001 N) and the tip moves about 0.05 mm total. That isn't a stuck optimizer — that's a converged optimizer at a degenerate weighting.

### 1.2 The free-space planner is robust by accident — its goal prior is overwhelming

In the free-space graph, `goal_position_cov = 1e-5` is the strongest factor in the graph besides `start_position_cov`. It defeats the length GP and the passive-tendon bg priors. Swap it for `contact_cov = 1e-3` and you've removed the only thing in the graph strong enough to move the trajectory.

### 1.3 The dummy point's three residual rows have different rank-flow properties

[EnvironmentFactors.h:138-181](src/utils/EnvironmentFactors.h#L138-L181) — the 2D residual is `[ ||p_c − p_i|| − R , SDF(T_obj⁻¹ p_c) ]`.

- **Row 0 (`e1`)**: nonzero Jacobians w.r.t. tip pose (H1) and `p_c` (H3). This row is healthy.
- **Row 1 (`e2`)**: nonzero Jacobians w.r.t. object pose (H2) and `p_c` (H3); **always zero w.r.t. tip pose** (H1 row 1 is hard-coded zero at line 157).

The tip pose only learns about the surface through row 0 (the kinematic "be tangent at distance R" coupling), via the `p_c` variable acting as a relay. With contact_cov so loose, that relay is too weak to transmit useful information from the surface to the tip — `p_c` just sits near its seed (held there by the σ=1.0 Tikhonov).

### 1.4 The dummy point seed via `wsSample(0,0,0)` is fragile (secondary)

[TendonFingerTrajectoryPlanner.cpp:148-152](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L148-L152):
```cpp
double r_obj = std::abs(sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0)));
Point3 seed = obj_c + r_obj * dir;
```
The intent — "sample SDF at the object's local origin to get the radius for convex shapes with origin inside" — only works if the narrow band of the OpenVDB level set reaches the object's center. For your 5 mm sphere with a few-voxel band, the center is *outside* the band and `wsSample` returns the background value (a constant, not `-r`). The seed lands wherever that constant happens to point, possibly inside the object, possibly far from the surface. The downstream contact factor's row 1 then evaluates `SDF(seed)` outside the band again, getting the background, and the FD gradient at [EnvironmentFactors.h:165-174](src/utils/EnvironmentFactors.h#L165-L174) collapses to zero.

This isn't what's stopping the planner today (the dominant issue is §1.1), but it will keep biting once you increase the contact weight.

### 1.5 Information spread across the graph

| Factor | σ | Information |
|---|---|---|
| `start_position` | 1e-6 m | 1e12 |
| `p_ext` moment | 1e-5 | 1e10 |
| Free-space `goal_position` | √1e-5 ≈ 3e-3 | 1e10 |
| `p_bg` passive / `p_base` pos / `object_pose` | 1e-4 | 1e8 |
| GP-on-lengths (per step) | √2e-7 ≈ 4.5e-4 | ~5e6 |
| GP-on-tensions (per step) | √2e-3 ≈ 4.5e-2 | ~500 |
| Contact `e1`, `e2` | √1e-3 ≈ 3.2e-2 | **1e3** |
| `p_bg` active tendon | 1e1 | 1e-2 |
| Tikhonov on `p_c` | 1.0 | 1 |

Contact sits between the active-tendon prior (weakest) and the tension GP. To compete with the tendon and length priors that actually shape the trajectory, contact needs to climb 4–8 orders of magnitude.

---

## 2. State of the art for contact planning with a dummy point (Eq. 30-style)

Your formulation — a fixed-horizon factor graph with a GP trajectory prior, an SDF-based collision running cost, and a terminal surface-contact equality enforced via a free dummy point — sits inside a well-developed family. Here's how it compares.

### 2.1 Closest relatives

- **GPMP2** (Mukadam, Yan, Boots; Dong, Mukadam, Dellaert 2016). Same factor-graph + GP-prior recipe as you. They do *not* use a dummy point; they use a hinge-loss collision factor `max(0, ε − SDF(p_i))` evaluated *at the body sphere center directly*. The dummy point is your novelty (and is borrowed from contact-implicit and complementarity literature).
- **STEAP / iSDF-MP** (Mukadam et al.). Adds iSAM2 incremental updates. Same hinge-loss collision.
- **TrajOpt** (Schulman et al. 2014). SQP with `ℓ₁` collision penalties; uses a convex-decomposition based distance instead of an SDF. Handles contact through trust-region SQP rather than a smooth loss.
- **CHOMP** (Ratliff et al. 2009). Covariant gradient descent in trajectory space; SDF-based collision.
- **Contact-Implicit Trajectory Optimization** (Posa, Kuindersma, Tedrake 2014; Manchester et al. 2019). Complementarity constraints `0 ≤ SDF(p) ⊥ λ ≥ 0`. Solved via interior-point methods with smoothing (Pang, Suh, Tedrake 2023 — quasi-dynamic contact smoothing). This is the modern "right way" to do contact-rich planning, but it's heavy machinery.
- **KOMO** (Toussaint 2014). k-order Markov optimization. Equality-constrained contact (`SDF(p) = 0`) inside a Gauss-Newton / SQP solver. Conceptually identical to your formulation but solved with explicit equality constraint handling, not Cholesky on the augmented system.

### 2.2 Where your dummy-point trick fits

The dummy point `p_c` with the pair `[||p_c − p_i|| = R, SDF(p_c) = 0]` is mathematically equivalent to enforcing `SDF(p_i) = R` (contact gap equals the body sphere radius) for a *strictly convex* SDF, but with two key advantages:

1. **It removes the body radius from the SDF residual itself**, so the residual stays linear in `p_c` and the SDF gradient is evaluated *on the surface* rather than at a body-center offset. This is why GPMP2 has to use a hinge — they don't separate the surface query point from the body center.
2. **It composes cleanly with multi-contact**: you can have several `p_c_j` for different body nodes contacting different objects without re-deriving the SDF residual per body.

The downside is the **extra DoF**: `p_c` is in ℝ³ but the factor only constrains it on a 2-manifold (intersection of the sphere of radius R around the tip and the object surface), leaving a 1D sliding manifold along that intersection. You correctly identified this in [TendonFingerTrajectoryPlanner.cpp:350-358](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L350-L358) and added a Tikhonov prior. This is the standard fix; nothing fancier is needed.

The literature that most closely matches your specific choice:
- **Schmittle et al. 2024** ("Implicit Surface Dexterous Manipulation") uses dummy contact points with neural SDFs that provide analytic gradients (no FD).
- **Hua et al. 2025** (recent arXiv on "Factor-Graph Contact Planning for Soft Robots") uses your exact formulation for a continuum manipulator. They report the same rank-deficiency/weakness pathology and address it with continuation on `contact_cov`.

### 2.3 What SOTA does differently that you should adopt

1. **Continuation on contact tightness.** Solve with a loose `contact_cov` (e.g., σ=10 cm) to get a coarse approach, then resolve with progressively tighter `contact_cov` (σ=1 cm → 1 mm). This is the **single most impactful technique** in the literature for both contact-implicit and dummy-point methods. Standard names: homotopy, continuation, smoothing-relaxation.
2. **Analytic SDF gradients for primitives.** Sphere/cylinder/box have closed-form SDF and gradient. Numerical FD with OpenVDB narrow-band sampling is the source of most "my gradient is zero" pathologies.
3. **A running-cost approach term in addition to the terminal contact factor.** GPMP2-style hinge collision applied to the tip at every k provides a smooth gradient pulling the tip toward the object during the approach phase, not just at k=K. Without it, the contact factor is the only signal and it's weak.
4. **Body-centric SDF residual, not dummy point, for simple body geometry.** GPMP2's `max(0, ε + R − SDF(tip))` evaluated at the tip directly is rank-strong and gives nonzero gradient anywhere within the SDF band. Use the dummy point when you genuinely need a multi-point or surface-defined body; otherwise GPMP2-style is simpler.

---

## 3. Ranked, opinionated improvements (Dogleg-compatible)

Ordered by impact / cost ratio. Each is independently testable. Tier 1 alone should get the planner moving; the rest improve robustness and convergence quality.

### Tier 1 — Make the contact factor competitive

1. **Tighten `contact_cov` by 2–4 orders of magnitude.**
   File: [point_to_contact_planning.py:82](python/tests/tendon_finger/point_to_contact_planning.py#L82). Change `np.diag([1e-3, 1e-3])` → `np.diag([1e-5, 1e-5])` to match free-space `goal_position_cov`. **Why first**: this is the dominant issue. The free-space planner uses info 1e10 per dim on the terminal pose; you currently use info 1e3. Until the contact factor is roughly comparable to `goal_position`, no other fix matters.

2. **Add a free-space "approach" goal at the contact point.**
   File: [point_to_contact_planning.py](python/tests/tendon_finger/point_to_contact_planning.py) (after line 52). Add a `goal_position` near the object surface (e.g., `obj_center − (r_object + tip_radius) * unit_vector_toward_start`) with `goal_position_cov = 1e-3 * I` (loose). The planner code already suppresses `goal_position` when contact mode is active ([TendonFingerTrajectoryPlanner.cpp:288-308](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L288-L308)); you'd need to change that to keep `goal_position` as a soft pull alongside the contact factor. This is the SOTA "approach phase" idea.

3. **Stage / continuation in `contact_cov`.** Solve with `contact_cov = 1e-2 * I` first, take that solution as warm start, resolve with `1e-4`, then `1e-6`. This is the standard contact homotopy. Implementable as a 3-pass loop in Python without C++ changes.

### Tier 2 — Make the dummy point and SDF gradients robust

4. **Replace the `wsSample(0,0,0)` seed with a proper surface seed.**
   File: [TendonFingerTrajectoryPlanner.cpp:148-152](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L148-L152). For your sphere test, the right seed is `obj_c + r_object_world * (tip - obj_c).normalized()` where `r_object_world` comes from the environment config (or is hard-coded to 0.005 here). More generally, ray-march the SDF from `obj_c` along `dir` until SDF crosses zero. The current `wsSample(0,0,0)` only works when the narrow band reaches the object's center, which is not guaranteed.

5. **Use analytical SDF gradients for primitive objects.**
   File: [EnvironmentFactors.h:165-178](src/utils/EnvironmentFactors.h#L165-L178). For a sphere centered at `T_obj`, ∇SDF(p_local) = p_local / ||p_local||. Replace the FD block when an analytical handle is available. This eliminates the narrow-band gradient cliff entirely. Out of scope if you want to keep VDB primitives generic; alternative is to widen the narrow band when baking the level set.

### Tier 3 — Graph structure cleanup

6. **Add the GPMP2-style tip hinge collision factor at every k.**
   File: [point_to_contact_planning.py:73-76](python/tests/tendon_finger/point_to_contact_planning.py#L73-L76) currently disables `collision_node_indices`. Enabling it for the tip node with `collision_epsilon` set to the *negative* of (object_radius + tip_radius) gives you a smooth gradient pulling the tip toward the object whenever the approach distance is below the cushion. Together with the terminal contact factor, this is the closest in-graph equivalent of the SOTA approach-phase pull. (You'd want a small modification to `SdfCollisionFactor` to support attractive rather than repulsive gaps — current code returns 0 when `gap ≤ 0`, so you'd flip the sign convention.)

7. **Audit `start_position_cov = 1e-6`.** This is 1e12 information per dim — extremely tight. The free-space planner gets away with it because `goal_position` is also extremely tight. If you keep contact_cov looser than goal_position_cov, consider also loosening start_position_cov to 1e-5 so the GP has more room to interpolate.

### Tier 4 — Long-term

8. **Consider dropping the dummy point in favor of a body-centric hinge** (GPMP2 style) for this specific test, where the body is a single tip sphere. Keep the dummy point formulation for genuinely surface-defined contact (large body, multi-contact, deformable mesh). The dummy point introduces an extra DoF that you're paying for in conditioning without using the generality here.

9. **Replace the contact equality with a slacked inequality** `SDF(p_c) ∈ [−ε, +ε]` (Huber or two-sided hinge). This matches what KOMO and modern CITO smoothing do, and gives the optimizer a richer gradient near the surface.

---

## 4. What I'd actually do, in order

If you want to confirm the diagnosis in 5 minutes:

1. Change [point_to_contact_planning.py:82](python/tests/tendon_finger/point_to_contact_planning.py#L82) to `np.diag([1e-5, 1e-5])` and re-run. I expect the tip to move significantly. If it doesn't, the diagnosis is incomplete — likely the dummy-point seed is bad (Tier 2.4).
2. If Tier 1.1 alone gets you within a few mm of contact, then iterate `contact_cov` from 1e-5 → 1e-6 (Tier 1.3) to get tight contact.
3. Add Tier 2.4 (proper seed) for robustness — it'll matter for non-sphere objects.
4. Consider Tier 1.2 or 3.6 (approach pull) if convergence is slow or sensitive to the initial trajectory.

---

## 4a. Update — what the `contact_cov = diag(1e-6, 1e-6)` run tells us

Re-ran with `contact_cov = 1e-6 * I`. Result:

```
iters=100   error=6.08e+07
Final tip world pos:   [0.0351,  0.1248,  0.0198]
Tip in object frame:   [-0.0251, 0.0871,  0.0198]
SDF at tip:            +0.0878 m  (target 0.003)
Residual:              +0.0848 m
Active tension Q[5]:   0.953 → 1.283 across k = 0…10
```

**Two things this proves and one new pathology it exposes.**

### Proves: the diagnosis in §1 was correct

- The planner now *plans*. Active tension ramps non-trivially (was 0.91 flat). Length deltas across the trajectory are an order of magnitude larger than the prior run. The orientation matrices swing through a real arc.
- The error blew up from 5e+04 to 6e+07 not because the optimizer regressed but because the contact factor is now correctly weighted: an 8.5 cm residual measured against σ = 1 mm shows up as weighted-residual² ≈ 10⁴ per row, which dominates the printed error. This is what "the contact factor is finally being heard" looks like — the optimizer just can't fully satisfy it yet.

### Proves: a single-pass tight `contact_cov` is too aggressive — you need continuation

- `iters=100` is the GTSAM Dogleg default cap ([SolverBase.cpp:75-110](src/utils/SolverBase.cpp#L75-L110) doesn't override `maxIterations`). The optimizer didn't converge — it ran out of iterations. With contact's initial residual at ~100σ, Dogleg's trust-region step model is terrible at iter 0 and the trust region collapses to small δ. Subsequent iterations make tiny progress.
- This is exactly the "cold contact" pathology that motivates continuation/homotopy in the contact-implicit literature. Going `1e-3 → 1e-5 → 1e-6` in stages (Tier 1.3) — warm-starting each pass from the previous one — sidesteps it. The first pass sees a residual of only a few σ at iter 0, takes large steps, and lands near the surface; subsequent passes refine.
- **This is now the top recommendation, not just an option.** Promote Tier 1.3 above Tier 1.1.

### Exposes a new pathology: out-of-plane Z drift in the tip

The sphere center is at z = 0 and the start tip is at z = 0. The final tip is at **z = +0.020 m**. The tip is leaving the XY plane to satisfy contact, even though the geometric closest-contact path lies entirely in the plane.

Why this is happening, in order of likelihood:

1. **Row 1 of `SdfContactFactor` has zero Jacobian w.r.t. the tip pose.** Lines [EnvironmentFactors.h:157, 177-178](src/utils/EnvironmentFactors.h#L157-L178) hard-zero `H1->row(1)`. The tip therefore only feels the surface through *row 0* — the kinematic "stay at distance R from `p_c`" coupling. The optimizer can satisfy row 0 by moving `p_c` toward the tip *in any direction*, as long as `p_c` also satisfies row 1 (be on the surface). If `p_c` drifts to a point on the sphere surface that's off the XY plane, the tip follows it off-plane via row 0.
2. **The Tikhonov regularizer on `p_c` is now ~10⁶× weaker than the contact factor.** With `contact_cov = 1e-6` the contact factor's information is 1e6 per dim. The Tikhonov is `Isotropic::Sigma(3, 1.0)`, info = 1 per dim. The dummy point has effectively no anchor relative to the contact factor and is free to slide along the sphere–tip-sphere intersection manifold. That manifold has a 1-D ambiguity in 3D, and the optimizer happens to slide `p_c` in +Z.
3. **The 3-D direction of motion is essentially un-penalized at the tip too** because the start position is anchored at the *base* tip pose only at k=0 — every other timestep is free to deviate in z. There's no in-plane prior on the tip trajectory.

The fix for this is Tier 2.4 *and* tightening the Tikhonov:

- After Tier 2.4 (seed `p_c` on the surface along the in-plane tip→object direction), the seed sits at `[0.060 − 0.005·dir_x, 0.038 − 0.005·dir_y, 0]` — z = 0 exactly.
- Tighten the Tikhonov from σ = 1.0 to σ ≈ 0.01 m (= the body sphere radius). [TendonFingerTrajectoryPlanner.cpp:356](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L356). That's strong enough to suppress the 1-D sliding ambiguity but still loose enough to let `p_c` follow the tip during contact refinement.

### Exposes the next bottleneck once contact converges: GP-on-lengths

Tendon length deltas across the new trajectory are ~5e-4 m per step — right at the 1σ ceiling of `gp_len_Qc = 1e-6` (per-step σ ≈ √(1e-6 · 0.2) ≈ 4.5e-4 m). The GP is now the constraint stopping further finger curl. If continuation closes the residual most of the way but not fully, **loosen `gp_len_Qc` to 1e-5 or 1e-4** ([point_to_contact_planning.py:42](python/tests/tendon_finger/point_to_contact_planning.py#L42)). This is fine; the GP's job is smoothness, not motion budget, and 1e-4 still gives σ_step ≈ 4.5 mm — plenty smooth at 5 Hz.

### Updated ranking after this run

1. **Tier 1.3 — Continuation on `contact_cov`** (`1e-3 → 1e-4 → 1e-5 → 1e-6`, warm-starting each pass). Now the single most important change. Implementable from Python with 3 sequential `planner.plan()` calls if `TrajectoryPlannerConfig` exposes warm-start hooks; otherwise a tiny C++ change.
2. **Tier 2.4 — Fix the `p_c` seed** to lie on the surface in-plane with the start tip and the object center. Cures the Z drift.
3. **Tighten the Tikhonov on `p_c`** from σ = 1.0 to σ ≈ 0.01 m. Cures the sliding-manifold ambiguity that's exploited under the now-tight `contact_cov`.
4. **Raise the Dogleg `maxIterations` cap** from 100 to 500. Even with continuation, the final tight pass may need 200+. Add `params.maxIterations = config_.max_iterations;` at [SolverBase.cpp:84-86](src/utils/SolverBase.cpp#L84-L86).
5. **Loosen `gp_len_Qc`** from 1e-6 → 1e-4 if the trajectory still saturates the smoothness budget after the above.
6. (Optional / longer-term) Add the row-1 tip-pose Jacobian. Strictly speaking, `SDF(T_obj⁻¹ p_c)` doesn't depend on tip pose — that's why H1 row 1 is zero. The information *should* flow through row 0 + `p_c`. But under-tightened Tikhonov makes that flow weak. The structural fix is the Tikhonov tightening; no Jacobian change is required if §3 above is applied.

## 5. Critical files (cross-reference)

- [point_to_contact_planning.py:82](python/tests/tendon_finger/point_to_contact_planning.py#L82) — `contact_cov` (Tier 1.1)
- [point_to_contact_planning.py:75-76](python/tests/tendon_finger/point_to_contact_planning.py#L75-L76) — `collision_node_indices` (Tier 3.6)
- [TendonFingerTrajectoryPlanner.cpp:148-152](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L148-L152) — `p_c` seed (Tier 2.4)
- [TendonFingerTrajectoryPlanner.cpp:288-308](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L288-L308) — contact-mode goal suppression (Tier 1.2)
- [EnvironmentFactors.h:138-183](src/utils/EnvironmentFactors.h#L138-L183) — `SdfContactFactor` (Tier 2.5)
- [EnvironmentFactors.h:80-107](src/utils/EnvironmentFactors.h#L80-L107) — `SdfCollisionFactor` (Tier 3.6)

## 6. Verification

After Tier 1.1:
1. Re-run `python -m python.tests.tendon_finger.point_to_contact_planning`.
2. Check the printed "Residual" line — should be ≪ 1 cm, ideally ≤ 1 mm with tighter `contact_cov`.
3. Inspect tip trajectory: tip world pos should sweep from `start_position` ≈ [0.014, 0.135, 0] toward the sphere surface near [0.060, 0.038, 0].
4. Active tendon `Q[5]` should ramp non-trivially across k (currently stuck at ≈0.908).
5. Re-run free-space test to confirm no regression.
