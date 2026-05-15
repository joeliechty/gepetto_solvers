# Diagnosing & Fixing `IndeterminantLinearSystemException` in `point_to_contact_planning.py`

## Context

You are extending the working free-space PAI planner (Eq. 22) to a contact-as-goal planner (Eq. 30) for a single tendon-driven finger. The free-space version converges; the contact version throws:

```
RuntimeError: Indeterminant linear system detected while working near
variable 5836665117072163816 (Symbol: Q1000).
```

`Q` is the symbol used for tendon tensions in this codebase, so GTSAM is telling you that **a tendon tension variable at one of the timesteps has no rank in the Hessian** — the linearized system has a null space along that direction. This is *not* generic numerical ill-conditioning (very wide eigenvalue spread) — it is a true rank deficiency. Cholesky cannot factor a rank-deficient SPD matrix and aborts.

This document gives:
1. The mechanical diagnosis grounded in your code (file:line),
2. A survey of state-of-the-art techniques that target each failure mode,
3. A ranked, opinionated list of fixes to try, easiest first.

---

## 1. Diagnosis — Why Q1000 Is Rank-Deficient

The root cause is a **broken information chain between active tendon tension and any factor that pins it down at intermediate timesteps**, made worse by an under-constrained terminal state. Three layered issues:

### 1.1 Active-tendon background prior is essentially zero information

[point_to_contact_planning.py:34](python/tests/tendon_finger/point_to_contact_planning.py#L34)
```python
bg_sigmas = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e1])
```

For passive tendons (indices 0–4), σ=1e-4 → information 1e8 — they are nailed to 0.5 N.
For the active tendon (index 5), σ=1e1 → information **1e-2** — effectively unanchored.

The free-space planner used σ=1e6 for the same slot (10× looser) but the system was rescued by the **goal_position** factor, which transmitted a strong constraint back through kinematics to Q[5]. In contact mode, that goal prior is *suppressed* by [TendonFingerTrajectoryPlanner.cpp:279-283](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L279-L283) and replaced by the much weaker contact factor (see §1.3).

### 1.2 The temporal GP on tensions is commented out

[TendonFingerTrajectoryPlanner.cpp:198-206](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L198-L206)
```cpp
// 5.1 GP on tensions (commented out - using GP on lengths instead)
// if (k < K) {
//     auto gp_noise = noiseModel::Gaussian::Covariance(Qc * config_.dt);
//     graph_.add(BetweenFactor<Eigen::Vector<double, N>>(
//         models_[k]->get_tensions_key(), ...
```

So `Q[k=1]` has **no direct factor coupling it to `Q[k=0]` or `Q[k=2]`**. The eq. 11 prior in your paper is missing from the actual graph. The only nominal anchors on Q[k=1, t=5] are:

- `p_bg` (info 1e-2 — negligible)
- `p_lim` (cubic barrier, identically zero when Q ≥ 0)
- `f_kin` `TendonDiscWrenchFactor` (couples Q to W and T, but W is itself a variable)

That last factor is an *equation* `W_j = Σ Q_t · SpatialWrench(v_t, T_j) - D_j`. Given any `Q[5]`, the optimizer can absorb the change into the disc wrenches W or the disc poses T. Whether that absorption costs anything depends on what *else* pins W and T at k=1.

### 1.3 The terminal contact factor only constrains 1 DoF of the tip pose

[EnvironmentFactors.h:138-163](src/utils/EnvironmentFactors.h#L138-L163)

`SdfContactFactor` returns the 2D residual `[ ||p_c − p_i|| − r , SDF(T_obj⁻¹ p_c) ]`. The first component constrains the *radial distance* between the tip center and the dummy point (1 DoF). The second component constrains the dummy point to lie on the SDF surface — but its Jacobian rows for the *finger pose* are all zero (lines 157–163 only fill `H1.row(0)`).

So the contact factor effectively pins **1 DoF of T[K, tip]** out of 6. The other 5 DoFs of the tip pose at k=K — and *all* DoFs of intermediate poses — are constrained only by kinematics + length GP + base prior. Backward propagation from contact through the chain is too weak to give Q[5] at k=1 a unique value.

### 1.4 The contact factor's e2 likely has zero gradient at initialization

[TendonFingerTrajectoryPlanner.cpp:138-141](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L138-L141) seeds `p_c = tip + 0.003 · (obj_center − tip)/‖obj_center − tip‖`. That places p_c **3 mm from the tip**, not on the object surface. If the tip starts ~5 cm from the sphere, p_c starts ~5 cm from the sphere too.

OpenVDB level sets store SDF values only inside a narrow band; outside that band, `wsSample` returns the constant background value, so the finite-difference gradient at [EnvironmentFactors.h:166-174](src/utils/EnvironmentFactors.h#L166-L174) is **zero**. The Jacobian normalization `if (norm > 1e-8) grad /= norm;` saves you from dividing by zero, but the resulting Jacobian rows for H2 and H3 are zero. So at iteration 0 the SDF row of the contact factor adds a large *residual* with *no gradient* — it inflates `error()` without contributing rank to the Hessian.

### 1.5 Mismatched information scales across the graph (secondary)

| Factor | σ | Information |
|---|---|---|
| `p_ext` moment | 1e-5 | 1e10 |
| `p_ext` force / `p_bg` passive / `p_base` pos / `object_pose` | 1e-4 | 1e8 |
| `p_base` rot | 1e-3 | 1e6 |
| `p_con` per-component | 1e-3 | 1e6 |
| `f_run` collision | 1e-3 | 1e6 |
| GP on lengths | √2e-6 ≈ 1.4e-3 | ~5e5 |
| `p_bg` active tendon | 1e1 | 1e-2 |
| `p_reg` dummy point | 1.0 | 1 |
| `p_lim` | 1.0 | 1 |

That is a **12-order-of-magnitude spread** in information. Even after the rank fix, this will hurt convergence and may need to be addressed via preconditioning or rescaling.

---

## 2. State-of-the-Art Survey

Three families of techniques are relevant, ordered by how directly they address rank deficiency vs. ill-conditioning vs. nonconvexity of contact.

### 2.1 Rank deficiency / underconstrained variables

- **Tikhonov damping (Levenberg–Marquardt)** — the textbook fix. LM solves `(JᵀJ + λI) Δ = Jᵀr` instead of `JᵀJ Δ = Jᵀr`. The added `λI` makes the system full-rank regardless of Jacobian rank. Dogleg (your current optimizer) does *not* damp — it requires a positive-definite Hessian and is brittle here.
- **Weak isotropic priors on every variable** (regularizers) — the equivalent of LM damping done at the factor level. You already do this for the dummy point ([TendonFingerTrajectoryPlanner.cpp:343-346](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L343-L346)). The fix is to do it for the *active tendon at every k*.
- **Variable elimination ordering** — rank deficiency manifests as a specific variable failing during elimination. GTSAM allows manual ordering (`COLAMD`, custom). Putting the rank-deficient variable last sometimes lets earlier eliminations regularize it. This is a workaround, not a fix.
- **Augmented-Lagrangian / penalty on equality constraints** — convert hard contact equality into a sequence of progressively tighter penalty solves (continuation), with LM damping inside.

### 2.2 Generic ill-conditioning (after rank is restored)

- **Block-Jacobi or SUBGRAPH preconditioned conjugate gradient (PCG)** — GTSAM exposes this via `LevenbergMarquardtParams::iterativeParams = ConjugateGradientParameters{...}`. Recommended when the factor information spans many orders of magnitude.
- **Variable rescaling** — rescale variables so that the diagonal of `JᵀJ` is ~1. For tendon tensions in N vs. lengths in m vs. poses, this matters. CHOMP and TrajOpt both use a metric-aware step.
- **QR linear solver instead of Cholesky** — `MULTIFRONTAL_QR` is slower but more numerically stable for ill-conditioned (but full-rank) systems. Will *not* save you when the system is rank-deficient.

### 2.3 Contact-implicit trajectory optimization (the field your paper sits in)

- **GPMP2** — Mukadam, Yan, Boots (2016). Closest to your formulation: factor-graph trajopt with a GP prior on the trajectory and an SDF-based hinge-loss collision factor. Differences from your code: they use a **piecewise-linear hinge** (not cubic), they use **interpolation factors** between sparse support states (sparser graph), and they use **iSAM2** for incremental updates. Their hinge loss has a discontinuous gradient but is rank-preserving because every support state is independently constrained.
- **TrajOpt** — Schulman et al. (2014). Sequential Convex Optimization with `ℓ₁` penalties on collision and a trust region in *configuration* space. Handles non-smooth collision robustly via trust region rather than via barrier smoothness.
- **CHOMP** — Ratliff et al. (2009). Covariant gradient descent in trajectory space using a smoothness metric as preconditioner. Worth knowing for the metric idea even if you don't switch.
- **Contact-Implicit Trajectory Optimization (CITO)** — Posa, Kuindersma, Tedrake (2014); Manchester, Doshi, et al. (2019). Models contact via complementarity constraints (LCP) and relaxes them. Modern variants smooth the complementarity (Pang, Suh, Tedrake, 2023 — *"Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasi-dynamic Contact Models"*). Probably too heavy a hammer for tangential surface contact, but the **smoothing-as-continuation** idea is directly applicable.
- **Implicit-surface dexterous manipulation** — Schmittle et al. (2024) and follow-ups use neural SDFs with analytic gradients (no FD), which sidesteps the narrow-band gradient issue entirely.
- **Homotopy / continuation methods for trajopt** — Toussaint's K-Order Markov Optimization, the classic numerical-continuation literature (Allgower & Georg). Also implemented in MoveIt's STOMP and in modern diffusion-style trajectory samplers as "noise scheduling."

### 2.4 SDF gradient quality (specific to your finite-difference SDF)

- **Analytic SDF gradients** for primitives (sphere, cylinder, box) — for the test scene this is trivial and removes a major source of noise.
- **Continuous SDF representations** — TSDFs, neural SDFs, or `nanoflann`-based query of triangle meshes — all give cleaner gradients than narrow-band VDB outside the band.
- **Background-aware sampling** — extend the narrow band, or detect when the sample falls outside the band and substitute a safe fallback (e.g. distance to the AABB of the band).

---

## 3. Ranked Fixes to Try

Ordered by **expected impact / cost ratio**. Try them roughly in order; each is independently testable. The first three are very likely to fix the immediate `IndeterminantLinearSystemException`; the rest improve robustness and convergence.

### Tier 1 — Direct fixes for the rank deficiency

1. **Switch optimizer Dogleg → LevenbergMarquardt with non-trivial initial λ.**
   File: [src/utils/SolverBase.cpp:84-99](src/utils/SolverBase.cpp#L84-L99). Replace `DoglegOptimizer` / `DoglegParams` with `LevenbergMarquardtOptimizer` / `LevenbergMarquardtParams`, set `params.lambdaInitial = 1e-3` (or 1e-1 for safety) and `params.lambdaUpperBound = 1e10`. **Why first:** this *immediately* makes the linear system full-rank (the `λI` term) and lets you see whether the optimizer can otherwise make progress. Lowest-risk, highest-information experiment.

2. **Enable the GP-on-tensions BetweenFactor** that's currently commented out.
   File: [src/tendon_finger/TendonFingerTrajectoryPlanner.cpp:198-206](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L198-L206). Uncomment and verify `config_.gp_tense_Qc` is set in the Python config (it is, [point_to_contact_planning.py:38](python/tests/tendon_finger/point_to_contact_planning.py#L38)). **Why second:** this is the missing eq. 11 from your paper. It directly provides temporal information to Q[5] at intermediate timesteps and matches the math you wrote.

3. **Tighten the active-tendon background prior** (or set goal_tensions).
   File: [point_to_contact_planning.py:34](python/tests/tendon_finger/point_to_contact_planning.py#L34). Change `bg_sigmas[5]` from `1e1` to something like `1e0` or `5e-1`, OR set `planner_config.goal_tensions` to a reasonable contact tension (e.g., 1.0 N). **Why third:** even with the GP enabled, if the *baseline* for the active tendon is at info 1e-2 the chain is still weak. The GP propagates information; some non-trivial information must enter somewhere.

### Tier 2 — Prevent the contact factor from being dead at init

4. **Re-seed the dummy point on the object surface, not 3 mm from the tip.**
   File: [src/tendon_finger/TendonFingerTrajectoryPlanner.cpp:130-141](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L130-L141). Change `seed = tip + r·dir` to `seed = obj_center − r_object·dir` (or query the SDF for the closest surface point along the line). **Why:** this puts p_c inside the OpenVDB narrow band, so the SDF row of the contact factor has a nonzero gradient from iteration 0 instead of contributing a large residual with no Jacobian.

5. **Replace the FD SDF gradient with the analytical gradient for known primitives**, or at least with the OpenVDB built-in `tools::Gradient` operator on the level set.
   Files: [src/utils/EnvironmentFactors.h:88-102](src/utils/EnvironmentFactors.h#L88-L102) and [165-178](src/utils/EnvironmentFactors.h#L165-L178). For the sphere test, `∇SDF(p) = (p − center)/‖p − center‖`. **Why:** removes a source of numerical noise and avoids the suspect `if (norm > 1e-8) grad /= norm` normalization which can produce a unit vector in a meaningless direction when the FD norm is tiny.

### Tier 3 — Continuation / homotopy (works *after* Tier 1 fixes)

6. **Stage the planner: free-space first, then add contact.**
   - Pass 1: solve eq. 22 with `goal_position = obj_center − r_object·n̂` (a guess at the contact point). This warm-starts the trajectory near the object.
   - Pass 2: re-solve eq. 30 with the previous solution as the initial values, contact factor enabled, but with `contact_cov` initially loose (e.g., `diag(1e-2, 1e-2)`).
   - Pass 3: re-solve with `contact_cov` tight (`diag(1e-6, 1e-6)`).
   This is the standard continuation trick for contact and dramatically reduces the "cold start" problem. Numerically equivalent to an outer Levenberg-Marquardt loop.

7. **Increase K from 2.** With only 3 timesteps the solver has very little freedom to "swing" toward contact. K=8–15 is typical for finger-reach motions and gives the optimizer interpolation room.

### Tier 4 — Conditioning improvements (after the system runs)

8. **Switch linear solver to PCG with Block-Jacobi preconditioner** if convergence is slow after the rank fix. Set `params.linearSolverType = LevenbergMarquardtParams::Iterative;` with `ConjugateGradientParameters`.

9. **Audit and rescale extreme noise models.** The 12-order-of-magnitude information spread (§1.5) will haunt you eventually. Consider σ_ext_wrench at 1e-2 instead of 1e-5 (you're not actually measuring wrenches with that precision), and review whether `object_pose_cov = 1e-8` is needed (could be 1e-4).

### Tier 5 — Long-term, paper-level changes

10. **Replace cubic barrier with hinge-loss collision (à la GPMP2)** and rely on LM damping for smoothness. Cubic-barrier C² claims hold mathematically, but in practice the second derivative `6(ε−φ)` is small enough near the boundary that finite-precision Cholesky still struggles.
11. **Add a 6-DoF goal-pose factor with a soft orientation prior** (e.g., σ_rot = 0.5 rad) so the contact factor isn't the *only* terminal pose constraint. Solves §1.3 directly.

---

## 4. Verification

After applying Tier-1 fixes, verify in this order:

1. **Run** `python -m python.tests.tendon_finger.point_to_contact_planning`. The `IndeterminantLinearSystemException` should be gone.
2. **Check `result.meta.iterations` and `result.meta.error`.** Expect non-zero iterations and finite error. If iterations = 0, LM rejected the first step — increase `lambdaInitial`.
3. **Check the printed "Contact check" residual** at [point_to_contact_planning.py:107](python/tests/tendon_finger/point_to_contact_planning.py#L107). It should be near zero (< 1 mm) if contact was achieved.
4. **Compare the Q[5] trajectory** printed at [line 122](python/tests/tendon_finger/point_to_contact_planning.py#L122) across timesteps to the free-space planner's output. They should be smoothly varying (because of the GP) and within physical limits (≥ 0).
5. **Re-run the existing free-space test** to confirm none of the changes break the working case.

## 5. Critical files to modify (cross-reference)

- [point_to_contact_planning.py:34](python/tests/tendon_finger/point_to_contact_planning.py#L34) — `bg_sigmas[5]` (Tier 1.3)
- [point_to_contact_planning.py:29](python/tests/tendon_finger/point_to_contact_planning.py#L29) — `K` (Tier 3.7)
- [src/utils/SolverBase.cpp:84-99](src/utils/SolverBase.cpp#L84-L99) — Dogleg → LM (Tier 1.1)
- [src/tendon_finger/TendonFingerTrajectoryPlanner.cpp:198-206](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L198-L206) — enable GP on tensions (Tier 1.2)
- [src/tendon_finger/TendonFingerTrajectoryPlanner.cpp:130-141](src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L130-L141) — re-seed dummy point (Tier 2.4)
- [src/utils/EnvironmentFactors.h:88-102, 165-178](src/utils/EnvironmentFactors.h#L88-L102) — SDF gradient (Tier 2.5)
