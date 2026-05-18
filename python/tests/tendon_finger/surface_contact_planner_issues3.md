# Why the finger penetrates the sphere at the end of the trajectory

## Context

The contact planner in [point_to_contact_planning.py](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py) is the latest evolution of a `SdfContactFactor`-based formulation. Two prior diagnostic docs in the same folder ([surface_contact_planner_issues.md](crest-sparse/python/tests/tendon_finger/surface_contact_planner_issues.md), [surface_contact_planner_issues_2.md](crest-sparse/python/tests/tendon_finger/surface_contact_planner_issues_2.md)) walked through earlier pathologies — rank deficiency, then a weighting problem where the planner converged but the tip never moved. Those fixes landed: GP-on-tensions enabled, ray-march `p_c` seed in object-local frame, Tikhonov tightened, continuation on `contact_cov` ([1e-3 → 1e-4 → 1e-5 → 1e-6]) implemented in the planning loop. The tip now moves toward the sphere.

**What the user observes today (different bug than before):** the trajectory looks correct early — the tip approaches the sphere and lands tangent. Towards the end of the trajectory, the tip slides along the surface and then crosses into the sphere interior. The screenshot shows clear interior penetration at the final timestep across multiple views.

**Why this is a new failure mode, not a recurrence:** the prior docs explained why the planner *didn't move*. Once those were fixed, the contact factor became strong enough to actually deform the rod into contact. But the contact factor as written has a geometric blind spot that only surfaces once it has the authority to drag the tip around — which is exactly where we are now.

## Mechanism — three compounding causes

### Cause 1: `SdfContactFactor` has no side-awareness (the main cause)

The 2D residual at [EnvironmentFactors.h:117-118, 181](crest-sparse/src/utils/EnvironmentFactors.h#L117-L181) is

```
e1 = ||p_c - p_tip|| - R
e2 = SDF(T_obj⁻¹ p_c)
```

Both are equality residuals with **no sign preference**. A solution where `p_c` sits on the object surface and the tip is **inside** the sphere at depth ≤ R satisfies `e1 = e2 = 0` exactly:

- `e2 = 0` is symmetric across the surface — the dummy point is on the surface from either side.
- `e1 = 0` only constrains the tip to lie on a sphere of radius R around `p_c`. Half of that sphere is interior to the object.

Concretely, for `r_obj = 0.025 m` and `R = 0.003 m` (per [point_to_contact_planning.py:81,117](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L81-L117) — note the inline comment at L58 says 0.005 but the verification uses 0.025, so the actual SDF is the 25 mm sphere), the tip can sit anywhere in the spherical shell `0.022 ≤ ||tip − obj_center|| ≤ 0.028` and still satisfy both residuals. **That admits 3 mm of penetration as a feasible minimum.** Once `contact_cov` is loose enough that residuals can be a few σ off, the admissible interior depth grows correspondingly.

### Cause 2: The collision running cost is disabled

[point_to_contact_planning.py:73-76](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L73-L76):

```python
# env.collision_node_indices = disc_node_indices
# env.collision_node_radii = [0.003] * len(disc_node_indices)
env.collision_node_indices = []
env.collision_node_radii = []
```

`SdfCollisionFactor` ([EnvironmentFactors.h:46-108](crest-sparse/src/utils/EnvironmentFactors.h#L46-L108)) is the only thing in the graph that would generate a repulsive gradient *anywhere along the trajectory* via `(ε − Φ_D)³` when a body sphere enters the safety margin. With the indices set to `[]`, nothing in the graph penalises penetration at any of the K+1 timesteps — not the tip, not the disc nodes, not even at the terminal step. The terminal contact factor is the only contact signal, and per Cause 1 it tolerates interior solutions.

### Cause 3: Sliding amplifies the asymmetry under tightening

At `contact_cov = 1e-6` (final continuation stage) the contact factor's information is 1e6 per dim, while the Tikhonov on `p_c` is `σ = 1e-2` → info 1e4 ([TendonFingerTrajectoryPlanner.cpp:388](crest-sparse/src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L388)). The Tikhonov is 100× weaker than the contact factor it's meant to regularize. The 1D sliding manifold (intersection of "sphere of radius R around tip" with "object surface") is therefore essentially free for `p_c` to slide along. As the rod curls under increasing flexor tension, the GP-on-lengths and background tension priors prefer the tip to keep moving along the established curl arc; once that arc grazes the sphere, the cheapest way to satisfy the contact factor is to let `p_c` slide tangentially and `p_tip` drift inward — both residuals stay near zero, and the rest of the graph is happier than if the tip were dragged radially outward against the curl direction. Hence the visible pattern: tangent contact early, sliding mid-trajectory, interior penetration at the end.

This matches the screenshot exactly — the finger starts curling toward the sphere, makes contact tangentially, slides as it continues to curl, and the terminal pose ends up with the tip a few mm inside.

## State of the art in object-contact planning (oriented to this codebase)

The current formulation belongs to the dummy-point-equality-contact family — most closely Schmittle 2024 and the recent factor-graph continuum-robot contact-planning literature. Compared to the surrounding SOTA:

| Family | Representative work | Contact representation | Side-awareness |
|---|---|---|---|
| **Hinge-loss SDF** (this codebase's `SdfCollisionFactor` style) | CHOMP (Ratliff 2009), GPMP2 (Mukadam/Dong 2016), STEAP | `max(0, ε − SDF(p))` evaluated at body center | **Yes** — penalises only the inside half. |
| **Equality dummy-point** (this codebase's `SdfContactFactor` style) | Schmittle 2024, Hua 2025, KOMO (Toussaint 2014) | `SDF(p_c) = 0` + `‖p_c − p‖ = R` | **No** — symmetric on the surface. |
| **Complementarity / CITO** | Posa, Cantu, Tedrake 2014; Manchester 2019; Pang/Suh/Tedrake 2023 ("quasi-dynamic contact smoothing") | `0 ≤ SDF(p) ⊥ λ ≥ 0`, force × gap = 0 | **Yes** — both `SDF ≥ 0` and complementarity are signed. |
| **Smoothed LCP / interior-point** | Howell & Manchester 2022, DIRTREL, TrajOptPlus | Smoothed `SDF(p) ≥ 0` with relaxed slack | **Yes**. |
| **Differentiable simulation** | Brax, MuJoCo MPC, DiffSim, Dojo (Howell et al. 2022) | Implicit time-stepping LCP, gradient through contact | **Yes** — sim physics excludes interior. |
| **Phase-aware sequence-of-modes** | TOWR (Winkler 2018), Mordatch/Todorov, Mode-graph SQP | Contact mode chosen per phase, sign-constrained per phase | **Yes**. |

The pattern across the modern literature is that contact representations are either *signed by construction* (CITO/LCP/diff-sim/hinge — penetration is impossible by sign) or *signed by an explicit non-penetration constraint added alongside the equality* (KOMO/SQP — equality `SDF=0` plus inequality `SDF ≥ 0` for non-contact bodies, sign of the contact normal supplied externally). The current `SdfContactFactor` is the only common formulation that's surface-equality with **no companion non-penetration constraint** — which is exactly the gap that shows up in the screenshot.

The closest fully-SOTA refactor for this codebase would be a smoothed complementarity contact (`λ`-augmented graph, with `SDF · λ = 0` enforced as a soft factor and `SDF ≥ 0` as a hinge). The closest minimally-invasive fix is to add the missing sign as either a hinge-collision running cost or as an extra row in the contact factor.

## Fix options — quick to SOTA, pick one or stack them

### Option A — Tactical, one-line: re-enable collision running cost
**Scope:** Python only. **Effort:** 5 minutes. **Coverage:** removes Cause 2; partially covers Cause 1 (collision factor catches interior solutions because it's a *one-sided* hinge).

Uncomment [point_to_contact_planning.py:73-74](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L73-L74) so `collision_node_indices = disc_node_indices` and `collision_node_radii = [0.003] * len`. Bump `env.collision_sigma` to `1e-4` so its information (1e8) sits between the contact factor (1e6 final stage) and the start-position anchor (1e12). Optionally also apply collision to the tip itself by adding `tip_node_index` to the list with radius 0.003.

**Tradeoff:** The collision factor's `(ε − Φ_D)³` cushion is only active for `Φ_D ≤ ε`. With `ε = 0.002`, intermediate-step penetrations of more than ~5 mm into the sphere fall back into the "always-zero" branch (because Φ_D = SDF − R is far negative, gap is large, but in [EnvironmentFactors.h:80-83](crest-sparse/src/utils/EnvironmentFactors.h#L80-L83) the branch is `gap ≤ 0` returns zero — i.e. far penetration produces a large repulsion, that's fine). It also doesn't fire *outside* the narrow band of the OpenVDB grid, so initial trajectories that don't graze the band don't get the gradient. For a small sphere with narrow band that's a real concern; widening the band when baking the SDF would help.

### Option B — Tactical: add a non-penetration row to `SdfContactFactor`
**Scope:** C++ (header only). **Effort:** ~15 lines. **Coverage:** removes Cause 1 directly.

Promote the contact factor from 2D residual to 3D by adding a third row that's a hinge on the *tip's* signed distance:

```
e3 = max(0, R − SDF(T_obj⁻¹ p_tip))    // positive iff tip is inside the surface offset by R
```

This is the signed analog of `e2`, evaluated at the tip itself rather than at the dummy point. Jacobian w.r.t. tip pose is the SDF gradient (already implemented in `SdfCollisionFactor`), so it's a copy-paste. Noise model becomes `Isotropic::Sigma(3, ...)` with row 3 weighted similarly to or slightly tighter than rows 1–2.

**Why this is the right minimal fix:** it directly closes the geometric symmetry. With `e3` active, the tip cannot satisfy `e1 = e2 = 0` from the interior side because `e3` rises linearly inside. Doesn't require enabling the global collision cost; doesn't touch the rest of the graph.

**Tradeoff:** mixes hinge (e3) and equality (e1, e2) into one factor — slightly unusual for GTSAM-style factor graphs but mathematically clean. Alternative: keep the contact factor 2D and add a *separate* `SdfNonPenetrationFactor` between the tip pose and the object pose at k=K (and optionally at k=K−1, K−2 to catch the slide).

### Option C — Tighten the Tikhonov + apply contact factor over a window
**Scope:** C++ (1 line for σ, ~20 lines to add window). **Effort:** 30 minutes. **Coverage:** Cause 3; partial coverage of Cause 1 by sheer weight.

Two sub-changes at [TendonFingerTrajectoryPlanner.cpp:369-391](crest-sparse/src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L369-L391):

1. Drop the Tikhonov σ from `1e-2` to `1e-3` or even `1e-4` so the dummy point cannot slide freely once `contact_cov` tightens. Currently the Tikhonov is 100–10000× weaker than `contact_cov` across the continuation; flipping that ratio kills the sliding mode.
2. Apply `SdfContactFactor` not just at `k = K` but at `k ∈ {K−2, K−1, K}` (or all `k` for which the rod is expected to be in contact). Use per-step dummy points `p_c_k` so each terminal-region step is constrained individually. This converts the "tangent at one instant" formulation into "tangent for a contact phase," which is the TOWR-style phase formulation and removes the late-trajectory sliding pathology.

**Tradeoff:** still doesn't fix the side-awareness — penetration at *all* contact-phase steps remains feasible. Pair with A or B for the actual cure.

### Option D — SOTA refactor: smoothed complementarity contact
**Scope:** C++ (new factor class) + planner orchestration. **Effort:** 1–2 days. **Coverage:** Causes 1+2+3 at once; aligns with Posa/Manchester CITO + Pang/Suh smoothing.

Introduce a per-step contact normal force `λ_k ∈ ℝ⁺` as a graph variable for each candidate contact body × object pair. Add three factors per active `(k, body, object)`:

- **Non-penetration hinge:** `r₁ = max(0, R − SDF(T_obj⁻¹ p_body_k))` with tight σ. (Same as Option B's e3, applied throughout the contact phase.)
- **Complementarity (smoothed):** `r₂ = λ_k · (SDF(T_obj⁻¹ p_body_k) − R)` — the gap-force product, driven toward zero with smoothing parameter κ that anneals across continuation passes alongside `contact_cov`. This is the Pang/Suh "quasi-dynamic smoothing" idea.
- **Friction cone (optional, frictionless if skipped):** `r₃ = max(0, ||λ_t|| − μ λ_n)` if you want tangential force balance.

Replace the dummy-point factor with this triad. The dummy point is no longer needed because non-penetration + complementarity together place the body on the surface with the correct sign. This is the formulation used by Drake's contact-implicit MPC and matches Howell & Manchester 2022.

**Tradeoff:** larger code change; introduces force variables which the rest of the rod model doesn't currently use for contact reactions (would need to wire `λ` into the rod's external wrench at the contact node, which is actually a desirable side effect — the rod will deform correctly under the contact reaction). The continuation loop now sweeps two parameters (`contact_cov` and the complementarity smoothing κ).

### Recommended bundle (the user said "mix of quick fixes + SOTA suggestions")

**Phase 1 (this week):** Option A + Option B together. A re-enables the safety net; B closes the geometric symmetry. Combined, they fix the penetration without any new variables.

**Phase 2 (next week):** Option C if any sliding remains visible. The Tikhonov tightening is a single-line change and the contact-window expansion turns "tangent at one instant" into "tangent for a phase," which also makes the rendered animation look qualitatively better.

**Phase 3 (refactor pass, when planning multi-contact / pushing / lifting):** Option D. The current dummy-point formulation will not scale gracefully to multi-contact (each new contact pair adds an extra DoF with weak regularisation). The complementarity formulation scales naturally and gives the rod model a real contact-reaction wrench.

## Critical files

- [point_to_contact_planning.py:73-76](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L73-L76) — re-enable collision running cost (Option A)
- [point_to_contact_planning.py:72](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L72) — tighten `collision_sigma` from 1e-3 to 1e-4 (Option A)
- [point_to_contact_planning.py:117](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L117) — fix the inline-comment/code disagreement on sphere radius (cosmetic but confusing)
- [EnvironmentFactors.h:120-183](crest-sparse/src/utils/EnvironmentFactors.h#L120-L183) — add 3rd row `e3` to `SdfContactFactor` (Option B). Reuse the FD-gradient block from `SdfCollisionFactor` for `H1` row 3.
- [TendonFingerTrajectoryPlanner.cpp:388](crest-sparse/src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L388) — Tikhonov σ on `p_c` (Option C, sub-change 1)
- [TendonFingerTrajectoryPlanner.cpp:369-391](crest-sparse/src/tendon_finger/TendonFingerTrajectoryPlanner.cpp#L369-L391) — extend contact factor to a k-window (Option C, sub-change 2). Will need a per-step `dummy_point_key(k)` if going this route.
- [EnvironmentFactors.h:46-108](crest-sparse/src/utils/EnvironmentFactors.h#L46-L108) — the existing `SdfCollisionFactor` whose FD-gradient block (lines 88-104) can be copy-pasted into the new `e3` row.

## Reusable patterns already in the codebase

- The hinge / one-sided collision logic is already implemented in `SdfCollisionFactor::evaluateError` ([EnvironmentFactors.h:60-107](crest-sparse/src/utils/EnvironmentFactors.h#L60-L107)) — Option B literally lifts its `gap = ε − Φ_D` branch.
- The FD SDF gradient block ([EnvironmentFactors.h:88-104](crest-sparse/src/utils/EnvironmentFactors.h#L88-L104) and [EnvironmentFactors.h:165-178](crest-sparse/src/utils/EnvironmentFactors.h#L165-L178)) is identical between the two factors; if doing Option D, factor it into a shared helper to avoid triplicating it.
- The continuation pattern ([point_to_contact_planning.py:98-103](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L98-L103)) is reusable as-is for Option D's smoothing parameter κ — just call `planner.set_contact_smoothing(κ)` in the same loop.
- `dummy_point_key()` will need a `(k)` variant if going to per-step contacts (Option C sub-2 or Option D). Look at how `object_key(k)` is templated for the precedent.

## Verification

1. Re-run `python -m python.tests.tendon_finger.point_to_contact_planning` after any of the options.
2. Check the printed `SDF at tip` and `Residual` lines at [point_to_contact_planning.py:117-122](crest-sparse/python/tests/tendon_finger/point_to_contact_planning.py#L117-L122):
   - Residual should be near zero (sub-mm under tight `contact_cov`).
   - `SDF at tip` should be ≈ `tip_radius` (i.e. 0.003), and **strictly positive** (i.e. tip-center outside the sphere by exactly the tip radius).
3. Add a per-step penetration check after the trajectory is extracted — for each `k`, compute `tip_in_obj_k = obj_pose_inv @ tip_world_k`, then `sdf_k = ||tip_in_obj_k|| − 0.025`. Assert `sdf_k ≥ −1e-4` for all `k` (small slack for floating-point). Print the worst (most negative) `sdf_k`.
4. Visually: re-run the four-pane PyVista viewer. The tip-sphere should sit *outside* the orange sphere at every k. The "slide-then-pierce" pattern in the screenshot should be gone.
5. No regression check: re-run `point_to_point_planning.py` to confirm free-space planning still converges with the same configs (only `EnvironmentFactors.h` changes if doing Option B, which doesn't affect free-space).
6. (If doing Option D) verify the contact reaction force `λ_K` is non-zero at terminal contact and zero at non-contact steps — confirms the complementarity is actually selecting the contact mode.
