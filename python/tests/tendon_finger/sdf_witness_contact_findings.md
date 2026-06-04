# SDF / witness-point tip-contact: `IndeterminantLinearSystem` findings

Diagnosis of why the witness-point contact solve
(`sdf_3dof_contact_kinematics_test.py`, and its analytic twin
`sphere_3dof_contact_kinematics_test.py`) threw
`gtsam::IndeterminantLinearSystemException`, and what changed after implementing
the C-frame gauge-fixing residuals (Section 3, Eq. 30–31).

---

## Update 2026-06-04 — implemented Option B (C-frame gauge fixing)

We implemented the gauge-fixing residuals proposed below as "Option B". The two
witness factors now emit **5 residuals** `[c_R, c_O, c_N, c_T1, c_T2]` instead of
3, where the two new terms pin the previously-free tangential gauge of the
witness point `p_c`.

### What changed in the code

- **`src/utils/EnvironmentFactors.h`**
  - Added `frisvad_tangent_basis(n, t1, t2)` — an inline, allocation-free
    Frisvad/Hughes–Möller Householder basis that maps `+Z` onto the unit normal
    `n` and returns the two orthonormal tangent vectors spanning its tangent
    plane (single south-pole singularity handled explicitly).
  - Both witness factors gained the two C-frame residuals
    `c_T1 = (p_c − p_i)·t1`, `c_T2 = (p_c − p_i)·t2`, with `t1, t2` built from the
    **contacted object's** surface normal (the SDF gradient normal for the SDF
    factor; body B's outward normal `n_b` for the sphere factor). `t1, t2` are
    treated as constant within the local Gauss–Newton step (C-frame held
    fixed — standard SOTA contact convention), so the new Jacobian rows are just
    the tangent vectors: `H_pc.row(3..4) = t1ᵀ, t2ᵀ` and
    `H_finger.row(3..4) = −t1ᵀ·D_center, −t2ᵀ·D_center`.
  - **Renamed** for a consistent `<ContactedGeometry>WitnessContactFactor`
    scheme (object geometry only, not the robot-side geometry):
    - `SdfContactFactor` → **`SdfWitnessContactFactor`**
    - `SphereSphereWitnessContactFactor` → **`SphereWitnessContactFactor`**
    - `SphereSphereContactFactor` (1-residual analytic) and `SdfCollisionFactor`
      keep their names.
- **`src/tendon_finger/TendonFingerSolver.cpp` / `TendonFingerTrajectoryPlanner.cpp`**
  - Construction sites updated to the new class names.
  - Constrained noise-model dimension bumped `Isotropic::Sigma(3,1.0)` →
    `Sigma(5,1.0)` at all three witness construction sites.
  - **Removed** the weak `Sigma(3,1.0)` `PriorFactor<Point3>` dummy-point anchors
    from both solver witness branches. They were a failed attempt to stabilize
    the gauge the new residuals now fix; the trajectory planner already ran its
    SDF witness contact without one.
- **`python/tests/tendon_finger/sdf_3dof_contact_kinematics_test.py`** — swapped
  the contacted object from `cylinder.vdb` to the wide-band `sphere.vdb`.

### Results

| Test | Factor | Result |
|------|--------|--------|
| `sphere_3dof_contact_kinematics_test.py` | `SphereWitnessContactFactor` (5 residual) | ✅ **converges** — signed gap `+0.00000 m`, iters 32, tendon 5 flexes in (3.0 → 3.331) |
| `sdf_3dof_contact_kinematics_test.py`    | `SdfWitnessContactFactor` (5 residual)    | ❌ still `IndeterminantLinearSystem`, now near **`Q0`** (was `O0`) |

**The gauge fix works.** The analytic sphere-sphere witness case — which
previously failed identically to the SDF case — now converges cleanly to a zero
gap. This confirms the root-cause diagnosis below for the *analytic* path: the
1-DoF tangential gauge of `p_c` was the problem, and the C-frame residuals
remove it.

**The SDF path still fails, but differently**, and the new evidence rules out the
earlier suspects:

- **Not the object geometry.** Fails identically with both `sphere.vdb` (6 cm
  band, r = 0.025) and `cylinder.vdb` (3 cm band, r = 0.005). So it is not the
  narrow band or the cylinder's axial symmetry.
- **Not the dummy-point prior.** Re-adding the removed `Sigma(3,1.0)` prior on
  `p_c` does **not** help — fails identically with and without it. So removing
  the prior was not a regression, and the residual problem is not about `p_c`
  being absent from a base cost factor.
- **Not the `p_c` gauge.** The same gauge fix that cured the analytic witness
  case is present and active in the SDF factor.
- **New failure locus is `Q0` (a tendon tension), not `O0` (object pose).** The
  exception now surfaces on the tension/rod side of the elimination clique, not
  the contact/object side, pointing at the SDF-factor ↔ rod coupling rather than
  the witness point.

This is the "I'll think more about it" territory flagged at the outset: the
requested math is implemented and validated (sphere passes), and the SDF case has
a remaining, distinct indeterminacy to investigate next. Leading hypothesis: the
SDF factor's **locally-constant-gradient approximation** (it treats `n_obj` as
`p_c`-independent and supplies no SDF-curvature term) leaves the tip-pose ↔
tension coupling rank-deficient at the operating point in a way the analytic
factor's exact normals do not. Worth checking next: the actual rank/condition of
the linearized contact block, and whether the FD SDF gradient is degenerate at
the iterates.

---

## Root cause (analytic path — now confirmed & fixed)

On a symmetric object the witness point `p_c` was under-determined by the
original 3 residuals:

1. `c_R = ‖p_c − c_tip‖ − R = 0` puts `p_c` on a sphere of radius `R` about the
   tip center.
2. `c_O` (surface) puts `p_c` on the object surface.
3. The intersection of those two surfaces is generically a **circle**.
4. `c_N = 1 + n_tip·n_obj` is **invariant** under rotating `p_c` about the
   tip-center → object-center axis — both normals rotate together, so their dot
   product is unchanged.

⟹ The three residuals pinned only **2 of `p_c`'s 3 DoF**. The remaining
tangential rotation was a free gauge that no fixed-sigma prior could stabilize
(the AL penalty grows like `√μ` toward ~1e6: a prior loose enough not to bias the
contact at low μ is swamped in the gauge direction at high μ → indeterminate,
while a prior tight enough to survive high μ overpowers the contact and stalls
short of the surface).

The C-frame residuals `c_T1, c_T2` pin that last DoF by forcing `(p_c − p_i)` to
lie along the contact normal (zero projection onto the object's tangent plane),
giving `p_c` a full-rank gradient. The sphere-sphere result above confirms this.

---

## Factor parity check (the original ask)

`SdfWitnessContactFactor` was verified to be a faithful counterpart of
`SphereWitnessContactFactor` (both in `src/utils/EnvironmentFactors.h`):

| Row | `SphereWitnessContactFactor` | `SdfWitnessContactFactor` | Match |
|-----|------------------------------|---------------------------|-------|
| c_R | `‖p_c − c_a‖ − r_a`          | `‖p_c − c_i‖ − R`         | ✅ identical (finger-sphere tangency) |
| c_O | `‖p_c − c_b‖ − r_b`          | `SDF(T_obj⁻¹ p_c)`        | ✅ SDF value *is* the signed distance — generalizes the sphere gap |
| c_N | `1 + n_a·n_b`                | `1 + n_i·n_obj_world`     | ✅ same antiparallel-at-contact convention (outward normals) |
| c_T1| `(p_c − c_a)·t1(n_b)`        | `(p_c − c_i)·t1(n_obj)`   | ✅ new — C-frame gauge fix |
| c_T2| `(p_c − c_a)·t2(n_b)`        | `(p_c − c_i)·t2(n_obj)`   | ✅ new — C-frame gauge fix |

The **one principled difference** remains row `c_N`: the sphere factor carries
both normals' dependence on `p_c` (`dn_a/dp_c = P_a`, `dn_b/dp_c = P_b`), while
the SDF factor uses a **locally-constant-gradient approximation** — it treats
`n_obj` as `p_c`-independent (no cheap closed-form derivative of a sampled SDF
gradient), keeping only the `n_i` term plus the object-rotation term
`−R_obj·skew(n_obj_local)`. Same convention `SdfCollisionFactor` already uses.
This is the leading suspect for the remaining SDF-only failure.

---

## Repro

```bash
# now converges (gap → 0):
python -m python.tests.tendon_finger.sphere_3dof_contact_kinematics_test

# still fails near Q0 (sphere.vdb and cylinder.vdb both):
python -m python.tests.tendon_finger.sdf_3dof_contact_kinematics_test

# 1-DoF analytic reference (gauge-free, always converged):
python -m python.tests.tendon_finger.sphere_1dof_contact_kinematics_test
```

## Relevant source

- `src/utils/EnvironmentFactors.h` — `frisvad_tangent_basis`,
  `SdfWitnessContactFactor`, `SphereWitnessContactFactor`,
  `SphereSphereContactFactor`, `SdfCollisionFactor`.
- `src/tendon_finger/TendonFingerSolver.cpp` — contact-graph build (sphere/SDF
  branches), dummy-point seeding.
- `src/tendon_finger/TendonFingerTrajectoryPlanner.cpp` — terminal contact mode.
- `src/utils/SolverBase.cpp` — AL optimize + marginals; `SolverBase.h` — LM/AL
  knob docs.
