# SDF / witness-point tip-contact: `IndeterminantLinearSystem` findings

Diagnosis of why the 3-residual witness-point contact solve
(`sdf_3dof_contact_kinematics_test.py`, and its analytic twin
`sphere_3dof_contact_kinematics_test.py`) throws
`gtsam::IndeterminantLinearSystemException` near the object pose **`O0`**, and
why a parameter sweep cannot fix it.

---

## TL;DR

- The failure is a **structural gauge freedom of the dummy witness point `p_c`**,
  not a tuning/conditioning bug.
- Confirmed by isolating the variable: the **1-residual** analytic contact
  (`SphereSphereContactFactor`, no dummy point) converges perfectly, while
  **both** 3-residual witness forms (analytic `SphereSphereWitnessContactFactor`
  *and* SDF `SdfContactFactor`) fail identically near `O0`. So the SDF is not
  the culprit — the dummy point is.
- A 36-combo parameter sweep over the AL/LM knobs and the object-pose prior
  found no robust fix. The single "success" found is illusory (AL stalled 4 cm
  short of contact). `cylinder.vdb` fails at every setting.
- **Robust fix is structural**: drop the dummy point and use a 1-residual SDF
  tip-contact factor `e = SDF(T_obj⁻¹·c_tip) − R` — the SDF analog of the
  1-DoF sphere factor that already works. (Not yet implemented.)

---

## Factor parity check (the original ask)

`SdfContactFactor` *was* verified to be a faithful counterpart of
`SphereSphereWitnessContactFactor` (both in
`crest-sparse/src/utils/EnvironmentFactors.h`):

| Row | `SphereSphereWitnessContactFactor` | `SdfContactFactor` | Match |
|-----|------------------------------------|--------------------|-------|
| e1  | `‖p_c − c_a‖ − r_a`                | `‖p_c − c_i‖ − R`  | ✅ identical (finger-sphere tangency) |
| e2  | `‖p_c − c_b‖ − r_b`                | `SDF(T_obj⁻¹ p_c)` | ✅ SDF value *is* the signed distance — generalizes the sphere gap |
| e3  | `1 + n_a·n_b`                      | `1 + n_i·n_obj_world` | ✅ same antiparallel-at-contact convention (outward normals) |

Jacobians:

- **e1 (row 0):** identical (`H1.row(0) = −n^T·D_center_pose`, `H3.row(0) = n^T`).
- **e2 (row 1):** sphere uses analytic `n_b`; SDF uses normalized FD gradient
  `n_obj_local` (both unit, consistent for a true SDF with ‖∇‖=1). SDF couples
  through the object pose via `D_plocal_obj`, the analog of the sphere's
  `D_cb_pose`.
- **e3 (row 2):** the **one principled difference**. The sphere factor carries
  both normals' dependence on `p_c` (`dn_a/dp_c = P_a`, `dn_b/dp_c = P_b`). The
  SDF factor uses a **locally-constant-gradient approximation** — treats
  `n_obj` as `p_c`-independent (no cheap closed-form derivative of a sampled SDF
  gradient), keeping only the `n_i` term plus the object-rotation term
  `−R_obj·skew(n_obj_local)`. Same convention `SdfCollisionFactor` already uses.

**Conclusion:** the factor math is consistent. The convergence failure is *not*
a factor-parity bug.

---

## Root cause: 1-DoF gauge freedom of the witness point

On a symmetric object the witness point `p_c` is under-determined:

1. `e1 = ‖p_c − c_tip‖ − R = 0` puts `p_c` on a sphere of radius `R` about the
   tip center.
2. `e2` (surface) puts `p_c` on the object surface.
3. The intersection of those two surfaces is generically a **circle**.
4. `e3 = 1 + n_tip·n_obj` is **invariant** under rotating `p_c` about the
   tip-center → object-center axis — both normals rotate together, so their dot
   product is unchanged.

⟹ The three residuals pin only **2 of `p_c`'s 3 DoF**. The remaining tangential
rotation is a free gauge. The 1 m stabilizing prior on `p_c` cannot fix it: the
AL penalty weight grows like `√μ` toward ~1e6, so a prior loose enough not to
bias the contact at low μ is swamped in the gauge direction at high μ
(→ `IndeterminantLinearSystem`), while a prior tight enough to survive high μ
overpowers the contact and stalls the finger short of the surface.

Compounding it for a **sphere object**: a sphere's SDF world-normal is
**rotation-independent** (`n_obj_world = normalize(p_world − c_obj)`), so the
contact factor gives the object pose `O0`'s 3 rotation DoF **zero information** —
they ride on the pose prior alone. During elimination the combined null space
(p_c gauge + O0 rotation symmetry) surfaces at `O0`, hence the exception names
`O0`.

This is exactly the gauge the C++ comment at
`crest-sparse/src/tendon_finger/TendonFingerSolver.cpp` (~L195–209) already
describes for the sphere-sphere witness case. The comment assumed the SDF path
was immune ("a general surface normal is unique"); a **sphere** SDF reintroduces
the gauge, and a **cylinder** has its own axial symmetry.

---

## Evidence

### Baselines (isolates the dummy point as the culprit)

| Test | Factor | Dummy pt? | Result |
|------|--------|-----------|--------|
| `sphere_1dof_contact_kinematics_test.py` | `SphereSphereContactFactor` (1 residual) | no  | ✅ gap `+0.00000 m`, err 0, iters 31, tendon flexes in (T5 3.0→3.33) |
| `sphere_3dof_contact_kinematics_test.py` | `SphereSphereWitnessContactFactor` (3 residual) | yes | ❌ `IndeterminantLinearSystem` near `O0` |
| `sdf_3dof_contact_kinematics_test.py`    | `SdfContactFactor` (3 residual)          | yes | ❌ `IndeterminantLinearSystem` near `O0` |

The analytic 3-residual form fails the same way as the SDF form ⟹ the SDF is not
the problem; the witness/dummy-point formulation is.

### Parameter sweep (`sdf_3dof_contact_param_search.py`, sphere.vdb)

36 combos over:

| Knob | Values |
|------|--------|
| `al_initial_mu`       | 1, 10, 100 |
| `al_mu_increase_rate` | 2 |
| `al_max_iterations`   | 40 |
| `max_iterations`      | 100 |
| `object_pose_cov`     | 1e-8, 1e-6 |
| `lambda_initial`      | 1e-5, 1e-2, 1 |
| `diagonal_damping`    | False, True |

Outcome:

- **Only** `mu0=100, object_pose_cov=1e-8, diagonal_damping=False` returned
  without throwing — and that "success" is **illusory**: signed gap **+0.041 m
  (4 cm short)**, err 89, iters 12. The AL stalled; the contact residuals were
  never driven to zero. (`lambda_initial` made no difference there.)
- `object_pose_cov=1e-6` (looser object prior) → **always fails** (confirms O0's
  rotation is held only by the prior).
- `diagonal_damping=True` → **always fails** (it scales LM damping by the 1e8
  prior diagonal, distorting the step).
- `al_initial_mu` ∈ {1, 10} → **always fails** regardless of other knobs.

### Cylinder (`cylinder.vdb`)

`object_pose_cov=1e-8, diagonal_damping=False`, `mu0` ∈ {1, 10, 100}:
**fails at every `mu0`** (`IndeterminantLinearSystem`). A cylinder's axial
symmetry leaves its own gauge; it is not a workaround.

**Verdict:** parameter tuning cannot fix this. There is no μ window that is both
well-posed (high enough μ to be full-rank) and able to flex the finger into the
surface (low enough μ not to freeze the start configuration).

---

## Fix options (none implemented yet)

### Option A — 1-residual SDF tip-contact factor (recommended)

Mirror the 1-DoF factor that already works. Drop the dummy point entirely:

```
e = SDF(T_obj⁻¹ · c_tip) − R          // scalar residual
```

where `c_tip` is the tip-node translation and `R` the tip radius. This places
the tip *center* at signed distance `R` from the surface (tip sphere tangent to
the surface), is **gauge-free**, fully determined by tip pose + object pose, and
the SDF gradient supplies the contact normal implicitly. It is essentially
`SdfCollisionFactor`'s `phi = SDF − radius` without the one-sided hinge.

- **Pros:** gauge-free; should converge like `sphere_1dof` (gap → 0); ~30 lines
  (new factor + solver branch in `TendonFingerSolver.cpp` + pybind + rebuild).
- **Cons:** pure surface-equality `SDF=0` has **no side-awareness** (a tip
  *inside* the object also satisfies it). For a kinematics test starting outside
  this is fine; in the planner a collision hinge supplies the missing sign.
  See `surface_contact_planner_issues3.md` in this folder.

### Option B — gauge-fix the witness factor

Keep the 3-residual `SdfContactFactor` / dummy point but add a 4th gauge-fixing
residual (or a μ-coupled prior) that pins `p_c`'s tangential rotation about the
center line. More complex; preserves the explicit witness-point/normal-alignment
formulation (useful if you need the contact point as an output or want a path
toward general non-symmetric surfaces with their own multi-contact structure).

### Not viable

- Looser/tighter fixed-sigma `p_c` prior (the `√μ` argument above).
- LM damping / `lambda_initial` / `diagonal_damping`.
- AL `mu`/rate schedule (illusory stall only).
- Switching the object mesh (sphere → cylinder makes it worse).

---

## Repro

```bash
# fails near O0:
python -m python.tests.tendon_finger.sdf_3dof_contact_kinematics_test
python -m python.tests.tendon_finger.sphere_3dof_contact_kinematics_test

# works (reference, gap → 0):
python -m python.tests.tendon_finger.sphere_1dof_contact_kinematics_test

# the sweep:
python -m python.tests.tendon_finger.sdf_3dof_contact_param_search
```

## Relevant source

- `crest-sparse/src/utils/EnvironmentFactors.h` — `SdfContactFactor`,
  `SphereSphereWitnessContactFactor`, `SphereSphereContactFactor`,
  `SdfCollisionFactor`.
- `crest-sparse/src/tendon_finger/TendonFingerSolver.cpp` — contact-graph build
  (~L164–256: sphere/SDF branches), dummy-point seeding (~L269–333), gauge
  comment (~L195–209).
- `crest-sparse/src/utils/SolverBase.cpp` — AL optimize + marginals
  (~L90–177); `SolverBase.h:41–62` — LM/AL knob docs.
