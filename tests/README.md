# Test suite

Two tiers, split by the `slow` marker:

```bash
pytest -m "not slow"    # pure functions, no solver          -- under a second
pytest -m slow          # golden solves + viz smoke          -- ~6 minutes
pytest                  # both
```

Run from the repo root. `conftest.py` puts the repo root on `sys.path`, so this works
before the package is renamed and installed, and stays inert afterwards.

## Layout

| Path | Covers |
|---|---|
| `core/test_geometry.py` | `scene.py` — analytic SDFs, witnesses, proxy ellipsoids, extents |
| `core/test_hand.py` | the tendon hand's morphology — bone/joint spec, digit set, disc indexing, pinch table |
| `core/test_hand_interface.py` | the `Hand` seam, driven by a STUB hand that is not the built-in one |
| `core/test_solver_helpers.py` | `solvers.py` — pose conventions, residual readouts, opposition sign, `capabilities()` |
| `test_golden_solves.py` | FK / IK / planner forward passes (`slow`) |
| `projects/test_viz_smoke.py` | the visualizer's five `--smoke` routines (`slow`) |

`_pkg.py` is the single place the suite names the application layer's import path, so a
package move changes that one file and no test. Its `config` is the five-digit tendon
hand's morphology package — one hand among (eventually) several, so a test asserting a
measured number there is asserting something about THAT hand.

## Two constraints worth knowing before adding a test

**No baked `.vdb` SDF grids.** They live in `_objects/`, total 54 MB, are gitignored,
and regenerating them needs conda-only `pyopenvdb` — so a test built on one is
unrunnable on a fresh checkout. Solve tests use the analytic ellipsoid primitives
(`coin`, `credit_card`, `pen`, `*_sphere_ellipsoid`, `megaminx`), which carry their
geometry inline. `core/test_geometry.py::test_analytic_primitives_need_no_vdb_grid`
guards this.

SDF-path coverage lives in `scripts/capture_baseline.py` instead, which runs against a
developer's working tree where the grids exist.

**Pin the hand.** `load_hand_dimensions()` prefers `gepetto_core`'s CAD-derived
geometry and silently falls back to the bundled `DEFAULT_HAND_DIMENSIONS`. The two are
*not* the same hand — the middle finger's first joint diameter is 9.8 mm from the CAD
and 14.0 mm in the fallback — so any test asserting a number must take the
`pinned_hand` fixture and pass it to the solver (`HandFKSolver(params, pinned_hand)`).
Otherwise the test passes or fails depending on whether `gepetto_core` happens to be
installed. `pinned_dims` is the dimension table itself, for tests that assert on it
directly.

**A test of the SOLVERS should not use the built-in hand.** `core/test_hand.py` covers
the tendon hand's own geometry, and every one of its assertions would still pass if the
solvers had `FINGER_NAMES` and the flexor index baked in — because the built-in hand
agrees with those constants. `core/test_hand_interface.py` is the counterweight: it
registers a two-digit stub with four actuators and no opposing digit, and asserts what
the solver stack does with it. Anything claiming the hand is pluggable belongs there.

## What the golden numbers are

They characterize current behavior; they are not targets. Notably the IK scenario does
**not** close its grasp — it stalls at a ~10 mm worst gap, the documented open problem
in `notes_5f_contact.md`. That is still a strong regression check: repeat solves are
bit-identical, and run-to-run float nondeterminism is ~1e-14 m, so the committed
`rtol=1e-6` has a wide margin and any real change to the graph moves the numbers.

If a change moves them deliberately, update the constants in the same commit and say
why in the message.
