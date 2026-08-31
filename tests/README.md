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
| `core/test_hand.py` | `config.py` — bone/joint spec, digit set, disc indexing, pinch table |
| `core/test_solver_helpers.py` | `solvers.py` — pose conventions, residual readouts, opposition sign, `capabilities()` |
| `test_golden_solves.py` | FK / IK / planner forward passes (`slow`) |
| `projects/test_viz_smoke.py` | the visualizer's five `--smoke` routines (`slow`) |

`_pkg.py` is the single place the suite names the application layer's import path.
When the package is renamed and the three-tier move happens, that one file changes and
no test does.

## Two constraints worth knowing before adding a test

**No baked `.vdb` SDF grids.** They live in `_objects/`, total 54 MB, are gitignored,
and regenerating them needs conda-only `pyopenvdb` — so a test built on one is
unrunnable on a fresh checkout. Solve tests use the analytic ellipsoid primitives
(`coin`, `credit_card`, `pen`, `*_sphere_ellipsoid`, `megaminx`), which carry their
geometry inline. `core/test_geometry.py::test_analytic_primitives_need_no_vdb_grid`
guards this.

SDF-path coverage lives in `scripts/capture_baseline.py` instead, which runs against a
developer's working tree where the grids exist.

**Pin the hand dimensions.** `config.load_hand_dimensions()` prefers `gepetto_core`'s
CAD-derived geometry and silently falls back to the bundled `DEFAULT_HAND_DIMENSIONS`.
The two are *not* the same hand — the middle finger's first joint diameter is 9.8 mm
from the CAD and 14.0 mm in the fallback — so any test asserting a number must take
the `pinned_dims` fixture, which forces the bundled copy. Otherwise the test passes or
fails depending on whether `gepetto_core` happens to be installed.

## What the golden numbers are

They characterize current behavior; they are not targets. Notably the IK scenario does
**not** close its grasp — it stalls at a ~10 mm worst gap, the documented open problem
in `notes_5f_contact.md`. That is still a strong regression check: repeat solves are
bit-identical, and run-to-run float nondeterminism is ~1e-14 m, so the committed
`rtol=1e-6` has a wide margin and any real change to the graph moves the numbers.

If a change moves them deliberately, update the constants in the same commit and say
why in the message.
