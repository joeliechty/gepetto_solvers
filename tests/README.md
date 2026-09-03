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
| `core/test_build_env.py` | the built conda envs — linkage, ABI, Boost/Eigen pins (`slow`) |
| `test_golden_solves.py` | FK / IK / planner forward passes (`slow`) |
| `projects/test_viz_smoke.py` | the visualizer's five `--smoke` routines (`slow`) |

`_pkg.py` is the single place the suite names the application layer's import path, so a
package move changes that one file and no test. Its `config` is the five-digit tendon
hand's morphology package — one hand among (eventually) several, so a test asserting a
measured number there is asserting something about THAT hand.

## `core/test_build_env.py` checks every env, not the active one

The three `conda_setup_py*.sh` scripts build `gepetto_py10` / `gepetto_py11` /
`gepetto_py12`, and they do **not** pin identically — py10 and py12 pin the linux-64
`libpinocchio` build `h2844b27_0` where the mac script pins `h9a60d09_0`, and py12
resolves OpenVDB 13.0 against py10's 12.1. A linkage regression that only bites one
interpreter is exactly what a suite run from whichever env happens to be active would
miss, so that file discovers every declared env and probes each in a subprocess.

Env names *and* their python versions are parsed out of the `conda create` lines in
the scripts themselves, so a fourth script needs no edit here. An env that is absent,
or present but not yet `pip install`ed into, is skipped with a reason naming it — the
file must stay runnable on a machine that set up only one.

What it guards, all of it a pin whose failure is silent rather than loud:

- **One Boost per prefix.** conda-forge froze the `boost` metapackage at 1.85.0 with a
  hard `libboost-python-devel ==1.85.0` pin while libpinocchio 4.0.0 needs ≥1.88, so
  `boost` and `libboost-devel` together are either an unsatisfiable solve or two Boosts
  on the loader path. GTSAM links Boost by unversioned soname, so the second case runs
  and corrupts memory across the serialization boundary.
- **Eigen 3.4 everywhere.** GTSAM static-asserts `GTSAM_EIGEN_VERSION_*`; that is a
  compile-time error and unreadable from Python, so the test asserts the consequence
  instead — Pinocchio and the extension loaded in one process, both still working.
- **Nothing from outside the prefix.** `CMakeLists.txt` finds GTSAM and OpenVDB through
  the interpreter prefix, so a system `libboost_serialization.so` on the loader path is
  an ABI mismatch the loader performs happily.

It needs `pytest` in the env, which `conda_setup_*.sh` does not install — the scripts
install `.[viz,web]`, not `.[dev]`. Add it with `pip install ".[dev]"`.

## Two constraints worth knowing before adding a test

**No baked `.vdb` SDF grids.** They live in `_objects/`, total 54 MB, are gitignored,
and regenerating them needs conda-only `pyopenvdb` — so a test built on one is
unrunnable on a fresh checkout. Solve tests use the analytic ellipsoid primitives
(`coin`, `credit_card`, `pen`, `*_sphere_ellipsoid`, `megaminx`), which carry their
geometry inline. `core/test_geometry.py::test_analytic_primitives_need_no_vdb_grid`
guards this.

SDF-path coverage lives in `scripts/capture_baseline.py` instead, which runs against a
developer's working tree where the grids exist.

**Pin the hand.** `load_hand_dimensions()` prefers `epfl_hand_core`'s CAD-derived
geometry and silently falls back to the bundled `DEFAULT_HAND_DIMENSIONS`. The two are
*not* the same hand — the middle finger's first joint diameter is 9.8 mm from the CAD
and 14.0 mm in the fallback — so any test asserting a number must take the
`pinned_hand` fixture and pass it to the solver (`HandFKSolver(params, pinned_hand)`).
Otherwise the test passes or fails depending on whether `epfl_hand_core` happens to be
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
