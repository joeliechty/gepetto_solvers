# gepetto_solvers

Factor graph–based solvers for a tendon-driven robotic hand, and for the continuum
structures it is built from.

*(Formerly `crest-sparse` / CREST-Sparse — Continuum Robot ESTimation with Sparse
Nonlinear Optimization. The package was renamed in 2026-08; the research the code
implements is the same.)*

## What this is

Continuum robot state estimation can be formulated like SLAM: variables connected
through spatial motion priors and measurement factors. This repository builds factor
graph representations of the conditional distribution over continuum robot
configurations, and leans on GTSAM's sparse nonlinear optimization to solve them.

The tree is the **tendon hand** and what it needs. A Cosserat rod builds a
tendon-driven finger, and a set of fingers sharing one floating wrist variable makes
the hand — three C++ solvers over that graph:

| | |
|---|---|
| `TendonHandSolver` | static FK, and single-shot IK against a contact surface |
| `TendonHandTrajectoryPlanner` | a K+1-step trajectory with GP temporal priors |
| `HandIKStepper` (Python) | the IK solve, one Augmented Lagrangian iteration per call |

**[docs/tendon_hand.md](docs/tendon_hand.md) is the real documentation** — what each
solver builds, which section of *Underactuated Object Manipulation* it implements,
and how everything wires up. Start there. This file only gets you running.

## Layout

```
include/gepetto_solvers/   public C++ headers
src/                       C++ implementation; src/bindings/ holds all the pybind11
python/gepetto_solvers/    the Python layer, in three tiers:
    core/                    foundational — hand, environment, geometry, solvers,
                             robot_plan, objects, plotting, diagnostics
    projects/                isolated research — grasp_pipeline, viz, robot_mount
    experimental/            ad hoc diagnostics, not maintained as examples
scripts/                   thin CLIs, one per demo
tests/                     the pytest suite
docs/                      the architecture doc, and why the tree is shaped this way
```

`core/` imports nothing from `projects/` or `experimental/`, and no project imports
another. That is the whole dependency rule.

[docs/refactor-2026-08.md](docs/refactor-2026-08.md) records why the tree is shaped
this way — the decisions taken during the restructure, and what is still open.

## Install

**On a fresh machine, use the setup scripts.** They do the part that is genuinely
hard: building a *forked* GTSAM (4.3a1 plus a constrained-module heap-overflow fix)
from source into a conda prefix, with the toolchain and channel pins that make
OpenVDB, Boost and TBB agree with it.

```bash
./conda_setup_py12.sh        # Linux, Python 3.12   (py10 is the same, 3.10)
./conda_setup_py11_mac.sh    # macOS, Python 3.11
```

Each creates the env, builds and installs GTSAM, sets the linker-path activate hook,
then installs this package. Read the script before running it: it removes an existing
env of the same name and deletes `../gtsam`.

**Once that env exists**, day-to-day work is just:

```bash
conda activate crest_py12          # or crest_py11 / crest_py10
rm -rf build && pip install -e ".[viz,web,dev]"
```

Delete `build/` after any structural change — a stale CMake cache outlives the tree
it describes.

### Extras

`pyproject.toml` is the single source of truth for Python dependencies. Solving needs
only numpy and scipy; everything that draws, fetches or fits is optional:

| extra | brings | for |
|---|---|---|
| `viz` | pyvista, vtk, matplotlib, pillow | the render windows the demo scripts open |
| `web` | viser, trimesh | the interactive workbench |
| `ycb` | trimesh, scikit-learn, coacd, requests | fetching and fitting YCB scans |
| `dev` | pytest, ruff, mypy | the test suite and the linters |

GTSAM, OpenVDB and pybind11 are deliberately **not** listed: they are C++ build
dependencies the conda scripts install into `$CONDA_PREFIX`, and CMake finds them
through the Python interpreter's prefix.

## Run

Scripts work from anywhere once the package is installed:

```bash
python scripts/ik_5f_contact.py big_sphere --no-viz   # single-shot grasp
python scripts/traj_5f_slide_grasp.py --no-viz        # slide-and-grasp trajectory
python scripts/fk_5f_sweep.py                         # FK (opens a window)
python scripts/dynamics_sim.py                        # rod dynamics
python scripts/viz_interactive.py                     # workbench on :8080
python scripts/viz_interactive.py --smoke             # its headless self-check
```

`--no-viz` skips the render window. `scripts/experimental/` holds the diagnostics —
AL traces, constraint sweeps — which answer one question each and are not examples.

The baked SDF grids (`.vdb`, ~54 MB) are gitignored, so a fresh checkout has none.
Rebuild them with `python scripts/objects/make_big_sphere.py` (needs conda-only
`pyopenvdb`), or use the analytic primitives — `coin`, `pen`, `mid_sphere_ellipsoid`,
`megaminx` — which need no grid at all.

## Tests

```bash
pytest -m "not slow"    # pure functions, no solver -- under a second
pytest -m slow          # golden solves + the viz smoke checks -- ~6 minutes
pytest                  # both
ruff check .
mypy
```

See [tests/README.md](tests/README.md) for the two constraints that shape the suite:
no committed test may need a `.vdb` grid, and any test asserting a number must pin
the hand dimensions (`load_hand_dimensions()` silently prefers `gepetto_core`'s CAD
geometry over the bundled fallback, and the two are not the same hand).

`scripts/capture_baseline.py` captures solver behavior to JSON and diffs two
captures. It covers the SDF paths the committed suite cannot, and is what a refactor
should be checked against:

```bash
python scripts/capture_baseline.py --out ~/before.json
...
python scripts/capture_baseline.py --diff ~/before.json ~/after.json
```

## Troubleshooting

**`ImportError: libgtsam.so.4: cannot open shared object file`** — GTSAM's install
directory is not visible to the dynamic linker. The setup scripts write an
`activate.d` hook for this; if you built GTSAM by hand, `sudo ldconfig` and re-run.

**A GUI control does nothing, or `capabilities()` reports `False`** — the installed
extension is older than the Python layer. `rm -rf build && pip install .`

**A `.vdb` is missing** — the error names the baker to run.
