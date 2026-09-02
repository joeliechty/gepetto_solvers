# The 2026-08 restructure

Why the tree looks the way it does, what was decided along the way and why, and
what is still open.

This is the record for the person who later asks "why is it like this" — or who
wants to do the next one. It is not a changelog; `git log` has that. It documents
the **decisions**, including the ones that went against the obvious answer.

Branch: `refactor/claude-md-architecture`. Eleven commits took the tree to its
new shape (`1dfa6f0`..`fb5b792`, 330 files); later commits fix what fell out of
it — see §6.

---

## 1. Why

The repo had been pruned to the tendon hand and was well documented, but it was
not organized the way `CLAUDE.md` prescribes. Concretely:

- `python/tests/` was **not tests**. It held the entire ~17k-line application
  layer — solvers, scene, config, robot plan, fifteen argparse entry points —
  with one real test file inside it. That file could not even be collected by
  pytest (relative imports, no `__init__.py`); it ran only through a hand-rolled
  `main()`.
- Seven source files were over 1000 lines. `viz_interactive.py` was 5186, of
  which one class was 4284 across 122 methods.
- Bindings were scattered as a `pybind.cpp` inside each module directory plus a
  separate `src/pybind/`. Headers sat beside their `.cpp`.
- Dependencies were split across `requirements.txt` and `pyproject.toml`, and had
  drifted: the former pinned `pybind11` (which conda already provides) while
  omitting `trimesh`, `pillow`, `scikit-learn`, `coacd` and `requests`, all of
  which the code imports.
- No lint config, no type checking, ~10 annotated signatures in 17k lines.

## 2. What changed, honestly

The headline number is real but partial:

| | before | after |
|---|---|---|
| largest Python file | 5186 | 993 |
| largest C++ file | 2218 | 775 |
| files over 1000 lines | 7 | **0** |
| files over 800 lines | 8 | 1 |
| files over 500 lines | 14 | 12 |
| files over 300 lines | 29 | **38** |
| total source files | 94 | 179 |

Read the bottom half of that table too. Splitting eliminated the *enormous*
files, which was the goal — nothing is now beyond what one sitting can hold. But
the count of merely-large files barely moved, and the count of medium files went
**up**, because a 3294-line module becomes twelve 200–550-line modules rather than
twelve 100-line ones. If the target is "no file over 300 lines", this refactor did
not get there and was not trying to.

Layout now:

```
include/gepetto_solvers/   public C++ headers (41)
src/                       C++ implementation (27), bindings/ holds all pybind11
python/gepetto_solvers/    110 modules in three tiers:
    core/                    foundational; imports nothing from the tiers below
    projects/                isolated; no project imports another
    experimental/            ad hoc diagnostics, not maintained as examples
scripts/                   26 thin CLIs
tests/                     114 tests in two tiers
```

## 3. The order it was done in, and why that order

**Phase 0 — the safety net, before anything moved.** `1dfa6f0`

This is the part worth copying. Nothing was refactored until there was a way to
prove behavior had not changed:

- `scripts/capture_baseline.py` runs six solver scenarios and writes every number
  that matters to JSON, then diffs two captures at `rtol=1e-6`. It covers the
  `.vdb` SDF paths the committed suite cannot.
- 69 pure-function tests over the geometry, the hand morphology, and the witness
  readouts — all of which are pure functions of solved poses, so they need no
  solver and run in 0.13 s.
- Golden forward-pass tests for FK / IK / planner.
- The five `--smoke` routines `viz_interactive.py` already had, wired into pytest
  so they run instead of sitting behind a CLI flag nobody types.

Three measurements from this phase shaped everything after it:

1. **Solves are reproducible.** Repeated IK solves are bit-identical; FK varies by
   ~1e-14 m from threaded linear algebra. So `rtol=1e-6` has enormous margin and a
   baseline diff is a trustworthy gate.
2. **`load_hand_dimensions()` silently switches hands.** It prefers
   `gepetto_core`'s CAD geometry and falls back to the bundled
   `DEFAULT_HAND_DIMENSIONS` — and they disagree (middle finger first joint
   diameter: 9.8 mm vs 14.0 mm). Every test asserting a number pins the bundled
   copy, or it would pass or fail depending on what happens to be installed.
3. **The IK scenario does not close.** Single-shot IK on `mid_sphere_ellipsoid`
   stalls at a ~10 mm worst gap. That is the open problem in
   `notes_5f_contact.md`, so the golden test *characterizes the stall* rather than
   asserting a convergence that does not happen.

**Phase 1 — rename and move.** `27259b9`

`crest_sparse` → `gepetto_solvers` across the pybind module, the CMake target, the
C++ namespace and every import; then `python/tests/` exploded into the three
tiers. Mechanical, and verified byte-identical against the baseline.

**Phase 2 — the splits.** `0eed6f2`..`d23a464`

Six commits, smallest first to establish the pattern. Each split was derived from
the file's own dependency graph rather than from where the section comments
happened to be; all four Python graphs turned out acyclic, which is why the
splits have no import cycles.

**Phase 3 — dependencies, lint, types.** `ab887ef`

**Phase 4 — C++ layout.** `b337c0e`

**Phase 5 — documentation.** `fb5b792`

---

## 4. Decisions

The durable part of this document.

### Compatibility re-exports are deliberate, not leftovers

Every split package's `__init__.py` re-exports the full public surface of the
module it replaced. `from gepetto_solvers.core.solvers import X` still reaches
anything that used to be in `solvers.py`, without the caller knowing which of the
twelve submodules it landed in.

This is why 330 files changed and almost no call site did. It also means the
`__init__.py` files are load-bearing API definitions, not boilerplate — deleting a
name from one is a breaking change.

A few **private** names are re-exported too, with an explicit alias and a comment
saying why: `_Rx/_Ry/_Rz` (`robot_mount.mount` builds its candidate mounting
rotations from them), `solvers._set_if` (several docstrings name that path), the
`robot_plan` rotation helpers, and the five `_smoke_*` routines. Private names
crossing a module boundary are invisible to a search for the public API, which is
how they were found: by collecting every name the whole tree imports from a module
and diffing against the new package.

### `ruff format` is not run

`CLAUDE.md` §6 asks for Ruff formatting. It is deliberately not applied, and
`pyproject.toml` records why at the config site.

Measured, `ruff format` reflows **9921 lines across 94 files**, and what it removes
is meaning. The clearest example is the stiffness diagonal in
`core/hand/finger_config.py`:

```python
# as written -- the six rows read as a table
K_inv[0,0] = 1 / (k_bending * lateral_stiffness_scale)  # out-of-plane bend (about x)
K_inv[1,1] = 1 / k_bending                              # flexion (about y) -- never scaled

# as ruff format would have it
K_inv[0, 0] = 1 / (
    k_bending * lateral_stiffness_scale
)  # out-of-plane bend (about x)
K_inv[1, 1] = 1 / k_bending  # flexion (about y) -- never scaled
```

`E501` follows from the same decision rather than being a separate one: 93 of the
100 violations are code, so clearing them means reflowing exactly the lines we
declined to reflow. Line length is advisory; `ruff check` is the gate.

### pep8-naming's case rules are off

188 findings, all of the same kind: `T` is an SE(3) pose, `R` a rotation, `K` the
stiffness, `I` the second moment of area, `H` the Hessian. `T`/`S`/`F`/`D`/`Q`/`L`/
`W`/`O`/`Y`/`U` are also the literal GTSAM key symbols the graph is built from.
Renaming `euler_to_R` to `euler_to_r` would break the correspondence between the
code and the equations it implements, which is the single thing making this system
readable.

### B905 is recorded as debt, not silenced

90 `zip()` calls have no explicit `strict=`. Python's default silently truncates
to the shortest input, which over the parallel per-finger arrays this code is full
of would hide a real mismatch. Choosing `strict=True` (unequal lengths are a bug)
or `strict=False` (the truncation is intended) is a judgement per site, by someone
who knows what that site iterates. Making it blind, inside a refactor that promises
no behavior change, was the wrong place. **This is a real follow-up**, not a
closed question.

### numpy and scipy carry bounds, not exact pins

`requirements.txt` pinned them exactly. The reason those pins existed is a cp310
wheel ceiling — numpy ≥2.3 and scipy ≥1.16 dropped Python 3.10 — which is a
compatibility ceiling, not a demand for one exact build.

The difference matters on macOS. `conda_setup_py11_mac.sh` deliberately replaces
pip's Accelerate-linked numpy/scipy with conda's openblas build, because the
Accelerate wheels lack LAPACK symbols this OS needs. With an exact pin, every later
`pip install -e .` would drag the Accelerate wheel back in and silently undo that.
A bound is satisfied by whatever conda already installed, so pip leaves it alone —
verified, numpy stayed at 2.4.4 across both editable and non-editable installs.

That same fact forced the one ordering change in the setup scripts: the swap now
runs **after** `pip install`, because numpy/scipy are declared dependencies now and
doing it first would simply have them put back.

### The conda setup scripts were treated as protected

They carry knowledge that cannot be recovered by reading the code: the forked GTSAM
at `release-4.3a1-fixes` (4.3a1 plus a constrained-module heap-overflow fix), the
`gcc_linux-64=13` pin (GCC 16 false-positives `-Wmaybe-uninitialized` on Eigen and
trips GTSAM's `-Werror`), `--override-channels` for the conda ToS and ABI, the
explicit `zlib` OpenVDB's `FindZLIB` needs, and the `activate.d` linker hooks.

Only the final `pip install` block changed in each. The diff was checked
explicitly for every one of those lines. `CMakeLists.txt`'s Python-prefix
resolution block is likewise byte-identical, because `conda_setup_py12.sh`'s
interpreter guard depends on that behavior.

### mypy is scoped, and not strict

It runs over `core/{environment,geometry,hand,robot_plan,solvers}` — the layer
everything builds on, where a wrong signature propagates. `plotting/` and
`objects/ycb/` are excluded as GUI glue over untyped third-party APIs, where
chasing strictness buys noise.

Not strict mode either: this is a numpy codebase where most parameters are
array-like and most returns are `ndarray`, and `disallow_untyped_defs` over 12k
lines of that would demand annotations that restate the docstrings without
checking anything. What is enabled catches the errors that bite — a name that does
not exist, a call that cannot work, an `Optional` used without a guard.

### The compiled extension is opaque to static analysis

pybind11 exports at runtime, so every `gepetto_solvers.TendonHandSolver` reads as a
missing attribute. There is a `[[tool.mypy.overrides]]` asserting that. Generating
stubs for the extension is worth doing one day.

---

## 5. Two things that are load-bearing and fail silently

Both are documented at their site as well; they are repeated here because they are
the things most likely to be broken by someone who does not know them.

### Constraint insertion order in `TendonHandGraph.cpp`

The Augmented Lagrangian indexes multipliers by a constraint's **position** in
`ConstrainedOptProblem`'s enumeration, which is graph insertion order. Reorder a
block in `build_graph` and every carried multiplier re-seats onto the wrong
constraint — and the solve still runs. The numbers are merely wrong.

Every hard constraint must enter through `add_eq`/`add_ineq`, which record the
semantic tag beside it in the same order.

`tests/core/test_constraint_tags.py` guards this, and it was written **before**
`TendonHandModel.cpp` was split, not after. The signal it watches for is the one
the architecture doc names: **0 matched against a non-empty carry** means a tag
drifted, not that the problem changed.

### The `.vdb` grids are not where a naive reading suggests

They are gitignored, 54 MB, and need conda-only `pyopenvdb` to regenerate — so a
fresh checkout has none, and a non-editable `pip install .` puts the package
somewhere that has none either. `core/objects/OBJECTS_DIR` is the single definition
of where they are looked for: `$GEPETTO_OBJECTS_DIR`, then the package directory,
then a source checkout at or above the working directory.

Before the refactor this never came up, because only the thin extension was
installed and the whole application layer always ran from the source tree.

---

## 6. Bugs found

Found *by the tests*, during the refactor. None were introduced by it; the first
four were latent, the rest were caught the moment they were created.

| where | what |
|---|---|
| `core/solvers/result.py` | `at_iterate` indexed `self.iterates` without checking it existed |
| `core/robot_plan/pacing.py` | `max(key=dict.get)` — `.get` can return `None` where `__getitem__` cannot |
| `core/solvers/fk.py` | `solve()` could fall off the end of a function annotated `-> HandResult` |
| `core/geometry/scene/surface.py` | a Taubin norm was a numpy scalar assigned a float floor |
| `robot_plan/hardware.py` | `from . import solvers` meant `core.solvers` before the split and `core.robot_plan.solvers` after — an ImportError reachable only on the hardware path |
| `projects/viz/…` | `from .constants import *` **silently skips underscore-prefixed names**, so every tolerance the smoke checks judge against vanished |
| three `core/solvers/` modules | imports inserted *inside* a parenthesised multi-line import — a `startswith()` scan for "the last import line" does not understand those |
| `experimental/sweeps/` | a `for i` → `for _i` rename broke a variable used *after* the loop; `B007` only inspects the loop body |

The two in the middle are the argument for the whole Phase 0 approach: neither is
visible to inspection, and both would have shipped.

### Three that got past the tests entirely

Found by a person opening the app, which is the honest way to describe it. All
three share a root cause: **nothing exercised the GUI.** The five `--smoke`
routines drive the solver half and never touch viser, so `_build_gui` — 912
lines, the largest function in the codebase — first ran in front of a user.

1. **Eight class attributes were dropped from `HandVizApp`.** The mixin split
   walked the class body for `FunctionDef` nodes and silently discarded
   everything else. `TENDON_IDLE`, `UNFITTED_SUFFIX`, `WRIST_PRIOR_GAUGE_LIMIT`,
   `FINGERTIP_SHELL_M`, `GRASPABLE_MAX_M`, `_WRIST_RANGE_MARGIN`,
   `_TENSION_BISECT_STEPS`, `_TENSION_BISECT_TOL_M`. `ViserHandScene` had no
   class attributes, so it escaped.

2. **Four function-local relative imports in `core/solvers/witness.py` pointed
   one package too shallow.** `from .hand.config import …` meant `core.hand.config`
   when `solvers.py` was a module in `core/`; after the split it means
   `core.solvers.hand.config`, which does not exist. Ruff cannot resolve module
   paths, and an import of `witness` does not execute a function body — so both
   gates were clean.

3. **A circular import between `core/environment/` and `core/hand/config/`.**
   Introduced by the compatibility re-export: `hand.config.__init__` eagerly
   imported the `attach_*` family from `environment`, while every `environment`
   module imports back into `hand.config`. Importing `core.environment.collision`
   *first* got a half-initialized package and failed. It worked in practice only
   because everything reached it from the other direction. Fixed with a PEP 562
   module `__getattr__`, so the re-export resolves on first access rather than at
   import time.

The gaps those exposed are now closed by `tests/projects/test_mixin_surface.py`:
it pins both composed classes' member surfaces, walks `_build_gui` for `self.X`
constant reads and checks each resolves, and — the one that matters —
**builds the entire GUI against a real viser server.**

The general lesson, and the one to carry into the next refactor: *an import
check proves nothing about a function body, and a class-body walk that only
collects methods is not a move.* Both were verified with static tools that
cannot see the thing being broken.

---

## 7. Still open

Ordered by how much they matter.

1. **`fk_5f_sweep.py` no longer sweeps the flexor.** The oscillation is commented
   out at the use site and the tension is pinned at `background + 1.0`:
   ```python
   flexor = background_tension + 1.0#flexor_amplitude * (np.cos(...) + 1.0)
   ```
   The architecture doc still advertises "warm-started wrist sweep + flexor
   sweep". Left alone because restoring it changes behavior; the dead knob is
   documented in place. **This is the only open item about the research rather
   than the plumbing.**

2. **90 bare `zip()` calls** — see the B905 decision above. Needs a per-site pass.

3. **`_build_gui` is 912 lines** inside `projects/viz/viz_interactive/_gui.py`. It
   builds the entire viser control tree as one function; splitting it means
   inventing folder boundaries that do not exist in the code yet. A design change,
   not a move.

4. **`build_graph` is 583 lines** in `src/tendon_hand/TendonHandGraph.cpp`. The
   plan sketched decomposing it into private `add_*` methods. Not done, because
   carving it up means deciding which locals cross block boundaries and getting
   that wrong reorders constraint insertion — the one change in that file that
   fails silently. It is *safe to attempt now* in a way it was not before, because
   `test_constraint_tags.py` exists.

5. **OpenMP is dead build machinery.** CMake finds it, links it, defines
   `GEPETTO_USE_OPENMP` and prints "enabling parallel collision detection" — and
   there is not one `#pragma omp` in the source. Either wire it up or drop it.

6. **`CREST_AL_VERBOSE` keeps its old name.** It is a user-facing environment
   variable; renaming it silently would break muscle memory for no gain. Rename it
   deliberately, with the docs, or leave it.

7. **No stubs for the compiled extension**, so static analysis cannot see any C++
   type. `pybind11-stubgen` would fix this.

---

## 8. How to verify a change to any of this

```bash
rm -rf build && pip install -e ".[viz,web,dev]"

pytest -m "not slow"     # 110 tests, under a second
pytest -m slow           # 23 tests: golden solves + viz smoke, ~6 min
ruff check .
mypy

# the real gate for anything that touches a solve path
python scripts/capture_baseline.py --out ~/after.json
python scripts/capture_baseline.py --diff ~/before.json ~/after.json
```

The baseline diff must be **empty**. It was, at the end of every phase of this
refactor, across all six scenarios including the SDF path.

Two things it will not catch, and what to do instead:

- **Rendering.** The smoke tests never start viser, so `core/plotting/viser_hand/`
  is covered by import and composition only. Run
  `python scripts/viz_interactive.py` and look at the page.
- **A `def_readwrite` to an unregistered pybind type** raises at *attribute
  access*, not at import — so an import check proves nothing about a binding
  change. Always run a solve to completion.

And note the diff refuses to compare captures taken with different hand
dimensions, for the reason in §3.


---

## 9. Follow-up: separating the hand from the solvers (2026-09)

The restructure above left one thing still fused: the solvers *were* the tendon
hand. `TendonHandModel` held a `std::variant<TendonFingerModel<1..10>>`, the
graph builder called into it directly, and the Python layer carried the hand's
identity as module constants — `FINGER_NAMES`, `FLEXOR_IDX` (declared twice, in
two packages, cross-checked at runtime), `HAND_PINCH_POSES`,
`HARDWARE_FINGER_NAMES`, and the literal string `"thumb"` matched in five Python
modules and three places in C++.

That is now split along the line the math already had:

* **`HandKinematics`** (C++) contributes the factors internal to a mechanism and
  answers `site_pose_key({digit, node})`. `HandModel::build_graph` builds the
  task constraints around it and never learns what it is posing. Registered by
  name; `TendonHandKinematics` is the first implementation and is a MOVE of what
  was inline in the graph builder, not a rewrite.
* **`Hand`** (Python) carries the digits, the actuation layout, the opposing
  digit and the measured tables, and names the kinematics to load.

Renames were complete, with no aliases: `tendon_hand/` → `hand/`,
`tendon_finger/` → `digits/tendon/`, `TendonHandSolver` → `HandSolver`,
`TendonHandMarginals` → `HandState` (`.fingers` → `.digits`),
`core/hand/config/` → `core/hands/tendon_5f/`. Every in-repo caller, script and
test moved in the same change.

**The gate held.** The golden sums, the constraint-tag ordering and the AL dual
transfer were all unchanged, and the baseline diff was empty across all six
scenarios — which is what proves the graph is the same graph.

Two things this deliberately did NOT generalize, both recorded where they live:

* **`HandState` still holds `std::vector<TendonFingerMarginals>`** — see the note
  in `HandState.h`. It is the transport for a finished solve, produced and
  consumed entirely behind `HandKinematics::extract` / `insert_from_state`, and
  the graph builder never touches it. A mechanism whose state is not
  rod-and-tendon shaped needs a variant payload there and a matching split in
  the Python readers of `state.digits[i].rod`; that is worth doing against a real
  second payload rather than against a guess.
* **Dynamics.** `HandModel` calls its kinematics through a contributor-shaped
  hook so a `HandDynamics` can join it in a fixed order without disturbing the
  constraint numbering. Nothing was built.

See [adding_a_hand.md](adding_a_hand.md).
