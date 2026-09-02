"""Golden forward-pass tests for the three solvers (CLAUDE.md section 7).

Every test here runs the real C++ solver, so they are all marked ``slow``::

    pytest -m slow tests/test_golden_solves.py

**Hermetic by construction.** Only analytic ellipsoid primitives are used, never a
baked ``.vdb`` SDF grid: the grids are 54 MB, gitignored, and regenerating them needs
conda-only ``pyopenvdb``, so a solve test built on one would be unrunnable on a fresh
checkout. The ``pinned_dims`` fixture additionally forces the bundled hand dimensions,
so the numbers do not depend on whether ``gepetto_core`` is installed.

**What the committed numbers mean.** They characterize what these solvers do *today*;
they are not targets. In particular the single-shot IK scenario does NOT close its
grasp -- it stalls with a ~10 mm worst gap -- which is the documented open problem in
``notes_5f_contact.md`` ("why small objects don't grasp"), not a regression. Asserting
the stall is still a strong refactor check: the stall point is a precise, reproducible
function of the inputs (measured bit-identical across repeat runs), so a split that
perturbs the graph moves it.

Regenerate after an intentional behavior change with::

    pytest -m slow tests/test_golden_solves.py -q   # read the reported actuals
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import solvers

pytestmark = pytest.mark.slow


# Loose enough to absorb the ~1e-14 m of run-to-run float nondeterminism that
# threaded GTSAM introduces (measured over repeated FK solves), tight enough that any
# real change in the graph shows up.
RTOL = 1e-6


def _tips(result, k=0):
    """Fingertip world positions at frame ``k``, ``(n_fingers, 3)``."""
    frame = result.frames[k]
    return np.array(
        [
            np.asarray(frame[n].marginals.sites[-1].pose.mean, float)[:3, 3]
            for n in result.finger_names
        ]
    )


def _assert_frame_well_formed(result, k, n_fingers=5):
    """Structural invariants every frame of every solve must satisfy."""
    tips = _tips(result, k)
    assert tips.shape == (n_fingers, 3)
    assert np.isfinite(tips).all()

    lengths = result.displacements(k)
    assert len(lengths) == n_fingers
    for arr in lengths:
        arr = np.asarray(arr, float)
        assert arr.shape == (6,), "anatomical routing is 6 tendons per digit"
        assert np.isfinite(arr).all()

    # Fingertips live within arm's reach of the wrist, not at 1e9.
    assert np.linalg.norm(tips, axis=1).max() < 0.5


# ---------------------------------------------------------------------------
# FK -- pure kinematics, no contact
# ---------------------------------------------------------------------------


def test_fk_shapes_and_ordering(pinned_dims):
    result = solvers.HandFKSolver(solvers.HandSolveParams()).solve()

    assert len(result.frames) == 1, "FK produces a single frame"
    assert result.finger_names == ["index", "middle", "ring", "pinky", "thumb"]
    _assert_frame_well_formed(result, 0)


def test_fk_golden_sums(pinned_dims):
    result = solvers.HandFKSolver(solvers.HandSolveParams()).solve()

    assert _tips(result).sum() == pytest.approx(0.941626180859, rel=RTOL)
    tendon_sum = float(np.sum([np.sum(x) for x in result.displacements(0)]))
    assert tendon_sum == pytest.approx(4.048889415094, rel=RTOL)


def test_fk_is_reproducible(pinned_dims):
    a = solvers.HandFKSolver(solvers.HandSolveParams()).solve()
    b = solvers.HandFKSolver(solvers.HandSolveParams()).solve()
    # Not bit-identical -- threaded linear algebra reorders sums -- but far below
    # any physically meaningful scale.
    np.testing.assert_allclose(_tips(a), _tips(b), atol=1e-12)


def test_fk_responds_to_flexor_tension(pinned_dims):
    """A sanity check that the tensions actually reach the graph: pulling the
    flexors harder must curl the fingers, moving every fingertip."""
    slack = solvers.HandFKSolver(
        solvers.HandSolveParams(flexor_tensions=[0.2] * 5)
    ).solve()
    pulled = solvers.HandFKSolver(
        solvers.HandSolveParams(flexor_tensions=[1.5] * 5)
    ).solve()

    motion = np.linalg.norm(_tips(pulled) - _tips(slack), axis=1)
    assert (motion > 1e-3).all(), f"some fingertip did not move: {motion}"


# ---------------------------------------------------------------------------
# IK -- single-shot contact (Augmented Lagrangian path)
# ---------------------------------------------------------------------------


def _ik_params():
    # Identity wrist, not the hover default: the hover pose pins the base ~0.11 m
    # off the object and the contact violation freezes (a 70 mm worst gap rather
    # than 10 mm). See the HandSolveParams.wrist_pose docstring.
    return solvers.HandSolveParams(
        primitive="mid_sphere_ellipsoid", wrist_pose=np.eye(4)
    )


def test_ik_shapes(pinned_dims):
    result = solvers.HandIKSolver(_ik_params()).solve()

    assert len(result.frames) == 1
    _assert_frame_well_formed(result, 0)
    assert set(result.surface_gaps()) == set(result.finger_names)


def test_ik_golden_sums(pinned_dims):
    result = solvers.HandIKSolver(_ik_params()).solve()

    assert _tips(result).sum() == pytest.approx(0.287816014344, rel=RTOL)
    assert sum(result.surface_gaps().values()) == pytest.approx(
        0.024360234813, rel=RTOL
    )
    tendon_sum = float(np.sum([np.sum(x) for x in result.displacements(0)]))
    assert tendon_sum == pytest.approx(3.866361765158, rel=RTOL)


def test_ik_stalls_at_the_documented_gap(pinned_dims):
    """CHARACTERIZATION, not a target. This scenario does not close: it stalls with
    the middle finger ~10.4 mm off the surface while the thumb slightly penetrates.

    If this starts passing at a much smaller gap, the grasp got BETTER and the
    number wants updating -- but deliberately, with the change understood."""
    result = solvers.HandIKSolver(_ik_params()).solve()

    assert result.worst_gap() == pytest.approx(0.010433433595, rel=RTOL)
    gaps = result.surface_gaps()
    assert gaps["middle"] == pytest.approx(0.010433433595, rel=RTOL)
    assert gaps["thumb"] < 0.0, "thumb is expected to be slightly inside the surface"


def test_ik_is_reproducible(pinned_dims):
    a = solvers.HandIKSolver(_ik_params()).solve()
    b = solvers.HandIKSolver(_ik_params()).solve()
    np.testing.assert_allclose(_tips(a), _tips(b), atol=1e-12)


# ---------------------------------------------------------------------------
# Planner -- K+1 steps tied by GP temporal priors
# ---------------------------------------------------------------------------


def test_planner_produces_k_plus_one_frames(pinned_dims):
    """K=3 rather than the demos' K=10: this is ~75 s and the frame-count contract
    is what the test is for. HandResult.frames is length 1 for FK/IK and K+1 here,
    so a step scrubber can index them identically."""
    result = solvers.HandPlannerSolver(
        solvers.HandSolveParams(
            primitive="mid_sphere_ellipsoid", wrist_pose=np.eye(4), K=3
        )
    ).solve()

    assert len(result.frames) == 4
    for k in range(4):
        _assert_frame_well_formed(result, k)


def test_planner_golden_terminal_state(pinned_dims):
    result = solvers.HandPlannerSolver(
        solvers.HandSolveParams(
            primitive="mid_sphere_ellipsoid", wrist_pose=np.eye(4), K=3
        )
    ).solve()

    last = len(result.frames) - 1
    assert _tips(result, last).sum() == pytest.approx(0.331450866209, rel=RTOL)
    assert result.worst_gap(last) == pytest.approx(0.057433628289, rel=RTOL)


def test_planner_trajectory_is_continuous(pinned_dims):
    """The GP temporal priors exist to make consecutive steps neighbours. Whatever
    the solve converges to, no fingertip may teleport between steps."""
    result = solvers.HandPlannerSolver(
        solvers.HandSolveParams(
            primitive="mid_sphere_ellipsoid", wrist_pose=np.eye(4), K=3
        )
    ).solve()

    for k in range(len(result.frames) - 1):
        step = np.linalg.norm(_tips(result, k + 1) - _tips(result, k), axis=1)
        assert step.max() < 0.10, f"fingertip jumped {step.max():.3f} m at step {k}"
