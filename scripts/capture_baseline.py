"""Capture the solvers' numeric behavior to JSON, and diff two captures.

This is the refactor safety net. The committed pytest suite deliberately avoids the
baked ``.vdb`` SDF grids (54 MB, gitignored, and regenerating them needs conda-only
``pyopenvdb``), so it cannot cover the SDF code paths at all. This script can, because
it runs against a developer's working tree where those grids exist.

Workflow::

    python scripts/capture_baseline.py --out ~/before.json      # before a refactor
    ...                                                          # move/split things
    python scripts/capture_baseline.py --out ~/after.json
    python scripts/capture_baseline.py --diff ~/before.json ~/after.json

The diff must be empty. Keep the JSON *outside* the repo -- it is a local measurement,
not an artifact, and the numbers are machine-specific at the tolerances used here.

Timings are recorded but never compared: they are the one field guaranteed to differ.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from gepetto_solvers.core.solvers import (
    HandFKSolver,
    HandIKSolver,
    HandPlannerSolver,
    HandSolveParams,
)

# Numbers below this move in the last bits of a double under any reordering of
# floating-point work, which a refactor can cause without changing behavior.
ATOL = 1e-9
RTOL = 1e-6


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
# Each entry is (name, builds-a-solver, needs-a-.vdb). The `vdb` flag lets a machine
# without the baked grids still capture the analytic half instead of erroring out.


def _fk_default() -> HandFKSolver:
    return HandFKSolver(HandSolveParams())


def _fk_flexed() -> HandFKSolver:
    p = HandSolveParams(flexor_tensions=[1.2, 1.2, 1.2, 1.2, 1.2])
    return HandFKSolver(p)


def _ik_mid_sphere() -> HandIKSolver:
    p = HandSolveParams(
        primitive="mid_sphere_ellipsoid",
        sigma_wrist_pos=1e-2,
        sigma_wrist_rot=1e-1,
    )
    return HandIKSolver(p)


def _ik_pen() -> HandIKSolver:
    p = HandSolveParams(primitive="pen", sigma_wrist_pos=1e-2, sigma_wrist_rot=1e-1)
    return HandIKSolver(p)


def _ik_big_sphere_sdf() -> HandIKSolver:
    p = HandSolveParams(
        primitive="big_sphere", sigma_wrist_pos=1e-2, sigma_wrist_rot=1e-1
    )
    return HandIKSolver(p)


def _planner_mid_sphere() -> HandPlannerSolver:
    p = HandSolveParams(
        primitive="mid_sphere_ellipsoid",
        K=5,
        sigma_wrist_pos=1e-2,
        sigma_wrist_rot=1e-1,
    )
    return HandPlannerSolver(p)


SCENARIOS: list[tuple[str, Callable[[], Any], bool]] = [
    ("fk_default", _fk_default, False),
    ("fk_flexed", _fk_flexed, False),
    ("ik_mid_sphere_ellipsoid", _ik_mid_sphere, False),
    ("ik_pen", _ik_pen, False),
    ("ik_big_sphere_sdf", _ik_big_sphere_sdf, True),
    ("planner_mid_sphere_K5", _planner_mid_sphere, False),
]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _tip_positions(result, k: int) -> list[list[float]]:
    """Fingertip world positions at frame ``k``, in finger order."""
    frame = result.frames[k]
    return [
        np.asarray(frame[name].marginals.rod.states[-1].pose.mean, float)[
            :3, 3
        ].tolist()
        for name in result.finger_names
    ]


def _record(result, params) -> dict:
    """Everything about a solve that a refactor must not change."""
    n = len(result.frames)
    out: dict[str, Any] = {
        "num_frames": n,
        "finger_names": list(result.finger_names),
        "tip_radii": [float(r) for r in result.tip_radii],
        "object_center": np.asarray(result.object_center, float).tolist(),
        "object_rotation": np.asarray(result.object_rotation, float).tolist(),
        "frames": [],
    }

    meta = result.meta
    for attr in ("iterations", "error"):
        if hasattr(meta, attr):
            out[f"meta_{attr}"] = float(getattr(meta, attr))
    # Recorded for context, excluded from the diff -- see _compare().
    for attr in ("total_time_ms", "build_time_ms", "optimize_time_ms"):
        if hasattr(meta, attr):
            out.setdefault("timings", {})[attr] = float(getattr(meta, attr))

    for k in range(n):
        frame: dict[str, Any] = {"tips": _tip_positions(result, k)}
        try:
            frame["surface_gaps"] = {
                name: float(g) for name, g in result.surface_gaps(k).items()
            }
            frame["worst_gap"] = float(result.worst_gap(k))
        except Exception as exc:  # FK attaches no contact; nothing to measure
            frame["surface_gaps"] = f"<unavailable: {type(exc).__name__}>"
        try:
            frame["tendon_lengths"] = [
                np.asarray(v, float).tolist() for v in result.tendon_lengths(k)
            ]
        except Exception as exc:
            frame["tendon_lengths"] = f"<unavailable: {type(exc).__name__}>"
        out["frames"].append(frame)

    return out


def capture(only: list[str] | None, skip_vdb: bool) -> dict:
    # Which hand morphology these numbers were produced with. load_hand_dimensions()
    # prefers gepetto_core's CAD-derived HandGeometry and silently falls back to the
    # bundled DEFAULT_HAND_DIMENSIONS, and the two do NOT agree (the middle finger's
    # first joint diameter is 9.8 mm from CAD, 14.0 mm in the fallback). Comparing a
    # capture taken with one source against a capture taken with the other is
    # meaningless, so the source is recorded and the diff refuses to cross it.
    try:
        import gepetto_core.geometry  # noqa: F401

        dims_source = "gepetto_core"
    except Exception:
        dims_source = "DEFAULT_HAND_DIMENSIONS"

    payload: dict[str, Any] = {
        "_meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "dims_source": dims_source,
        "scenarios": {},
    }

    for name, build, needs_vdb in SCENARIOS:
        if only and name not in only:
            continue
        if needs_vdb and skip_vdb:
            print(f"  {name}: skipped (--skip-vdb)")
            continue

        print(f"  {name}: ", end="", flush=True)
        t0 = time.time()
        try:
            solver = build()
            result = solver.solve()
            payload["scenarios"][name] = _record(result, solver.params)
            print(f"ok ({time.time() - t0:.1f}s)")
        except Exception as exc:
            # A scenario that cannot run is itself a fact worth recording -- if it
            # starts or stops failing across a refactor, that is a behavior change.
            payload["scenarios"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"FAILED ({type(exc).__name__}: {exc})")

    return payload


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def _compare(a: Any, b: Any, path: str, out: list[str]) -> None:
    """Structural compare with a numeric tolerance. Appends human-readable diffs."""
    # Timings and the environment block are expected to differ.
    if path.endswith(("/timings", "/_meta")):
        return

    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{path}/{key}: only in AFTER")
            elif key not in b:
                out.append(f"{path}/{key}: only in BEFORE")
            else:
                _compare(a[key], b[key], f"{path}/{key}", out)
        return

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
            return
        # strict=True is free here: the length mismatch is reported and returned
        # above, so reaching this point means the two lists are the same length.
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _compare(x, y, f"{path}[{i}]", out)
        return

    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append(f"{path}: {a} -> {b}")
        return

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            out.append(f"{path}: {a!r} -> {b!r}")
        return

    if a != b:
        out.append(f"{path}: {a!r} -> {b!r}")


def diff(before_path: str, after_path: str) -> int:
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    b_src = before.get("dims_source")
    a_src = after.get("dims_source")
    if b_src != a_src:
        print(
            f"REFUSING TO DIFF: hand dimensions came from {b_src!r} before and "
            f"{a_src!r} after.\nThe two morphologies differ, so every number below "
            "would differ for reasons\nunrelated to the refactor. Re-capture both "
            "with the same source."
        )
        return 2

    findings: list[str] = []
    _compare(before.get("scenarios", {}), after.get("scenarios", {}), "", findings)

    if not findings:
        print(f"No differences (rtol={RTOL}, atol={ATOL}).")
        return 0

    print(f"{len(findings)} difference(s) (rtol={RTOL}, atol={ATOL}):\n")
    for line in findings:
        print(f"  {line}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write a capture to this JSON path")
    ap.add_argument(
        "--diff", nargs=2, metavar=("BEFORE", "AFTER"), help="compare two captures"
    )
    ap.add_argument("--only", nargs="+", help="run only these scenarios")
    ap.add_argument(
        "--skip-vdb",
        action="store_true",
        help="skip scenarios needing a baked .vdb SDF grid",
    )
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args()

    if args.list:
        for name, _, needs_vdb in SCENARIOS:
            print(f"  {name}{'   (needs .vdb)' if needs_vdb else ''}")
        return 0

    if args.diff:
        return diff(*args.diff)

    if not args.out:
        ap.error("one of --out, --diff or --list is required")

    print("Capturing baseline:")
    payload = capture(args.only, args.skip_vdb)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
