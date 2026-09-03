"""Make a checkout's object data complete: bake every object's exact form.

Every object in the registry carries an ellipsoid form and an exact one. The
first is derived or committed; the second is a baked OpenVDB grid, which is
build output and is not in version control -- roughly 50 MB of it, from ~0.6 GB
of YCB scans that are not committed either. So a fresh clone can plan approaches
and cannot contact exact geometry, and nothing about that is discoverable until
a phase turns out to be unavailable.

This module is what closes that gap, and it is deliberately the ONLY documented
way to do so. Baking grids one at a time invites the failure this whole area is
prone to: a grid produced with a different voxel size, band or fillet than its
siblings is indistinguishable from a correct one until a solve behaves oddly
against that one object.

Its exit status answers "is this machine set up" -- it finishes by checking the
invariant rather than by reporting what it happened to do.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from . import OBJECTS_DIR, has_exact_form, names_exact_form, vdb_path
from .sdf import DEFAULT_VOXEL_SIZE


def _specs():
    from ..geometry.scene import get_primitive_specs
    return get_primitive_specs()


def _needs_download(spec) -> bool:
    """A YCB object whose scan is not in the local cache yet.

    Separated from "not baked" because the two have different costs and
    different remedies: baking is a minute of CPU, fetching is a download that
    the user may not want right now."""
    if "ycb" not in spec:
        return False
    try:
        return not _cache().is_cached(spec["ycb"],
                                      spec.get("source") or "google_16k")
    except Exception:
        # No `ycb` extra installed, or a source this catalog does not list.
        # Conservative answer: the bake itself will say so precisely if one is
        # actually attempted, and a status query must not raise.
        return True


def _cache():
    """One shared :class:`YcbCache`, since building it parses the catalog."""
    global _CACHE
    if _CACHE is None:
        from .ycb.data import DEFAULT_CACHE, Catalog, YcbCache
        _CACHE = YcbCache(Catalog(), DEFAULT_CACHE)
    return _CACHE


_CACHE = None


def check(specs=None):
    """``(complete, pending, blocked)`` -- object names by state.

    ``pending`` can be baked right now; ``blocked`` needs a scan downloaded
    first."""
    specs = specs if specs is not None else _specs()
    complete, pending, blocked = [], [], []
    for name, spec in sorted(specs.items()):
        if not names_exact_form(spec):
            continue          # nothing to bake; not a failure
        if has_exact_form(spec):
            complete.append(name)
        elif _needs_download(spec):
            blocked.append(name)
        else:
            pending.append(name)
    return complete, pending, blocked


def _bake_one(name, voxel_size):
    """Bake one object by NAME, re-resolving its spec.

    Takes a name rather than a spec so it can be handed to a worker process:
    specs carry lambdas (their ``plot`` entry) and do not pickle."""
    from .sdf import bake_spec
    spec = _specs()[name]
    started = time.time()
    path = bake_spec(name, spec, voxel_size=voxel_size)
    return name, str(path), os.path.getsize(path), time.time() - started


def _require_tools(need_ycb):
    """Fail before the first bake, not partway through it.

    A run of this is long enough that discovering a missing dependency at object
    60 is a genuinely different experience from discovering it at object 0."""
    from .sdf import require_openvdb
    require_openvdb()
    if not need_ycb:
        return
    missing = [m for m in ("trimesh", "requests")
               if not _importable(m)]
    if missing:
        raise ImportError(
            f"baking the scanned objects needs {', '.join(missing)}: "
            f"pip install '.[ycb]'")


def _importable(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


def _report(complete, pending, blocked, total_bytes=0):
    print(f"\n{len(complete)} object(s) have their exact form"
          + (f"  ({total_bytes / 1e6:.0f} MB on disk)" if total_bytes else ""))
    if pending:
        print(f"{len(pending)} not baked: {', '.join(pending[:6])}"
              + (" ..." if len(pending) > 6 else ""))
    if blocked:
        print(f"{len(blocked)} need their scan downloaded first "
              f"(re-run with --download): {', '.join(blocked[:6])}"
              + (" ..." if len(blocked) > 6 else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="report what is missing and exit, baking nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-bake objects that already have a grid")
    parser.add_argument("--objects", nargs="+", metavar="NAME",
                        help="bake only these (registry names, e.g. cube "
                             "ycb:010_potted_meat_can)")
    parser.add_argument("--download", action="store_true",
                        help="fetch the YCB scans that are not cached yet "
                             "(~0.6 GB for the full set)")
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE,
                        help=f"grid resolution in metres (default "
                             f"{DEFAULT_VOXEL_SIZE}). Changing it changes the "
                             f"finite-difference steps every contact factor "
                             f"takes, so bake the whole set at one value")
    parser.add_argument("--jobs", type=int, default=1,
                        help="bake this many objects in parallel")
    args = parser.parse_args(argv)

    specs = _specs()
    if args.objects:
        unknown = [n for n in args.objects if n not in specs]
        if unknown:
            parser.error(f"unknown object(s): {', '.join(unknown)}")
        specs = {n: specs[n] for n in args.objects}

    complete, pending, blocked = check(specs)
    if args.check:
        _report(complete, pending, blocked, _disk_usage(specs))
        return 0 if not (pending or blocked) else 1

    todo = list(pending) + (list(complete) if args.force else [])
    if args.download:
        todo += blocked
        blocked = []
    if not todo:
        print("Nothing to do -- every requested object already has its grid.")
        _report(complete, pending, blocked, _disk_usage(specs))
        return 0 if not blocked else 1

    _require_tools(need_ycb=any("ycb" in specs[n] for n in todo))
    print(f"Baking {len(todo)} object(s) at a {args.voxel_size * 1000:g} mm "
          f"voxel into {OBJECTS_DIR}")

    failures = _run(sorted(todo), args.voxel_size, args.jobs)

    complete, pending, blocked = check(_specs())
    _report(complete, pending, blocked, _disk_usage(_specs()))
    if failures:
        print(f"\n{len(failures)} object(s) failed:")
        for name, why in failures:
            print(f"  {name}: {why}")
    # The invariant, not a tally of what this run happened to do: a machine is
    # set up when every object that names an exact form has one.
    return 0 if not (pending or blocked or failures) else 1


def _run(todo, voxel_size, jobs):
    """Bake ``todo``, serially or across a process pool. Returns the failures.

    One object's failure never stops the rest: a scan that will not close or a
    primitive with an unroundable feature is a fact about that object, and
    aborting the run would leave the other hundred unbaked for it."""
    failures = []

    def done(result):
        name, path, size, secs = result
        print(f"  {name:34s} {size / 1e6:6.1f} MB  {secs:5.1f}s  "
              f"{os.path.basename(path)}")

    if jobs <= 1:
        for name in todo:
            try:
                done(_bake_one(name, voxel_size))
            except Exception as exc:          # noqa: BLE001 - reported, not raised
                failures.append((name, str(exc)))
                print(f"  {name:34s} FAILED: {exc}")
        return failures

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_bake_one, n, voxel_size): n for n in todo}
        for future, name in ((f, futures[f]) for f in futures):
            try:
                done(future.result())
            except Exception as exc:          # noqa: BLE001 - reported, not raised
                failures.append((name, str(exc)))
                print(f"  {name:34s} FAILED: {exc}")
    return failures


def _disk_usage(specs):
    total = 0
    for spec in specs.values():
        if has_exact_form(spec):
            total += os.path.getsize(vdb_path(spec))
    return total
