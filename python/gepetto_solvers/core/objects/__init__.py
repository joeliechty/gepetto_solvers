"""Object data: the baked ``.vdb`` SDF grids and the YCB ellipsoid fits.

This package is also the ONE definition of where that data lives. Every caller that
needs the grid directory imports :data:`OBJECTS_DIR` from here rather than walking
``os.path.dirname(__file__)`` upwards, which is what the modules used to do -- and
which silently broke whenever a file changed depth in the tree.

The grids themselves are NOT in version control: they total ~54 MB and are rebuilt by
the bakers in ``scripts/objects/make_*.py``, which need conda-only ``pyopenvdb``. The
analytic primitives (``coin``, ``credit_card``, ``pen``, ``*_sphere_ellipsoid``,
``megaminx``) need no grid at all, which is why the test suite uses them.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve_objects_dir() -> str:
    """Locate the grid directory, in either install mode.

    The grids are gitignored developer artifacts, so they are NOT shipped in the
    wheel -- a non-editable ``pip install .`` therefore leaves this package directory
    without them, while an editable install points straight back at the checkout that
    has them. Rather than force one install mode, look in the plausible places:

    1. ``$GEPETTO_OBJECTS_DIR`` if set -- the escape hatch for grids kept out of tree
       (they are 54 MB, and a shared machine may well want one copy).
    2. Next to this file: an editable install, or a tree that does ship them.
    3. A source checkout at or above the working directory -- the case where the
       package was installed non-editable but the demos are being run from the repo,
       which is how every documented invocation works.

    Falling through all three is not an error: every analytic primitive
    (``coin``, ``pen``, ``*_sphere_ellipsoid``, ``megaminx``) needs no grid at all.
    Only an SDF primitive does, and those already raise a pointed error naming the
    baker to run.
    """
    env = os.environ.get("GEPETTO_OBJECTS_DIR")
    if env:
        return env

    if any(_PACKAGE_DIR.glob("*.vdb")):
        return str(_PACKAGE_DIR)

    rel = Path("python") / "gepetto_solvers" / "core" / "objects"
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        candidate = base / rel
        if candidate.is_dir() and any(candidate.glob("*.vdb")):
            return str(candidate)

    return str(_PACKAGE_DIR)


#: Directory holding the baked ``.vdb`` SDF grids and the ``ycb/`` fits.
OBJECTS_DIR: str = _resolve_objects_dir()

#: Where ``ycb/browser.py`` writes its committed ellipsoid decompositions.
YCB_FITS_DIR: str = os.path.join(OBJECTS_DIR, "ycb", "fits")


def vdb_path(spec: dict) -> str:
    """Absolute path to the baked grid a primitive spec names.

    Raises ``KeyError`` for a spec with no ``vdb`` key -- callers that support
    both should check :func:`names_exact_form` first.
    """
    return os.path.normpath(os.path.join(OBJECTS_DIR, spec["vdb"]))


def names_exact_form(spec: dict) -> bool:
    """Does this object HAVE an exact (SDF) form at all, baked or not?

    A property of the object, and separate from :func:`has_exact_form` on
    purpose. Every object in the registry should answer True here; whether the
    file is actually on this machine is a different question with a different
    remedy, and conflating the two turns "you have not run the setup script" into
    "this object cannot be grasped precisely".
    """
    return "vdb" in spec


def has_exact_form(spec: dict) -> bool:
    """Is this object's baked grid present ON THIS MACHINE?

    The grids are gitignored build output, so a fresh checkout answers False for
    everything and ``scripts/objects/setup_objects.py`` is what changes that. The
    pipeline phases that contact the exact geometry gate on this, and say so by
    name rather than failing when a grid turns out to be missing.
    """
    return names_exact_form(spec) and os.path.exists(vdb_path(spec))


__all__ = ["OBJECTS_DIR", "YCB_FITS_DIR", "has_exact_form", "names_exact_form",
           "vdb_path"]
