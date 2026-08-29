"""Suite-wide fixtures and the sys.path bootstrap.

Two things every test in this tree depends on:

1. The repo root on ``sys.path``. Before the refactor the application layer is the
   implicit namespace package ``python.tests.tendon_hand``, importable only from the
   repo root. After the refactor ``gepetto_solvers`` is installed and this is inert.

2. Pinned hand dimensions. See :func:`pinned_dims`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# tests/_pkg.py is imported by name from every test module.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def pinned_dims(monkeypatch):
    """Force the bundled ``DEFAULT_HAND_DIMENSIONS``, ignoring ``gepetto_core``.

    ``config.load_hand_dimensions()`` prefers ``gepetto_core.geometry.HandGeometry``
    and silently falls back to the bundled constant when that import fails. The two
    are NOT the same hand: the middle finger's first joint diameter is 9.8 mm from
    the CAD and 14.0 mm in the fallback. Any test asserting a committed number would
    therefore pass or fail depending on whether ``gepetto_core`` happens to be
    installed, which is exactly the hermeticity CLAUDE.md section 7 rules out.

    Pinning the fallback is the right direction: it is the copy that ships with this
    repo, so it is the one a checkout can reproduce anywhere.

    The patch is applied to every module that imported the function into its own
    namespace (``from .config import load_hand_dimensions`` binds a new name, so
    patching ``config`` alone would not reach ``solvers``).
    """
    from _pkg import config, solvers

    def _fallback():
        return config.DEFAULT_HAND_DIMENSIONS

    monkeypatch.setattr(config, "load_hand_dimensions", _fallback)
    monkeypatch.setattr(solvers, "load_hand_dimensions", _fallback, raising=False)
    return config.DEFAULT_HAND_DIMENSIONS


@pytest.fixture
def hand_configs(pinned_dims):
    """The anatomical 5-digit hand, built from the pinned dims."""
    from _pkg import config

    return config.get_default_hand_configs(pinned_dims)


def assert_allclose(actual, desired, rtol=1e-9, atol=1e-12, err_msg=""):
    """``np.testing.assert_allclose`` with tolerances suited to these solvers."""
    np.testing.assert_allclose(
        np.asarray(actual, float),
        np.asarray(desired, float),
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
    )
