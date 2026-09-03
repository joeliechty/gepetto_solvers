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
    """Force the bundled ``DEFAULT_HAND_DIMENSIONS``, ignoring ``epfl_hand_core``.

    ``config.load_hand_dimensions()`` prefers ``epfl_hand_core.geometry.HandGeometry``
    and silently falls back to the bundled constant when that import fails. The two
    are NOT the same hand: the middle finger's first joint diameter is 9.8 mm from
    the CAD and 14.0 mm in the fallback. Any test asserting a committed number would
    therefore pass or fail depending on whether ``epfl_hand_core`` happens to be
    installed, which is exactly the hermeticity CLAUDE.md section 7 rules out.

    Pinning the fallback is the right direction: it is the copy that ships with this
    repo, so it is the one a checkout can reproduce anywhere.

    Nothing needs patching any more: a hand takes its dimensions as a
    constructor argument (see :func:`pinned_hand`), so a test that wants the
    bundled ones asks for them rather than intercepting a lookup. This fixture
    is the dimension TABLE, for tests that assert on it directly.
    """
    from _pkg import config

    return config.DEFAULT_HAND_DIMENSIONS


@pytest.fixture
def pinned_hand(pinned_dims):
    """A :class:`TendonHand5F` built from the bundled dimensions.

    The hand to hand a solver in any test that asserts a measured number. Pass
    it as ``HandFKSolver(params, pinned_hand)``; without it the solver builds the
    DEFAULT hand, whose dimensions come from ``epfl_hand_core`` when that is
    installed and from the bundled table when it is not -- two different hands,
    so the assertion would pass or fail depending on the machine.
    """
    from _pkg import config

    return config.TendonHand5F(pinned_dims)


@pytest.fixture
def hand_configs(pinned_hand):
    """The anatomical 5-digit hand's per-digit solver configs, pinned."""
    return pinned_hand.digit_configs()


def assert_allclose(actual, desired, rtol=1e-9, atol=1e-12, err_msg=""):
    """``np.testing.assert_allclose`` with tolerances suited to these solvers."""
    np.testing.assert_allclose(
        np.asarray(actual, float),
        np.asarray(desired, float),
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
    )
