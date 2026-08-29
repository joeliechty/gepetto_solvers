#!/usr/bin/env python3
"""Headless sweep of phase0 settings looking for a phase1 warm start.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.experimental.sweeps.sweep_pregrasp_pen
"""

from gepetto_solvers.experimental.sweeps.sweep_pregrasp_pen import main

if __name__ == "__main__":
    raise SystemExit(main())
