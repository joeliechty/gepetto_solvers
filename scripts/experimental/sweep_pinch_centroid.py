#!/usr/bin/env python3
"""Verification harness for the pinch-centroid constraint.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.experimental.sweeps.sweep_pinch_centroid
"""

from gepetto_solvers.experimental.sweeps.sweep_pinch_centroid import main

if __name__ == "__main__":
    raise SystemExit(main())
