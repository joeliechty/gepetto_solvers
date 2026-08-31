#!/usr/bin/env python3
"""Measure T_flange<-wrist from an Onshape assembly.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.experimental.onshape.mount_onshape_fit
"""

from gepetto_solvers.experimental.onshape.mount_onshape_fit import main

if __name__ == "__main__":
    raise SystemExit(main())
