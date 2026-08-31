#!/usr/bin/env python3
"""Cosserat rod dynamics simulation.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.cosserat_demo.dynamics_sim
"""

from gepetto_solvers.projects.cosserat_demo.dynamics_sim import main

if __name__ == "__main__":
    raise SystemExit(main())
