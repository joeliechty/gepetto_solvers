#!/usr/bin/env python3
"""Collision only, no contact -- isolates the inequality path.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.ik_5f_collision
"""

from gepetto_solvers.projects.grasp_pipeline.ik_5f_collision import main

if __name__ == "__main__":
    raise SystemExit(main())
