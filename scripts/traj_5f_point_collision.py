#!/usr/bin/env python3
"""Point goals plus collision.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.traj_5f_point_collision
"""

from gepetto_solvers.projects.grasp_pipeline.traj_5f_point_collision import main

if __name__ == "__main__":
    raise SystemExit(main())
