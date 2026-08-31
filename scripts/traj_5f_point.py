#!/usr/bin/env python3
"""Terminal tip-position goals instead of contact (non-AL path).

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.traj_5f_point
"""

from gepetto_solvers.projects.grasp_pipeline.traj_5f_point import main

if __name__ == "__main__":
    raise SystemExit(main())
