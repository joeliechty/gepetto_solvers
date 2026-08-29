#!/usr/bin/env python3
"""Table plus --k-touch three-phase slide and grasp.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.traj_5f_slide_grasp
"""

from gepetto_solvers.projects.grasp_pipeline.traj_5f_slide_grasp import main

if __name__ == "__main__":
    raise SystemExit(main())
