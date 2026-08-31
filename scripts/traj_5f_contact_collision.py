#!/usr/bin/env python3
"""Grasp trajectory with collision at every plannable step.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.traj_5f_contact_collision
"""

from gepetto_solvers.projects.grasp_pipeline.traj_5f_contact_collision import main

if __name__ == "__main__":
    raise SystemExit(main())
