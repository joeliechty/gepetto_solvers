#!/usr/bin/env python3
"""K+1-step grasp trajectory with GP temporal priors.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.traj_5f_contact
"""

from gepetto_solvers.projects.grasp_pipeline.traj_5f_contact import main

if __name__ == "__main__":
    raise SystemExit(main())
