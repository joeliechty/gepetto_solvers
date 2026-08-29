#!/usr/bin/env python3
"""FK sweep: live animation of a warm-started wrist + flexor sweep.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.fk_5f_sweep
"""

from gepetto_solvers.projects.grasp_pipeline.fk_5f_sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
