#!/usr/bin/env python3
"""Single-shot five-finger grasp.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.ik_5f_contact
"""

from gepetto_solvers.projects.grasp_pipeline.ik_5f_contact import main

if __name__ == "__main__":
    raise SystemExit(main())
