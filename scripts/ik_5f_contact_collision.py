#!/usr/bin/env python3
"""Grasp plus collision together.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.ik_5f_contact_collision
"""

from gepetto_solvers.projects.grasp_pipeline.ik_5f_contact_collision import main

if __name__ == "__main__":
    raise SystemExit(main())
