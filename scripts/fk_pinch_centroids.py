#!/usr/bin/env python3
"""Brute-force where every thumb-opposition digit combination meets.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.grasp_pipeline.fk_pinch_centroids
"""

from gepetto_solvers.projects.grasp_pipeline.fk_pinch_centroids import main

if __name__ == "__main__":
    raise SystemExit(main())
