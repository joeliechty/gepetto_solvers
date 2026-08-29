#!/usr/bin/env python3
"""Interactive viser workbench (--smoke for the headless self-check).

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.viz.viz_interactive
"""

from gepetto_solvers.projects.viz.viz_interactive import main

if __name__ == "__main__":
    raise SystemExit(main())
