#!/usr/bin/env python3
"""Read-only Augmented Lagrangian trace dumper and parameter sweeper.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.experimental.traces.debug_al_trace
"""

from gepetto_solvers.experimental.traces.debug_al_trace import main

if __name__ == "__main__":
    raise SystemExit(main())
