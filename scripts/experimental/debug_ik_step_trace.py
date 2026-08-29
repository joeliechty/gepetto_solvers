#!/usr/bin/env python3
"""Headless replay of the GUI Step button, one AL iteration at a time.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.experimental.traces.debug_ik_step_trace
"""

from gepetto_solvers.experimental.traces.debug_ik_step_trace import main

if __name__ == "__main__":
    raise SystemExit(main())
