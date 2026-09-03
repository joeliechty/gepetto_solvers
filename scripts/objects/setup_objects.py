#!/usr/bin/env python3
"""Bake every object's exact (SDF) form, so this checkout is ready to run.

Run once per machine; `--check` reports what is missing without baking. Thin
CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.core.objects.setup
"""

from gepetto_solvers.core.objects.setup import main

if __name__ == "__main__":
    raise SystemExit(main())
