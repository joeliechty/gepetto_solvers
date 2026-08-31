#!/usr/bin/env python3
"""Browse the YCB catalog and author ellipsoid decompositions (viser GUI).

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.core.objects.ycb.browser
"""

from gepetto_solvers.core.objects.ycb.browser import main

if __name__ == "__main__":
    raise SystemExit(main())
