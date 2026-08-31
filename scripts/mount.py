#!/usr/bin/env python3
"""Offline self-check of the hand-to-robot mounting transform.

Thin CLI wrapper; all arguments are handled by the implementation in::

    gepetto_solvers.projects.robot_mount.mount
"""

from gepetto_solvers.projects.robot_mount.mount import main

if __name__ == "__main__":
    raise SystemExit(main())
