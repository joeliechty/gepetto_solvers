"""Where the support plane sits: seated in HEIGHT from the object, fixed in-plane.

The split is the whole point of :func:`solvers.auto_table_origin`. An object
resting on a table must ride the table's height, and sliding that object ACROSS
the table must leave the table -- and therefore its corner frame, its calibration
grid and the robot registration hung off them -- exactly where it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import scene, solvers


def _params(center, **kw):
    p = solvers.HandSolveParams()
    p.primitive = "big_sphere"
    p.object_center = np.asarray(center, float)
    p.plane_normal = np.array(scene.TABLE_NORMAL, float)
    p.plane_origin = None
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _origin(center, **kw):
    p = _params(center, **kw)
    spec, obj_center, _rot, _pose = solvers.resolve_scene(p)
    return np.asarray(solvers.auto_table_origin(p, spec, obj_center), float)


def test_moving_the_object_in_plane_leaves_the_table_alone():
    base = np.array(scene.GRASP_SPHERE_CENTER, float)
    origin = _origin(base)
    for offset in ([0.07, 0.0, 0.0], [0.0, -0.11, 0.0], [0.05, 0.09, 0.0]):
        np.testing.assert_allclose(_origin(base + offset), origin, atol=1e-12)


def test_moving_the_object_along_the_normal_carries_the_table():
    base = np.array(scene.GRASP_SPHERE_CENTER, float)
    origin = _origin(base)
    lifted = _origin(base + [0.0, 0.0, 0.04])
    np.testing.assert_allclose(lifted - origin, [0.0, 0.0, 0.04], atol=1e-12)


def test_the_slab_is_anchored_in_plane_whatever_the_object():
    """In-plane the origin is TABLE_ANCHOR, height and object notwithstanding."""
    origin = _origin(np.array(scene.GRASP_SPHERE_CENTER, float) + [0.3, -0.2, 0.5])
    np.testing.assert_allclose(origin[:2], scene.TABLE_ANCHOR[:2], atol=1e-12)


def test_the_object_still_rests_on_the_table():
    """burial=0 puts the top face tangent to the object's underside."""
    center = np.array(scene.GRASP_SPHERE_CENTER, float) + [0.06, 0.06, 0.0]
    p = _params(center, table_burial=0.0)
    spec, obj_center, rot, _pose = solvers.resolve_scene(p)
    origin = np.asarray(solvers.auto_table_origin(p, spec, obj_center), float)
    half = scene.object_extent_along(spec, np.array([0.0, 0.0, 1.0]), rot)
    assert origin[2] == pytest.approx(center[2] - half, abs=1e-12)


def test_burial_of_a_half_puts_the_plane_through_the_centroid():
    center = np.array(scene.GRASP_SPHERE_CENTER, float) + [-0.08, 0.02, 0.0]
    origin = _origin(center, table_burial=0.5)
    assert origin[2] == pytest.approx(center[2], abs=1e-12)


def test_the_corner_moves_only_with_the_height():
    """The registration landmark inherits the same split (``_corner_viz``)."""
    normal = np.array(scene.TABLE_NORMAL, float)
    base = np.array(scene.GRASP_SPHERE_CENTER, float)
    corner = scene.table_corner(_origin(base), normal)
    slid = scene.table_corner(_origin(base + [0.1, 0.1, 0.0]), normal)
    np.testing.assert_allclose(slid, corner, atol=1e-12)
