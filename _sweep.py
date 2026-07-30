"""Is the phase-1 plateau the object footprint (fingers land ON the ball) rather
than the wrist trust region? Compare objects, and a base shifted clear."""
import numpy as np

from python.tests.tendon_hand.scene import primitive_surface_gap
from python.tests.tendon_hand.config import tip_node_index, disc_node_indices
from python.tests.tendon_hand.solvers import (
    HandControllerSolver, HandSolveParams, capabilities, free_space_start_pose,
    resolve_table_origin)
from python.tests.tendon_hand.viz_controller import (
    pregrasp_flexor, start_orientation)
from python.tests.tendon_hand.viz_interactive import FINGER_LABELS


def run(tag, primitive, ticks=15, shift=None, **over):
    p = HandSolveParams()
    p.primitive = primitive
    p.collision = True
    p.table = True
    p.flexor_tensions = [pregrasp_flexor(p)] * len(FINGER_LABELS)
    p.wrist_pose, _ = free_space_start_pose(p, orientation=start_orientation(),
                                            center_on_object=True)
    if shift is not None:
        p.wrist_pose = np.array(p.wrist_pose, float)
        p.wrist_pose[:3, 3] += np.asarray(shift, float)
    p.phase = 1
    p.sigma_wrist_pos_step = over.pop("sigma_wrist_pos_step", 1e-1)
    p.sigma_wrist_rot_step = over.pop("sigma_wrist_rot_step", 1.0)
    for k, v in over.items():
        setattr(p, k, v)

    s = HandControllerSolver(p)
    origin = np.asarray(resolve_table_origin(p, s.spec, s.object_center), float)
    n = np.asarray(p.plane_normal, float); n /= np.linalg.norm(n)
    T0 = np.array(p.wrist_pose, float)
    for _ in range(ticks):
        res = s.step()
    T = np.array(s._base_pose, float)
    frame = res.frames[0]
    hs, worst_o = [], np.inf
    for (name, cfg), tip_r in zip(s.configs, s.tip_radii):
        fm = frame[name].marginals
        pos = np.array(fm.rod.states[tip_node_index(cfg)].pose.mean, float)[:3, 3]
        hs.append(float((pos - origin) @ n) - tip_r)
        for nd in disc_node_indices(cfg):
            if nd == 0:
                continue
            q = np.array(fm.rod.states[nd].pose.mean, float)[:3, 3]
            worst_o = min(worst_o, primitive_surface_gap(
                s.object_rotation.T @ (q - s.object_center), s.spec)
                - p.collision_radius)
    v = dict(s.phase_violations())
    print(f"{tag:<40} sup={v['support_equality']:.4f} m  "
          f"tips above plane [{', '.join(f'{h*1000:+5.1f}' for h in hs)}] mm  "
          f"obj-clear {worst_o*1000:+6.2f} mm  base {np.linalg.norm(T[:3,3]-T0[:3,3])*1000:5.1f} mm")


if __name__ == "__main__":
    print("== object size (loose wrist 1e-1 / 1.0, 15 ticks) ==")
    for prim in ("mid_sphere_ellipsoid", "small_sphere_ellipsoid", "coin"):
        run(prim, prim)
    print("\n== mid_sphere, base pre-shifted clear of the footprint ==")
    for dx in (-0.02, -0.04, -0.06):
        run(f"shift x {dx:+.2f} m", "mid_sphere_ellipsoid", shift=(dx, 0, 0))
    print("\n== mid_sphere, thumb+index pinch only ==")
    run("contact 1,0,0,0,1", "mid_sphere_ellipsoid",
        contact_fingers=[True, False, False, False, True])
