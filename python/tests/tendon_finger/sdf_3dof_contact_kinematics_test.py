import os
import argparse
import numpy as np
import time

import crest_sparse
from .._plotting.tendon_finger_plotter import TendonFingerPlotter

from .config import get_6tendon_config


# Object center, shared by all primitives: the p2p goal position used in
# point_to_point_planning.py, mirrored across x=0 (X negated). The 6-tendon
# routing was rotated 180 deg about the finger axis to match the gepetto_core CAD
# convention, which flips the flexor curl from world +X to -X; the object moves
# with it. The SDF lives at the VDB local origin (see the _objects/make_*.py
# generators); we place it in the world by translating the object pose to this center.
OBJECT_CENTER = np.array([-6.02088876e-02, 3.77734425e-02, 0.0])


def Rx(theta):
    """Rotation matrix about the X axis (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


# Registry of supported object primitives. "vdb" is the level-set file produced
# by the matching _objects/make_*.py script; the geometry fields must match the
# parameters those scripts were generated with. "plot" describes how the
# TendonFingerPlotter should render the primitive.
def get_primitive_specs():
    return {
        "sphere": {
            "type": "sphere",
            "vdb": "sphere.vdb",         # make_sphere.py (radius 0.025)
            "radius": 0.025,
            "plot": lambda c: {"type": "sphere", "center": c, "radius": 0.025},
        },
        "cylinder": {
            "type": "cylinder",
            "vdb": "cylinder.vdb",       # make_cylinder.py (radius 0.025, height 0.04, local Y axis)
            "radius": 0.025,
            "height": 0.04,
            # Rotate the (local Y-aligned) cylinder 90 deg about X so its axis is
            # vertical (world +Z). The finger moves in the z~0 plane, so it
            # contacts the curved side of this upright cylinder (radius 0.025 from
            # the center axis -- same reach as the sphere, so it's touchable).
            "rotation": Rx(np.pi / 2),
            "plot": lambda c: {"type": "cylinder", "center": c,
                               "radius": 0.025, "height": 0.04,
                               "direction": (0.0, 0.0, 1.0)},
        },
        "cube": {
            "type": "cube",
            # half_extents match the cylinder's footprint (radius 0.025 in X/Z,
            # half-height 0.02 in Y) so the finger contacts the flat +Y face the
            # same way it does the cylinder's flat cap.
            "vdb": "cube.vdb",           # make_cube.py (half_extents 0.025, 0.02, 0.025)
            "half_extents": (0.025, 0.02, 0.025),
            "plot": lambda c: {"type": "cube", "center": c,
                               "extents": (0.05, 0.04, 0.05)},
        },
    }


def primitive_surface_gap(p_local, spec):
    """Analytic signed distance from a point (in the object's local frame) to
    the primitive surface. Mirrors the SDFs in the _objects/make_*.py scripts so
    we can report the achieved contact gap independently of the solver."""
    ptype = spec["type"]
    if ptype == "sphere":
        return float(np.linalg.norm(p_local) - spec["radius"])
    if ptype == "cylinder":
        # Axis along Y.
        r = spec["radius"]
        half_h = spec["height"] / 2.0
        dist_xz = np.hypot(p_local[0], p_local[2])
        dx = dist_xz - r
        dy = abs(p_local[1]) - half_h
        out_dist = np.hypot(max(dx, 0.0), max(dy, 0.0))
        in_dist = min(max(dx, dy), 0.0)
        return float(out_dist + in_dist)
    if ptype == "cube":
        hx, hy, hz = spec["half_extents"]
        d = np.abs(p_local) - np.array([hx, hy, hz])
        out_dist = np.linalg.norm(np.maximum(d, 0.0))
        in_dist = min(max(d[0], max(d[1], d[2])), 0.0)
        return float(out_dist + in_dist)
    raise ValueError(f"Unknown primitive type: {ptype!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve the SDF tip-contact kinematics against a primitive object.")
    parser.add_argument(
        "primitive", nargs="?", default="sphere",
        choices=["sphere", "cylinder", "cube"],
        help="Object primitive to load the SDF for and solve contact against.")
    args = parser.parse_args()

    primitive_specs = get_primitive_specs()
    spec = primitive_specs[args.primitive]

    config = get_6tendon_config()
    num_tendons = config.num_tendons  # 6

    # Tip node index (last rod node). The C++ side also accepts -1 as a tip
    # alias via CosseratRodModel::clamp_node_idx; we compute the explicit
    # index here for use as the contact target and as the plotter's
    # contact_node_index too.
    num_nodes = config.num_discs + (config.num_discs - 1) * config.num_between_nodes
    tip_node_index = num_nodes - 1

    tip_radius = 0.003

    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python -m tests._objects.make_{args.primitive} (run from the python/ dir).")

    # Use the SDF-backed 3-residual SdfContactFactor ([c_R, c_O, c_N]) with an
    # explicit dummy witness point, instead of the analytic sphere contact. This
    # is the SDF counterpart of SphereSphereWitnessContactFactor: e1 is the same
    # finger-object tangency, but the object surface and its outward normal come
    # from the SDF (value + normalized gradient) rather than a closed-form
    # primitive. EnvironmentConfig is the carrier for sdf_contact.
    env = crest_sparse.EnvironmentConfig()
    env.load_sdf(vdb_path)

    # Optional fixed rotation that orients the SDF asset in the world (e.g. the
    # cylinder is stored Y-aligned but placed with its axis vertical).
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[0:3, 0:3] = object_rotation
    object_pose[0:3, 3] = OBJECT_CENTER
    env.object_pose_mean = object_pose
    env.object_pose_cov = 1e-8 * np.eye(6)   # rigidly anchored
    env.object_pose_per_step = False

    # Terminal contact: the tip sphere (radius tip_radius) must land tangent to
    # the SDF surface. Wrapped as a hard AL equality constraint on the tip node.
    env.target_contact_node = tip_node_index
    env.contact_node_radius = tip_radius

    config.sdf_contact = env

    # Contact is a hard equality constraint solved with GTSAM's Augmented
    # Lagrangian optimizer (auto-enabled whenever sdf_contact is set): it drives
    # all three residuals ([c_R, c_O, c_N]) to ~0 exactly, instead of
    # approximating contact with a tight covariance. Convergence is governed by
    # the AL params on config.base. A gradual penalty schedule
    # (al_mu_increase_rate=2) run for enough outer iterations converges to a
    # ~micron gap; growing the penalty too fast (rate >= 5) trips the
    # relative-convergence check early and stalls the finger short of the surface.
    config.base.al_initial_mu = 1.0
    config.base.al_mu_increase_rate = 2.0
    config.base.al_max_iterations = 40

    solver = crest_sparse.TendonFingerSolver(config)

    # Pin passive tendons tight, leave the flexor (index 5) loose so the
    # optimizer can adjust it freely to satisfy the contact factor. Mirrors
    # the goal-tension covariance used in point_to_contact_planning.py.
    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    tensions_cov = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
    tip_wrench = crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)

    t0 = time.time()
    solution = solver.solve(tensions, tip_wrench, None)
    dt_ms = (time.time() - t0) * 1000.0

    tip_pose = np.array(solution.marginals.rod.states[-1].pose.mean)
    tip_pos = tip_pose[:3, 3]
    # Signed gap between the tip sphere surface and the object surface: the
    # object SDF value at the tip center (object-local frame) minus the tip
    # radius. ~0 means tangent contact. Map the world tip into the object's
    # local frame (R^T (p - t)) so the analytic SDF (defined in local coords)
    # is valid even when the object is rotated.
    tip_local = object_rotation.T @ (tip_pos - OBJECT_CENTER)
    gap = primitive_surface_gap(tip_local, spec) - tip_radius
    print(f"Primitive: {args.primitive}")
    print(f"Solved in {dt_ms:.1f} ms | iters={solution.meta.iterations} | error={solution.meta.error:.4g}")
    print(f"  tip position: {tip_pos}")
    print(f"  signed surface gap (sdf(tip) - tip_radius): {gap:+.5f} m  (target ~0)")
    print(f"  active tendon (5) tension: {solution.marginals.tensions.mean[5]:.3f}")

    plotter = TendonFingerPlotter(
        plot_backbone_frames=True,
        contact_node_index=tip_node_index,
        contact_node_radius=tip_radius,
        primitives=[dict(spec["plot"](OBJECT_CENTER),
                         color="goldenrod", opacity=0.35)],
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0, 0.1, 0],
    )
    plotter.update(solution)
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
