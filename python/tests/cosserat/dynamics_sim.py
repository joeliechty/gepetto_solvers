import time
import numpy as np

import crest_sparse
from .._plotting.cosserat_rod_plotter import CosseratRodPlotter


def get_K_inv(rod_diameter=0.003, youngs_modulus=45.0e9, shear_modulus=18.0e9):
    radius = rod_diameter / 2.0
    area = np.pi * radius**2
    moment = np.pi * radius**4 / 4.0
    polar_moment = 2.0 * moment

    k_bending = youngs_modulus * moment
    k_torsion = shear_modulus * polar_moment
    k_shear = shear_modulus * area
    k_extension = youngs_modulus * area

    # shear_scale = 0.03
    # ext_scale = 0.03

    # k_shear *= shear_scale
    # k_extension *= ext_scale

    return np.diag(
        1.0 / np.array([
            k_bending, 
            k_bending, 
            k_torsion, 
            k_shear, 
            k_shear, 
            k_extension
        ])
    )


def main():
    config = crest_sparse.CosseratRodDynamicsConfig()
    config.rod_config.rod_length = 0.3
    config.rod_config.num_nodes = 12
    config.delta_initial = 1.0e-3
    config.rod_config.K_inv = get_K_inv()
    config.rod_config.sigma_twist_pos = 1.0e-5
    config.rod_config.sigma_twist_rot = 1.0e-3
    config.rod_config.sigma_small_force = 1.0e-3
    config.rod_config.sigma_small_moment = 1.0e-4
    config.rod_config.sigma_base_pose_pos = 1.0e-5
    config.rod_config.sigma_base_pose_rot = 1.0e-3

    config.num_time_steps = 100
    config.dt = 0.1
    config.linear_damping = 0.5
    config.rotational_damping = 1e-2
    config.linear_inertia = 1
    config.rotational_inertia = 1e-2
    config.initial_tip_wrench = np.array([0, 0.0, 0, -0.1, 0, 0])

    solver = crest_sparse.CosseratRodDynamicsSolver(config)
    solution = solver.solve()

    plotter = CosseratRodPlotter(
        plot_wrenches=False,
        plot_backbone_frames=True,
        plot_internal_wrenches=True,
        camera_azimuth=60, 
        camera_distance=1.5, 
        camera_focal_point=np.array([0, 0, 0.25]))

    print(f"iter:  {solution.meta.iterations}\n"
          f"err:   {solution.meta.error:.3e}\n"
          f"build: {solution.meta.build_time_ms:.2f}\n"
          f"opt:   {solution.meta.optimize_time_ms:.2f}\n"
          f"total: {solution.meta.total_time_ms:.2f}\n")

    for rod_t in solution.marginals.rods_t:
        plotter.update(rod_t)
        time.sleep(0.1)

if __name__ == "__main__":
    main()