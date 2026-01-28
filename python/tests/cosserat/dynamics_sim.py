import time
import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import crest_sparse
from .._plotting.cosserat_rod_plotter import CosseratRodPlotter


def get_K_inv(rod_diameter, youngs_modulus=40.0e9, shear_modulus=15.0e9):
    radius = rod_diameter / 2.0
    area = np.pi * radius**2
    moment = np.pi * radius**4 / 4.0
    polar_moment = 2.0 * moment

    k_bending = youngs_modulus * moment
    k_torsion = shear_modulus * polar_moment
    k_shear = shear_modulus * area
    k_extension = youngs_modulus * area

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


def get_config(rod_diameter, rod_length, num_nodes, density=6500.0):
    config = crest_sparse.CosseratRodDynamicsConfig()

    config.rod.rod_length = rod_length
    config.rod.num_nodes = num_nodes

    config.rod.K_inv = get_K_inv(rod_diameter)

    config.rod.sigma_twist_pos = 1.0e-4
    config.rod.sigma_twist_rot = 1.0e-3
    config.rod.sigma_small_force = 1.0e-2
    config.rod.sigma_small_moment = 1.0e-3
    config.rod.sigma_base_pose_pos = 1.0e-4
    config.rod.sigma_base_pose_rot = 1.0e-3
    
    config.num_time_steps = 100
    config.dt = 0.01
    config.linear_damping = 0.001
    config.rotational_damping = 0.0001

    segment_radius = rod_diameter / 2
    segment_area = np.pi * segment_radius**2
    segment_length = rod_length / num_nodes
    segment_mass = density * segment_area * segment_length

    config.linear_inertia = segment_mass
    config.rotational_inertia = 1.0 / 12.0 * config.linear_inertia * (3 * segment_radius ** 2 + segment_length ** 2)

    config.initial_tip_wrench = np.array([0, 0.3, 0, -0.9, 0, 0])

    return config


def main():
    rod_diameter = 0.003
    rod_length = 0.7
    num_nodes = 10
    config = get_config(rod_diameter, rod_length, num_nodes)

    solver = crest_sparse.CosseratRodDynamicsSolver(config)

    plotter = CosseratRodPlotter(
        plot_wrenches=False,
        plot_backbone_frames=True,
        plot_internal_wrenches=False,
        camera_azimuth=60, 
        camera_distance=1.7, 
        camera_focal_point=np.array([0, 0, 0.35]))

    solution = solver.solve()

    for s in solution.marginals.rods_t:
        s.meta = solution.meta
        plotter.update(s)
        time.sleep(config.dt)


    # print(f"iter:    {solution.meta.iterations}\n"
    #       f"err:     {solution.meta.error:.3e}\n"
    #       f"build:   {solution.meta.build_time_ms:.2f}\n"
    #       f"opt:     {solution.meta.optimize_time_ms:.2f}\n"
    #       f"extract: {solution.meta.extract_time_ms:.2f}\n"
    #       f"total:   {solution.meta.total_time_ms:.2f}\n")

    # x = []
    # x_sigma = []

    # for rod_t in solution.marginals.rods_t:
        
        
    #     x_s = [state.pose.mean[0,3] for state in rod_t.marginals.states]
    #     x_s_sigma = [np.sqrt(state.pose.cov[3,3]) for state in rod_t.marginals.states]
        
    #     x.append(x_s)
    #     x_sigma.append(x_s_sigma)
        
    # x = np.array(x)
    # x_sigma = np.array(x_sigma)

    # num_t, num_s = x.shape
    # s = np.linspace(0, rod_length, num_s)
    # t = np.linspace(0, num_t * config.dt, num_t)

    # T, S = np.meshgrid(t, s, indexing="xy")

    # fig = plt.figure(figsize=(8, 5))
    # ax = fig.add_subplot(111, projection="3d")

    # surf = ax.plot_surface(
    #     T, S, x.T,
    #     facecolors=plt.cm.viridis(x_sigma.T / np.max(x_sigma)),
    #     linewidth=0,
    #     antialiased=True
    # )

    # ax.set_xlabel("time (sec)")
    # ax.set_ylabel("arc length (m)")
    # ax.set_zlabel("x (m)")

    # plt.tight_layout()
    # plt.savefig("cosserat_dynamics.png", dpi=300, bbox_inches="tight")
    # plt.close()

if __name__ == "__main__":
    main()