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


def get_base_config():
    config = crest_sparse.CosseratRodDynamicsConfig()

    rod_diameter = 0.003
    density = 6500.0
    total_time = 1.0
    frame_rate = 100

    config.rod_length = 1.0
    config.num_nodes = 20
    config.K_inv = get_K_inv(rod_diameter)

    config.dynamics_noise_sigma = 1e-4
    config.twist_noise_sigma = 1e-4
    config.wrench_noise_sigma = 1e-4

    config.init_velocity_mean = 10.0
    config.init_velocity_sigma = 1e-4
    config.init_velocity_idx = 5

    config.num_time_steps = int(frame_rate * total_time)
    config.dt = 1.0 / frame_rate

    config.linear_damping = 0.005
    config.rotational_damping = 1e-5

    segment_radius = rod_diameter / 2
    segment_area = np.pi * segment_radius**2
    segment_length = config.rod_length / config.num_nodes
    segment_mass = density * segment_area * segment_length

    config.linear_inertia = segment_mass
    config.rotational_inertia = 1.0 / 12.0 * config.linear_inertia * (3 * segment_radius ** 2 + segment_length ** 2)

    return config


def solve():
    config = get_base_config()
    solution = crest_sparse.CosseratRodDynamicsSolver(config).solve()

    return solution, config


def render_movie(solution, save=False):
    plotter = CosseratRodPlotter(
        save_frames_dir_name="cosserat_dynamics" if save else None,
        plot_wrenches=False,
        plot_backbone_frames=True,
        plot_internal_wrenches=False,
        backbone_radius=0.005,
        camera_azimuth=60, 
        camera_distance=2.5, 
        camera_focal_point=np.array([0, 0, 0.5]))

    for s in solution.marginals.rods_t:
        s.meta = solution.meta
        plotter.update(s)
    

def make_pde_heatmaps(t, s, x_mean, x_sigma):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharey=True, sharex=True)

    im0 = axes[0].imshow(
        x_mean.T,
        extent=[t[0], t[-1], s[0], s[-1]],
        origin="lower",
        aspect="auto",
        cmap="bwr",
    )

    axes[0].set_ylabel("arc length (m)")

    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        x_sigma.T,
        extent=[t[0], t[-1], s[0], s[-1]],
        origin="lower",
        aspect="auto",
        cmap="binary",
    )

    axes[1].set_xlabel("time (sec)")
    axes[1].set_ylabel("arc length (m)")

    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig("pde_heatmaps.png", dpi=300)
    plt.close()


def make_tip_response_plot(t, x_tip_mean, x_tip_sigma):
    plt.figure(figsize=(5, 3))

    plt.plot(t, x_tip_mean, label="mean")

    plt.fill_between(
        t, 
        x_tip_mean - 2.0 * x_tip_sigma,
        x_tip_mean + 2.0 * x_tip_sigma,
        alpha=0.3,
        label="2-σ"
    )
    
    plt.xlabel("time (sec)")
    plt.ylabel("tip x position (m)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("tip_response.png", dpi=300)
    plt.close()


def plot_solution(solution, config):
    x_mean = []
    x_sigma = []

    for rod_t in solution.marginals.rods_t:
        x_s_mean = [state.pose.mean[0,3] for state in rod_t.marginals.states]
        x_s_sigma = [np.sqrt(state.pose.cov[3,3]) for state in rod_t.marginals.states]
        
        x_mean.append(x_s_mean)
        x_sigma.append(x_s_sigma)
        
    x_mean = np.array(x_mean)
    x_sigma = np.array(x_sigma)

    num_t, num_s = x_mean.shape
    s = np.linspace(0, config.rod_length, num_s)
    t = np.linspace(0, config.dt * config.num_time_steps, num_t)

    make_pde_heatmaps(t, s, x_mean, x_sigma)
    make_tip_response_plot(t, x_mean[:,-1], x_sigma[:,-1])


def main():
    # solution, config = solve([0, 0, 0, -1, 0, 0])  # one mode of oscillation
    solution, config = solve()  # two modes of oscillation
    # solution = solve([0.3, 0.3, 0.8, 0, 0, 0])  # weird initial z moment

    render_movie(solution, save=False)
    plot_solution(solution, config)

    
if __name__ == "__main__":
    main()