import numpy as np
import matplotlib.pyplot as plt

from crest_sparse import TendonRobotSolver, Vector6Gaussian, VectorXGaussian, Vector3Gaussian

from .._plotting.tendon_robot_plotter import TendonRobotPlotter
from .._plotting.utils import setup_plt
from .config import get_base_config
from .utils import TipForceFunction, generate_waypoint_trajectory, GaussianProcessNoiseModel
from .benchmark import TendonRobotSolver as Benchmark


def run_sim(
        sim_time=30.0,
        frame_rate=30.0,
        plot=True,
        do_baseline=True,
        tip_force_prior_sigma=0.1,
        tensions_meas_sigma=0.01, 
        tip_position_meas_sigma=0.001):
    
    config = get_base_config()

    t, positions, tensions_nominal, _ = generate_waypoint_trajectory(sim_time, frame_rate=frame_rate, seed=7)
    tip_force_function = TipForceFunction(max_magnitude=2 * tip_force_prior_sigma, framerate=frame_rate, seed=8)

    # A solver to simulate the nominal trajectory of the robot, given open loop tensions
    simulator_nominal = TendonRobotSolver(config)
    plotter_nominal = TendonRobotPlotter(
        save_frames_dir_name='tendon_robot_nominal', 
        plot_tip_force=True, 
        plot_backbone_ellipsoids=False
    )
    
    # Setup baseline solver (get holes from dummy solution)
    dummy_solution = simulator_nominal.solve(VectorXGaussian(np.zeros(4), np.eye(4)), Vector6Gaussian(np.zeros(6), np.eye(6)), None)
    solver_baseline = Benchmark(config, dummy_solution.marginals.tendon_config.hole_locations)

    # Simulator to sample tip pose data
    simulator_tracking = TendonRobotSolver(config)

    # Solver that does the actual inference using tip pose data
    solver_tracking = TendonRobotSolver(config)
    plotter_tracking = TendonRobotPlotter(
        save_frames_dir_name='tendon_robot_posterior',
        plot_rviz_coords=True,
        plot_tip_force=True
    )

    # Use a separate solver for these so it doesn't mess up our performance metrics
    solver_jacobian = TendonRobotSolver(config)
    solver_prior = TendonRobotSolver(config)
    plotter_prior = TendonRobotPlotter(save_frames_dir_name='tendon_robot_prior', plot_tip_force=False)

    # Setup Jacobian control
    damping = 6e-2
    tensions_min = np.array([0.5, 0.1, 0.1, 0.1])
    tensions_cmd_current = tensions_min
    
    # Continuous time noise models for smoothness
    tensions_noise_model = GaussianProcessNoiseModel(4, frame_rate, sim_time, seed=0)
    p_noise_model = GaussianProcessNoiseModel(3, frame_rate, sim_time, seed=1)

    # Setup data collection
    data = {
        'p_mean': [], 'p_std': [], 'p_meas': [], 'p_nominal': [], 'p_goal': [], 'p_baseline': [],
        'f_mean': [], 'f_std': [], 'f_gt': []
    }

    small_tensions_cov = 1e-6 * np.eye(4)
    small_wrench_cov = 1e-6 * np.eye(6)

    wrench_prior_cov = small_wrench_cov.copy()
    wrench_prior_cov[3:,3:] = tip_force_prior_sigma ** 2 * np.eye(3)
    position_meas_cov = tip_position_meas_sigma ** 2 * np.eye(3)

    for ti, p_goal, tensions_nominal_i in zip(t, positions, tensions_nominal):
        f_gt = tip_force_function(ti)

        # Nominal solution with no force correction, still need to add noise and re solve
        tensions_noise = tensions_noise_model.step(tensions_meas_sigma ** 2 * np.eye(4))
        tensions_nominal_gt = tensions_nominal_i + tensions_noise
        solution_nominal = simulator_nominal.solve(
            VectorXGaussian(tensions_nominal_gt, small_tensions_cov),
            Vector6Gaussian(np.hstack((np.zeros(3), f_gt)), small_wrench_cov),
            None
        )
        p_nominal = solution_nominal.marginals.rod.states[-1].pose.mean[:3,3]

        # Prior solution, using only prior wrench, no measurement
        solution_prior = solver_prior.solve(
            VectorXGaussian(tensions_cmd_current, tensions_meas_sigma ** 2 * np.eye(4)),
            Vector6Gaussian(np.zeros(6), wrench_prior_cov),
            None 
        )

        # Simulated solution to sample position from, small covariances
        tensions_gt = tensions_cmd_current + tensions_noise
        solution_sim = simulator_tracking.solve(
            VectorXGaussian(tensions_gt, small_tensions_cov),
            Vector6Gaussian(np.hstack((np.zeros(3), f_gt)), small_wrench_cov),
            None
        )

        # Sample the position
        p_sim_mean = solution_sim.marginals.rod.states[-1].pose.mean[:3,3]
        R_sim_mean = solution_sim.marginals.rod.states[-1].pose.mean[:3,:3]
        p_sim_cov = R_sim_mean @ solution_sim.marginals.rod.states[-1].pose.cov[3:,3:] @ R_sim_mean.T
        p_meas = p_sim_mean + p_noise_model.step(p_sim_cov + position_meas_cov)

        # Compare simulator solver to baseline solver if requested
        if do_baseline:
            solution_baseline = solver_baseline.solve(tensions_nominal_gt, f_gt)
            p_baseline = solution_baseline[-1]['p']
            print(f"baseline error: {np.linalg.norm(p_baseline - p_nominal)}")
            data['p_baseline'].append(p_baseline)

        # Use the sampled position as a prior on tip pose
        solution_post = solver_tracking.solve(
            VectorXGaussian(tensions_cmd_current, small_tensions_cov),
            Vector6Gaussian(np.zeros(6), wrench_prior_cov),
            Vector3Gaussian(p_meas, position_meas_cov)
        )
        
        # Evaluate the Jacobian for control using the estimated tip wrench
        wrench_post = solution_post.marginals.external_wrenches[-1]
        solution_jacobian = solver_jacobian.solve(
            VectorXGaussian(tensions_cmd_current, tensions_meas_sigma ** 2 * np.eye(4)),
            wrench_post,
            None
        )
        
        J_position = solution_jacobian.marginals.J_pose_tensions[3:]
        p_error = R_sim_mean.T @ (p_goal - p_meas)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions_cmd_current = tensions_cmd_current + d_tensions
        tensions_cmd_current = np.maximum(tensions_cmd_current, tensions_min)

        data['p_mean'].append(solution_post.marginals.rod.states[-1].pose.mean[:3,3])
        data['p_std'].append(np.sqrt(np.diag(solution_post.marginals.rod.states[-1].pose.cov[3:,3:])))
        data['p_meas'].append(p_meas)
        data['p_goal'].append(p_goal)
        data['p_nominal'].append(p_nominal)
        data['f_gt'].append(f_gt)
        data['f_mean'].append(wrench_post.mean[3:])
        data['f_std'].append(np.sqrt(np.diag(wrench_post.cov[3:,3:])))

        if plot:
            plotter_tracking.update(solution_post, p_desired=p_goal, tip_force_gt=f_gt)
            plotter_nominal.update(solution_nominal, p_desired=p_goal, tip_force_gt=f_gt)
            plotter_prior.update(solution_prior)

        progress = 100.0 * ti / t[-1]
        print(f"Progress: {progress:5.1f}%", end="\r")

    return t, {k: np.asarray(v) for k, v in data.items()}


if __name__ == "__main__":
    plot = False
    do_baseline = False
    t, data = run_sim(sim_time=30.0, plot=plot, do_baseline=do_baseline)

    if do_baseline:
        setup_plt()
        plt.figure()
        baseline_err = 1000 * np.linalg.norm(data['p_nominal'] - data['p_baseline'], axis=1)
        plt.plot(t, baseline_err, label=f"baseline (mean={np.mean(baseline_err):.3f} mm)")
        plt.xlabel("time (sec)")
        plt.ylabel("baseline error (mm)")
        plt.grid(True, alpha=0.25)
        plt.legend()

        plt.tight_layout()
        plt.savefig("figures/tendon_robot_baseline.pdf", dpi=300)
        plt.close()

    

    color_cycle = ['r', 'g', 'b', 'c']

    setup_plt(width=3.7, height=4, grid=True)

    fig, axes = plt.subplots(3, 2, sharex=True)
    axes = axes.flatten()  # flatten to 1D for easy indexing

    for ii in range(3):
        ax = axes[ii*2]  # left column
        ax.plot(t, 1000 * data['p_goal'][:, ii], 'k--', label='desired')
        ax.plot(t, 1000 * data['p_meas'][:, ii], linestyle='-', color=color_cycle[ii], label='tracking')
        ax.plot(t, 1000 * data['p_nominal'][:, ii], linestyle=':', color=color_cycle[ii], label='OL')
        ax.set_xlim([t[0], t[-1]+1e-1])
        if ii == 1:
            ax.legend(ncol=3, columnspacing=0.2, borderpad=0.0, borderaxespad=0.2,
                    handlelength=1.0, handletextpad=0.2)

    # plot forces in second column (axes[1], axes[3], axes[5])
    for ii in range(3):
        ax = axes[ii*2+1]  # right column
        ax.plot(t, data['f_gt'][:, ii], 'k--', label='truth')
        ax.plot(t, data['f_mean'][:, ii], color=color_cycle[ii], label='mean')
        ax.fill_between(t, 
            data['f_mean'][:, ii] - 2 * data['f_std'][:, ii],
            data['f_mean'][:, ii] + 2 * data['f_std'][:, ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')
        ax.set_xlim([t[0], t[-1]+1e-1])
        if ii == 0:
            ax.legend(ncol=3, columnspacing=0.2, borderpad=0.0, borderaxespad=0.2,
                    handlelength=1.0, handletextpad=0.2)

    axes[-2].set_xlabel("time (sec)")  # bottom left
    axes[-1].set_xlabel("time (sec)")  # bottom right

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # Get left and right column centers in figure coordinates
    left_center = (axes[0].get_position().x0 + axes[0].get_position().x1)/2
    right_center = (axes[1].get_position().x0 + axes[1].get_position().x1)/2

    # Place column titles
    fig.text(left_center, 0.97, 'Position (mm)', ha='center', va='bottom', fontsize=8)
    fig.text(right_center, 0.97, 'Force (N)', ha='center', va='bottom', fontsize=8)

    plt.savefig("figures/tendon_robot_results.pdf", bbox_inches="tight")



        # fig, axes = plt.subplots(3, 1, sharex=True)

    
    
    # for ii, ax in enumerate(axes):
    #     ax.plot(t, 1000 * tip_position_tracking_mean[:, ii], linestyle='-', color=color_cycle[ii], label='mean')
    #     ax.plot(t, 1000 * tip_position_tracking_gt[:, ii], linestyle='--', color=color_cycle[ii], label='truth')
    #     ax.fill_between(t, 
    #         1000 * tip_position_tracking_mean[:,ii] - 2000 * tip_position_tracking_std[:,ii],
    #         1000 * tip_position_tracking_mean[:,ii] + 2000 * tip_position_tracking_std[:,ii], 
    #         alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')

    #     ax.set_ylabel(position_labels[ii])
    #     if ii == 1:
    #         ax.legend(ncol=3, columnspacing=0.5, handletextpad=0.5)

    # fig.align_ylabels()
    # plt.tight_layout()
    
    # plt.savefig("figures/position_noise_model.pdf", bbox_inches="tight")