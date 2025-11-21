#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "cosserat_rod/CosseratRodSolver.h"
#include "cosserat_dynamics/CosseratDynamicsSolver.h"
#include "cosserat_shell/CosseratShellModel.h"
#include "cosserat_shell/CosseratShellSolver.h"
#include "parallel_robot/ParallelRobotModel.h"
#include "parallel_robot/ParallelRobotSolver.h"

namespace py = pybind11;


PYBIND11_MODULE(crest_sparse, m) {
    py::class_<SolutionMetadata>(m, "SolutionMetadata")
        .def(py::init<>())
        .def_readwrite("total_time_ms", &SolutionMetadata::total_time_ms)
        .def_readwrite("build_time_ms", &SolutionMetadata::build_time_ms)
        .def_readwrite("optimize_time_ms", &SolutionMetadata::optimize_time_ms)
        .def_readwrite("marginalize_time_ms", &SolutionMetadata::marginalize_time_ms)
        .def_readwrite("extract_time_ms", &SolutionMetadata::extract_time_ms)
        
        .def_readwrite("iterations", &SolutionMetadata::iterations)
        .def_readwrite("error", &SolutionMetadata::error);

    py::class_<CosseratRodMarginals>(m, "CosseratRodMarginals")
        .def(py::init<>())
        .def_readwrite("pose_mean", &CosseratRodMarginals::pose_mean)
        .def_readwrite("pose_cov", &CosseratRodMarginals::pose_cov)
        .def_readwrite("stress_mean", &CosseratRodMarginals::stress_mean)
        .def_readwrite("stress_cov", &CosseratRodMarginals::stress_cov)
        .def_readwrite("wrench_mean", &CosseratRodMarginals::wrench_mean)
        .def_readwrite("wrench_cov", &CosseratRodMarginals::wrench_cov);
    
    py::class_<Solution<CosseratRodMarginals>>(m, "CosseratRodSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<CosseratRodMarginals>::meta)
        .def_readwrite("marginals", &Solution<CosseratRodMarginals>::marginals);

    py::class_<CosseratShellMarginals>(m, "CosseratShellMarginals")
        .def(py::init<>())
        .def_readwrite("pose_mean", &CosseratShellMarginals::pose_mean)
        .def_readwrite("pose_cov", &CosseratShellMarginals::pose_cov)
        .def_readwrite("stress_mean", &CosseratShellMarginals::stress_mean)
        .def_readwrite("stress_cov", &CosseratShellMarginals::stress_cov);

    py::class_<Solution<CosseratDynamicsMarginals>>(m, "CosseratDynamicsSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<CosseratDynamicsMarginals>::meta)
        .def_readwrite("marginals", &Solution<CosseratDynamicsMarginals>::marginals);
    
    py::class_<Solution<CosseratShellMarginals>>(m, "CosseratShellSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<CosseratShellMarginals>::meta)
        .def_readwrite("marginals", &Solution<CosseratShellMarginals>::marginals);

    py::class_<CosseratRodSolverConfig>(m, "CosseratRodSolverConfig")
        .def(py::init<>())  // default constructor
        .def_readwrite("rod_length", &CosseratRodSolverConfig::rod_length)
        .def_readwrite("num_nodes", &CosseratRodSolverConfig::num_nodes)
        .def_readwrite("K_inv", &CosseratRodSolverConfig::K_inv)
        .def_readwrite("sigma_twist_pos", &CosseratRodSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_twist_rot", &CosseratRodSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_small_force", &CosseratRodSolverConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &CosseratRodSolverConfig::sigma_small_moment)
        .def_readwrite("sigma_base_pose_pos", &CosseratRodSolverConfig::sigma_base_pose_pos)
        .def_readwrite("sigma_base_pose_rot", &CosseratRodSolverConfig::sigma_base_pose_rot);

    py::class_<CosseratDynamicsConfig>(m, "CosseratRodDynamicsConfig")
        .def(py::init<>())  // default constructor
        .def_readwrite("rod_config", &CosseratDynamicsConfig::rod_config)
        .def_readwrite("num_time_steps", &CosseratDynamicsConfig::num_time_steps)
        .def_readwrite("dt", &CosseratDynamicsConfig::dt)
        .def_readwrite("linear_damping", &CosseratDynamicsConfig::linear_damping)
        .def_readwrite("rotational_damping", &CosseratDynamicsConfig::rotational_damping)
        .def_readwrite("linear_inertia", &CosseratDynamicsConfig::linear_inertia)
        .def_readwrite("rotational_inertia", &CosseratDynamicsConfig::rotational_inertia)
        .def_readwrite("initial_tip_wrench", &CosseratDynamicsConfig::initial_tip_wrench);

    py::class_<CosseratShellSolverConfig>(m, "CosseratShellSolverConfig")
        .def(py::init<>())
        .def_readwrite("num_nodes_x", &CosseratShellSolverConfig::num_nodes_x)
        .def_readwrite("num_nodes_y", &CosseratShellSolverConfig::num_nodes_y)
        .def_readwrite("element_size", &CosseratShellSolverConfig::element_size)
        .def_readwrite("K_inv", &CosseratShellSolverConfig::K_inv)
        .def_readwrite("sigma_twist_pos", &CosseratShellSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_twist_rot", &CosseratShellSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_stress_force", &CosseratShellSolverConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &CosseratShellSolverConfig::sigma_stress_moment);

    py::class_<CosseratRodSolver>(m, "CosseratRodSolver")
        .def(py::init<const CosseratRodSolverConfig&>())
        .def("solve", &CosseratRodSolver::solve,
            py::arg("tip_force_mean"),
            py::arg("tip_force_cov"),
            py::arg("tip_pos_mean"),
            py::arg("tip_pos_cov"),
            py::arg("nominal_strain"),
            py::call_guard<py::gil_scoped_release>());

    py::class_<CosseratDynamicsSolver>(m, "CosseratRodDynamicsSolver")
        .def(py::init<const CosseratDynamicsConfig&>())
        .def("solve", &CosseratDynamicsSolver::solve,
            py::call_guard<py::gil_scoped_release>());
    
    py::class_<CosseratShellSolver>(m, "CosseratShellSolver")
        .def(py::init<const CosseratShellSolverConfig&>())
        .def("solve", &CosseratShellSolver::solve,
            py::arg("displacement_mean"),
            py::arg("displacement_cov"),
            py::call_guard<py::gil_scoped_release>());

    py::class_<ParallelRobotSolverConfig>(m, "ParallelRobotSolverConfig")
        .def(py::init<>())
        .def_readwrite("nodes_per_rod", &ParallelRobotSolverConfig::nodes_per_rod)
        .def_readwrite("K_inv", &ParallelRobotSolverConfig::K_inv)
        .def_readwrite("sigma_twist_pos", &ParallelRobotSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_twist_rot", &ParallelRobotSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_small_force", &ParallelRobotSolverConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &ParallelRobotSolverConfig::sigma_small_moment)
        .def_readwrite("base_end_poses", &ParallelRobotSolverConfig::base_end_poses)
        .def_readwrite("tip_end_poses", &ParallelRobotSolverConfig::tip_end_poses)
        .def_readwrite("sigma_end_pose_pos", &ParallelRobotSolverConfig::sigma_end_pose_pos)
        .def_readwrite("sigma_end_pose_rot", &ParallelRobotSolverConfig::sigma_end_pose_rot);
    
    py::class_<ParallelRobotMarginals>(m, "ParallelRobotMarginals")
        .def(py::init<>())
        .def_readwrite("rods", &ParallelRobotMarginals::rods)
        .def_readwrite("platform_pose_mean", &ParallelRobotMarginals::platform_pose_mean)
        .def_readwrite("platform_pose_cov", &ParallelRobotMarginals::platform_pose_cov)
        .def_readwrite("platform_wrench_mean", &ParallelRobotMarginals::platform_wrench_mean)
        .def_readwrite("platform_wrench_cov", &ParallelRobotMarginals::platform_wrench_cov);
        
    py::class_<Solution<ParallelRobotMarginals>>(m, "ParallelRobotSolution")
        .def_readwrite("meta", &Solution<ParallelRobotMarginals>::meta)
        .def_readwrite("marginals", &Solution<ParallelRobotMarginals>::marginals);

    py::class_<ParallelRobotSolver>(m, "ParallelRobotSolver")
        .def(py::init<const ParallelRobotSolverConfig&>(), py::arg("config"))
        .def("solve", &ParallelRobotSolver::solve, 
            py::arg("rod_lengths"),
            py::arg("sigma_rod_lengths"),
            py::arg("wrench_mean"),
            py::arg("wrench_cov"),
            py::call_guard<py::gil_scoped_release>());
        
}