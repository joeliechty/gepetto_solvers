#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "cosserat_rod.h"
#include "cosserat_rod_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(crest_sparse, m) {
    py::class_<SolutionMetadata>(m, "SolutionMetadata")
        .def(py::init<>())
        .def_readwrite("solve_time_ms", &SolutionMetadata::solve_time_ms)
        .def_readwrite("total_time_ms", &SolutionMetadata::total_time_ms)
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
    
    py::class_<CosseratRodSolution>(m, "CosseratRodSolution")
        .def(py::init<>())
        .def_readwrite("meta", &CosseratRodSolution::meta)
        .def_readwrite("marginals", &CosseratRodSolution::marginals);
    
    py::class_<CosseratRodSolverConfig>(m, "CosseratRodSolverConfig")
        .def(py::init<>())  // default constructor
        .def_readwrite("rod_length", &CosseratRodSolverConfig::rod_length)
        .def_readwrite("num_nodes", &CosseratRodSolverConfig::num_nodes)
        .def_readwrite("k_bending", &CosseratRodSolverConfig::k_bending)
        .def_readwrite("k_torsion", &CosseratRodSolverConfig::k_torsion)
        .def_readwrite("k_shear", &CosseratRodSolverConfig::k_shear)
        .def_readwrite("k_extension", &CosseratRodSolverConfig::k_extension)
        .def_readwrite("sigma_twist_pos", &CosseratRodSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_twist_rot", &CosseratRodSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_small_force", &CosseratRodSolverConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &CosseratRodSolverConfig::sigma_small_moment)
        .def_readwrite("sigma_base_pose_pos", &CosseratRodSolverConfig::sigma_base_pose_pos)
        .def_readwrite("sigma_base_pose_rot", &CosseratRodSolverConfig::sigma_base_pose_rot);

    py::class_<CosseratRodSolver>(m, "CosseratRodSolver")
        .def(py::init<const CosseratRodSolverConfig&>())
        .def("solve", &CosseratRodSolver::solve,
            py::arg("tip_force_mean"),
            py::arg("tip_force_cov"),
            py::arg("tip_pos_mean"),
            py::arg("tip_pos_cov"),
            py::call_guard<py::gil_scoped_release>());
}