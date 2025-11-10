#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include"cosserat_rod.h"

namespace py = pybind11;

PYBIND11_MODULE(crest_sparse, m) {
    py::class_<CosseratRodSolution>(m, "CosseratRodSolution")
        .def(py::init<>())
        .def_readwrite("pose_mean", &CosseratRodSolution::pose_mean)
        .def_readwrite("pose_cov", &CosseratRodSolution::pose_cov)
        .def_readwrite("wrench_mean", &CosseratRodSolution::wrench_mean)
        .def_readwrite("wrench_cov", &CosseratRodSolution::wrench_cov);
    
    py::class_<CosseratRodConfig>(m, "CosseratRodConfig")
        .def(py::init<>())  // default constructor
        .def_readwrite("rod_length", &CosseratRodConfig::rod_length)
        .def_readwrite("num_backbone_nodes", &CosseratRodConfig::num_backbone_nodes)
        .def_readwrite("k_bending", &CosseratRodConfig::k_bending)
        .def_readwrite("k_torsion", &CosseratRodConfig::k_torsion)
        .def_readwrite("k_shear", &CosseratRodConfig::k_shear)
        .def_readwrite("k_extension", &CosseratRodConfig::k_extension)
        .def_readwrite("sigma_twist_position", &CosseratRodConfig::sigma_twist_position)
        .def_readwrite("sigma_twist_rotation", &CosseratRodConfig::sigma_twist_rotation)
        .def_readwrite("sigma_stress_force", &CosseratRodConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &CosseratRodConfig::sigma_stress_moment)
        .def_readwrite("sigma_small_force", &CosseratRodConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &CosseratRodConfig::sigma_small_moment)
        .def_readwrite("sigma_base_position", &CosseratRodConfig::sigma_base_position)
        .def_readwrite("sigma_base_rotation", &CosseratRodConfig::sigma_base_rotation);

    py::class_<BasicCosseratSolver>(m, "BasicCosseratSolver")
        .def(py::init<const CosseratRodConfig&>())
        .def("solve", &BasicCosseratSolver::solve,
            py::arg("tip_force_mean"),
            py::arg("tip_force_cov"),
            py::call_guard<py::gil_scoped_release>());
}