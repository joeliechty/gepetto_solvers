#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "CosseratDynamicsSolver.h"

namespace py = pybind11;


void bind_cosserat_dynamics(py::module& m) {
    py::class_<CosseratDynamicsMarginals>(m, "CosseratRodDynamicsMarginals")
        .def(py::init<>())
        .def_readwrite("rods_t", &CosseratDynamicsMarginals::rods_t);

    py::class_<Solution<CosseratDynamicsMarginals>>(m, "CosseratRodDynamicsSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<CosseratDynamicsMarginals>::meta)
        .def_readwrite("marginals", &Solution<CosseratDynamicsMarginals>::marginals);
    
    py::class_<CosseratDynamicsConfig>(m, "CosseratRodDynamicsConfig")
        .def(py::init<>())  // default constructor
        .def_readwrite("rod_length", &CosseratDynamicsConfig::rod_length)
        .def_readwrite("num_nodes", &CosseratDynamicsConfig::num_nodes)
        .def_readwrite("K_inv", &CosseratDynamicsConfig::K_inv)
        .def_readwrite("dynamics_noise_sigma", &CosseratDynamicsConfig::dynamics_noise_sigma)
        .def_readwrite("twist_noise_sigma", &CosseratDynamicsConfig::twist_noise_sigma)
        .def_readwrite("wrench_noise_sigma", &CosseratDynamicsConfig::wrench_noise_sigma)
        .def_readwrite("init_velocity_sigma", &CosseratDynamicsConfig::init_velocity_sigma)
        .def_readwrite("init_velocity_mean", &CosseratDynamicsConfig::init_velocity_mean)
        .def_readwrite("init_velocity_idx", &CosseratDynamicsConfig::init_velocity_idx)
        .def_readwrite("num_time_steps", &CosseratDynamicsConfig::num_time_steps)
        .def_readwrite("dt", &CosseratDynamicsConfig::dt)
        .def_readwrite("linear_damping", &CosseratDynamicsConfig::linear_damping)
        .def_readwrite("rotational_damping", &CosseratDynamicsConfig::rotational_damping)
        .def_readwrite("linear_inertia", &CosseratDynamicsConfig::linear_inertia)
        .def_readwrite("rotational_inertia", &CosseratDynamicsConfig::rotational_inertia);

    py::class_<CosseratDynamicsSolver>(m, "CosseratRodDynamicsSolver")
        .def(py::init<const CosseratDynamicsConfig&>())
        .def("solve", &CosseratDynamicsSolver::solve,
        py::call_guard<py::gil_scoped_release>());
}