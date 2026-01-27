#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "CosseratDynamicsSolver.h"

namespace py = pybind11;


void bind_cosserat_dynamics(py::module& m) {
    py::class_<CosseratDynamicsMarginals>(m, "CosseratRodDynamicsMarginals")
        .def_readonly("rod", &CosseratDynamicsMarginals::rod)
        .def_readonly("velocities", &CosseratDynamicsMarginals::velocities);

    py::class_<Solution<CosseratDynamicsMarginals>>(m, "CosseratRodDynamicsSolution")
        .def_readonly("meta", &Solution<CosseratDynamicsMarginals>::meta)
        .def_readonly("marginals", &Solution<CosseratDynamicsMarginals>::marginals);
    
    py::class_<CosseratDynamicsConfig>(m, "CosseratRodDynamicsConfig")
        .def(py::init<>())  // default constructor
        .def_readwrite("rod", &CosseratDynamicsConfig::rod)
        .def_readwrite("dt", &CosseratDynamicsConfig::dt)
        .def_readwrite("linear_damping", &CosseratDynamicsConfig::linear_damping)
        .def_readwrite("rotational_damping", &CosseratDynamicsConfig::rotational_damping)
        .def_readwrite("linear_inertia", &CosseratDynamicsConfig::linear_inertia)
        .def_readwrite("rotational_inertia", &CosseratDynamicsConfig::rotational_inertia)
        .def_readwrite("acceleration_noise_sigma", &CosseratDynamicsConfig::acceleration_noise_sigma)
        .def_readwrite("initial_tip_wrench", &CosseratDynamicsConfig::initial_tip_wrench);

    py::class_<CosseratDynamicsSolver>(m, "CosseratRodDynamicsSolver")
        .def(py::init<const CosseratDynamicsConfig&>())
        .def("solve", &CosseratDynamicsSolver::solve,
        py::call_guard<py::gil_scoped_release>());
}