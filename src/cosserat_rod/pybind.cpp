#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "CosseratRodModel.h"
#include "utils/SolverBase.h"

namespace py = pybind11;


// The rod's RESULT types only. CosseratRodSolver -- the rod as a standalone
// solver -- is gone; the hand drives CosseratRodModel through TendonFingerModel
// instead. But every finger's marginals carry a CosseratRodMarginals, and the
// hand scripts read `fm.rod.states[n].pose.mean` constantly, so these two
// classes have to stay registered even with no rod solver to produce them.
void bind_cosserat_rod(py::module& m) {
    py::class_<CosseratRodState>(m, "CosseratRodState")
        .def(py::init<>())
        .def_readwrite("pose", &CosseratRodState::pose)
        .def_readwrite("stress", &CosseratRodState::stress)
        .def_readwrite("wrench", &CosseratRodState::wrench);

    py::class_<CosseratRodMarginals>(m, "CosseratRodMarginals")
        .def(py::init<>())
        .def_readwrite("states", &CosseratRodMarginals::states);

    // CosseratDynamicsSolver's marginals are a list of these, one per time step.
    py::class_<Solution<CosseratRodMarginals>>(m, "CosseratRodSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<CosseratRodMarginals>::meta)
        .def_readwrite("marginals", &Solution<CosseratRodMarginals>::marginals);
}
