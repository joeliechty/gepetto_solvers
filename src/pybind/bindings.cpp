#include "bindings.h"

#include "utils/SolverBase.h"
#include "utils/Gaussians.h"

namespace py = pybind11;


void bind_utils(py::module& m) {
    // Augmented Lagrangian outer-loop state, carried between solves. Opaque on
    // purpose: a caller moves it from one solver to another and never builds one
    // -- the multipliers only mean anything paired with the tags beside them.
    py::class_<crest_sparse::WarmALState>(m, "ALDuals")
        .def(py::init<>())
        .def_property_readonly(
            "num_equality",
            [](const crest_sparse::WarmALState& s) { return s.lambda_eq.size(); })
        .def_property_readonly(
            "num_inequality",
            [](const crest_sparse::WarmALState& s) { return s.lambda_ineq.size(); })
        .def_property_readonly("mu", [](const crest_sparse::WarmALState& s) {
            return s.mu_eq_at;
        })
        .def_property_readonly("tagged", &crest_sparse::WarmALState::tagged,
                               "True when every multiplier carries the identity "
                               "of its constraint, which is what a transfer "
                               "across a rebuilt graph requires.")
        .def_readonly("tags_equality", &crest_sparse::WarmALState::tag_eq)
        .def_readonly("tags_inequality", &crest_sparse::WarmALState::tag_ineq)
        .def("__bool__", [](const crest_sparse::WarmALState& s) {
            return !s.empty();
        })
        .def("__repr__", [](const crest_sparse::WarmALState& s) {
            return "<ALDuals eq=" + std::to_string(s.lambda_eq.size()) +
                   " ineq=" + std::to_string(s.lambda_ineq.size()) +
                   " mu=" + std::to_string(s.mu_eq_at) +
                   (s.tagged() ? " tagged>" : " untagged>");
        });

    py::class_<crest_sparse::ALTransferReport>(m, "ALTransferReport")
        .def(py::init<>())
        .def_readonly("matched_equality",
                      &crest_sparse::ALTransferReport::matched_eq)
        .def_readonly("total_equality", &crest_sparse::ALTransferReport::total_eq)
        .def_readonly("matched_inequality",
                      &crest_sparse::ALTransferReport::matched_ineq)
        .def_readonly("total_inequality",
                      &crest_sparse::ALTransferReport::total_ineq)
        .def_property_readonly("matched", &crest_sparse::ALTransferReport::matched)
        .def_property_readonly("total", &crest_sparse::ALTransferReport::total)
        .def("__repr__", [](const crest_sparse::ALTransferReport& r) {
            return "<ALTransferReport " + std::to_string(r.matched()) + "/" +
                   std::to_string(r.total()) + " constraints matched>";
        });

    py::class_<SolverBaseConfig>(m, "SolverBaseConfig")
        .def(py::init<>())
        .def_readwrite("linear_solver_type", &SolverBaseConfig::linear_solver_type)
        .def_readwrite("optimizer_type", &SolverBaseConfig::optimizer_type)
        .def_readwrite("use_dense", &SolverBaseConfig::use_dense)
        .def_readwrite("delta_initial", &SolverBaseConfig::delta_initial)
        .def_readwrite("lambda_initial", &SolverBaseConfig::lambda_initial)
        .def_readwrite("lambda_upper_bound", &SolverBaseConfig::lambda_upper_bound)
        .def_readwrite("diagonal_damping", &SolverBaseConfig::diagonal_damping)
        .def_readwrite("max_iterations", &SolverBaseConfig::max_iterations)
        .def_readwrite("al_initial_mu", &SolverBaseConfig::al_initial_mu)
        .def_readwrite("al_mu_increase_rate", &SolverBaseConfig::al_mu_increase_rate)
        .def_readwrite("al_max_iterations", &SolverBaseConfig::al_max_iterations)
        .def_readwrite("al_max_dual_step", &SolverBaseConfig::al_max_dual_step)
        .def_readwrite("al_inner_rel_tol_initial", &SolverBaseConfig::al_inner_rel_tol_initial)
        .def_readwrite("al_abs_violation_tol", &SolverBaseConfig::al_abs_violation_tol)
        .def_readwrite("al_abs_cost_tol", &SolverBaseConfig::al_abs_cost_tol)
        .def_readwrite("al_rel_violation_tol", &SolverBaseConfig::al_rel_violation_tol)
        .def_readwrite("al_rel_cost_tol", &SolverBaseConfig::al_rel_cost_tol)
        .def_readwrite("al_warm_start_duals", &SolverBaseConfig::al_warm_start_duals)
        .def_readwrite("al_warm_mu_max", &SolverBaseConfig::al_warm_mu_max)
        .def_readwrite("al_transfer_mu_max", &SolverBaseConfig::al_transfer_mu_max)
        .def_readwrite("record_iterations", &SolverBaseConfig::record_iterations)
        .def_readwrite("iteration_sample_interval", &SolverBaseConfig::iteration_sample_interval)
        .def_readwrite("skip_marginals", &SolverBaseConfig::skip_marginals);

    py::class_<SolutionMetadata>(m, "SolutionMetadata")
        .def(py::init<>())
        .def_readwrite("total_time_ms", &SolutionMetadata::total_time_ms)
        .def_readwrite("build_time_ms", &SolutionMetadata::build_time_ms)
        .def_readwrite("optimize_time_ms", &SolutionMetadata::optimize_time_ms)
        .def_readwrite("marginalize_time_ms", &SolutionMetadata::marginalize_time_ms)
        .def_readwrite("extract_time_ms", &SolutionMetadata::extract_time_ms)
        .def_readwrite("iterations", &SolutionMetadata::iterations)
        .def_readwrite("error", &SolutionMetadata::error)
        .def_readwrite("iteration_errors", &SolutionMetadata::iteration_errors)
        .def_readwrite("iteration_trust_region", &SolutionMetadata::iteration_trust_region)
        .def_readwrite("iteration_step_norms", &SolutionMetadata::iteration_step_norms)
        .def_readwrite("al_iteration_costs", &SolutionMetadata::al_iteration_costs)
        .def_readwrite("al_iteration_violations", &SolutionMetadata::al_iteration_violations)
        .def_readwrite("al_iteration_mus", &SolutionMetadata::al_iteration_mus);

    py::class_<Vector6Gaussian>(m, "Vector6Gaussian")
        .def(py::init<>())
        .def(py::init<const gtsam::Vector6&, const gtsam::Matrix6&>(),
            py::arg("mean"), py::arg("cov"))
        .def_readwrite("mean", &Vector6Gaussian::mean)
        .def_readwrite("cov", &Vector6Gaussian::cov);

    py::class_<Pose3Gaussian>(m, "Pose3Gaussian")
        .def(py::init<>())
        .def(py::init<const gtsam::Matrix4&, const gtsam::Matrix6&>(),
            py::arg("mean"), py::arg("cov"))
        .def_readwrite("mean", &Pose3Gaussian::mean)
        .def_readwrite("cov", &Pose3Gaussian::cov);

    py::class_<Vector3Gaussian>(m, "Vector3Gaussian")
        .def(py::init<>())
        .def(py::init<const gtsam::Vector3&, const gtsam::Matrix3&>(),
            py::arg("mean"), py::arg("cov"))
        .def_readwrite("mean", &Vector3Gaussian::mean)
        .def_readwrite("cov", &Vector3Gaussian::cov);

    py::class_<Vector4Gaussian>(m, "Vector4Gaussian")
        .def(py::init<>())
        .def(py::init<const gtsam::Vector4&, const gtsam::Matrix4&>(),
            py::arg("mean"), py::arg("cov"))
        .def_readwrite("mean", &Vector4Gaussian::mean)
        .def_readwrite("cov", &Vector4Gaussian::cov);

    py::class_<VectorXGaussian>(m, "VectorXGaussian")
        .def(py::init<>())
        .def(py::init([](const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
            VectorXGaussian g;
            g.mean = mean;
            g.cov = cov;
            return g;
        }), py::arg("mean"), py::arg("cov"))
        .def_readwrite("mean", &VectorXGaussian::mean)
        .def_readwrite("cov", &VectorXGaussian::cov);
}


PYBIND11_MODULE(_crest_sparse, m) {
    bind_cosserat_rod(m);
    bind_cosserat_dynamics(m);
    bind_tendon_finger(m);
    bind_tendon_hand(m);
    bind_utils(m);

    // Whether the solve() bindings drop the GIL for the duration of the C++
    // solve. A capability flag rather than something a caller can introspect,
    // because a py::call_guard leaves no trace hasattr can find -- and the
    // difference matters to an interactive caller: without it the interpreter
    // is frozen for the whole solve and a stop button cannot be serviced.
    m.attr("solve_releases_gil") = true;
}