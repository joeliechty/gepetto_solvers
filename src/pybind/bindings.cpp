#include "bindings.h"

#include "utils/SolverBase.h"
#include "utils/Gaussians.h"

namespace py = pybind11;


void bind_utils(py::module& m) {
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
        .def_readwrite("record_iterations", &SolverBaseConfig::record_iterations)
        .def_readwrite("iteration_sample_interval", &SolverBaseConfig::iteration_sample_interval);

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
        .def_readwrite("iteration_step_norms", &SolutionMetadata::iteration_step_norms);

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
    bind_cosserat_shell(m);
    bind_parallel_robot(m);
    bind_tendon_robot(m);
    bind_multi_robot(m);
    bind_tendon_finger(m);
    bind_utils(m);
}