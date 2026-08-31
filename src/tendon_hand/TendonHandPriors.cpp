// Temporal priors, step priors, and marginal extraction.

#include "gepetto_solvers/tendon_hand/TendonHandModel.h"

#include "gepetto_solvers/utils/MiscInline.h"

#include <gtsam/constrained/NonlinearEqualityConstraint.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <openvdb/tools/Interpolation.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>

using namespace gtsam;


void TendonHandModel::add_temporal_gp(
    NonlinearFactorGraph& graph,
    const TendonHandModel& next,
    const Eigen::MatrixXd& gp_tense_Qc,
    const Eigen::MatrixXd& gp_len_Qc,
    double dt) const
{
    const bool has_len_gp = gp_len_Qc.size() > 0;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            // Tension GP (Eq 1.11): identity transition, zero-mean between factor.
            Eigen::Matrix<double, N, N> Qc = gp_tense_Qc.topLeftCorner<N, N>();
            graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                fp->get_tensions_key(),
                next.finger_tension_key(static_cast<int>(i)),
                Eigen::Vector<double, N>::Zero(),
                noiseModel::Gaussian::Covariance(Qc * dt)));

            // Length GP (Eq 1.13), optional.
            if (has_len_gp) {
                Eigen::Matrix<double, N, N> Qc_len = gp_len_Qc.topLeftCorner<N, N>();
                graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                    fp->get_lengths_key(),
                    next.finger_length_key(static_cast<int>(i)),
                    Eigen::Vector<double, N>::Zero(),
                    noiseModel::Gaussian::Covariance(Qc_len * dt)));
            }
        }, fingers_[i]);
    }
}


void TendonHandModel::add_length_priors(
    NonlinearFactorGraph& graph,
    const std::vector<VectorXGaussian>& lengths) const
{
    if (lengths.size() != fingers_.size())
        throw std::invalid_argument(
            "add_length_priors: lengths size must match number of fingers");

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (lengths[i].mean.size() != N)
                throw std::invalid_argument(
                    "add_length_priors: finger " + std::to_string(i) + " expects " +
                    std::to_string(N) + " tendon lengths, got " +
                    std::to_string(lengths[i].mean.size()));

            Eigen::Vector<double, N> mean = lengths[i].mean;
            Eigen::Matrix<double, N, N> cov = lengths[i].cov.topLeftCorner<N, N>();
            graph.add(PriorFactor<Eigen::Vector<double, N>>(
                fp->get_lengths_key(), mean, noiseModel::Gaussian::Covariance(cov)));
        }, fingers_[i]);
    }
}


void TendonHandModel::add_tension_priors(
    NonlinearFactorGraph& graph,
    const std::vector<VectorXGaussian>& tensions) const
{
    if (tensions.size() != fingers_.size())
        throw std::invalid_argument(
            "add_tension_priors: tensions size must match number of fingers");

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (tensions[i].mean.size() != N)
                throw std::invalid_argument(
                    "add_tension_priors: finger " + std::to_string(i) + " expects " +
                    std::to_string(N) + " tendon tensions, got " +
                    std::to_string(tensions[i].mean.size()));

            Eigen::Vector<double, N> mean = tensions[i].mean;
            Eigen::Matrix<double, N, N> cov = tensions[i].cov.topLeftCorner<N, N>();
            graph.add(PriorFactor<Eigen::Vector<double, N>>(
                fp->get_tensions_key(), mean, noiseModel::Gaussian::Covariance(cov)));
        }, fingers_[i]);
    }
}


TendonHandMarginals TendonHandModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    TendonHandMarginals out;
    out.fingers.reserve(fingers_.size());
    out.finger_names = finger_names_;
    for (const auto& finger : fingers_) {
        std::visit([&](const auto& fp) {
            out.fingers.push_back(fp->get_marginals(values, marginals));
        }, finger);
    }
    return out;
}


TendonHandMarginals TendonHandModel::get_marginals_means_only(
    const Values& values) const
{
    TendonHandMarginals out;
    out.fingers.reserve(fingers_.size());
    out.finger_names = finger_names_;
    // Zero-returning functors: extract means only, skipping the Marginals solve.
    // cov_of returns a 6x6 (pose block; the tension cov it also feeds is unused
    // for visualization). joint_of must be sized (6+N)x(6+N) per finger because
    // TendonFingerModel::get_J_pose_tensions reads block<6,N>(0,6)/block<N,N>(6,6),
    // so it is built inside the visit where the finger's N (NumTendons) is known.
    auto zero_cov = [](gtsam::Key) { return gtsam::Matrix::Zero(6, 6); };
    for (const auto& finger : fingers_) {
        std::visit([&](const auto& fp) {
            constexpr int N = std::remove_reference_t<decltype(*fp)>::NumTendons;
            auto zero_joint = [N](gtsam::Key, gtsam::Key) {
                return gtsam::Matrix::Zero(6 + N, 6 + N);
            };
            out.fingers.push_back(fp->get_marginals(values, zero_cov, zero_joint));
        }, finger);
    }
    return out;
}
