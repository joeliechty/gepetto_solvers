#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>

template<int N>
class TensionLimitFactor : public gtsam::NoiseModelFactor1<Eigen::Vector<double, N>> {
private:
    double alpha_;
    double q_min_;
    std::vector<int> active_indices_; // e.g., {5} from the PDF

public:
    TensionLimitFactor(gtsam::Key key, double alpha, double q_min,
                       const std::vector<int>& active_indices,
                       gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor1<Eigen::Vector<double, N>>(model, key),
          alpha_(alpha), q_min_(q_min), active_indices_(active_indices) {}

    gtsam::Vector evaluateError(const Eigen::Vector<double, N>& Q,
                                gtsam::OptionalMatrixType H = nullptr) const override {
        gtsam::Vector error = gtsam::Vector::Zero(active_indices_.size());

        if (H) *H = gtsam::Matrix::Zero(active_indices_.size(), N);

        for (size_t i = 0; i < active_indices_.size(); ++i) {
            int t = active_indices_[i];
            // e = exp(-alpha * (Q - Q_min))
            double e = std::exp(-alpha_ * (Q(t) - q_min_));
            error(i) = e;

            if (H) {
                // de/dQ = -alpha * exp(-alpha * (Q - Q_min))
                (*H)(i, t) = -alpha_ * e;
            }
        }
        return error;
    }
};