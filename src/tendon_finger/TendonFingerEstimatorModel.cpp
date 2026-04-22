#include "TendonFingerEstimatorModel.h"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

template<int N>
gtsam::NonlinearFactorGraph TendonFingerEstimatorModel<N>::build_estimation_graph(
    const VectorNGaussian<N>& tensions,
    double measured_bend,
    const gtsam::SharedNoiseModel& bend_noise) const
{
    // 1. Generate the standard tendon graph from the base model
    gtsam::NonlinearFactorGraph graph = this->build_graph(tensions);

    // 2. Prevent segment fualts if only a one or two disk robot config is passed in
    if (this->num_discs_ < 3) {
        throw std::invalid_argument(
            "TendonFingerEstimatorModel requires at least 3 discs to place a factor "
            "between index 1 and 2. Currently has: " + std::to_string(this->num_discs_));
    }

    // 3. Extract the pose indicies for the first knuckle (disk_1 to disk_2)
    int pose_idx_proximal = this->tendon_config_.disc_pose_idx[1];
    int pose_idx_distal = this->tendon_config_.disc_pose_idx[2];

    // 4. Generate the corresponding gtsam symbol keys using the rod model
    gtsam::Key key_1 = this->rod_->get_pose_key(pose_idx_proximal);
    gtsam::Key key_2 = this->rod_->get_pose_key(pose_idx_distal);

    // 5. Inject the sensor reading into the graph
    graph.add(KnuckleBendFactor(key_1, key_2, measured_bend, bend_noise));

    return graph;

}

// Explicit instantiations
template class TendonFingerEstimatorModel<1>;
template class TendonFingerEstimatorModel<2>;
template class TendonFingerEstimatorModel<3>;
template class TendonFingerEstimatorModel<4>;
template class TendonFingerEstimatorModel<5>;
template class TendonFingerEstimatorModel<6>;
template class TendonFingerEstimatorModel<7>;
template class TendonFingerEstimatorModel<8>;
template class TendonFingerEstimatorModel<9>;
template class TendonFingerEstimatorModel<10>;