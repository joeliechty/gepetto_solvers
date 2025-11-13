using TendonWrenchBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector4, gtsam::Vector6>;

class TendonDiscWrenchFactor: public TendonWrenchBase {
    using TendonWrenchBase::evaluateError;

public:
    TendonDiscWrenchFactor(
        gtsam::Key pose_prev_key,
        gtsam::Key pose_key,
        gtsam::Key pose_next_key, // Set to dummy key if we are at the tip
        gtsam::Key wrench_key,
        gtsam::Key tensions_key,
        gtsam::Key external_wrench_key,
        const bool is_tip,
        const std::vector<gtsam::Point3>& holes_prev,
        const std::vector<gtsam::Point3>& holes,
        const std::vector<gtsam::Point3>& holes_next, // Not used if we are at the tip
        const gtsam::SharedNoiseModel& model);
        
    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_prev, 
        const gtsam::Pose3& pose, 
        const gtsam::Pose3& pose_next, 
        const gtsam::Vector6& wrench, 
        const gtsam::Vector4& tensions,
        const gtsam::Vector6& wrench_external,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4, 
        gtsam::OptionalMatrixType H5,
        gtsam::OptionalMatrixType H6) const override;

private:
    gtsam::Vector6 get_single_tendon_wrench(
        const double tension, 
        const gtsam::Pose3& pose, 
        const gtsam::Pose3& pose_other, 
        const gtsam::Point3& hole, 
        const gtsam::Point3& hole_other,
        gtsam::OptionalJacobian<6, 1> H_tension = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {},
        gtsam::OptionalJacobian<6, 6> H_pose_other = {}) const;

    bool is_tip_;
    std::vector<gtsam::Point3> holes_prev_;  // Previous disc hole location in local frame of previous disc, z = 0
    std::vector<gtsam::Point3> holes_;       // Tip disc hole locations in local frame of tip disc
    std::vector<gtsam::Point3> holes_next_;  // Disc hole locations in the next frame 
};
