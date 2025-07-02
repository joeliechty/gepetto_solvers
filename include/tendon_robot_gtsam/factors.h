







namespace gtsam{

Vector6 propagate_wrench_backward(
    const Pose3& pose,
    const Pose3& tip_pose,
    const Vector6& tip_wrench,
    const OptionalMatrixType d_wrench_d_pose,
    const OptionalMatrixType d_wrench_d_tip_pose,
    const OptionalMatrixType d_wrench_d_tip_wrench)
{
    Matrix66 d_tip_pose_inv_d_tip_pose;
    Matrix66 d_delta_d_tip_pose_inv, d_delta_d_pose;
    Matrix66 d_wrench_d_delta, d_wrench_d_tip_wrench_;

    Pose3 tip_pose_inv = tip_pose.inverse(
        (d_wrench_d_tip_pose ? &d_tip_pose_inv_d_tip_pose : nullptr));

    Pose3 delta = tip_pose_inv.compose(pose,
        (d_wrench_d_tip_pose ? &d_delta_d_tip_pose_inv : nullptr),
        (d_wrench_d_pose ? &d_delta_d_pose : nullptr));

    Vector6 wrench = delta.AdjointTranspose(
        tip_wrench,
        (d_wrench_d_pose || d_wrench_d_tip_pose ? &d_wrench_d_delta : nullptr),
        (d_wrench_d_tip_wrench ? &d_wrench_d_tip_wrench_ : nullptr));

    // Assign Jacobians if needed
    if (d_wrench_d_pose) {
        *d_wrench_d_pose = d_wrench_d_delta * d_delta_d_pose;
    }
    if (d_wrench_d_tip_pose) {
        *d_wrench_d_tip_pose = d_wrench_d_delta * d_delta_d_tip_pose_inv * d_tip_pose_inv_d_tip_pose;
    }
    if (d_wrench_d_tip_wrench) {
        *d_wrench_d_tip_wrench = d_wrench_d_tip_wrench_;
    }

    return wrench;
}

class CosseratTwistFactor: public NoiseModelFactorN<Pose3, Pose3, Pose3, Vector6> {
    double ds_;  // segment length
    gtsam::Matrix66 K_inv_;  // Assuming constant stiffness inverse per factor

public:

    using NoiseModelFactorN<Pose3, Pose3, Pose3, Vector6>::evaluateError;
  
    CosseratTwistFactor(Key pose_i_key,
                        Key pose_ip1_key,
                        Key pose_tip_key,
                        Key tip_wrench_key,
                        double ds,
                        const Matrix66& K_inv,
                        const SharedNoiseModel& model): 
        NoiseModelFactorN(model, pose_i_key, pose_ip1_key, pose_tip_key, tip_wrench_key),
        ds_(ds),
        K_inv_(K_inv) {}

    Vector evaluateError(const Pose3& pose_i, const Pose3& pose_ip1, const Pose3& pose_tip, const Vector6& tip_wrench,
        OptionalMatrixType H1, OptionalMatrixType H2, OptionalMatrixType H3, OptionalMatrixType H4) const override {
        
        Matrix6 d_delta_d_pose_i, d_delta_d_pose_ip1;
        Pose3 delta = pose_i.between(pose_ip1, d_delta_d_pose_i, d_delta_d_pose_ip1);

        Matrix6 d_twist_d_delta;
        Vector6 twist = Pose3::Logmap(delta, d_twist_d_delta);
        
        Matrix d_stress_i_d_pose_i, d_stress_i_d_pose_tip, d_stress_i_d_tip_wrench;
        Vector6 stress_i = propagate_wrench_backward(
            pose_i,
            pose_tip,
            tip_wrench,
            H1 ? &d_stress_i_d_pose_i : nullptr,
            H3 ? &d_stress_i_d_pose_tip : nullptr,
            H4 ? &d_stress_i_d_tip_wrench : nullptr);

        Matrix d_stress_ip1_d_pose_ip1, d_stress_ip1_d_pose_tip, d_stress_ip1_d_tip_wrench;
        Vector6 stress_ip1 = propagate_wrench_backward(
            pose_ip1, 
            pose_tip, 
            tip_wrench, 
            H2 ? &d_stress_ip1_d_pose_ip1 : nullptr,
            H3 ? &d_stress_ip1_d_pose_tip : nullptr,
            H4 ? &d_stress_ip1_d_tip_wrench : nullptr);
        
        Vector6 stress = 0.5 * (stress_i + stress_ip1);  // More accurate than just stress_i: kind of like a midpoint rule
        
        Vector6 nominal_strain;
        nominal_strain.head<3>() << 0.0, 0.0, 0.0;          // First three elements from curvature
        nominal_strain.tail<3>() << 0.0, 0.0, 1.0;

        Vector6 predicted_twist = ds_ * (K_inv_ * stress + nominal_strain);
        Vector6 twist_error = predicted_twist - twist;
    

        if (H1) {
            *H1 = ds_ * K_inv_ * 0.5 * d_stress_i_d_pose_i - d_twist_d_delta * d_delta_d_pose_i;

            // Matrix6 H1_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_i_) {
            //         return this->evaluateError(pose_i_, pose_ip1, pose_tip, tip_wrench, curvature,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_i);

            // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H2) {
            *H2 = ds_ * K_inv_ * 0.5 * d_stress_ip1_d_pose_ip1 - d_twist_d_delta * d_delta_d_pose_ip1;

            // Matrix6 H2_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_ip1_) {
            //         return this->evaluateError(pose_i, pose_ip1_, pose_tip, tip_wrench, curvature,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_ip1);

            // std::cout << "H2 check: " << (*H2 - H2_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H3) {
            *H3 = ds_ * K_inv_ * 0.5 * (d_stress_i_d_pose_tip + d_stress_ip1_d_pose_tip);

            // Matrix6 H3_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_tip_) {
            //         return this->evaluateError(pose_i, pose_ip1, pose_tip_, tip_wrench, curvature,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_tip);

            // std::cout << "H3 check: " << (*H3 - H3_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H4) {
            *H4 = ds_ * K_inv_ * 0.5 * (d_stress_i_d_tip_wrench + d_stress_ip1_d_tip_wrench);

            // Matrix6 H4_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& tip_wrench_) {
            //         return this->evaluateError(pose_i, pose_ip1, pose_tip, tip_wrench_, curvature,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, tip_wrench);

            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }
        
        return twist_error;
    }
};

class TipTendonDiscWrenchFactor: public NoiseModelFactorN<Pose3, Pose3, Vector6, Vector4> {
    std::vector<Point3> hole_loc_prev_;  // Previous disc hole location in local frame of previous disc, z = 0
    std::vector<Point3> hole_loc_tip_;   // Tip disc hole locations in local frame of tip disc

public:

    using NoiseModelFactorN<Pose3, Pose3, Vector6, Vector4>::evaluateError;
  
    TipTendonDiscWrenchFactor(Key pose_prev_key,
                              Key pose_tip_key,
                              Key tip_wrench_key,
                              Key tensions_key,
                              std::vector<Point2> xy_hole_loc_prev,
                              std::vector<Point2> xy_hole_loc_tip,
                              const SharedNoiseModel& model): 
        NoiseModelFactor4(model, pose_prev_key, pose_tip_key, tip_wrench_key, tensions_key) {

        auto lift_to_3D = [](const Point2& v) {
            return Point3(v(0), v(1), 0.0);
        };

        for (const auto& h : xy_hole_loc_prev)
            hole_loc_prev_.push_back(lift_to_3D(h));

        for (const auto& h : xy_hole_loc_tip)
            hole_loc_tip_.push_back(lift_to_3D(h));  
    }

    Vector evaluateError(const Pose3& pose_prev, const Pose3& pose_tip, const Vector6& tip_wrench, const Vector4& tensions,
        OptionalMatrixType H1, OptionalMatrixType H2, OptionalMatrixType H3, OptionalMatrixType H4) const override {
        
        // Get disc routing hole locations in world frame
        Point3 hole_0_prev_world = pose_prev.transformFrom(hole_loc_prev_[0]);
        Point3 hole_0_tip_world = pose_tip.transformFrom(hole_loc_tip_[0]);

        Vector3 force_0_world = tensions[0] * (hole_0_prev_world - hole_0_tip_world).normalized();
        Vector3 force_0 = pose_tip.rotation().unrotate(force_0_world);
        
        Vector3 moment_0 = hole_loc_tip_[0].cross(force_0);
        
        Vector6 wrench_0;
        wrench_0 << moment_0, force_0;

        Vector6 wrench_error = tip_wrench - wrench_0;
    

        if (H1) {
            *H1 = numericalDerivative11<Vector6, Pose3>(
                [&](const Pose3& pose_prev_) {
                    return this->evaluateError(pose_prev_, pose_tip, tip_wrench, tensions,
                                               nullptr, nullptr, nullptr, nullptr);
                }, pose_prev);
        }

        if (H2) {
            *H2 = numericalDerivative11<Vector6, Pose3>(
                [&](const Pose3& pose_tip_) {
                    return this->evaluateError(pose_prev, pose_tip_, tip_wrench, tensions,
                                               nullptr, nullptr, nullptr, nullptr);
                }, pose_tip);
        }

        if (H3) {
            *H3 = numericalDerivative11<Vector6, Vector6>(
                [&](const Vector6& tip_wrench_) {
                    return this->evaluateError(pose_prev, pose_tip, tip_wrench_, tensions,
                                               nullptr, nullptr, nullptr, nullptr);
                }, tip_wrench);
        }

        if (H4) {
            *H4 = numericalDerivative11<Vector6, Vector4>(
                [&](const Vector4& tensions_) {
                    return this->evaluateError(pose_prev, pose_tip, tip_wrench, tensions_,
                                               nullptr, nullptr, nullptr, nullptr);
                }, tensions);
        }

        return wrench_error;
    }
};
}