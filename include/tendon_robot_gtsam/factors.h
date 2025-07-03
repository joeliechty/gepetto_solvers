
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

class CosseratRodFactor: public NoiseModelFactorN<Pose3, Pose3, Vector6, Vector6, Vector6> {
    double ds_;  // segment length
    gtsam::Matrix66 K_inv_;  // Assuming constant stiffness inverse per factor

public:

    using NoiseModelFactorN<Pose3, Pose3, Vector6, Vector6, Vector6>::evaluateError;
  
    CosseratRodFactor(Key pose_0_key,
                      Key pose_1_key,
                      Key stress_0_key,
                      Key stress_1_key,
                      Key wrench_key,
                      double ds,
                      const Matrix66& K_inv,
                      const SharedNoiseModel& model): 
        NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key),
        ds_(ds),
        K_inv_(K_inv) {}

    Vector evaluateError(
        const Pose3& pose_0, 
        const Pose3& pose_1, 
        const Vector6& stress_0, 
        const Vector6& stress_1, 
        const Vector6& wrench_1,
        OptionalMatrixType H1, 
        OptionalMatrixType H2, 
        OptionalMatrixType H3, 
        OptionalMatrixType H4,
        OptionalMatrixType H5) const override {
        
        Pose3 delta = pose_0.between(pose_1);
        Vector6 twist = Pose3::Logmap(delta);
        
        Vector6 stress_mid = 0.5 * (stress_0 + stress_1);
        
        Vector6 nominal_strain = Vector6::Zero();
        nominal_strain[5] = 1.0;  // Straight rod: linear velocity in z direction only

        Vector6 twist_predicted = ds_ * (K_inv_ * stress_mid + nominal_strain);

        Vector6 stress_predicted = propagate_wrench_backward(pose_0, pose_1, wrench_1 + stress_1, nullptr, nullptr, nullptr);
        
        Vector12 error;
        error.head<6>() = twist_predicted - twist;
        error.tail<6>() = stress_predicted - stress_0;
    

        if (H1) {
            *H1 = numericalDerivative11<Vector12, Pose3>(
                [&](const Pose3& pose_0_) {
                    return this->evaluateError(pose_0_, pose_1, stress_0, stress_1, wrench_1,
                                            nullptr, nullptr, nullptr, nullptr, nullptr);
                }, pose_0);

            // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H2) {
            *H2 = numericalDerivative11<Vector12, Pose3>(
                [&](const Pose3& pose_1_) {
                    return this->evaluateError(pose_0, pose_1_, stress_0, stress_1, wrench_1,
                                            nullptr, nullptr, nullptr, nullptr, nullptr);
                }, pose_1);

            // std::cout << "H2 check: " << (*H2 - H2_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H3) {
            *H3 = numericalDerivative11<Vector12, Vector6>(
                [&](const Vector6& stress_0_) {
                    return this->evaluateError(pose_0, pose_1, stress_0_, stress_1, wrench_1,
                                            nullptr, nullptr, nullptr, nullptr, nullptr);
                }, stress_0);

            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }
        
        if (H4) {
            *H4 = numericalDerivative11<Vector12, Vector6>(
                [&](const Vector6& stress_1_) {
                    return this->evaluateError(pose_0, pose_1, stress_0, stress_1_, wrench_1,
                                            nullptr, nullptr, nullptr, nullptr, nullptr);
                }, stress_1);

            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H5) {
            *H5 = numericalDerivative11<Vector12, Vector6>(
                [&](const Vector6& wrench_) {
                    return this->evaluateError(pose_0, pose_1, stress_0, stress_1, wrench_,
                                            nullptr, nullptr, nullptr, nullptr, nullptr);
                }, wrench_1);

            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }


        return error;
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
                              std::vector<Point3> xy_hole_loc_prev,
                              std::vector<Point3> xy_hole_loc_tip,
                              const SharedNoiseModel& model): 
        NoiseModelFactor4(model, pose_prev_key, pose_tip_key, tip_wrench_key, tensions_key) {
            hole_loc_prev_ = xy_hole_loc_prev;
            hole_loc_tip_ = xy_hole_loc_tip;
    }

    Vector evaluateError(const Pose3& pose_prev, const Pose3& pose_tip, const Vector6& tip_wrench, const Vector4& tensions,
        OptionalMatrixType H1, OptionalMatrixType H2, OptionalMatrixType H3, OptionalMatrixType H4) const override {
        
        // Get disc routing hole locations in world frame
        Vector6 wrench_total = Vector6::Zero();

        for (int tendon_idx = 0; tendon_idx < tensions.size(); ++tendon_idx) {
            Point3 hole_prev_world = pose_prev.transformFrom(hole_loc_prev_[tendon_idx]);
            Point3 hole_tip_world  = pose_tip.transformFrom(hole_loc_tip_[tendon_idx]);

            Vector3 delta = hole_prev_world - hole_tip_world;
            Vector3 force_world = tensions[tendon_idx] * delta.normalized();
            Vector3 force = pose_tip.rotation().unrotate(force_world);

            Vector3 moment = hole_loc_tip_[tendon_idx].cross(force);

            Vector6 wrench;
            wrench << moment, force;
            wrench_total += wrench;
        }

        Vector6 wrench_error = tip_wrench - wrench_total;
    
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