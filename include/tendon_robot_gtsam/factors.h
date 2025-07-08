
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
        
        Matrix66 d_delta_d_pose_0, d_delta_d_pose_1;
        Pose3 delta = pose_0.between(pose_1,
            H1 ? OptionalJacobian<6, 6>(d_delta_d_pose_0) : std::nullopt,
            H2 ? OptionalJacobian<6, 6>(d_delta_d_pose_1) : std::nullopt);

        Matrix66 d_twist_d_delta;
        Vector6 twist = Pose3::Logmap(delta, 
            H1 || H2 ? OptionalJacobian<6, 6>(d_twist_d_delta) : std::nullopt);
        
        Vector6 stress_mid = 0.5 * (stress_0 + stress_1);
        Vector6 nominal_strain = Vector6::Zero();
        nominal_strain[5] = 1.0;  // Straight rod: linear velocity in z direction only
        Vector6 twist_p = ds_ * (K_inv_ * stress_mid + nominal_strain);
        
        Matrix d_stress_p_d_pose_0, d_stress_p_d_pose_1, d_stress_p_d_wrench_sum;
        Vector6 stress_p = propagate_wrench_backward(pose_0, pose_1, wrench_1 + stress_1, 
            H1 ? &d_stress_p_d_pose_0 : nullptr,
            H2 ? &d_stress_p_d_pose_1 : nullptr,
            H4 || H5 ? &d_stress_p_d_wrench_sum : nullptr);
        
        Vector12 error;
        error.head<6>() = twist_p - twist;
        error.tail<6>() = stress_p - stress_0;

        if (H1) {
            *H1 = (Eigen::Matrix<double, 12, 6>() <<
                -d_twist_d_delta * d_delta_d_pose_0,
                d_stress_p_d_pose_0).finished();

            // Eigen::Matrix<double, 12, 6> H1_check = numericalDerivative11<Vector12, Pose3>(
            //     [&](const Pose3& pose_0_) {
            //         return this->evaluateError(pose_0_, pose_1, stress_0, stress_1, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_0);

            // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H2) {
            *H2 = (Eigen::Matrix<double, 12, 6>() <<
                -d_twist_d_delta * d_delta_d_pose_1,
                d_stress_p_d_pose_1).finished();

            // Eigen::Matrix<double, 12, 6> H2_check = numericalDerivative11<Vector12, Pose3>(
            //     [&](const Pose3& pose_1_) {
            //         return this->evaluateError(pose_0, pose_1_, stress_0, stress_1, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_1);

            // std::cout << "H2 check: " << (*H2 - H2_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H3) {
            double d_stress_mid_d_stress_0 = 0.5;

            *H3 = (Eigen::Matrix<double, 12, 6>() << 
                ds_ * K_inv_ * d_stress_mid_d_stress_0, 
                -Matrix6::Identity()).finished();

            // Eigen::Matrix<double, 12, 6> H3_check = numericalDerivative11<Vector12, Vector6>(
            //     [&](const Vector6& stress_0_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0_, stress_1, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, stress_0);

            // std::cout << "H3 check: " << (*H3 - H3_check).cwiseAbs().maxCoeff() << std::endl;
        }
        
        if (H4) {
            double d_stress_mid_d_stress_1 = 0.5;

            *H4 = (Eigen::Matrix<double, 12, 6>() << 
                ds_ * K_inv_ * d_stress_mid_d_stress_1,
                d_stress_p_d_wrench_sum).finished();

            // Eigen::Matrix<double, 12, 6> H4_check = numericalDerivative11<Vector12, Vector6>(
            //     [&](const Vector6& stress_1_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0, stress_1_, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, stress_1);

            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H5) {
            *H5 = (Eigen::Matrix<double, 12, 6>() << 
                Matrix6::Zero(),
                d_stress_p_d_wrench_sum).finished();

            // Eigen::Matrix<double, 12, 6> H5_check = numericalDerivative11<Vector12, Vector6>(
            //     [&](const Vector6& wrench_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0, stress_1, wrench_,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, wrench_1);

            // std::cout << "H5 check: " << (*H5 - H5_check).cwiseAbs().maxCoeff() << std::endl;
        }

        return error;
    }
};

Vector6 get_single_tendon_wrench(
    const double tension, 
    const Pose3 pose, 
    const Pose3 pose_other, 
    const Point3 hole, 
    const Point3 hole_other,
    const OptionalMatrixType d_wrench_d_tension,
    const OptionalMatrixType d_wrench_d_pose,
    const OptionalMatrixType d_wrench_d_pose_other)
{
    Matrix36 d_hole_other_world_d_pose_other;
    Point3 hole_other_world = pose_other.transformFrom(hole_other, d_hole_other_world_d_pose_other);
    
    Matrix36 d_hole_other_local_d_pose;
    Matrix3 d_hole_other_local_d_hole_other_world;
    Point3 hole_other_local = pose.transformTo(
        hole_other_world,
        d_hole_other_local_d_pose,
        d_hole_other_local_d_hole_other_world);

    Vector3 hole_diff = hole_other_local - hole;
    double norm = hole_diff.norm();
    Vector3 force_dir;
    Matrix3 d_force_dir_d_hole_diff;

    if (norm > 1e-9 && std::isfinite(norm)) {
        force_dir = normalize(hole_diff, d_force_dir_d_hole_diff);
    } else {
        force_dir = Vector3::Zero();
        d_force_dir_d_hole_diff = Matrix3::Zero();
    }

    Vector3 force = tension * force_dir;
    Matrix31 d_force_d_tension = force_dir;
    Matrix33 d_force_d_force_dir = tension * Matrix3::Identity();

    Matrix33 d_moment_d_force;
    Vector3 moment = cross(hole, force, nullptr, d_moment_d_force);

    Vector6 wrench;
    wrench << moment, force;

    if (d_wrench_d_tension) {
        *d_wrench_d_tension = (Vector6() << 
            d_moment_d_force * d_force_d_tension,
            d_force_d_tension).finished();  // 6×1
    }

    // d_wrench/d_pose
    if (d_wrench_d_pose) {
        // chain rule: d_force_dir/d_pose = d_force_dir/d_hole_diff * d_hole_diff/d_pose
        //            = d_force_dir/d_hole_diff * d_hole_other_local/d_pose
        Matrix36 d_force_dir_d_pose = d_force_dir_d_hole_diff * d_hole_other_local_d_pose;  // 3×6
        Matrix36 d_force_d_pose = d_force_d_force_dir * d_force_dir_d_pose;  // 3×6
        Matrix36 d_moment_d_pose = d_moment_d_force * d_force_d_pose;        // 3×6

        *d_wrench_d_pose = Matrix66();
        d_wrench_d_pose->block<3,6>(0,0) = d_moment_d_pose;
        d_wrench_d_pose->block<3,6>(3,0) = d_force_d_pose;
    }

    // d_wrench/d_pose_other
    if (d_wrench_d_pose_other) {
    Matrix36 d_force_dir_d_pose_other =
        d_force_dir_d_hole_diff *
        d_hole_other_local_d_hole_other_world *
        d_hole_other_world_d_pose_other;

    Matrix36 d_force_d_pose_other = d_force_d_force_dir * d_force_dir_d_pose_other;
    Matrix36 d_moment_d_pose_other = d_moment_d_force * d_force_d_pose_other;

    *d_wrench_d_pose_other = Matrix66();
    d_wrench_d_pose_other->block<3,6>(0,0) = d_moment_d_pose_other;
    d_wrench_d_pose_other->block<3,6>(3,0) = d_force_d_pose_other;
}
    return wrench;
}

class TendonDiscWrenchFactor: public NoiseModelFactorN<Pose3, Pose3, Pose3, Vector6, Vector4> {
    std::vector<Point3> holes_prev_;  // Previous disc hole location in local frame of previous disc, z = 0
    std::vector<Point3> holes_;   // Tip disc hole locations in local frame of tip disc
    std::vector<Point3> holes_next_;
    bool is_tip_;
public:

    using NoiseModelFactorN<Pose3, Pose3, Pose3, Vector6, Vector4>::evaluateError;
  
    TendonDiscWrenchFactor(Key pose_prev_key,
                           Key pose_key,
                           Key pose_next_key, // Set to dummy key if we are at the tip
                           Key wrench_key,
                           Key tensions_key,
                           bool is_tip,
                           std::vector<Point3> xy_holes_prev,
                           std::vector<Point3> xy_holes,
                           std::vector<Point3> xy_holes_next, // Not used if we are at the tip
                           const SharedNoiseModel& model): 
        NoiseModelFactor5(model, pose_prev_key, pose_key, pose_next_key, wrench_key, tensions_key) {
            is_tip_ = is_tip;
            holes_prev_ = xy_holes_prev;
            holes_ = xy_holes;
            holes_next_ = xy_holes_next;
    }

    Vector evaluateError(
        const Pose3& pose_prev, 
        const Pose3& pose, 
        const Pose3& pose_next, 
        const Vector6& wrench, 
        const Vector4& tensions,
        OptionalMatrixType H1, 
        OptionalMatrixType H2, 
        OptionalMatrixType H3, 
        OptionalMatrixType H4, 
        OptionalMatrixType H5) const override 
    {
        Vector6 wrench_total = Vector6::Zero();
        
        Matrix64 d_wrench_d_tensions;
        d_wrench_d_tensions.setZero();

        Matrix66 d_wrench_d_pose;
        d_wrench_d_pose.setZero();

        Matrix66 d_wrench_d_pose_prev;
        d_wrench_d_pose_prev.setZero();

        Matrix66 d_wrench_d_pose_next;
        d_wrench_d_pose_next.setZero();

        // Sum up all tendon wrenches on this disc
        for (int tendon_idx = 0; tendon_idx < tensions.size(); ++tendon_idx) {
            // Wrench from previous disc
            Matrix d_wrench_prev_d_tension, d_wrench_prev_d_pose, d_wrench_prev_d_pose_prev;

            Vector6 wrench_prev = get_single_tendon_wrench(
                tensions[tendon_idx],
                pose,
                pose_prev,
                holes_[tendon_idx],
                holes_prev_[tendon_idx],
                &d_wrench_prev_d_tension,
                &d_wrench_prev_d_pose,
                &d_wrench_prev_d_pose_prev);
            
            wrench_total += wrench_prev;
            Vector6 d_wrench_d_tension = d_wrench_prev_d_tension;
            d_wrench_d_pose += d_wrench_prev_d_pose;
            d_wrench_d_pose_prev += d_wrench_prev_d_pose_prev;
            
            // Wrench from next disc. Ignore if we are at the tip
            if (!is_tip_){
                Matrix d_wrench_next_d_tension, d_wrench_next_d_pose, d_wrench_next_d_pose_next;

                Vector6 wrench_next = get_single_tendon_wrench(
                    tensions[tendon_idx], 
                    pose,
                    pose_next, 
                    holes_[tendon_idx],
                    holes_next_[tendon_idx],
                    &d_wrench_next_d_tension,
                    &d_wrench_next_d_pose,
                    &d_wrench_next_d_pose_next);
                
                wrench_total += wrench_next;
                d_wrench_d_tension += d_wrench_next_d_tension;
                d_wrench_d_pose += d_wrench_next_d_pose;
                d_wrench_d_pose_next += d_wrench_next_d_pose_next;
            }

            d_wrench_d_tensions.col(tendon_idx) = d_wrench_d_tension;
        }

        Vector6 wrench_error = wrench - wrench_total;
    
        if (H1) {
            *H1 = -d_wrench_d_pose_prev;

            // Matrix6 H1_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_prev_) {
            //         return this->evaluateError(pose_prev_, pose, pose_next, wrench, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_prev);

            // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H2) {
            *H2 = -d_wrench_d_pose;

            // Matrix6 H2_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_) {
            //         return this->evaluateError(pose_prev, pose_, pose_next, wrench, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose);
            
            // std::cout << "H2 check: " << (*H2 - H2_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H3) {
            *H3 = -d_wrench_d_pose_next;

            // Matrix6 H3_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_next_) {
            //         return this->evaluateError(pose_prev, pose, pose_next_, wrench, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_next);

            // std::cout << "H3 check: " << (*H3 - H3_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H4) {
            *H4 = Matrix6::Identity();
            
            // Matrix6 H4_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& wrench_) {
            //         return this->evaluateError(pose_prev, pose, pose_next, wrench_, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, wrench);
            
            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H5) {
            *H5 = -d_wrench_d_tensions;

            // Matrix64 H5_check = numericalDerivative11<Vector6, Vector4>(
            //     [&](const Vector4& tensions_) {
            //         return this->evaluateError(pose_prev, pose, pose_next, wrench, tensions_,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, tensions);

            // std::cout << "H5 check: " << (*H5 - H5_check).cwiseAbs().maxCoeff() << std::endl;
        }

        return wrench_error;
    }
};
}