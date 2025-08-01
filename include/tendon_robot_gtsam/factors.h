
namespace gtsam{

Vector6 propagate_wrench_backward(
    const Pose3& pose,
    const Pose3& tip_pose,
    const Vector6& tip_wrench,
    OptionalJacobian<6, 6> H_pose = {},
    OptionalJacobian<6, 6> H_tip_pose = {},
    OptionalJacobian<6, 6> H_tip_wrench = {})
{
    Matrix66 d_tip_pose_inv_d_tip_pose;
    Matrix66 d_delta_d_tip_pose_inv, d_delta_d_pose;
    Matrix66 d_wrench_d_delta, d_wrench_d_tip_wrench_;

    Pose3 tip_pose_inv = tip_pose.inverse(
        (H_pose ? &d_tip_pose_inv_d_tip_pose : 0));

    Pose3 delta = tip_pose_inv.compose(pose,
        (H_tip_pose ? &d_delta_d_tip_pose_inv : 0),
        (H_pose ? &d_delta_d_pose : 0));

    Vector6 wrench = delta.AdjointTranspose(
        tip_wrench,
        (H_pose || H_tip_pose ? &d_wrench_d_delta : 0),
        (H_tip_wrench ? &d_wrench_d_tip_wrench_ : 0));

    // Assign Jacobians if needed
    if (H_pose) {
        *H_pose = d_wrench_d_delta * d_delta_d_pose;
    }
    if (H_tip_pose) {
        *H_tip_pose = d_wrench_d_delta * d_delta_d_tip_pose_inv * d_tip_pose_inv_d_tip_pose;
    }
    if (H_tip_wrench) {
        *H_tip_wrench = d_wrench_d_tip_wrench_;
    }

    return wrench;
}

class CosseratRodTwistFactor: public NoiseModelFactorN<Pose3, Pose3, Vector6, Vector6> {
    double ds_;  // segment length
    gtsam::Matrix66 K_inv_;  // Assuming constant stiffness inverse per factor

public:

    using NoiseModelFactorN<Pose3, Pose3, Vector6, Vector6>::evaluateError;
  
    CosseratRodTwistFactor(Key pose_0_key,
                           Key pose_1_key,
                           Key stress_0_key,
                           Key stress_1_key,
                           double ds,
                           const Matrix66& K_inv,
                           const SharedNoiseModel& model): 
        NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key),
        ds_(ds),
        K_inv_(K_inv) {}

    Vector evaluateError(
        const Pose3& pose_0, 
        const Pose3& pose_1, 
        const Vector6& stress_0, 
        const Vector6& stress_1, 
        OptionalMatrixType H1, 
        OptionalMatrixType H2, 
        OptionalMatrixType H3, 
        OptionalMatrixType H4) const override {
        
        Matrix66 d_delta_d_pose_0, d_delta_d_pose_1;
        Pose3 delta = pose_0.between(pose_1,
            H1 ? &d_delta_d_pose_0 : 0,
            H2 ? &d_delta_d_pose_1 : 0);

        Matrix66 d_twist_d_delta;
        Vector6 twist = Pose3::Logmap(delta, 
            H1 || H2 ? &d_twist_d_delta : 0);
        
        Vector6 stress_mid = 0.5 * (stress_0 + stress_1);
        Vector6 nominal_strain = Vector6::Zero();
        nominal_strain[5] = 1.0;  // Straight rod: linear velocity in z direction only
        Vector6 twist_p = ds_ * (K_inv_ * stress_mid + nominal_strain);
        
        Vector6 twist_error = twist_p - twist;

        if (H1) {
            *H1 = -d_twist_d_delta * d_delta_d_pose_0;

            // Eigen::Matrix<double, 6, 6> H1_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_0_) {
            //         return this->evaluateError(pose_0_, pose_1, stress_0, stress_1,
            //                                 nullptr, nullptr, nullptr, nullptr);
            //     }, pose_0);
            
            // *H1 = H1_check;

            // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H2) {
            *H2 = -d_twist_d_delta * d_delta_d_pose_1;

            // Eigen::Matrix<double, 6, 6> H2_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_1_) {
            //         return this->evaluateError(pose_0, pose_1_, stress_0, stress_1,
            //                                 nullptr, nullptr, nullptr, nullptr);
            //     }, pose_1);
            
            // *H2 = H2_check;

            // std::cout << "H2 check: " << (*H2 - H2_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H3) {
            double d_stress_mid_d_stress_0 = 0.5;
            *H3 = ds_ * K_inv_ * d_stress_mid_d_stress_0;

            // Eigen::Matrix<double, 6, 6> H3_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& stress_0_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0_, stress_1,
            //                                 nullptr, nullptr, nullptr, nullptr);
            //     }, stress_0);

            // *H3 = H3_check;

            // std::cout << "H3 check: " << (*H3 - H3_check).cwiseAbs().maxCoeff() << std::endl;
        }
        
        if (H4) {
            double d_stress_mid_d_stress_1 = 0.5;
            *H4 = ds_ * K_inv_ * d_stress_mid_d_stress_1;

            // Eigen::Matrix<double, 6, 6> H4_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& stress_1_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0, stress_1_,
            //                                 nullptr, nullptr, nullptr, nullptr);
            //     }, stress_1);
            
            // *H4 = H4_check;

            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }

        return twist_error;
    }
};

class CosseratRodStressFactor: public NoiseModelFactorN<Pose3, Pose3, Vector6, Vector6, Vector6> {

public:

    using NoiseModelFactorN<Pose3, Pose3, Vector6, Vector6, Vector6>::evaluateError;
  
    CosseratRodStressFactor(Key pose_0_key,
                            Key pose_1_key,
                            Key stress_0_key,
                            Key stress_1_key,
                            Key wrench_key,
                            const SharedNoiseModel& model): 
        NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key) {}

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
        
        Matrix6 d_stress_p_d_pose_0, d_stress_p_d_pose_1, d_stress_p_d_wrench_sum;
        Vector6 stress_p = propagate_wrench_backward(pose_0, pose_1, wrench_1 + stress_1, 
            H1 ? &d_stress_p_d_pose_0 : 0,
            H2 ? &d_stress_p_d_pose_1 : 0,
            H4 || H5 ? &d_stress_p_d_wrench_sum : 0);
        
        Vector6 stress_error = stress_p - stress_0;

        if (H1) {
            *H1 = d_stress_p_d_pose_0;

            // Eigen::Matrix<double, 6, 6> H1_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_0_) {
            //         return this->evaluateError(pose_0_, pose_1, stress_0, stress_1, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_0);
            
            // *H1 = H1_check;

            // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H2) {
            *H2 = d_stress_p_d_pose_1;

            // Eigen::Matrix<double, 6, 6> H2_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_1_) {
            //         return this->evaluateError(pose_0, pose_1_, stress_0, stress_1, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_1);
            
            // *H2 = H2_check;

            // std::cout << "H2 check: " << (*H2 - H2_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H3) {
            *H3 = -Matrix6::Identity();

            // Eigen::Matrix<double, 6, 6> H3_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& stress_0_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0_, stress_1, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, stress_0);
            
            // *H3 = H3_check;

            // std::cout << "H3 check: " << (*H3 - H3_check).cwiseAbs().maxCoeff() << std::endl;
        }
        
        if (H4) {
            *H4 = d_stress_p_d_wrench_sum;

            // Eigen::Matrix<double, 6, 6> H4_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& stress_1_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0, stress_1_, wrench_1,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, stress_1);
            
            // *H4 = H4_check;

            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H5) {
            *H5 = d_stress_p_d_wrench_sum;

            // Eigen::Matrix<double, 6, 6> H5_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& wrench_) {
            //         return this->evaluateError(pose_0, pose_1, stress_0, stress_1, wrench_,
            //                                 nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, wrench_1);
            
            // *H5 = H5_check;

            // std::cout << "H5 check: " << (*H5 - H5_check).cwiseAbs().maxCoeff() << std::endl;
        }

        return stress_error;
    }
};

Vector6 get_single_tendon_wrench(
    const double tension, 
    const Pose3 pose, 
    const Pose3 pose_other, 
    const Point3 hole, 
    const Point3 hole_other,
    OptionalJacobian<6, 1> H_tension = {},
    OptionalJacobian<6, 6> H_pose = {},
    OptionalJacobian<6, 6> H_pose_other = {})
{
    Matrix36 d_hole_other_world_d_pose_other;
    Point3 hole_other_world = pose_other.transformFrom(hole_other, 
        H_pose_other ? &d_hole_other_world_d_pose_other : 0);
    
    Matrix36 d_hole_other_local_d_pose;
    Matrix3 d_hole_other_local_d_hole_other_world;
    Point3 hole_other_local = pose.transformTo(hole_other_world,
        H_pose? &d_hole_other_local_d_pose : 0,
        d_hole_other_local_d_hole_other_world);

    Vector3 hole_diff = hole_other_local - hole;
    double norm = hole_diff.norm();

    Vector3 force_dir;
    Matrix3 d_force_dir_d_hole_diff = Matrix3::Zero();
    
    bool valid = hole_diff.allFinite() && norm > 1e-3;
    
    if (valid) {
        force_dir = normalize(hole_diff, H_pose || H_pose_other ? &d_force_dir_d_hole_diff : 0);
    } else {
        force_dir = Vector3::Zero();
    }

    Vector3 force = tension * force_dir;
    Matrix31 d_force_d_tension = force_dir;
    Matrix33 d_force_d_force_dir = tension * Matrix3::Identity();

    Matrix33 d_moment_d_force;
    Vector3 moment = cross(hole, force, nullptr, 
         H_tension || H_pose || H_pose_other ? &d_moment_d_force : 0);

    Vector6 wrench;
    wrench << moment, force;

    if (H_tension) {
        H_tension->head<3>() = d_moment_d_force * d_force_d_tension;
        H_tension->tail<3>() = d_force_d_tension;
    }

    if (H_pose) {
        Matrix36 d_force_dir_d_pose = d_force_dir_d_hole_diff * d_hole_other_local_d_pose;
        Matrix36 d_force_d_pose = d_force_d_force_dir * d_force_dir_d_pose;
        Matrix36 d_moment_d_pose = d_moment_d_force * d_force_d_pose;

        H_pose->block<3,6>(0,0) = d_moment_d_pose;
        H_pose->block<3,6>(3,0) = d_force_d_pose;
    }

    if (H_pose_other) {
        Matrix36 d_force_dir_d_pose_other =
            d_force_dir_d_hole_diff *
            d_hole_other_local_d_hole_other_world *
            d_hole_other_world_d_pose_other;

        Matrix36 d_force_d_pose_other = d_force_d_force_dir * d_force_dir_d_pose_other;
        Matrix36 d_moment_d_pose_other = d_moment_d_force * d_force_d_pose_other;

        H_pose_other->block<3,6>(0,0) = d_moment_d_pose_other;
        H_pose_other->block<3,6>(3,0) = d_force_d_pose_other;
    }

    return wrench;
}

class TendonDiscWrenchFactor: public NoiseModelFactorN<Pose3, Pose3, Pose3, Vector6, Vector4, Vector6> {
    std::vector<Point3> holes_prev_;  // Previous disc hole location in local frame of previous disc, z = 0
    std::vector<Point3> holes_;   // Tip disc hole locations in local frame of tip disc
    std::vector<Point3> holes_next_;
    bool is_tip_;
public:

    using NoiseModelFactorN<Pose3, Pose3, Pose3, Vector6, Vector4, Vector6>::evaluateError;
  
    TendonDiscWrenchFactor(Key pose_prev_key,
                           Key pose_key,
                           Key pose_next_key, // Set to dummy key if we are at the tip
                           Key wrench_key,
                           Key tensions_key,
                           Key external_wrench_key,
                           bool is_tip,
                           std::vector<Point3> xy_holes_prev,
                           std::vector<Point3> xy_holes,
                           std::vector<Point3> xy_holes_next, // Not used if we are at the tip
                           const SharedNoiseModel& model): 
        NoiseModelFactor5(model, pose_prev_key, pose_key, pose_next_key, wrench_key, tensions_key, external_wrench_key) {
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
        const Vector6& wrench_external,
        OptionalMatrixType H1, 
        OptionalMatrixType H2, 
        OptionalMatrixType H3, 
        OptionalMatrixType H4, 
        OptionalMatrixType H5,
        OptionalMatrixType H6) const override 
    {
        Vector6 wrench_tendons = Vector6::Zero();
        
        Matrix64 d_wrench_d_tensions = Matrix64::Zero();
        Matrix66 d_wrench_d_pose = Matrix66::Zero();
        Matrix66 d_wrench_d_pose_prev = Matrix66::Zero();
        Matrix66 d_wrench_d_pose_next = Matrix66::Zero();

        // Sum up all tendon wrenches on this disc
        for (int tendon_idx = 0; tendon_idx < tensions.size(); ++tendon_idx) {
            // Wrench from previous disc
            Vector6 d_wrench_prev_d_tension;
            Matrix6 d_wrench_prev_d_pose, d_wrench_prev_d_pose_prev;

            Vector6 wrench_prev = get_single_tendon_wrench(
                tensions[tendon_idx],
                pose,
                pose_prev,
                holes_[tendon_idx],
                holes_prev_[tendon_idx],
                H5 ? &d_wrench_prev_d_tension : 0,
                H2 ? &d_wrench_prev_d_pose : 0,
                H1 ? &d_wrench_prev_d_pose_prev : 0);
            
            wrench_tendons += wrench_prev;
            Vector6 d_wrench_d_tension = d_wrench_prev_d_tension;
            d_wrench_d_pose += d_wrench_prev_d_pose;
            d_wrench_d_pose_prev += d_wrench_prev_d_pose_prev;
            
            // Wrench from next disc. Ignore if we are at the tip
            if (!is_tip_){
                Vector6 d_wrench_next_d_tension;
                Matrix6 d_wrench_next_d_pose, d_wrench_next_d_pose_next;

                Vector6 wrench_next = get_single_tendon_wrench(
                    tensions[tendon_idx], 
                    pose,
                    pose_next, 
                    holes_[tendon_idx],
                    holes_next_[tendon_idx],
                    H5 ? &d_wrench_next_d_tension : 0,
                    H2 ? &d_wrench_next_d_pose : 0,
                    H3 ? &d_wrench_next_d_pose_next : 0);
                
                wrench_tendons += wrench_next;
                d_wrench_d_tension += d_wrench_next_d_tension;
                d_wrench_d_pose += d_wrench_next_d_pose;
                d_wrench_d_pose_next += d_wrench_next_d_pose_next;
            }

            d_wrench_d_tensions.col(tendon_idx) = d_wrench_d_tension;
        }

        Vector6 wrench_error = wrench - wrench_tendons - wrench_external;
    
        if (H1) {
            *H1 = -d_wrench_d_pose_prev;

            // Matrix6 H1_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_prev_) {
            //         return this->evaluateError(pose_prev_, pose, pose_next, wrench, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_prev);
            
            // *H1 = H1_check;

            // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H2) {
            *H2 = -d_wrench_d_pose;

            // Matrix6 H2_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_) {
            //         return this->evaluateError(pose_prev, pose_, pose_next, wrench, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose);
            
            // *H2 = H2_check;

            // std::cout << "H2 check: " << (*H2 - H2_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H3) {
            *H3 = -d_wrench_d_pose_next;

            // Matrix6 H3_check = numericalDerivative11<Vector6, Pose3>(
            //     [&](const Pose3& pose_next_) {
            //         return this->evaluateError(pose_prev, pose, pose_next_, wrench, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, pose_next);
            
            // *H3 = H3_check;

            // std::cout << "H3 check: " << (*H3 - H3_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H4) {
            *H4 = Matrix6::Identity();
            
            // Matrix6 H4_check = numericalDerivative11<Vector6, Vector6>(
            //     [&](const Vector6& wrench_) {
            //         return this->evaluateError(pose_prev, pose, pose_next, wrench_, tensions,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, wrench);

            // *H4 = H4_check;
            
            // std::cout << "H4 check: " << (*H4 - H4_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H5) {
            *H5 = -d_wrench_d_tensions;

            // Matrix64 H5_check = numericalDerivative11<Vector6, Vector4>(
            //     [&](const Vector4& tensions_) {
            //         return this->evaluateError(pose_prev, pose, pose_next, wrench, tensions_,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, tensions);
            
            // *H5 = H5_check;

            // std::cout << "H5 check: " << (*H5 - H5_check).cwiseAbs().maxCoeff() << std::endl;
        }

        if (H6) {
            *H6 = -Matrix6::Identity();

            // Matrix64 H5_check = numericalDerivative11<Vector6, Vector4>(
            //     [&](const Vector4& tensions_) {
            //         return this->evaluateError(pose_prev, pose, pose_next, wrench, tensions_,
            //                                    nullptr, nullptr, nullptr, nullptr, nullptr);
            //     }, tensions);
            
            // *H5 = H5_check;

            // std::cout << "H5 check: " << (*H5 - H5_check).cwiseAbs().maxCoeff() << std::endl;
        }

        return wrench_error;
    }
};

class LastTipStateFactor: public NoiseModelFactorN<Pose3, Pose3, Pose3, Pose3> {
public:
    using NoiseModelFactorN<Pose3, Pose3, Pose3, Pose3>::evaluateError;
  
    LastTipStateFactor(Key pose_im3_key,
                       Key pose_im2_key,
                       Key pose_im1_key,
                       Key pose_i_key,
                       const SharedNoiseModel& model): 
        NoiseModelFactor4(model, pose_im3_key, pose_im2_key, pose_im1_key, pose_i_key) {}

    Vector evaluateError(
        const Pose3& pose_im3,
        const Pose3& pose_im2,
        const Pose3& pose_im1,
        const Pose3& pose_i,
        OptionalMatrixType H1, 
        OptionalMatrixType H2,
        OptionalMatrixType H3,
        OptionalMatrixType H4) const override 
    {  
        Matrix36 d_p_im3;
        Vector3 p_im3 = pose_im3.translation(d_p_im3);

        Matrix36 d_p_im2;
        Vector3 p_im2 = pose_im2.translation(d_p_im2);

        Matrix36 d_p_im1;
        Vector3 p_im1 = pose_im1.translation(d_p_im1);

        Matrix36 d_p_i;
        Vector3 p_i   = pose_i.translation(d_p_i);

        Vector3 jerk = p_i - 3.0 * p_im1 + 3.0 * p_im2 - p_im3;

        if (H1) {
            *H1 = -d_p_im3;
        }

        if (H2) {
            *H2 = 3.0 * d_p_im2;
        }

        if (H3) {
            *H3 = -3.0 * d_p_im1;
        }

        if (H4) {
            *H4 = d_p_i;
        }
        

        return jerk;
    }
};
}