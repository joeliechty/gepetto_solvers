#include "PlatformWrenchBalanceFactor.h"

#include <gtsam/base/Matrix.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

using namespace gtsam;


PlatformWrenchBalanceFactor::PlatformWrenchBalanceFactor(
    Key w0_key, Key p0_key,
    Key w1_key, Key p1_key,
    Key w2_key, Key p2_key,
    Key w3_key, Key p3_key,
    Key w4_key, Key p4_key,
    Key w5_key, Key p5_key,
    Key w_platform_key,
    Key p_platform_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, 
        w0_key, p0_key,
        w1_key, p1_key,
        w2_key, p2_key,
        w3_key, p3_key,
        w4_key, p4_key,
        w5_key, p5_key,
        w_platform_key, p_platform_key) {}



Vector PlatformWrenchBalanceFactor::evaluateError(
    const Vector6& w0, const Pose3& p0,
    const Vector6& w1, const Pose3& p1,
    const Vector6& w2, const Pose3& p2,
    const Vector6& w3, const Pose3& p3,
    const Vector6& w4, const Pose3& p4,
    const Vector6& w5, const Pose3& p5,
    const Vector6& w_platform, const Pose3& p_platform,
    OptionalMatrixType H1, OptionalMatrixType H2,
    OptionalMatrixType H3, OptionalMatrixType H4,
    OptionalMatrixType H5, OptionalMatrixType H6,
    OptionalMatrixType H7, OptionalMatrixType H8,
    OptionalMatrixType H9, OptionalMatrixType H10,
    OptionalMatrixType H11, OptionalMatrixType H12,
    OptionalMatrixType H13, OptionalMatrixType H14) const
{
    Matrix36 d_tp_d_p_platform;
    const Vector3 tp = p_platform.translation(d_tp_d_p_platform);

    const Vector6* ws[6] = {&w0,&w1,&w2,&w3,&w4,&w5};
    const Pose3* ps[6]   = {&p0,&p1,&p2,&p3,&p4,&p5};
    
    OptionalMatrixType Hw[6] = {H1,H3,H5,H7,H9,H11};
    OptionalMatrixType Hp[6] = {H2,H4,H6,H8,H10,H12};

    Vector3 e_force = Vector3::Zero();
    Vector3 e_moment = Vector3::Zero();
    Matrix36 d_moment_d_platform_pose = Matrix36::Zero();

    for(int i=0;i<6;i++)
    {
        const Vector6& w = *ws[i];
        const Vector3 f = w.tail<3>();
        const Vector3 m = w.head<3>();

        Matrix36 d_f_d_w = Matrix36::Zero();
        d_f_d_w.rightCols(3) = Matrix3::Identity();

        Matrix36 d_m_d_w = Matrix36::Zero();
        d_m_d_w.leftCols(3) = Matrix3::Identity();

        Matrix36 d_ti_d_pi;
        const Vector3 ti = ps[i]->translation(d_ti_d_pi);
        Vector3 r = ti - tp;

        Matrix3 d_m_d_r, d_m_d_f;
        Vector3 rxf = cross(r, f, d_m_d_r, d_m_d_f);

        e_force  += f;
        e_moment += m + rxf;

        if(Hw[i])
        {
            Matrix6 H = Matrix6::Zero();
            H.topRows<3>() = d_m_d_w + d_m_d_f * d_f_d_w;
            H.bottomRows<3>() = d_f_d_w;

            *Hw[i] = H;
        }

        if(Hp[i])
        {
            Matrix6 H = Matrix6::Zero();
            H.topRows<3>() = d_m_d_r * d_ti_d_pi;

            *Hp[i] = H;
        }

        d_moment_d_platform_pose += d_m_d_r * (-d_tp_d_p_platform);
    }

    e_force  -= w_platform.tail<3>();
    e_moment -= w_platform.head<3>();

    Vector6 error;
    error << e_moment, e_force;

    if(H13)
        *H13 = -Matrix6::Identity();

    if(H14)
    {
        Matrix6 H = Matrix6::Zero();
        H.topRows<3>() = d_moment_d_platform_pose;
        *H14 = H;
    }

    return error;
}