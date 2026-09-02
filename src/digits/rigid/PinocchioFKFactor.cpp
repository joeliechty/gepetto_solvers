// The kinematics likelihood factor, and the shared model behind it.
//
// This is the ONLY translation unit that includes Pinocchio. That is not an
// accident of layering: Pinocchio needs BOOST_MPL_LIMIT_LIST_SIZE raised (its
// joint variant has more alternatives than Boost.MPL's default list) and it must
// see the same Eigen GTSAM was built against, so keeping its headers in one
// place keeps both constraints in one place too. See the notes in CMakeLists.txt
// and the conda setup scripts.

#include "gepetto_solvers/digits/rigid/PinocchioFKFactor.h"

#include <pinocchio/multibody.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <stdexcept>

using namespace gtsam;

namespace gepetto_solvers {

namespace {

// Pinocchio stacks a spatial velocity [v; w]; GTSAM's Pose3 tangent is [w; v].
// Multiplying on the left by this exchanges the two blocks.
//
// Applied as a row swap rather than a matrix product -- same result, and it
// makes the operation legible at the call site.
inline void swap_linear_angular_rows(Matrix& J) {
    Matrix top = J.topRows(3);
    J.topRows(3) = J.bottomRows(3);
    J.bottomRows(3) = top;
}

}  // namespace


// --- RigidChainModel ------------------------------------------------------

RigidChainModel::RigidChainModel() : model_(std::make_unique<pinocchio::Model>()) {}
RigidChainModel::~RigidChainModel() = default;


std::shared_ptr<RigidChainModel> RigidChainModel::from_urdf_xml(
    const std::string& urdf_xml)
{
    // Not make_shared: the constructor is private.
    std::shared_ptr<RigidChainModel> chain(new RigidChainModel());
    pinocchio::urdf::buildModelFromXML(urdf_xml, *chain->model_);
    if (chain->model_->njoints <= 1)
        throw std::invalid_argument(
            "RigidChainModel: the URDF parsed to no movable joints. Check that "
            "its joints are revolute/prismatic rather than fixed.");
    return chain;
}


std::shared_ptr<RigidChainModel> RigidChainModel::from_urdf_file(
    const std::string& path)
{
    std::shared_ptr<RigidChainModel> chain(new RigidChainModel());
    pinocchio::urdf::buildModel(path, *chain->model_);
    if (chain->model_->njoints <= 1)
        throw std::invalid_argument(
            "RigidChainModel: " + path + " parsed to no movable joints.");
    return chain;
}


int RigidChainModel::frame_id(const std::string& name) const {
    if (!model_->existFrame(name))
        throw std::invalid_argument(
            "RigidChainModel: no frame named \"" + name + "\" in this model. A "
            "mistyped frame would otherwise read as a pose at the origin.");
    return static_cast<int>(model_->getFrameId(name));
}


std::pair<int, int> RigidChainModel::joint_indices(const std::string& name) const {
    if (!model_->existJointName(name))
        throw std::invalid_argument(
            "RigidChainModel: no joint named \"" + name + "\" in this model.");
    const auto jid = model_->getJointId(name);
    const int nq = model_->joints[jid].nq();
    const int nv = model_->joints[jid].nv();
    if (nq != 1 || nv != 1)
        throw std::invalid_argument(
            "RigidChainModel: joint \"" + name + "\" has nq=" +
            std::to_string(nq) + ", nv=" + std::to_string(nv) +
            "; only 1-DOF joints are supported, because everything downstream "
            "indexes one scalar per joint. A `continuous` URDF joint is nq=2 "
            "(cos, sin) -- give it limits to make it `revolute`.");
    return {static_cast<int>(model_->joints[jid].idx_q()),
            static_cast<int>(model_->joints[jid].idx_v())};
}


int RigidChainModel::nq() const { return model_->nq; }
int RigidChainModel::nv() const { return model_->nv; }

std::vector<double> RigidChainModel::lower_position_limits() const {
    return {model_->lowerPositionLimit.data(),
            model_->lowerPositionLimit.data() + model_->lowerPositionLimit.size()};
}

std::vector<double> RigidChainModel::upper_position_limits() const {
    return {model_->upperPositionLimit.data(),
            model_->upperPositionLimit.data() + model_->upperPositionLimit.size()};
}


// --- PinocchioFKFactor ----------------------------------------------------

PinocchioFKFactor::PinocchioFKFactor(Key wrist_key,
                                     Key joint_key,
                                     Key site_key,
                                     std::shared_ptr<const RigidChainModel> chain,
                                     int frame_id,
                                     std::vector<int> q_index,
                                     std::vector<int> v_index,
                                     const SharedNoiseModel& noise)
:
    Base(noise, wrist_key, joint_key, site_key),
    chain_(std::move(chain)),
    frame_id_(frame_id),
    q_index_(std::move(q_index)),
    v_index_(std::move(v_index))
{
    if (!chain_)
        throw std::invalid_argument("PinocchioFKFactor: null model");
    if (q_index_.size() != v_index_.size())
        throw std::invalid_argument(
            "PinocchioFKFactor: q_index has " + std::to_string(q_index_.size()) +
            " entries and v_index " + std::to_string(v_index_.size()) +
            "; they index the same joints and must match");
    if (q_index_.empty())
        throw std::invalid_argument("PinocchioFKFactor: no joints");
    for (int i : q_index_)
        if (i < 0 || i >= chain_->nq())
            throw std::invalid_argument(
                "PinocchioFKFactor: configuration index " + std::to_string(i) +
                " out of range for nq=" + std::to_string(chain_->nq()));
    for (int i : v_index_)
        if (i < 0 || i >= chain_->nv())
            throw std::invalid_argument(
                "PinocchioFKFactor: velocity index " + std::to_string(i) +
                " out of range for nv=" + std::to_string(chain_->nv()));

    data_ = std::make_unique<pinocchio::Data>(chain_->model());
}


PinocchioFKFactor::~PinocchioFKFactor() = default;


NonlinearFactor::shared_ptr PinocchioFKFactor::clone() const {
    // Not the default copy: Data is per-instance scratch and must not be
    // shared, so the fresh factor builds its own via the constructor.
    return std::make_shared<PinocchioFKFactor>(
        key<1>(), key<2>(), key<3>(), chain_, frame_id_, q_index_, v_index_,
        noiseModel());
}


Pose3 PinocchioFKFactor::frame_placement(const Vector& q, Matrix* H_local) const {
    const pinocchio::Model& model = chain_->model();

    if (static_cast<size_t>(q.size()) != q_index_.size())
        throw std::invalid_argument(
            "PinocchioFKFactor: joint vector has " + std::to_string(q.size()) +
            " entries, this factor owns " + std::to_string(q_index_.size()));

    // Scatter this factor's joints into a full-model configuration. The frame's
    // placement depends only on its own ancestors, so the joints owned by other
    // digits can hold anything -- neutral is the cheapest well-defined choice.
    Eigen::VectorXd q_full = pinocchio::neutral(model);
    for (size_t i = 0; i < q_index_.size(); ++i)
        q_full[q_index_[i]] = q[static_cast<Eigen::Index>(i)];

    pinocchio::forwardKinematics(model, *data_, q_full);
    pinocchio::updateFramePlacement(model, *data_, frame_id_);
    const auto& oMf = data_->oMf[frame_id_];
    const Pose3 T_fk(Rot3(oMf.rotation()), oMf.translation());

    if (H_local) {
        // Body-frame ("LOCAL") Jacobian: the frame that Pose3::compose takes its
        // second-argument derivative in.
        Matrix J = Matrix::Zero(6, model.nv);
        pinocchio::computeFrameJacobian(model, *data_, q_full, frame_id_,
                                        pinocchio::LOCAL, J);
        swap_linear_angular_rows(J);

        // Narrow to the columns this factor's q owns.
        H_local->resize(6, static_cast<Eigen::Index>(v_index_.size()));
        for (size_t i = 0; i < v_index_.size(); ++i)
            H_local->col(static_cast<Eigen::Index>(i)) = J.col(v_index_[i]);
    }

    return T_fk;
}


Pose3 PinocchioFKFactor::predict(const Pose3& wrist, const Vector& q) const {
    return wrist * frame_placement(q, nullptr);
}


Vector PinocchioFKFactor::evaluateError(const Pose3& wrist,
                                        const Vector& q,
                                        const Pose3& site,
                                        OptionalMatrixType H_wrist,
                                        OptionalMatrixType H_q,
                                        OptionalMatrixType H_site) const
{
    const bool want_pose_jac = (H_wrist != nullptr) || (H_q != nullptr);

    Matrix J_local;
    const Pose3 T_fk = frame_placement(q, H_q ? &J_local : nullptr);

    // f_fk,i(T_w, q) = T_w * T_fk,i(q)
    Matrix6 Hc_wrist, Hc_fk;
    const Pose3 T_pred = wrist.compose(
        T_fk,
        want_pose_jac ? &Hc_wrist : nullptr,
        H_q ? &Hc_fk : nullptr);

    // e = T_i (-) f_fk,i = Log( f_fk,i^-1 * T_i ), GTSAM's Local(f, T_i).
    Matrix6 Hl_pred, Hl_site;
    const Vector6 e = T_pred.localCoordinates(
        site,
        (H_wrist || H_q) ? &Hl_pred : nullptr,
        H_site ? &Hl_site : nullptr);

    if (H_wrist) *H_wrist = Hl_pred * Hc_wrist;
    if (H_q)     *H_q     = Hl_pred * Hc_fk * J_local;
    if (H_site)  *H_site  = Hl_site;

    return e;
}

}  // namespace gepetto_solvers
