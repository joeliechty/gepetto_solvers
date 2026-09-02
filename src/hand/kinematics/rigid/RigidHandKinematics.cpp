// A URDF-described rigid-body hand, behind the HandKinematics interface.
//
// Note what is NOT here: no pinocchio header. All of that lives in
// PinocchioFKFactor.cpp, and this file reaches the model through
// RigidChainModel's small const interface. That keeps Pinocchio's Boost.MPL and
// Eigen constraints in one translation unit.

#include "gepetto_solvers/hand/kinematics/rigid/RigidHandKinematics.h"

#include "gepetto_solvers/hand/HandKinematicsRegistry.h"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <stdexcept>

using namespace gtsam;

namespace gepetto_solvers {

RigidHandKinematics::RigidHandKinematics(
    const RigidHandKinematicsConfig& config,
    const std::vector<std::string>& digit_names,
    const Pose3& wrist_pose,
    Key wrist_key)
:
    digit_names_(digit_names),
    wrist_pose_(wrist_pose),
    wrist_key_(wrist_key),
    id_(next_id_++)
{
    if (config.digits.size() != digit_names.size())
        throw std::invalid_argument(
            "RigidHandKinematics: " + std::to_string(config.digits.size()) +
            " digit specs for " + std::to_string(digit_names.size()) + " digits");
    if (config.urdf_xml.empty() && config.urdf_path.empty())
        throw std::invalid_argument(
            "RigidHandKinematics: give either urdf_xml or urdf_path");

    chain_ = config.urdf_xml.empty()
                 ? RigidChainModel::from_urdf_file(config.urdf_path)
                 : RigidChainModel::from_urdf_xml(config.urdf_xml);

    digits_.reserve(config.digits.size());
    for (size_t d = 0; d < config.digits.size(); ++d) {
        const RigidDigitSpec& spec = config.digits[d];
        Digit digit;
        digit.name = spec.name.empty() ? digit_names[d] : spec.name;

        if (spec.joints.empty())
            throw std::invalid_argument(
                "RigidHandKinematics: digit \"" + digit.name + "\" has no joints");
        if (spec.site_frames.empty())
            throw std::invalid_argument(
                "RigidHandKinematics: digit \"" + digit.name + "\" has no site "
                "frames; it needs at least a fingertip to be contactable");
        if (static_cast<int>(spec.site_frames.size()) + 1 > kDigitStride)
            throw std::invalid_argument(
                "RigidHandKinematics: digit \"" + digit.name + "\" has more "
                "sites than the key stride allows");

        for (const auto& j : spec.joints) {
            auto [qi, vi] = chain_->joint_indices(j);   // throws, naming the joint
            digit.q_index.push_back(qi);
            digit.v_index.push_back(vi);
        }
        for (const auto& f : spec.site_frames)
            digit.frame_id.push_back(chain_->frame_id(f));  // throws, naming it

        // Per-site relaxation covariance: the formulation's Sigma_fk,i.
        const bool has_override =
            d < config.site_sigma_fk.size() && !config.site_sigma_fk[d].empty();
        if (has_override &&
            config.site_sigma_fk[d].size() != spec.site_frames.size())
            throw std::invalid_argument(
                "RigidHandKinematics: digit \"" + digit.name + "\" has " +
                std::to_string(config.site_sigma_fk[d].size()) +
                " site sigmas for " + std::to_string(spec.site_frames.size()) +
                " sites");
        for (size_t s = 0; s < spec.site_frames.size(); ++s)
            digit.sigma.push_back(has_override ? config.site_sigma_fk[d][s]
                                               : config.sigma_fk);

        // The fixed mount: where this digit hangs off the palm. Read from the
        // model at its neutral configuration, which is the placement chain with
        // every joint transform identity.
        //
        // That is only the CONSTANT offset if no joint sits between the palm and
        // this digit's first joint. RigidChainModel checks that and says so,
        // rather than letting a wrist-mounted sub-chain silently produce a mount
        // that moves with q.
        digit.mount = Pose3(chain_->fixed_placement_of_joint(spec.joints.front()));

        digit.q_init = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(spec.joints.size()));
        if (d < config.q_init.size() && !config.q_init[d].empty()) {
            if (config.q_init[d].size() != spec.joints.size())
                throw std::invalid_argument(
                    "RigidHandKinematics: digit \"" + digit.name + "\" seeds " +
                    std::to_string(config.q_init[d].size()) + " joints but has " +
                    std::to_string(spec.joints.size()));
            for (size_t i = 0; i < config.q_init[d].size(); ++i)
                digit.q_init[static_cast<Eigen::Index>(i)] = config.q_init[d][i];
        }

        digits_.push_back(std::move(digit));
    }
}


int RigidHandKinematics::num_sites(int digit) const {
    if (digit < 0 || digit >= static_cast<int>(digits_.size()))
        throw std::out_of_range(
            "RigidHandKinematics: digit " + std::to_string(digit) +
            " out of range for " + std::to_string(digits_.size()) + " digits");
    return static_cast<int>(digits_[digit].frame_id.size()) + 1;  // + the mount
}


int RigidHandKinematics::resolve_site(HandSite site) const {
    const int n = num_sites(site.digit);
    const int s = site.node < 0 ? n + site.node : site.node;
    if (s < 0 || s >= n)
        throw std::out_of_range(
            "RigidHandKinematics: site " + std::to_string(site.node) +
            " out of range for digit " + std::to_string(site.digit) +
            ", which has " + std::to_string(n) + " sites");
    return s;
}


Key RigidHandKinematics::site_pose_key(HandSite site) const {
    const int s = resolve_site(site);
    // Site 0 is the fixed mount: rigidly placed by the wrist, so it IS the
    // wrist variable rather than one pinned to it.
    if (s == 0) return wrist_key_;
    return Symbol('K', id_ * 1000 + site.digit * kDigitStride + s);
}


bool RigidHandKinematics::site_is_root(HandSite site) const {
    return resolve_site(site) == 0;
}


Pose3 RigidHandKinematics::digit_base_offset(int digit) const {
    if (digit < 0 || digit >= static_cast<int>(digits_.size()))
        throw std::out_of_range("RigidHandKinematics::digit_base_offset: digit " +
                                std::to_string(digit) + " out of range");
    return digits_[digit].mount;
}


Key RigidHandKinematics::actuation_key(int digit) const {
    if (digit < 0 || digit >= static_cast<int>(digits_.size()))
        throw std::out_of_range("RigidHandKinematics::actuation_key: digit " +
                                std::to_string(digit) + " out of range");
    return Symbol('J', id_ * 1000 + digit);
}


std::optional<Key> RigidHandKinematics::displacement_key(int) const {
    // Actuation IS position on a joint-space hand, so there is no second
    // variable for a displacement GP or a length prior to attach to.
    return std::nullopt;
}


void RigidHandKinematics::add_kinematics_factors(
    NonlinearFactorGraph& graph,
    ConstraintTagger& /*tags*/,
    const std::vector<VectorXGaussian>& actuation,
    const std::vector<Vector6Gaussian>& /*tip_wrenches*/)
{
    // tip_wrenches is ignored: a rigid-body hand carries no external-wrench
    // variables, and the interface says a mechanism without them may skip it.
    //
    // `tags` is unused because nothing here is a HARD constraint. Joint limits
    // would be, and would go through it -- see the class note.
    if (actuation.size() != digits_.size())
        throw std::invalid_argument(
            "RigidHandKinematics: " + std::to_string(actuation.size()) +
            " actuation priors for " + std::to_string(digits_.size()) + " digits");

    for (size_t d = 0; d < digits_.size(); ++d) {
        const Digit& digit = digits_[d];
        const auto dof = static_cast<Eigen::Index>(digit.q_index.size());
        const Key q_key = actuation_key(static_cast<int>(d));

        // p(q) -- the joint-state prior, a soft pull toward the seeded
        // configuration q_S with covariance Sigma_q.
        if (actuation[d].mean.size() != dof)
            throw std::invalid_argument(
                "RigidHandKinematics: digit \"" + digit.name + "\" has " +
                std::to_string(dof) + " joints but its prior has " +
                std::to_string(actuation[d].mean.size()) + " entries");
        graph.add(PriorFactor<Vector>(
            q_key, Vector(actuation[d].mean),
            noiseModel::Gaussian::Covariance(actuation[d].cov)));

        // p(T_i | T_w, q) -- one kinematics likelihood per site. Site 0 is the
        // mount, which is the wrist variable itself and needs no factor.
        for (size_t s = 0; s < digit.frame_id.size(); ++s) {
            graph.add(std::make_shared<PinocchioFKFactor>(
                wrist_key_,
                q_key,
                site_pose_key({static_cast<int>(d), static_cast<int>(s) + 1}),
                chain_,
                digit.frame_id[s],
                digit.q_index,
                digit.v_index,
                noiseModel::Diagonal::Sigmas(digit.sigma[s])));
        }
    }
}


void RigidHandKinematics::insert_initial_values(Values& values,
                                                const Values* warm) const {
    // The shared wrist. Inserted here because this kinematics owns every
    // variable in the graph apart from the task ones, and nothing else would.
    if (!values.exists(wrist_key_)) values.insert(wrist_key_, wrist_pose_);

    for (size_t d = 0; d < digits_.size(); ++d) {
        const Digit& digit = digits_[d];
        const Key q_key = actuation_key(static_cast<int>(d));
        if (!values.exists(q_key)) values.insert(q_key, Vector(digit.q_init));
    }

    // Adopt warm-start values BEFORE seeding the sites, so each site starts at
    // the prediction from the posture actually being carried in rather than
    // from the cold configuration.
    if (warm) {
        for (Key k : values.keys())
            if (warm->exists(k)) values.update(k, warm->at(k));
    }

    const Pose3 wrist = values.at<Pose3>(wrist_key_);
    for (size_t d = 0; d < digits_.size(); ++d) {
        const Digit& digit = digits_[d];
        const Vector q = values.at<Vector>(actuation_key(static_cast<int>(d)));
        for (size_t s = 0; s < digit.frame_id.size(); ++s) {
            const Key key =
                site_pose_key({static_cast<int>(d), static_cast<int>(s) + 1});
            if (values.exists(key)) continue;
            // Seed ON the kinematics manifold: each site at exactly what its own
            // factor predicts, so the FK likelihood starts at zero error and the
            // first iterations go into the task constraints rather than into
            // pulling the hand back onto its own kinematics.
            PinocchioFKFactor probe(wrist_key_, actuation_key(static_cast<int>(d)),
                                    key, chain_, digit.frame_id[s],
                                    digit.q_index, digit.v_index,
                                    noiseModel::Diagonal::Sigmas(digit.sigma[s]));
            values.insert(key, probe.predict(wrist, q));
        }
    }

    if (warm) {
        for (Key k : values.keys())
            if (warm->exists(k)) values.update(k, warm->at(k));
    }
}


void RigidHandKinematics::insert_from_state(Values& values,
                                            const HandState& state) const {
    if (state.digits.size() != digits_.size())
        throw std::invalid_argument(
            "insert_from_state: state has " +
            std::to_string(state.digits.size()) + " digits, this hand has " +
            std::to_string(digits_.size()));

    values.insert(wrist_key_, Pose3(state.wrist_pose));

    for (size_t d = 0; d < digits_.size(); ++d) {
        const Digit& digit = digits_[d];
        const DigitState& ds = state.digits[d];
        const auto dof = static_cast<Eigen::Index>(digit.q_index.size());

        if (ds.actuation.mean.size() != dof)
            throw std::invalid_argument(
                "insert_from_state: digit \"" + digit.name + "\" has " +
                std::to_string(ds.actuation.mean.size()) +
                " joints in the state, this hand has " + std::to_string(dof));
        if (ds.sites.size() != digit.frame_id.size() + 1)
            throw std::invalid_argument(
                "insert_from_state: digit \"" + digit.name + "\" has " +
                std::to_string(ds.sites.size()) + " sites in the state, this "
                "hand has " + std::to_string(digit.frame_id.size() + 1));

        values.insert(actuation_key(static_cast<int>(d)), Vector(ds.actuation.mean));
        for (size_t s = 0; s < digit.frame_id.size(); ++s)
            values.insert(
                site_pose_key({static_cast<int>(d), static_cast<int>(s) + 1}),
                Pose3(ds.sites[s + 1].pose.mean));
    }
}


HandState RigidHandKinematics::extract(const Values& values,
                                       const Marginals* marginals) const {
    HandState out;
    out.digit_names = digit_names_;
    out.digits.reserve(digits_.size());

    const Pose3 wrist = values.exists(wrist_key_) ? values.at<Pose3>(wrist_key_)
                                                  : wrist_pose_;
    out.wrist_pose = wrist.matrix();

    auto pose_cov = [&](Key k) {
        return marginals ? Matrix(marginals->marginalCovariance(k))
                         : Matrix(Matrix6::Zero());
    };

    for (size_t d = 0; d < digits_.size(); ++d) {
        const Digit& digit = digits_[d];
        DigitState ds;

        // Site 0, the mount: not a variable, so it is composed rather than read.
        SiteState mount;
        mount.pose.mean = (wrist * digit.mount).matrix();
        mount.pose.cov = Matrix6::Zero();
        mount.stress.mean = Vector6::Zero();
        mount.stress.cov = Matrix6::Zero();
        mount.wrench.mean = Vector6::Zero();
        mount.wrench.cov = Matrix6::Zero();
        ds.sites.push_back(mount);

        for (size_t s = 0; s < digit.frame_id.size(); ++s) {
            const Key key =
                site_pose_key({static_cast<int>(d), static_cast<int>(s) + 1});
            SiteState st;
            st.pose.mean = values.at<Pose3>(key).matrix();
            st.pose.cov = pose_cov(key);
            // No continuum state on a rigid body; zero rather than absent, as
            // SiteState documents.
            st.stress.mean = Vector6::Zero();
            st.stress.cov = Matrix6::Zero();
            st.wrench.mean = Vector6::Zero();
            st.wrench.cov = Matrix6::Zero();
            ds.sites.push_back(st);
        }

        const Key q_key = actuation_key(static_cast<int>(d));
        const Vector q = values.at<Vector>(q_key);
        ds.actuation.mean = q;
        ds.actuation.cov = marginals ? Matrix(marginals->marginalCovariance(q_key))
                                     : Matrix(Matrix::Zero(q.size(), q.size()));

        // Actuation IS position here, so there is no separate displacement.
        ds.displacement.clear();

        // Every site but the mount can carry a collision sphere; the mount is
        // rigidly placed by the wrist and the collision passes skip root sites.
        for (size_t s = 1; s <= digit.frame_id.size(); ++s)
            ds.collision_sites.push_back(static_cast<int>(s));

        ds.extras = nullptr;   // no mechanism-specific state
        out.digits.push_back(std::move(ds));
    }

    return out;
}


void RigidHandKinematics::add_temporal_gp(
    NonlinearFactorGraph& graph,
    const HandKinematics& next,
    const Eigen::MatrixXd& gp_actuation_Qc,
    const Eigen::MatrixXd& gp_displacement_Qc,
    double dt) const
{
    if (gp_displacement_Qc.size() > 0)
        throw std::invalid_argument(
            "RigidHandKinematics: a displacement GP was requested, but a "
            "joint-space hand has no displacement variable distinct from its "
            "actuation -- put the process noise on gp_actuation_Qc instead.");

    for (size_t d = 0; d < digits_.size(); ++d) {
        const auto dof = static_cast<Eigen::Index>(digits_[d].q_index.size());
        const Eigen::MatrixXd Qc = gp_actuation_Qc.topLeftCorner(dof, dof);
        graph.add(BetweenFactor<Vector>(
            actuation_key(static_cast<int>(d)),
            next.actuation_key(static_cast<int>(d)),
            Vector(Vector::Zero(dof)),
            noiseModel::Gaussian::Covariance(Qc * dt)));
    }
}


void RigidHandKinematics::add_actuation_priors(
    NonlinearFactorGraph& graph,
    const std::vector<VectorXGaussian>& actuation) const
{
    if (actuation.size() != digits_.size())
        throw std::invalid_argument(
            "add_actuation_priors: " + std::to_string(actuation.size()) +
            " priors for " + std::to_string(digits_.size()) + " digits");

    for (size_t d = 0; d < digits_.size(); ++d) {
        const auto dof = static_cast<Eigen::Index>(digits_[d].q_index.size());
        if (actuation[d].mean.size() != dof)
            throw std::invalid_argument(
                "add_actuation_priors: digit \"" + digits_[d].name + "\" has " +
                std::to_string(dof) + " joints, prior has " +
                std::to_string(actuation[d].mean.size()));
        graph.add(PriorFactor<Vector>(
            actuation_key(static_cast<int>(d)), Vector(actuation[d].mean),
            noiseModel::Gaussian::Covariance(actuation[d].cov)));
    }
}


void RigidHandKinematics::add_displacement_priors(
    NonlinearFactorGraph&, const std::vector<VectorXGaussian>&) const
{
    throw std::invalid_argument(
        "RigidHandKinematics: this hand has no displacement variable -- "
        "actuation IS position. Use add_actuation_priors.");
}


// --- registration ---------------------------------------------------------

namespace {

const bool registered_rigid = [] {
    register_hand_kinematics(
        "rigid_urdf",
        [](const HandKinematicsConfig& config,
           const std::vector<std::string>& digit_names,
           const Pose3& wrist_pose,
           Key wrist_key) -> std::unique_ptr<HandKinematics> {
            const auto* rc = dynamic_cast<const RigidHandKinematicsConfig*>(&config);
            if (!rc)
                throw std::invalid_argument(
                    "the \"rigid_urdf\" kinematics needs a "
                    "RigidHandKinematicsConfig, but the HandSpec carried a "
                    "different payload");
            return std::make_unique<RigidHandKinematics>(
                *rc, digit_names, wrist_pose, wrist_key);
        });
    return true;
}();

}  // namespace

}  // namespace gepetto_solvers
