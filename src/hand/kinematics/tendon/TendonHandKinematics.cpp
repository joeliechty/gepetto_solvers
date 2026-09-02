// The tendon hand's kinematics, behind the HandKinematics interface.
//
// Everything here was previously inline in the hand model itself: the per-digit
// construction switch, the rod+tendon factor emission, the key accessors, the
// values/state round trip and the GP temporal priors. TendonFingerModel and its
// factors are untouched -- they are simply no longer visible to the graph
// builder.

#include "gepetto_solvers/hand/kinematics/tendon/TendonHandKinematics.h"

#include "gepetto_solvers/hand/HandKinematicsRegistry.h"
#include "gepetto_solvers/utils/MiscInline.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <stdexcept>
#include <type_traits>

using namespace gtsam;

namespace gepetto_solvers {

namespace {

// Construct a TendonFingerModel<N> from a TendonFingerSolverConfig, choosing the
// per-disc vs. simple routing path and the K_inv vs. per-segment path exactly as
// TendonFingerSolver does. base_pose_mean is supplied by the caller (= wrist o offset).
template<int N>
std::unique_ptr<TendonFingerModel<N>> make_finger_impl(
    const TendonFingerSolverConfig& c,
    const Pose3& base_pose_mean,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    SharedDiagonal base_pose_noise)
{
    if (c.per_disc_tendon_input.is_populated()) {
        if (c.K_inv_per_segment.empty()) {
            return std::make_unique<TendonFingerModel<N>>(
                c.rod_length, c.num_discs, c.num_between_nodes,
                c.per_disc_tendon_input, c.K_inv, twist_noise, stress_noise,
                base_pose_mean, base_pose_noise, c.disc_positions_normalized);
        }
        return std::make_unique<TendonFingerModel<N>>(
            c.rod_length, c.num_discs, c.num_between_nodes,
            c.per_disc_tendon_input, c.K_inv_per_segment, twist_noise, stress_noise,
            base_pose_mean, base_pose_noise, c.disc_positions_normalized);
    }
    if (c.K_inv_per_segment.empty()) {
        return std::make_unique<TendonFingerModel<N>>(
            c.rod_length, c.num_discs, c.num_between_nodes,
            c.tendon_input, c.K_inv, twist_noise, stress_noise,
            base_pose_mean, base_pose_noise, c.disc_positions_normalized);
    }
    return std::make_unique<TendonFingerModel<N>>(
        c.rod_length, c.num_discs, c.num_between_nodes,
        c.tendon_input, c.K_inv_per_segment, twist_noise, stress_noise,
        base_pose_mean, base_pose_noise, c.disc_positions_normalized);
}

}  // namespace


TendonHandKinematics::TendonHandKinematics(
    const TendonHandKinematicsConfig& config,
    const std::vector<std::string>& digit_names,
    const Pose3& wrist_pose,
    Key wrist_key)
:
    digit_names_(digit_names),
    wrist_key_(wrist_key)
{
    if (config.fingers.size() != digit_names.size())
        throw std::invalid_argument(
            "TendonHandKinematics: " + std::to_string(config.fingers.size()) +
            " finger configs for " + std::to_string(digit_names.size()) + " digits");

    const size_t n = config.fingers.size();
    fingers_.reserve(n);
    hand_base_offsets_.reserve(n);
    small_wrench_noises_.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const TendonFingerSolverConfig& c = config.fingers[i];

        SharedDiagonal twist_noise = get_noise_model_rot_pos(
            c.sigma_twist_rot, c.sigma_twist_pos);
        SharedDiagonal stress_noise = get_noise_model_rot_pos(
            c.sigma_stress_moment, c.sigma_stress_force);
        SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
            c.sigma_base_rot, c.sigma_base_pos);
        // Tight per-digit external-wrench prior noise, matching TendonFingerSolver.
        small_wrench_noises_.push_back(
            get_noise_model_rot_pos(c.sigma_stress_moment, c.sigma_stress_force));

        // This digit's fixed attachment to the wrist. Its node-0 pose mean is
        // T_wrist o T_offset, so the shared base variable seeds to T_wrist for
        // every digit (see insert_initial_values / set_root_reparameterization).
        Pose3 offset(c.hand_base_offset);
        hand_base_offsets_.push_back(offset);
        Pose3 base_pose_mean = wrist_pose * offset;

        int Nt = c.num_tendons;
        switch (Nt) {
            case 1:  fingers_.push_back(make_finger_impl<1>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 2:  fingers_.push_back(make_finger_impl<2>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 3:  fingers_.push_back(make_finger_impl<3>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 4:  fingers_.push_back(make_finger_impl<4>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 5:  fingers_.push_back(make_finger_impl<5>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 6:  fingers_.push_back(make_finger_impl<6>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 7:  fingers_.push_back(make_finger_impl<7>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 8:  fingers_.push_back(make_finger_impl<8>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 9:  fingers_.push_back(make_finger_impl<9>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 10: fingers_.push_back(make_finger_impl<10>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            default: throw std::invalid_argument(
                "num_tendons must be between 1 and 10, got " + std::to_string(Nt));
        }

        // Share the one wrist variable and let the hand model own the wrist
        // prior. `cfg` rather than the loop variable directly: capturing a
        // structured binding in a lambda is C++20, and this builds as C++17.
        const auto& cfg = c;
        const Key shared_wrist_key = wrist_key_;
        std::visit([&](auto& fp) {
            fp->set_hand_base(offset, shared_wrist_key);
            fp->set_emit_base_prior(false);
            // Per-digit, like the rod sigmas above: the planar-bending switch
            // rides on the DIGIT config, so a hand can mix keyed and free rods.
            if (cfg.planar_bending)
                fp->set_planar_bending(cfg.sigma_planar_bend, cfg.sigma_planar_twist);
        }, fingers_.back());
    }
}


const TendonHandKinematics::FingerVariant&
TendonHandKinematics::finger_at(int digit) const {
    if (digit < 0 || digit >= static_cast<int>(fingers_.size()))
        throw std::out_of_range(
            "TendonHandKinematics: digit " + std::to_string(digit) +
            " out of range for " + std::to_string(fingers_.size()) + " digits");
    return fingers_[static_cast<size_t>(digit)];
}


// --- the factors that ARE this hand's kinematics -------------------------
//
// `tags` is unused: every factor the tendon kinematics emits is a soft one (the
// rod chain, the tendon coupling and length factors, the wrench priors), so
// there is no hard constraint to number. It stays in the signature because the
// ordering guarantee belongs to the interface, not to this implementation -- a
// mechanism with a closed linkage loop will need it.
void TendonHandKinematics::add_kinematics_factors(
    NonlinearFactorGraph& graph,
    ConstraintTagger& /*tags*/,
    const std::vector<VectorXGaussian>& actuation,
    const std::vector<Vector6Gaussian>& tip_wrenches)
{
    if (actuation.size() != fingers_.size())
        throw std::invalid_argument("tensions size must match number of fingers");
    if (tip_wrenches.size() != fingers_.size())
        throw std::invalid_argument("tip_wrenches size must match number of fingers");

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (actuation[i].mean.size() != N)
                throw std::invalid_argument(
                    "Finger " + std::to_string(i) + " expects " + std::to_string(N) +
                    " tendons, got " + std::to_string(actuation[i].mean.size()));

            VectorNGaussian<N> t;
            t.mean = actuation[i].mean;
            t.cov  = actuation[i].cov;

            // Rod + tendon factors (base prior suppressed via set_emit_base_prior).
            graph.add(fp->build_graph(t));

            // Constrain interior external wrenches to zero; the tip wrench uses
            // the caller's value (mirrors TendonFingerSolver::build_graph).
            int num_nodes = fp->get_num_nodes();
            for (int j = 1; j + 1 < num_nodes; ++j) {
                graph.add(PriorFactor<Vector6>(
                    fp->get_external_wrench_key(j), Vector6::Zero(),
                    small_wrench_noises_[i]));
            }
            graph.add(PriorFactor<Vector6>(
                fp->get_external_wrench_key(num_nodes - 1),
                tip_wrenches[i].mean,
                noiseModel::Gaussian::Covariance(tip_wrenches[i].cov)));
        }, fingers_[i]);
    }
}


// --- what the task constraints key off -----------------------------------

Key TendonHandKinematics::site_pose_key(HandSite site) const {
    return std::visit(
        [&](const auto& fp) { return fp->rod_->get_pose_key(site.node); },
        finger_at(site.digit));
}


bool TendonHandKinematics::site_is_root(HandSite site) const {
    return std::visit([&](const auto& fp) {
        return fp->rod_->uses_root() &&
               fp->rod_->get_pose_key(site.node) == fp->rod_->get_pose_key(0);
    }, finger_at(site.digit));
}


Pose3 TendonHandKinematics::digit_base_offset(int digit) const {
    if (digit < 0 || digit >= static_cast<int>(hand_base_offsets_.size()))
        throw std::out_of_range(
            "TendonHandKinematics::digit_base_offset: digit " +
            std::to_string(digit) + " out of range");
    return hand_base_offsets_[static_cast<size_t>(digit)];
}


Key TendonHandKinematics::actuation_key(int digit) const {
    return std::visit([](const auto& fp) { return fp->get_tensions_key(); },
                      finger_at(digit));
}


std::optional<Key> TendonHandKinematics::displacement_key(int digit) const {
    return std::visit(
        [](const auto& fp) { return std::optional<Key>(fp->get_lengths_key()); },
        finger_at(digit));
}


// --- values / state round trip -------------------------------------------

void TendonHandKinematics::insert_initial_values(Values& values,
                                                 const Values* warm) const {
    // Merge each digit's values; the shared wrist variable appears in every
    // digit's values (identical), so keep only the first and drop the rest.
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            Values fv = fp->get_initial_values();
            if (i > 0) fv.erase(wrist_key_);
            values.insert(fv);
        }, fingers_[i]);
    }

    // Adopt any warm-start poses here, BEFORE the caller's witness seeding, so
    // the projections that derive each witness from its contact node start from
    // where the digit actually converged rather than from a straight hand.
    if (warm) {
        for (Key k : values.keys())
            if (warm->exists(k)) values.update(k, warm->at(k));
    }
}


void TendonHandKinematics::insert_from_state(Values& values,
                                             const HandState& state) const {
    if (state.digits.size() != fingers_.size())
        throw std::invalid_argument(
            "insert_from_state: state has " +
            std::to_string(state.digits.size()) + " digits, this hand has " +
            std::to_string(fingers_.size()));

    for (size_t i = 0; i < fingers_.size(); ++i) {
        const DigitState& fm = state.digits[i];
        // The tendon-only half of the bundle. A state produced by another
        // kinematics has none, and cannot seed this hand.
        const auto* extras = dynamic_cast<const TendonDigitExtras*>(fm.extras.get());
        if (!extras)
            throw std::invalid_argument(
                "insert_from_state: digit " + std::to_string(i) +
                " carries no tendon state, so it did not come from a tendon "
                "hand; a posture can only seed the kinematics that produced it");

        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            const int num_nodes = fp->get_num_nodes();
            if (static_cast<int>(fm.sites.size()) != num_nodes)
                throw std::invalid_argument(
                    "insert_from_state: digit " + std::to_string(i) +
                    " has " + std::to_string(fm.sites.size()) +
                    " sites, this hand has " + std::to_string(num_nodes) +
                    " nodes");
            if (fm.actuation.mean.size() != N)
                throw std::invalid_argument(
                    "insert_from_state: digit " + std::to_string(i) +
                    " has " + std::to_string(fm.actuation.mean.size()) +
                    " tendons, this hand has " + std::to_string(N));

            // Rod chain. Node 0's pose is NOT a variable under the hand-base
            // reparameterization; its stress and wrench still are, so only the
            // pose insert is skipped.
            const bool skip_node0_pose = fp->rod_->uses_root();
            for (int j = 0; j < num_nodes; ++j) {
                const auto& s = fm.sites[j];
                if (!(skip_node0_pose && j == 0))
                    values.insert(fp->rod_->get_pose_key(j), Pose3(s.pose.mean));
                values.insert(fp->rod_->get_stress_key(j), Vector6(s.stress.mean));
                values.insert(fp->rod_->get_wrench_key(j), Vector6(s.wrench.mean));
            }

            // External disc wrenches. external_wrenches is indexed by NODE and
            // resolves through get_external_wrench_key, which aliases the rod's
            // own wrench key at non-disc nodes -- already written above. Only the
            // genuine Symbol('D', ...) variables are left, so walk the discs.
            const auto& disc_pose_idx = fp->get_tendon_config().disc_pose_idx;
            for (size_t d = 1; d < disc_pose_idx.size(); ++d) {
                const int node = disc_pose_idx[d];
                if (node < 0 ||
                    node >= static_cast<int>(extras->external_wrenches.size()))
                    continue;
                values.insert(fp->get_disc_wrench_key(static_cast<int>(d)),
                              Vector6(extras->external_wrenches[node].mean));
            }

            values.insert(fp->get_tensions_key(),
                          Eigen::Vector<double, N>(fm.actuation.mean));

            if (static_cast<int>(fm.displacement.size()) == N) {
                Eigen::Vector<double, N> L;
                for (int t = 0; t < N; ++t) L(t) = fm.displacement[t];
                values.insert(fp->get_lengths_key(), L);
            }
        }, fingers_[i]);
    }

    // The shared wrist, taken straight from the bundle.
    //
    // It has to be carried rather than skipped: under the hand-base
    // reparameterization node 0's pose is not a variable but the composition
    // T_0 = T_wrist o T_offset, so the loop above deliberately did not insert
    // it. A warm start missing the wrist would hold every rod pose from the
    // state and the wrist at whatever the receiving model was constructed with
    // -- an inconsistent guess that the Root factors and the wrist prior tear
    // back apart on the first iteration, i.e. the hand snapping to the
    // commanded base pose.
    //
    // extract() reads it off the wrist variable directly, so there is no longer
    // an offset inversion here (or in Python) to get it wrong.
    const bool uses_root = !fingers_.empty() && std::visit(
        [](const auto& fp) { return fp->rod_->uses_root(); }, fingers_[0]);
    if (uses_root)
        values.insert(wrist_key_, Pose3(state.wrist_pose));
}


DigitState TendonHandKinematics::to_digit_state(const TendonFingerMarginals& fm) {
    DigitState d;

    d.sites.reserve(fm.rod.states.size());
    for (const auto& s : fm.rod.states)
        d.sites.push_back(SiteState{s.pose, s.stress, s.wrench});

    d.actuation = fm.tensions;
    d.displacement = fm.tendon_lengths;
    // The disc set IS the collision-sphere set on this hand.
    d.collision_sites = fm.tendon_config.disc_pose_idx;

    auto extras = std::make_shared<TendonDigitExtras>();
    extras->tendon_config = fm.tendon_config;
    extras->external_wrenches = fm.external_wrenches;
    extras->J_pose_tensions = fm.J_pose_tensions;
    d.extras = std::move(extras);

    return d;
}


HandState TendonHandKinematics::extract(const Values& values,
                                        const Marginals* marginals) const {
    HandState out;
    out.digits.reserve(fingers_.size());
    out.digit_names = digit_names_;

    // The shared wrist, read straight off the variable that carries it. Node 0
    // is not a variable under the root reparameterization, so this is the only
    // place it exists -- and reading it here is what spares every caller the
    // offset inversion they used to do for themselves.
    if (values.exists(wrist_key_))
        out.wrist_pose = values.at<Pose3>(wrist_key_).matrix();

    if (marginals) {
        for (const auto& finger : fingers_) {
            std::visit([&](const auto& fp) {
                out.digits.push_back(
                    to_digit_state(fp->get_marginals(values, *marginals)));
            }, finger);
        }
        return out;
    }

    // Means-only: zero-returning functors extract the same state but skip the
    // expensive Marginals factorization, so this is cheap enough to call once
    // per solver-iteration snapshot. cov_of returns a 6x6 (pose block; the
    // tension cov it also feeds is unused for visualization). joint_of must be
    // sized (6+N)x(6+N) per digit because TendonFingerModel::get_J_pose_tensions
    // reads block<6,N>(0,6)/block<N,N>(6,6), so it is built inside the visit
    // where N is known.
    auto zero_cov = [](Key) { return Matrix::Zero(6, 6); };
    for (const auto& finger : fingers_) {
        std::visit([&](const auto& fp) {
            constexpr int N = std::remove_reference_t<decltype(*fp)>::NumTendons;
            auto zero_joint = [N](Key, Key) {
                return Matrix::Zero(6 + N, 6 + N);
            };
            out.digits.push_back(
                to_digit_state(fp->get_marginals(values, zero_cov, zero_joint)));
        }, finger);
    }
    return out;
}


// --- trajectory support ---------------------------------------------------

void TendonHandKinematics::add_temporal_gp(
    NonlinearFactorGraph& graph,
    const HandKinematics& next,
    const Eigen::MatrixXd& gp_actuation_Qc,
    const Eigen::MatrixXd& gp_displacement_Qc,
    double dt) const
{
    const bool has_len_gp = gp_displacement_Qc.size() > 0;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            // Tension GP (Eq 1.11): identity transition, zero-mean between factor.
            Eigen::Matrix<double, N, N> Qc = gp_actuation_Qc.topLeftCorner<N, N>();
            graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                fp->get_tensions_key(),
                next.actuation_key(static_cast<int>(i)),
                Eigen::Vector<double, N>::Zero(),
                noiseModel::Gaussian::Covariance(Qc * dt)));

            // Length GP (Eq 1.13), optional.
            if (has_len_gp) {
                std::optional<Key> next_len = next.displacement_key(static_cast<int>(i));
                if (!next_len.has_value())
                    throw std::invalid_argument(
                        "add_temporal_gp: a displacement GP was requested but the "
                        "next step's kinematics has no displacement variable for "
                        "digit " + std::to_string(i));
                Eigen::Matrix<double, N, N> Qc_len =
                    gp_displacement_Qc.topLeftCorner<N, N>();
                graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                    fp->get_lengths_key(), *next_len,
                    Eigen::Vector<double, N>::Zero(),
                    noiseModel::Gaussian::Covariance(Qc_len * dt)));
            }
        }, fingers_[i]);
    }
}


void TendonHandKinematics::add_displacement_priors(
    NonlinearFactorGraph& graph,
    const std::vector<VectorXGaussian>& displacement) const
{
    if (displacement.size() != fingers_.size())
        throw std::invalid_argument(
            "add_length_priors: lengths size must match number of fingers");

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (displacement[i].mean.size() != N)
                throw std::invalid_argument(
                    "add_length_priors: finger " + std::to_string(i) + " expects " +
                    std::to_string(N) + " tendon lengths, got " +
                    std::to_string(displacement[i].mean.size()));

            Eigen::Vector<double, N> mean = displacement[i].mean;
            Eigen::Matrix<double, N, N> cov = displacement[i].cov.topLeftCorner<N, N>();
            graph.add(PriorFactor<Eigen::Vector<double, N>>(
                fp->get_lengths_key(), mean, noiseModel::Gaussian::Covariance(cov)));
        }, fingers_[i]);
    }
}


void TendonHandKinematics::add_actuation_priors(
    NonlinearFactorGraph& graph,
    const std::vector<VectorXGaussian>& actuation) const
{
    if (actuation.size() != fingers_.size())
        throw std::invalid_argument(
            "add_tension_priors: tensions size must match number of fingers");

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (actuation[i].mean.size() != N)
                throw std::invalid_argument(
                    "add_tension_priors: finger " + std::to_string(i) + " expects " +
                    std::to_string(N) + " tendon tensions, got " +
                    std::to_string(actuation[i].mean.size()));

            Eigen::Vector<double, N> mean = actuation[i].mean;
            Eigen::Matrix<double, N, N> cov = actuation[i].cov.topLeftCorner<N, N>();
            graph.add(PriorFactor<Eigen::Vector<double, N>>(
                fp->get_tensions_key(), mean, noiseModel::Gaussian::Covariance(cov)));
        }, fingers_[i]);
    }
}


// --- registration ---------------------------------------------------------
//
// Self-registering at static init, so linking this translation unit in is all it
// takes to make "tendon" loadable. The anonymous-namespace object exists only
// for its constructor.
namespace {

const bool registered_tendon = [] {
    register_hand_kinematics(
        "tendon",
        [](const HandKinematicsConfig& config,
           const std::vector<std::string>& digit_names,
           const Pose3& wrist_pose,
           Key wrist_key) -> std::unique_ptr<HandKinematics> {
            const auto* tc = dynamic_cast<const TendonHandKinematicsConfig*>(&config);
            if (!tc)
                throw std::invalid_argument(
                    "the \"tendon\" kinematics needs a TendonHandKinematicsConfig, "
                    "but the HandSpec carried a different payload");
            return std::make_unique<TendonHandKinematics>(
                *tc, digit_names, wrist_pose, wrist_key);
        });
    return true;
}();

}  // namespace

}  // namespace gepetto_solvers
