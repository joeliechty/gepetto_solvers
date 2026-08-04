#pragma once

#include <gtsam/constrained/AugmentedLagrangianOptimizer.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>


namespace crest_sparse {

// The Augmented Lagrangian outer-loop state that survives between solves.
//
// Two penalty weights per constraint class, and the distinction matters. GTSAM's
// loop carries mu in a local that means "the penalty the NEXT merit function
// will use", while State::muEq means "the penalty that produced these values" --
// and the dual-ascent step size is taken from the latter
// (updateLagrangeMultiplier reads previousState.muEq). Storing one and using it
// for both takes a full-sized dual step scaled by a penalty that has not solved
// anything yet, which on the first tick means stepping the multipliers by the
// violation at the raw initial guess.
struct WarmALState {
    double mu_eq_next   = 0.0;  // -> the next merit function
    double mu_ineq_next = 0.0;
    double mu_eq_at     = 0.0;  // -> the dual-ascent step size on resume
    double mu_ineq_at   = 0.0;
    std::vector<gtsam::Vector> lambda_eq;
    std::vector<double>        lambda_ineq;

    // Stable identity of the constraint each multiplier belongs to, in the same
    // order (see TendonHandModel::constraint_tags). Empty on a solver that does
    // not tag its constraints -- such a state can still be carried WITHIN one
    // solver, where positions do not move, just not across a rebuild.
    std::vector<std::string> tag_eq;
    std::vector<std::string> tag_ineq;

    // Empty multipliers mean "nothing carried yet": start exactly as GTSAM does.
    bool empty() const { return lambda_eq.empty() && lambda_ineq.empty(); }

    bool tagged() const {
        return tag_eq.size() == lambda_eq.size() &&
               tag_ineq.size() == lambda_ineq.size() && !empty();
    }

    void clear() { *this = WarmALState{}; }
};


// How much of a transfer survived, for reporting. A run of 0 matched against a
// non-empty carried state means the tags drifted, which is a bug rather than a
// legitimately different problem, so the number is worth surfacing.
struct ALTransferReport {
    size_t matched_eq = 0, total_eq = 0;
    size_t matched_ineq = 0, total_ineq = 0;

    size_t matched() const { return matched_eq + matched_ineq; }
    size_t total() const { return total_eq + total_ineq; }
};


// Re-seat multipliers carried from one problem onto a DIFFERENT one by matching
// constraint identity, so that the constraints the two problems share keep their
// multipliers and everything new starts at zero.
//
// This is what makes "change a constraint and carry on" possible at all. lambda
// is indexed by POSITION, so inserting one constraint renumbers every multiplier
// after it -- the count check in optimizeWarm() is a guard against exactly that,
// and it can only ever answer "all or nothing". Matching by tag answers "these
// 47 of 52", which is the honest answer when a constraint set changes.
//
// mu is global to the whole problem and cannot be per-constraint, so it is
// carried but CLAMPED by mu_max: measured on a table-contact grasp, restarting at
// the mu the previous solve reached (5e5) pins the new constraint as rigidly as
// the old ones and the hand cannot move to satisfy it (stalls in 2 iterations,
// 36 mm from the object), while ~1e4 keeps the old contact and still converges.
inline WarmALState remap_al_state(const WarmALState& carried,
                                  const std::vector<std::string>& eq_tags,
                                  const std::vector<size_t>& eq_dims,
                                  const std::vector<std::string>& ineq_tags,
                                  double mu_max,
                                  ALTransferReport* report = nullptr) {
    WarmALState out;
    out.tag_eq   = eq_tags;
    out.tag_ineq = ineq_tags;
    out.mu_eq_at     = std::min(carried.mu_eq_at, mu_max);
    out.mu_ineq_at   = std::min(carried.mu_ineq_at, mu_max);
    out.mu_eq_next   = std::min(carried.mu_eq_next, mu_max);
    out.mu_ineq_next = std::min(carried.mu_ineq_next, mu_max);

    ALTransferReport rep;
    rep.total_eq = eq_tags.size();
    rep.total_ineq = ineq_tags.size();

    // Repeated tags are matched in order of appearance, so a family that merely
    // grew or shrank still pairs its common members rather than giving up.
    auto index_of = [](const std::vector<std::string>& tags) {
        std::map<std::string, std::vector<size_t>> m;
        for (size_t i = 0; i < tags.size(); ++i) m[tags[i]].push_back(i);
        return m;
    };
    auto take = [](std::map<std::string, std::vector<size_t>>& m,
                   const std::string& tag, size_t* idx) {
        auto it = m.find(tag);
        if (it == m.end() || it->second.empty()) return false;
        *idx = it->second.front();
        it->second.erase(it->second.begin());
        return true;
    };

    auto eq_pool = index_of(carried.tag_eq);
    out.lambda_eq.resize(eq_tags.size());
    for (size_t k = 0; k < eq_tags.size(); ++k) {
        const size_t dim = k < eq_dims.size() ? eq_dims[k] : 0;
        size_t src = 0;
        // The dimension must agree too: the same fingertip contact is a 5-row
        // witness constraint or a 1-row center-direct one depending on the
        // surface, and a 5-vector multiplier means nothing to the 1-row form.
        if (take(eq_pool, eq_tags[k], &src) && src < carried.lambda_eq.size() &&
            static_cast<size_t>(carried.lambda_eq[src].size()) == dim) {
            out.lambda_eq[k] = carried.lambda_eq[src];
            ++rep.matched_eq;
        } else {
            out.lambda_eq[k] = gtsam::Vector::Zero(static_cast<int>(dim));
        }
    }

    auto ineq_pool = index_of(carried.tag_ineq);
    out.lambda_ineq.assign(ineq_tags.size(), 0.0);
    for (size_t k = 0; k < ineq_tags.size(); ++k) {
        size_t src = 0;
        if (take(ineq_pool, ineq_tags[k], &src) &&
            src < carried.lambda_ineq.size()) {
            out.lambda_ineq[k] = carried.lambda_ineq[src];
            ++rep.matched_ineq;
        }
    }

    if (report) *report = rep;
    return out;
}


// An AugmentedLagrangianOptimizer whose outer loop can be SEEDED with a penalty
// weight and a set of Lagrange multipliers, and which hands both back when it
// finishes.
//
// Why this exists. GTSAM's optimize() always starts a fresh outer loop: muEq =
// initialMuEq and every multiplier at zero. That is correct for a one-shot
// solve, but the Section 1.8 controller re-solves the SAME constraint set every
// control tick, and each tick is allowed only a handful of outer iterations. Its
// penalty therefore cannot exceed al_initial_mu * al_rate^(iters-1) -- mu ~ 8 at
// the defaults, against the mu ~ 8e3 an offline solve reaches -- so the hard
// constraints are never actually enforced, no matter how many ticks run. The AL
// method's whole mechanism is that lambda accumulates across outer iterations;
// throwing it away every tick reduces the controller to a weak penalty method.
//
// Nothing here changes the algorithm. iterate() is GTSAM's, and this loop is a
// transcription of GTSAM's optimize() with three additions: the initial state may
// carry duals, mu is clamped, and the final state is handed back. With an empty
// WarmALState it is behaviourally identical to optimize().
//
// Implemented as a subclass rather than as a fork edit because everything needed
// is already reachable: iterate() and progress() are public, and problem_,
// checkConvergence(), logInitialState() and logIteration() are protected.
class WarmAugmentedLagrangianOptimizer
    : public gtsam::AugmentedLagrangianOptimizer {
public:
    using gtsam::AugmentedLagrangianOptimizer::AugmentedLagrangianOptimizer;

    // (equality, inequality) constraint counts of the problem this optimizer was
    // built from. A caller holding duals from a previous solve compares against
    // these before reusing them: lambda is indexed by a constraint's POSITION in
    // the problem, so a differing count means the correspondence is gone.
    std::pair<size_t, size_t> constraint_counts() const {
        return {problem_.eConstraints().size(), problem_.iConstraints().size()};
    }

    // Row count of each equality constraint, in the same order. A tag-matched
    // transfer checks these: a multiplier only means something to a constraint
    // of the same dimension (see remap_al_state).
    std::vector<size_t> eq_dims() const {
        std::vector<size_t> dims;
        dims.reserve(problem_.eConstraints().size());
        for (const auto& c : problem_.eConstraints()) dims.push_back(c->dim());
        return dims;
    }

    // Run the outer loop, seeded from `w` and reporting back through it.
    //
    // mu_max clamps both penalties every iteration. Without it mu compounds
    // across ticks without bound and the constraint penalty eventually drowns
    // the priors and the kinematics it is supposed to be balanced against.
    gtsam::Values optimizeWarm(WarmALState& w, double mu_max) const {
        const size_t n_eq = problem_.eConstraints().size();
        const size_t n_ineq = problem_.iConstraints().size();

        State previousState;
        State state(0, initialValues_, problem_);

        // Reuse the carried duals only if they still describe THIS problem. Any
        // size disagreement means they came from a different one, so fall back
        // to a cold start rather than pairing a multiplier with whatever
        // constraint now happens to sit at that index.
        const bool usable = !w.empty() && w.lambda_eq.size() == n_eq &&
                            w.lambda_ineq.size() == n_ineq;

        double mu_eq, mu_ineq;
        if (usable) {
            state.lambdaEq   = w.lambda_eq;
            state.lambdaIneq = w.lambda_ineq;
            // The penalty that PRODUCED initialValues_, so the first dual step
            // of this solve is scaled the way the last step of the previous one
            // was -- continuing the ascent rather than restarting it.
            state.muEq   = std::min(w.mu_eq_at, mu_max);
            state.muIneq = std::min(w.mu_ineq_at, mu_max);
            mu_eq   = std::min(w.mu_eq_next > 0.0 ? w.mu_eq_next : p_->initialMuEq,
                               mu_max);
            mu_ineq = std::min(w.mu_ineq_next > 0.0 ? w.mu_ineq_next : p_->initialMuIneq,
                               mu_max);
        } else {
            state.initializeLagrangeMultipliers(problem_);
            // state.muEq/muIneq stay at their 0.0 default, exactly as GTSAM's
            // optimize() leaves them. That zero is load-bearing: it makes the
            // first dual step of a cold solve zero-sized, so the multipliers are
            // not stepped by the violation at an initial guess no penalty has
            // been applied to yet.
            mu_eq   = p_->initialMuEq;
            mu_ineq = p_->initialMuIneq;
        }

        logInitialState(state);

        do {
            previousState = std::move(state);
            std::tie(state, mu_eq, mu_ineq) =
                iterate(previousState, mu_eq, mu_ineq);
            mu_eq   = std::min(mu_eq, mu_max);
            mu_ineq = std::min(mu_ineq, mu_max);
            logIteration(state);
        } while (!checkConvergence(state, previousState, *p_));

        w.lambda_eq    = state.lambdaEq;
        w.lambda_ineq  = state.lambdaIneq;
        w.mu_eq_at     = state.muEq;
        w.mu_ineq_at   = state.muIneq;
        w.mu_eq_next   = mu_eq;
        w.mu_ineq_next = mu_ineq;
        return state.values;
    }
};

}  // namespace crest_sparse
