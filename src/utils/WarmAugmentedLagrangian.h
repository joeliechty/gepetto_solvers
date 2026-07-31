#pragma once

#include <gtsam/constrained/AugmentedLagrangianOptimizer.h>

#include <algorithm>
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

    // Empty multipliers mean "nothing carried yet": start exactly as GTSAM does.
    bool empty() const { return lambda_eq.empty() && lambda_ineq.empty(); }

    void clear() { *this = WarmALState{}; }
};


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
