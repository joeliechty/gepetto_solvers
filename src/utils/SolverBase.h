#pragma once

#include "utils/WarmAugmentedLagrangian.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <string>
#include <tuple>
#include <vector>


struct SolutionMetadata {
    double total_time_ms;
    double build_time_ms;
    double optimize_time_ms;
    double marginalize_time_ms;
    double extract_time_ms;

    int iterations;
    double error;

    // Populated when SolverBaseConfig::record_iterations == true.
    std::vector<double> iteration_errors;        // error after each iterate() call
    std::vector<double> iteration_trust_region;  // Dogleg delta or LM lambda after each iterate()
    std::vector<double> iteration_step_norms;    // ||Δx|| in localCoordinates after each iterate()

    // Populated on the Augmented Lagrangian path when record_iterations == true,
    // one entry per AL outer iteration (from AugmentedLagrangianOptimizer::progress()).
    // The LM/Dogleg iterate-loop above is skipped on the AL path, so these are the
    // AL analogues of that trace.
    std::vector<double> al_iteration_costs;       // objective cost per AL outer iter
    std::vector<double> al_iteration_violations;  // constraint violation (eq+ineq) per AL outer iter
    std::vector<double> al_iteration_mus;         // penalty weight muEq per AL outer iter
};


template<typename MarginalType>
struct Solution {
    MarginalType marginals;
    SolutionMetadata meta;
};


struct SolverBaseConfig {
    std::string linear_solver_type = "MULTIFRONTAL_QR";
    // Nonlinear optimizer: "DOGLEG" or "LM" (Levenberg-Marquardt).
    std::string optimizer_type = "DOGLEG";
    bool use_dense = false;

    // Dogleg trust-region initial radius.
    double delta_initial = 1.0;

    // LM tuning. Defaults match GTSAM's, which are tuned for problems whose
    // initial values are already near the optimum; on highly nonlinear
    // problems started far from the solution, bumping lambda_initial up
    // (e.g. 1.0) and enabling diagonal_damping is often what gets LM to
    // accept a first step instead of bailing out at iters=0.
    double lambda_initial = 1e-5;
    double lambda_upper_bound = 1e5;
    bool diagonal_damping = false;

    // GTSAM's default is 100, which silently caps stiff contact problems
    // mid-descent. Expose so callers can bump.
    int max_iterations = 100;

    // Augmented Lagrangian (constrained) optimization tuning. Only consulted
    // when a subclass enables the AL path (i.e. when a hard contact constraint
    // is configured); the free-space Dogleg/LM path ignores these. The inner
    // nonlinear sub-solver is LM and reuses linear_solver_type / lambda_initial
    // / lambda_upper_bound / max_iterations above. Defaults match GTSAM's
    // AugmentedLagrangianParams.
    double al_initial_mu       = 1.0;  // initialMuEq: starting penalty weight
    double al_mu_increase_rate = 2.0;  // muEqIncreaseRate: outer-loop growth
    int    al_max_iterations   = 20;   // maxIterations: outer AL loop steps

    // Dual-ascent step cap (maxDualStepSize{Eq,Ineq}). The proper AL
    // multiplier update is lambda += mu * violation; GTSAM's default cap of 10
    // freezes the multipliers once mu grows past ~10, silently degrading the
    // method to a pure quadratic penalty that needs mu ~ 1e8+ (and ~30 outer
    // iterations of mu-doubling) to reach small violations. Default here is
    // effectively uncapped so the multipliers can do their job.
    double al_max_dual_step = 1e12;

    // Inexact inner solves: when > 0, the inner LM's relativeErrorTol starts
    // here on the first outer iteration and tightens ~ (initial mu / mu) down
    // to lm relativeErrorTol (1e-5). Early merit functions are about to change
    // anyway; solving them to 1e-5 wastes inner iterations. 0 disables.
    double al_inner_rel_tol_initial = 0.0;

    // Outer-loop stopping tolerances (ConstrainedOptimizerParams). The loop
    // stops when (violation < al_abs_violation_tol && cost < al_abs_cost_tol)
    // or (|d violation| < al_rel_violation_tol && |d cost| < al_rel_cost_tol).
    // Note al_abs_cost_tol is an ABSOLUTE cost threshold: with the GTSAM
    // default (1e-5) the first test never fires for problems whose converged
    // cost is O(1), so the loop only stops on stagnation — long after the
    // violation is already tiny. Set al_abs_cost_tol large (e.g. 1e12) to
    // stop on violation alone.
    double al_abs_violation_tol = 1e-5;
    double al_abs_cost_tol      = 1e-5;
    double al_rel_violation_tol = 1e-5;
    double al_rel_cost_tol      = 1e-5;

    // Carry the AL penalty weight and the Lagrange multipliers from one
    // optimize() call to the next, instead of restarting the outer loop at
    // al_initial_mu with zero multipliers every time.
    //
    // Off by default because it is only meaningful when successive solves pose
    // the SAME constrained problem — which is exactly the Section 1.8
    // controller (one graph, re-solved per control tick, warm-started from the
    // last) and exactly not the one-shot solvers. Where it does apply it is not
    // a tuning knob but a correctness fix: a tick capped at al_max_iterations
    // outer steps can only reach al_initial_mu * al_mu_increase_rate^(n-1), so
    // without carried duals the penalty resets before it is ever large enough
    // to enforce anything.
    //
    // The caller is responsible for calling reset_al_duals() whenever the
    // constraint set changes (see there).
    bool al_warm_start_duals = false;

    // Ceiling on the carried penalty weight. Applied every outer iteration once
    // al_warm_start_duals is on. mu compounds across ticks by design, and
    // unbounded it eventually dominates the whole graph: the step priors and
    // the rod kinematics stop being able to influence the solution at all, and
    // the linear system's conditioning degrades with it. With working
    // multipliers feasibility does not need an enormous mu anyway — that is the
    // point of the method — so this is a guard, not a limit to tune against.
    double al_warm_mu_max = 1e4;

    // Ceiling on a mu carried across a REBUILD (set_initial_duals), as opposed
    // to al_warm_mu_max which clamps mu within one solver's life. Separate
    // because the two want different values: a solver that keeps tightening the
    // same problem can be allowed a large mu, but a NEW problem inherits that mu
    // for constraints it has never seen. Measured on a table-contact grasp,
    // inheriting the 5e5 the previous solve reached froze the hand (it stalled 2
    // iterations in, 36 mm from the object it had just been asked to touch)
    // while 1e4 held the old contact and still converged.
    double al_transfer_mu_max = 1e4;

    // When true, optimize() uses a manual iterate() loop instead of
    // optimizer.optimize(). Populates SolutionMetadata::iteration_errors and
    // ::iteration_trust_region. Required for get_intermediate_solutions().
    bool record_iterations = false;

    // If > 0 (and record_iterations == true), store a gtsam::Values snapshot
    // every N iterations so intermediate solutions can be extracted afterwards.
    // 0 = disabled (no snapshots stored).
    int iteration_sample_interval = 0;

    // Skip the gtsam::Marginals factorization at the end of optimize(). That
    // factorization is the most expensive step after the optimizer itself, and
    // the Section 1.8 real-time controller re-runs the whole loop every control
    // tick while only ever consuming the MEANS. When true, marginals_ is left
    // empty and extract_solution() must use a means-only path — reading a
    // covariance from the resulting solution is invalid. Off by default, so
    // every existing solver keeps returning full marginals.
    bool skip_marginals = false;
};


class SolverBase {
public:
    SolverBase(const SolverBaseConfig& params);

    SolutionMetadata optimize();

    // Walk the factor graph and group factors by C++ type, summing the
    // unweighted error contribution (0.5 * ||whitened_residual||^2 per factor,
    // matching GTSAM's NonlinearFactor::error semantics). Returns entries
    // sorted by total error descending: (demangled_type_name, count, total_error).
    // Use to diagnose which factor type dominates the residual.
    std::vector<std::tuple<std::string, int, double>>
        get_factor_error_summary() const;

    // Same grouping as get_factor_error_summary(), but returns the list of
    // individual factor errors per type (in graph traversal order). Used for
    // per-factor-type residual histograms.
    std::vector<std::pair<std::string, std::vector<double>>>
        get_factor_errors_by_type() const;

    // Like get_factor_error_summary(), but evaluated at initial_values_ (the
    // snapshot taken at the start of optimize()) rather than at the final
    // values_. Use to diagnose how poor the initial guess was per factor type.
    std::vector<std::tuple<std::string, int, double>>
        get_initial_factor_error_summary() const;

    // Discard the AL penalty weight and Lagrange multipliers carried between
    // optimize() calls, so the next one starts a fresh outer loop.
    //
    // Must be called whenever the CONSTRAINT SET changes — a different phase, a
    // re-seeded state, a new object. lambda is indexed by a constraint's
    // position in the problem, so applying one problem's multipliers to
    // another's residuals is not a degraded warm start, it is wrong. (The
    // optimizer also refuses duals whose count disagrees, but a set that
    // changed CONTENT while keeping its size would slip past that check.)
    // No-op unless al_warm_start_duals is set.
    void reset_al_duals();

    // Seed the NEXT solve's multipliers from another solver's final state,
    // matched by constraint identity (see remap_al_state). The carried state
    // must be tagged and this solver must tag its own constraints
    // (constraint_tags_eq/ineq), or the transfer is refused -- pairing
    // multipliers by position across a rebuilt graph is not a degraded warm
    // start, it is wrong. Consumed by the first solve after the call.
    void set_initial_duals(const crest_sparse::WarmALState& duals) {
        al_transfer_in_ = duals;
    }

    // The AL state of the last solve, tagged so another solver can take it.
    const crest_sparse::WarmALState& get_al_duals() const { return al_warm_; }

    // How much of the last transfer matched (matched/total, per class). All
    // zeros when nothing was carried in.
    const crest_sparse::ALTransferReport& al_transfer_report() const {
        return al_transfer_report_;
    }

    // Linearize the factor graph at current values_ and return the dense
    // Hessian H and gradient g of the linearized quadratic.
    // For diagnostics: sparsity pattern, condition number, smallest singular
    // values to detect gauge freedom / ill-conditioning.
    std::pair<Eigen::MatrixXd, Eigen::VectorXd>
        get_hessian_and_gradient() const;

private:
    void optimize_dense_benchmark(
        const gtsam::DoglegParams& params, SolutionMetadata& meta);

    virtual void build_graph() = 0;

    virtual void extract_solution() = 0;

    virtual void get_initial_values() = 0;

protected:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    // When true, optimize() solves the graph with GTSAM's Augmented Lagrangian
    // optimizer, treating factors that derive from gtsam::NonlinearConstraint
    // (e.g. ZeroCostConstraint-wrapped contact factors) as hard equality
    // constraints. Subclasses set this in their constructor body when a hard
    // contact constraint is configured. Defaults false => legacy Dogleg/LM.
    bool use_augmented_lagrangian_ = false;

    // Final equality-penalty weight (mu) reached by the AL outer loop on the
    // last optimize() call. Reused to build well-conditioned Gaussian surrogates
    // for the hard constraints (penaltyFactor(mu)) in both the marginals graph
    // and get_hessian_and_gradient(); a raw linearization of a constrained-noise
    // factor cannot form a Hessian. Only meaningful when use_augmented_lagrangian_.
    double al_final_mu_ = 1.0;

    // AL outer-loop state carried between optimize() calls when
    // config_.al_warm_start_duals is set; see reset_al_duals(). Default
    // constructed (empty multipliers) means "cold start", which is the state
    // after construction and after a reset.
    crest_sparse::WarmALState al_warm_;

    // Duals handed in by set_initial_duals(), pending remap onto this solver's
    // own constraints. Cleared once consumed, so a transfer seeds the first
    // solve and every later one continues from al_warm_ as usual.
    crest_sparse::WarmALState al_transfer_in_;
    crest_sparse::ALTransferReport al_transfer_report_;

    // This solver's constraint identities, in graph-insertion order (= the order
    // ConstrainedOptProblem enumerates them). Empty by default: a solver that
    // does not tag its constraints simply cannot transfer duals across a
    // rebuild, which is the safe answer rather than a positional guess.
    virtual std::vector<std::string> constraint_tags_eq() const { return {}; }
    virtual std::vector<std::string> constraint_tags_ineq() const { return {}; }

    // Snapshot of values_ at the start of each optimize() call (the initial
    // guess for that solve). Used by get_initial_factor_error_summary() and
    // by subclass extractors of the "initial solution".
    gtsam::Values initial_values_;

    // Populated during the manual iterate() loop when
    // config_.record_iterations == true and config_.iteration_sample_interval > 0.
    // Cleared at the start of each optimize() call.
    std::vector<gtsam::Values> intermediate_values_;

    const SolverBaseConfig config_;
};