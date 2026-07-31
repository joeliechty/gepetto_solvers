#include "SolverBase.h"

#include "utils/WarmAugmentedLagrangian.h"

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/constrained/AugmentedLagrangianOptimizer.h>
#include <gtsam/constrained/NonlinearConstraint.h>
#include <gtsam/constrained/NonlinearInequalityConstraint.h>

#include <gtsam/base/types.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <typeinfo>

using namespace gtsam;


SolverBase::SolverBase(const SolverBaseConfig& config) 
:
    config_(config)
{}


inline auto now() {
    return std::chrono::high_resolution_clock::now();
}


template <typename ClockTimePoint>
inline double ms(const ClockTimePoint& start, const ClockTimePoint& stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}


void SolverBase::optimize_dense_benchmark(const DoglegParams& params, SolutionMetadata& meta) {
    double last_error = std::numeric_limits<double>::infinity();
    double error = last_error;
    int iterations = 0;

    // A more fair comparison might include comparing SparseMatrix ldlt to gtsam elimination
    for (int i = 0; i < params.maxIterations; i++) {
        // Linearize each factor
        auto linear = graph_.linearize(values_);

        // Form and solve normal equations for this iter
        auto [A, b] = linear->hessian();
        Eigen::LLT<Matrix, Eigen::Upper> llt(A);
        Vector delta = llt.solve(b);

        /*
        Note doing it this way is much slower due to the dense multiplication
        But it is perfectly possible to accumulate the hessian like above without gtsam.
        Therefore, the following would be an unfair comparison:

        auto [J, e] = linear->jacobian();
        Matrix A = J.transpose() * J;
        Vector b = J.transpose() * e;
        */

        // Apply big vector to each element in values
        VectorValues delta_values(delta, Scatter(*linear));
        values_ = values_.retract(delta_values);

        // Stopping conditions
        error = graph_.error(values_);
        double abs_error_change = last_error - error;
        double rel_error_change = abs_error_change / last_error;
        last_error = error;
        iterations++;

        meta.iteration_errors.push_back(error);
        meta.iteration_step_norms.push_back(delta_values.norm());

        if (abs_error_change < params.absoluteErrorTol ||
            rel_error_change < params.relativeErrorTol)
            break;
    }

    meta.error = error;
    meta.iterations = iterations;
}


SolutionMetadata SolverBase::optimize() {
    auto start = now();
    auto start_build = start;

    build_graph();
    intermediate_values_.clear();
    initial_values_ = values_;

    auto stop_build = now();
    auto start_optimize = stop_build;

    DoglegParams params;
    params.setLinearSolverType(config_.linear_solver_type);
    params.deltaInitial = config_.delta_initial;
    params.maxIterations = config_.max_iterations;
    // params.absoluteErrorTol = 1e-12;
    // params.relativeErrorTol = 1e-12;

    SolutionMetadata meta;
    // Final equality penalty weight reached by the AL outer loop; reused to
    // build a well-conditioned marginals graph below and by
    // get_hessian_and_gradient(). Stored on the instance so diagnostics run
    // after optimize() can rebuild the same penalty surrogate.
    al_final_mu_ = config_.al_initial_mu;
    // Augmented Lagrangian path: a subclass set use_augmented_lagrangian_
    // because the graph carries hard equality constraints (NonlinearConstraint
    // factors). GTSAM's optimizer splits graph_ into cost factors vs.
    // constraints internally and drives the constraint residuals to zero via
    // its own mu/lambda schedule.
    if (use_augmented_lagrangian_) {
        auto p = std::make_shared<AugmentedLagrangianParams>();
        p->initialMuEq      = config_.al_initial_mu;
        p->muEqIncreaseRate = config_.al_mu_increase_rate;
        // Inequality constraints (Section 1.5 collision avoidance) get the same
        // outer-loop penalty schedule as the equality constraints; GTSAM's
        // defaults (1.0 / 2.0) leave the collision penalty orders of magnitude
        // weaker than the contact penalty otherwise.
        p->initialMuIneq      = config_.al_initial_mu;
        p->muIneqIncreaseRate = config_.al_mu_increase_rate;
        p->maxIterations    = config_.al_max_iterations;
        // Uncap dual ascent so the multiplier update is the textbook
        // lambda += mu * violation (see SolverBaseConfig::al_max_dual_step).
        p->maxDualStepSizeEq   = config_.al_max_dual_step;
        p->maxDualStepSizeIneq = config_.al_max_dual_step;
        // Inexact inner solves (fork extension): loosen the inner LM tolerance
        // while mu is small, tighten as 1/mu. 0 disables.
        p->innerRelTolInitial = config_.al_inner_rel_tol_initial;
        // Outer-loop stopping tolerances.
        p->absoluteViolationTolerance = config_.al_abs_violation_tol;
        p->absoluteCostTolerance      = config_.al_abs_cost_tol;
        p->relativeViolationTolerance = config_.al_rel_violation_tol;
        p->relativeCostTolerance      = config_.al_rel_cost_tol;
        p->storeOptProgress = true;  // populate progress() for iteration count
        // Outer-loop trace (iter | muEq | muIneq | cost | eq/ineq violation |
        // inner LM iters) straight from GTSAM, for debugging AL stalls.
        if (std::getenv("CREST_AL_VERBOSE")) p->verbose = true;

        // The AL path always uses a Cholesky-based inner linear solver.
        // Inequality constraints REQUIRE it: GTSAM's AL optimizer builds the
        // inequality Lagrange-multiplier term as a BiasedFactor/AntiFactor
        // pair, and AntiFactor linearizes to a NEGATED HessianFactor; QR
        // elimination cannot consume negative information, so the inner LM
        // silently makes no progress and the outer loop "converges" at the
        // initial values. For equality-only problems it is an optimization:
        // Cholesky elimination measured ~25% faster than QR per inner LM
        // iteration on the hand trajectory solves with an identical result,
        // and the AL path runs thousands of inner iterations.
        std::string linear_solver_type = config_.linear_solver_type;
        if (linear_solver_type.find("QR") != std::string::npos) {
            std::string cholesky =
                (linear_solver_type.find("SEQUENTIAL") != std::string::npos)
                    ? "SEQUENTIAL_CHOLESKY" : "MULTIFRONTAL_CHOLESKY";
            std::cerr << "[SolverBase] AL path: switching linear solver "
                      << linear_solver_type << " -> " << cholesky
                      << " (required for inequality AntiFactors, faster for "
                      << "equality-only)." << std::endl;
            linear_solver_type = cholesky;
        }
        p->lm_params.setLinearSolverType(linear_solver_type);
        p->lm_params.lambdaInitial    = config_.lambda_initial;
        p->lm_params.lambdaUpperBound = config_.lambda_upper_bound;
        p->lm_params.diagonalDamping  = config_.diagonal_damping;
        p->lm_params.maxIterations    = config_.max_iterations;

        // Warm-started outer loop (Section 1.8 controller): carry mu and the
        // multipliers from the previous call so successive solves of the same
        // constrained problem keep tightening it, instead of each restarting at
        // al_initial_mu with zero duals. See SolverBaseConfig::al_warm_start_duals
        // for why a short-budget loop cannot enforce anything without this.
        crest_sparse::WarmAugmentedLagrangianOptimizer optimizer(graph_, values_, p);
        if (config_.al_warm_start_duals) {
            values_ = optimizer.optimizeWarm(al_warm_, config_.al_warm_mu_max);
        } else {
            values_ = optimizer.optimize();
        }
        const auto& progress = optimizer.progress();
        if (!progress.empty()) {
            meta.iterations = progress.back().iteration;
            if (progress.back().muEq > 0.0) al_final_mu_ = progress.back().muEq;
        }
        // Surface the per-outer-iteration trace for debug visualization. The
        // LM/Dogleg iterate-loop below is skipped on the AL path, so we mirror
        // it here from progress(): scalar convergence curves (cost / constraint
        // violation / penalty mu) plus full-trajectory Values snapshots for
        // step-by-step replay (see get_intermediate_solutions()).
        if (config_.record_iterations) {
            for (size_t i = 0; i < progress.size(); ++i) {
                const auto& st = progress[i];
                meta.al_iteration_costs.push_back(st.cost);
                meta.al_iteration_violations.push_back(st.violation());
                meta.al_iteration_mus.push_back(st.muEq);
                if (config_.iteration_sample_interval <= 0 ||
                    i % config_.iteration_sample_interval == 0)
                    intermediate_values_.push_back(st.values);
            }
        }
        // meta.error is set below from the cost-only graph; the full-graph
        // error would be dominated by the (large) hard-constraint penalty.
    } else if (config_.use_dense) {
        // If we want to use dense solver, e.g. for comparison
        optimize_dense_benchmark(params, meta);
    } else if (config_.optimizer_type == "LM") {
        LevenbergMarquardtParams lm_params;
        lm_params.setLinearSolverType(config_.linear_solver_type);
        lm_params.lambdaInitial = config_.lambda_initial;
        lm_params.lambdaUpperBound = config_.lambda_upper_bound;
        lm_params.diagonalDamping = config_.diagonal_damping;
        lm_params.maxIterations = config_.max_iterations;
        LevenbergMarquardtOptimizer optimizer(graph_, values_, lm_params);
        if (config_.record_iterations) {
            double prev_error = optimizer.error();
            meta.iteration_errors.push_back(prev_error);
            meta.iteration_trust_region.push_back(optimizer.lambda());
            Values prev_vals = optimizer.values();
            for (int i = 0; i < lm_params.maxIterations - 1; i++) {
                optimizer.iterate();
                double curr_error = optimizer.error();
                Values curr_vals = optimizer.values();
                meta.iteration_errors.push_back(curr_error);
                meta.iteration_trust_region.push_back(optimizer.lambda());
                meta.iteration_step_norms.push_back(
                    prev_vals.localCoordinates(curr_vals).norm());
                if (config_.iteration_sample_interval > 0 &&
                    i % config_.iteration_sample_interval == 0)
                    intermediate_values_.push_back(curr_vals);
                double abs_change = prev_error - curr_error;
                if (std::abs(abs_change) < lm_params.absoluteErrorTol ||
                    (prev_error > 0 && std::abs(abs_change) / prev_error < lm_params.relativeErrorTol))
                    break;
                prev_error = curr_error;
                prev_vals = std::move(curr_vals);
            }
            values_ = optimizer.values();
        } else {
            values_ = optimizer.optimize();
        }
        meta.error = optimizer.error();
        meta.iterations = optimizer.iterations();
    } else {
        // Default: Dogleg
        DoglegOptimizer optimizer(graph_, values_, params);
        if (config_.record_iterations) {
            double prev_error = optimizer.error();
            meta.iteration_errors.push_back(prev_error);
            meta.iteration_trust_region.push_back(optimizer.getDelta());
            Values prev_vals = optimizer.values();
            for (int i = 0; i < params.maxIterations - 1; i++) {
                optimizer.iterate();
                double curr_error = optimizer.error();
                Values curr_vals = optimizer.values();
                meta.iteration_errors.push_back(curr_error);
                meta.iteration_trust_region.push_back(optimizer.getDelta());
                meta.iteration_step_norms.push_back(
                    prev_vals.localCoordinates(curr_vals).norm());
                if (config_.iteration_sample_interval > 0 &&
                    i % config_.iteration_sample_interval == 0)
                    intermediate_values_.push_back(curr_vals);
                double abs_change = prev_error - curr_error;
                if (std::abs(abs_change) < params.absoluteErrorTol ||
                    (prev_error > 0 && std::abs(abs_change) / prev_error < params.relativeErrorTol))
                    break;
                prev_error = curr_error;
                prev_vals = std::move(curr_vals);
            }
            values_ = optimizer.values();
        } else {
            values_ = optimizer.optimize();
        }
        meta.error = optimizer.error();
        meta.iterations = optimizer.iterations();
    }

    auto stop_optimize = now();
    auto start_marginalize = stop_optimize;

    // A hard-constraint (Constrained-noise) factor left in the graph makes the
    // Cholesky/QR factorization behind Marginals singular. But simply dropping
    // the contact constraint also drops the information it carries — e.g. an
    // underactuated tendon whose tension is pinned *only* by the contact would
    // become indeterminate. So we instead replace each NonlinearConstraint with
    // its finite-weight penalty factor at the final AL penalty (mu): a
    // well-conditioned Gaussian (info ~ mu) that preserves the constraint's
    // information contribution. The free-space path has no constraints and so
    // builds marginals over the full graph unchanged.
    // skip_marginals: the factorization above is the single most expensive step
    // after the optimizer itself, and a real-time controller (Section 1.8) re-runs
    // this loop every control tick while only ever using the MEANS. Subclasses
    // that honor this flag must extract means-only marginals in extract_solution()
    // (e.g. TendonHandModel::get_marginals_means_only); marginals_ is left as the
    // default-constructed empty object, so reading covariances from it is invalid.
    if (config_.skip_marginals) {
        // Still report the objective-only error, which is cheap and is what the
        // caller's convergence reporting expects on the AL path.
        if (use_augmented_lagrangian_) {
            NonlinearFactorGraph err_graph;
            for (const auto& factor : graph_) {
                if (auto c = std::dynamic_pointer_cast<NonlinearConstraint>(factor)) {
                    err_graph.add(c->penaltyFactor(al_final_mu_));
                } else {
                    err_graph.add(factor);
                }
            }
            meta.error = err_graph.error(values_);
        }
    } else if (use_augmented_lagrangian_) {
        NonlinearFactorGraph marg_graph;
        for (const auto& factor : graph_) {
            if (auto c = std::dynamic_pointer_cast<NonlinearConstraint>(factor)) {
                marg_graph.add(c->penaltyFactor(al_final_mu_));
            } else {
                marg_graph.add(factor);
            }
        }
        // Report the objective error (the penalty term is ~0 at convergence).
        meta.error = marg_graph.error(values_);
        marginals_ = Marginals(marg_graph, values_);
    } else {
        marginals_ = Marginals(graph_, values_);
    }

    auto stop_marginalize = now();
    auto start_extract = stop_marginalize;

    extract_solution();

    auto stop_extract = now();
    auto stop = stop_extract;

    
    meta.total_time_ms       = ms(start, stop);
    meta.build_time_ms       = ms(start_build, stop_build);
    meta.optimize_time_ms    = ms(start_optimize, stop_optimize);
    meta.marginalize_time_ms = ms(start_marginalize, stop_marginalize);
    meta.extract_time_ms     = ms(start_extract, stop_extract);



    return meta;
}


void SolverBase::reset_al_duals() {
    al_warm_.clear();
}


std::vector<std::tuple<std::string, int, double>>
SolverBase::get_factor_error_summary() const {
    std::map<std::string, std::pair<int, double>> by_type;
    for (const auto& factor : graph_) {
        if (!factor) continue;
        const auto& f = *factor;
        const std::string name = gtsam::demangle(typeid(f).name());
        const double e = factor->error(values_);
        auto& entry = by_type[name];
        entry.first  += 1;
        entry.second += e;
    }
    std::vector<std::tuple<std::string, int, double>> out;
    out.reserve(by_type.size());
    for (const auto& [name, ce] : by_type) {
        out.emplace_back(name, ce.first, ce.second);
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return std::get<2>(a) > std::get<2>(b);
    });
    return out;
}


std::vector<std::pair<std::string, std::vector<double>>>
SolverBase::get_factor_errors_by_type() const {
    // Accumulate per-type lists in a map, then materialize sorted by total.
    std::map<std::string, std::vector<double>> by_type;
    for (const auto& factor : graph_) {
        if (!factor) continue;
        const auto& f = *factor;
        const std::string name = gtsam::demangle(typeid(f).name());
        by_type[name].push_back(factor->error(values_));
    }
    std::vector<std::pair<std::string, std::vector<double>>> out;
    out.reserve(by_type.size());
    for (auto& [name, errs] : by_type) {
        out.emplace_back(name, std::move(errs));
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        double sa = 0.0, sb = 0.0;
        for (double v : a.second) sa += v;
        for (double v : b.second) sb += v;
        return sa > sb;
    });
    return out;
}


std::vector<std::tuple<std::string, int, double>>
SolverBase::get_initial_factor_error_summary() const {
    std::map<std::string, std::pair<int, double>> by_type;
    for (const auto& factor : graph_) {
        if (!factor) continue;
        const auto& f = *factor;
        const std::string name = gtsam::demangle(typeid(f).name());
        const double e = factor->error(initial_values_);
        auto& entry = by_type[name];
        entry.first  += 1;
        entry.second += e;
    }
    std::vector<std::tuple<std::string, int, double>> out;
    out.reserve(by_type.size());
    for (const auto& [name, ce] : by_type) {
        out.emplace_back(name, ce.first, ce.second);
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return std::get<2>(a) > std::get<2>(b);
    });
    return out;
}


std::pair<Eigen::MatrixXd, Eigen::VectorXd>
SolverBase::get_hessian_and_gradient() const {
    // A raw linearization of a constrained-noise (hard-equality) factor cannot
    // form a Hessian ("cannot update information with constrained noise model").
    // On the AL path the graph still carries such factors, so mirror the
    // marginals rebuild: replace each NonlinearConstraint with its finite-weight
    // penalty factor at the final AL penalty (mu) before linearizing. This gives
    // a well-conditioned diagnostic Hessian that reflects the converged problem.
    const NonlinearFactorGraph* graph_ptr = &graph_;
    NonlinearFactorGraph penalty_graph;
    if (use_augmented_lagrangian_) {
        for (const auto& factor : graph_) {
            if (auto c = std::dynamic_pointer_cast<NonlinearConstraint>(factor)) {
                penalty_graph.add(c->penaltyFactor(al_final_mu_));
            } else {
                penalty_graph.add(factor);
            }
        }
        graph_ptr = &penalty_graph;
    }
    auto linear = graph_ptr->linearize(values_);
    // GaussianFactorGraph::hessian() returns (Hessian, information vector g)
    // where the linearized quadratic is 0.5 dx^T H dx - g^T dx + const.
    auto [H, g] = linear->hessian();
    return {std::move(H), std::move(g)};
}