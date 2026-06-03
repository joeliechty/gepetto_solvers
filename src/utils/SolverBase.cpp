#include "SolverBase.h"

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/constrained/AugmentedLagrangianOptimizer.h>
#include <gtsam/constrained/NonlinearConstraint.h>

#include <gtsam/base/types.h>

#include <algorithm>
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
    double error;
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
    // build a well-conditioned marginals graph below.
    double al_final_mu = config_.al_initial_mu;
    // Augmented Lagrangian path: a subclass set use_augmented_lagrangian_
    // because the graph carries hard equality constraints (NonlinearConstraint
    // factors). GTSAM's optimizer splits graph_ into cost factors vs.
    // constraints internally and drives the constraint residuals to zero via
    // its own mu/lambda schedule.
    if (use_augmented_lagrangian_) {
        auto p = std::make_shared<AugmentedLagrangianParams>();
        p->initialMuEq      = config_.al_initial_mu;
        p->muEqIncreaseRate = config_.al_mu_increase_rate;
        p->maxIterations    = config_.al_max_iterations;
        p->storeOptProgress = true;  // populate progress() for iteration count
        p->lm_params.setLinearSolverType(config_.linear_solver_type);
        p->lm_params.lambdaInitial    = config_.lambda_initial;
        p->lm_params.lambdaUpperBound = config_.lambda_upper_bound;
        p->lm_params.diagonalDamping  = config_.diagonal_damping;
        p->lm_params.maxIterations    = config_.max_iterations;

        AugmentedLagrangianOptimizer optimizer(graph_, values_, p);
        values_ = optimizer.optimize();
        const auto& progress = optimizer.progress();
        if (!progress.empty()) {
            meta.iterations = progress.back().iteration;
            if (progress.back().muEq > 0.0) al_final_mu = progress.back().muEq;
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
        values_ = optimizer.optimize();
        meta.error = optimizer.error();
        meta.iterations = optimizer.iterations();
    } else {
        // Default: Dogleg
        DoglegOptimizer optimizer(graph_, values_, params);
        values_ = optimizer.optimize();
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
    if (use_augmented_lagrangian_) {
        NonlinearFactorGraph marg_graph;
        for (const auto& factor : graph_) {
            if (auto c = std::dynamic_pointer_cast<NonlinearConstraint>(factor)) {
                marg_graph.add(c->penaltyFactor(al_final_mu));
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