#pragma once

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>

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
    int error;
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

    const SolverBaseConfig config_;
};