#include "SolverBase.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

using namespace gtsam;


SolverBase::SolverBase() {}


inline auto now() {
    return std::chrono::high_resolution_clock::now();
}


template <typename ClockTimePoint>
inline double ms(const ClockTimePoint& start, const ClockTimePoint& stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}


SolutionMetadata SolverBase::optimize() {
    auto start = now();
    auto start_build = start;

    build_graph();
    
    auto stop_build = now();
    auto start_optimize = stop_build;

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    params.deltaInitial = dogleg_delta_.value_or(1.0);  // If we have a delta from previous solve, use it
    DoglegOptimizer optimizer(graph_, values_, params);

    // LevenbergMarquardtParams params;
    // params.setLinearSolverType("MULTIFRONTAL_QR");
    // LevenbergMarquardtOptimizer optimizer(graph_, values_, params);

    values_ = optimizer.optimize();
    dogleg_delta_ = optimizer.getDelta(); // Save delta for next solve
    
    auto stop_optimize = now();
    auto start_marginalize = stop_optimize;

    marginals_ = Marginals(graph_, values_);

    auto stop_marginalize = now();
    auto start_extract = stop_marginalize;

    extract_solution();

    auto stop_extract = now();
    auto stop = stop_extract;

    SolutionMetadata meta;
    meta.total_time_ms       = ms(start, stop);
    meta.build_time_ms       = ms(start_build, stop_build);
    meta.optimize_time_ms    = ms(start_optimize, stop_optimize);
    meta.marginalize_time_ms = ms(start_marginalize, stop_marginalize);
    meta.extract_time_ms     = ms(start_extract, stop_extract);

    meta.error= optimizer.error();
    meta.iterations = optimizer.iterations();

    return meta;
}