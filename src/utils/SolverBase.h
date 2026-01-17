#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>


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


class SolverBase {
public:
    SolverBase();

    SolutionMetadata optimize();

private:
    virtual void build_graph() = 0;

    virtual void extract_solution() = 0;

    virtual void get_initial_values() = 0;

protected:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;
};