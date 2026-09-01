cmake_minimum_required(VERSION 3.18)
project(gepetto_solvers LANGUAGES CXX)

# Set release build flags for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Use Python's prefix to find conda packages (works even when CONDA_PREFIX isn't set)
get_filename_component(PYTHON_PREFIX "${Python_EXECUTABLE}" DIRECTORY)
get_filename_component(PYTHON_PREFIX "${PYTHON_PREFIX}" DIRECTORY)
list(APPEND CMAKE_PREFIX_PATH ${PYTHON_PREFIX})
list(APPEND CMAKE_MODULE_PATH ${PYTHON_PREFIX}/lib/cmake/OpenVDB)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)

# OpenMP for parallel collision detection
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel collision detection")
else()
    message(WARNING "OpenMP not found, collision detection will be single-threaded")
endif()

# Sources are listed explicitly rather than globbed. `file(GLOB_RECURSE)` means a
# deleted or added file needs no build-file edit, which sounds convenient and is
# how the last prune produced a build that still referenced a deleted directory
# from a stale cache -- documented in the architecture doc as a trap worth
# avoiding. An explicit list makes the file set part of the reviewed diff, and
# makes a missing file a configure-time error instead of a link-time one.
set(GEPETTO_SOLVERS_SOURCES
    src/cosserat_dynamics/CosseratDynamicsFactor.cpp
    src/cosserat_dynamics/CosseratDynamicsSolver.cpp

    src/cosserat_rod/BoundaryStressFactor.cpp
    src/cosserat_rod/CosseratRodModel.cpp
    src/cosserat_rod/CosseratStressFactor.cpp
    src/cosserat_rod/CosseratTwistFactor.cpp
    src/cosserat_rod/PlanarBendFactor.cpp
    src/cosserat_rod/RootBoundaryStressFactor.cpp
    src/cosserat_rod/RootCosseratStressFactor.cpp
    src/cosserat_rod/RootCosseratTwistFactor.cpp

    src/measurement/PositionPriorFactor.cpp

    src/digits/tendon/TendonDiscWrenchFactor.cpp
    src/digits/tendon/TendonFingerModel.cpp
    src/digits/tendon/TendonLengthFactor.cpp

    # The kinematics-agnostic hand: the graph builder, its constraint tagger and
    # the registry the kinematics is loaded from.
    src/hand/ConstraintTagger.cpp
    src/hand/HandGraph.cpp
    src/hand/HandKinematicsRegistry.cpp
    src/hand/HandModel.cpp
    src/hand/HandSolver.cpp
    src/hand/HandTrajectoryPlanner.cpp
    src/hand/HandValues.cpp

    # One translation unit per hand kinematics. Each self-registers at static
    # init, so listing it here is what makes it loadable by name.
    src/hand/kinematics/tendon/TendonHandKinematics.cpp

    src/utils/SolverBase.cpp
    src/utils/WrenchTransforms.cpp

    # The pybind11 layer, all of it in one place.
    src/bindings/bind_cosserat_dynamics.cpp
    src/bindings/bind_cosserat_rod.cpp
    src/bindings/bind_digits_tendon.cpp
    src/bindings/bind_hand.cpp
    src/bindings/module.cpp
)

pybind11_add_module(_gepetto_solvers ${GEPETTO_SOLVERS_SOURCES})

target_compile_features(_gepetto_solvers PRIVATE cxx_std_17)

# Public headers live in include/gepetto_solvers/<module>/, so every internal
# include reads `#include "gepetto_solvers/<module>/X.h"` -- the same path a
# consumer outside this build would use. src/ is on the path too, but only so the
# bindings can reach their own bindings.h; nothing else in src/ is includable.
target_include_directories(_gepetto_solvers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(_gepetto_solvers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GTSAM_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)

target_link_libraries(_gepetto_solvers PRIVATE gtsam OpenVDB::openvdb)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_gepetto_solvers PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_gepetto_solvers PRIVATE GEPETTO_USE_OPENMP)
endif()

install(TARGETS _gepetto_solvers LIBRARY DESTINATION gepetto_solvers)

# target_compile_features(gepetto_solvers PRIVATE cxx_std_17)
# target_compile_options(gepetto_solvers PRIVATE -O3 -fPIC)

# set_target_properties(gepetto_solvers PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/gepetto_solvers")
