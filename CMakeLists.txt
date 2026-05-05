cmake_minimum_required(VERSION 3.18)
project(crest_sparse LANGUAGES CXX)

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

file(GLOB_RECURSE SOURCES "src/*.cpp")

pybind11_add_module(_crest_sparse ${SOURCES})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GTSAM_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)

target_link_libraries(_crest_sparse PRIVATE gtsam OpenVDB::openvdb)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_crest_sparse PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_crest_sparse PRIVATE CREST_USE_OPENMP)
endif()

install(TARGETS _crest_sparse LIBRARY DESTINATION crest_sparse)

# target_compile_features(crest_sparse PRIVATE cxx_std_17)
# target_compile_options(crest_sparse PRIVATE -O3 -fPIC)

# set_target_properties(crest_sparse PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/crest_sparse")
