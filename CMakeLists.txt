cmake_minimum_required(VERSION 3.15)
project(tendon_robot_gtsam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # for shared libs

find_package(GTSAM REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    include
    ${GTSAM_INCLUDE_DIR}
)

file(GLOB MY_SOURCES src/*.cpp)
add_library(tendon_robot_gtsam SHARED ${MY_SOURCES})
target_link_libraries(tendon_robot_gtsam PRIVATE gtsam)

pybind11_add_module(tendon_robot_gtsam_py python/src/bindings.cpp)

target_link_libraries(tendon_robot_gtsam_py PRIVATE tendon_robot_gtsam gtsam)
