cmake_minimum_required(VERSION 3.15)
project(tendon_robot_gtsam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure position-independent code for shared libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(GTSAM REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(
    include
    ${GTSAM_INCLUDE_DIR}
)

# Collect source files
file(GLOB MY_SOURCES src/*.cpp)

# Build the shared library
add_library(tendon_robot_gtsam SHARED ${MY_SOURCES})
target_link_libraries(tendon_robot_gtsam PRIVATE gtsam)

# Now build the PyBind11 module
# pybind11_add_module(tendon_robot_gtsam src/bindings.cpp)
# target_link_libraries(tendon_robot_gtsam PRIVATE tendon_robot_gtsam gtsam)

# Optional: install targets
# install(TARGETS tendon_robot_gtsam tendon_robot_gtsam
#     LIBRARY DESTINATION lib
# )
# install(DIRECTORY include/ DESTINATION include)
