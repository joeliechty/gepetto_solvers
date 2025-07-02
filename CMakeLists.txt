cmake_minimum_required(VERSION 3.8)
project(tendon_robot_gtsam)

#set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#set(CMAKE_INSTALL_RPATH "/usr/local/lib")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include/tendon_robot_gtsam
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)


add_executable(tendon_robot_solver src/tendon_robot_solver.cpp)

target_link_libraries(tendon_robot_solver
  gtsam
) 

ament_target_dependencies(tendon_robot_solver
  rclcpp
  geometry_msgs
  sensor_msgs
)



add_executable(tension_publisher src/tension_publisher.cpp)

ament_target_dependencies(tension_publisher
  rclcpp
  sensor_msgs
)



add_executable(tendon_robot_viz src/tendon_robot_viz.cpp)

target_link_libraries(tendon_robot_viz 
  gtsam
)

ament_target_dependencies(tendon_robot_viz
  rclcpp
  geometry_msgs
  visualization_msgs
)



install(TARGETS
  tension_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  tendon_robot_solver
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  tendon_robot_viz
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()

