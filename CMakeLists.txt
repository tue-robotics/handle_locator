cmake_minimum_required(VERSION 3.0.2)
project(handle_locator)

find_package(catkin REQUIRED COMPONENTS
    actionlib
    geometry_msgs
    pcl_ros
    roscpp
    sensor_msgs
    tf2
    tf2_geometry_msgs
    tue_msgs
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

catkin_package(
    LIBRARIES handle_locator
    CATKIN_DEPENDS actionlib geometry_msgs pcl_ros roscpp sensor_msgs tf2 tf2_geometry_msgs tue_msgs
)

add_library(handle_locator src/handle_locator.cpp)
target_link_libraries(handle_locator ${catkin_LIBRARIES})

add_executable(locate_handle src/locate_handle.cpp)
target_link_libraries(locate_handle handle_locator)
