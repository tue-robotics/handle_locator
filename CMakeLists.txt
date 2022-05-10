cmake_minimum_required(VERSION 3.0.2)
project(handle_locator)

find_package(catkin REQUIRED COMPONENTS
    actionlib
    geometry_msgs
    pcl_ros
    roscpp
    sensor_msgs
    tf
    tue_msgs
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

catkin_package(
    LIBRARIES handle_locator
    CATKIN_DEPENDS actionlib geometry_msgs pcl_ros roscpp sensor_msgs tf tue_msgs
)

add_executable(handle_locator src/locate_handle.cpp)

target_link_libraries(handle_locator ${catkin_LIBRARIES})
