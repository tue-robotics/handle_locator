cmake_minimum_required(VERSION 2.8.3)
project(handle_locator)
add_definitions(-std=c++14)

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
    CATKIN_DEPENDS
        actionlib
        geometry_msgs
        pcl_ros
        roscpp
        sensor_msgs
        tf
        tue_msgs
)

add_executable(locate_handle_action src/handle_locator/locate_handle.cpp)

target_link_libraries(locate_handle_action ${catkin_LIBRARIES})
