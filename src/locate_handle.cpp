#include <ros/init.h>

#include <handle_locator/handle_locator.h>


int main(int argc, char** argv)
{
    ros::init(argc, argv, "locate_handle_server");
    handle_locator::HandleLocator hl;

    ROS_INFO("Handle locator server is active and spinning...");

    ros::spin();

    return 0;
}
