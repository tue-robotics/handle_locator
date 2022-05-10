#ifndef HANDLE_LOCATOR_HANDLE_LOCATOR_H_
#define HANDLE_LOCATOR_HANDLE_LOCATOR_H_

#include <actionlib/server/simple_action_server.h>

#include <pcl/point_types.h>

#include <pcl_ros/point_cloud.h>

#include <ros/callback_queue.h>
#include <ros/subscriber.h>

#include <tf2_ros/buffer.h>

#include <tue_msgs/LocateDoorHandleAction.h>

#include <memory>

namespace tf2_ros
{
class TransformListener;
}

namespace handle_locator
{

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class HandleLocator
{

public:
    HandleLocator();

    virtual ~HandleLocator();

private:

    void executeCB(const tue_msgs::LocateDoorHandleGoalConstPtr& goal);

    void pcCB(const sensor_msgs::PointCloud2ConstPtr& msg);

    std::unique_ptr<actionlib::SimpleActionServer<tue_msgs::LocateDoorHandleAction>> as_;

    ros::Subscriber pc_sub_;
    ros::CallbackQueue cb_queue_;

    sensor_msgs::PointCloud2ConstPtr pc_;

    tf2_ros::Buffer tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

};


}

#endif // HANDLE_LOCATOR_HANDLE_LOCATOR_H_
