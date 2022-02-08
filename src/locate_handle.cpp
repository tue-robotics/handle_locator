#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <tue_msgs/LocateDoorHandleAction.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#define MAX_ERROR 1.0  // if the error is larger than 1m, it's unlikely we found the handle
#define BOUNDING_BOX_SIZE 0.5

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

std::unique_ptr<actionlib::SimpleActionServer<tue_msgs::LocateDoorHandleAction>> as_;
std::unique_ptr<tf::TransformListener> tf_listener_;

void executeCB(const tue_msgs::LocateDoorHandleGoalConstPtr& goal) {
    ros::NodeHandle n;

    const sensor_msgs::PointCloud2ConstPtr msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(
            "/hero/head_rgbd_sensor/depth_registered/rectified_points", n, ros::Duration(1.0));

    if (!msg)
    {
        as_->setAborted();
        ROS_ERROR("No pointcloud message received within %.2f seconds", 1.0);
        return;
    }

    // Convert the msg to a real cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    // transform the two frame points of the door to the frame of the ros message
    geometry_msgs::PointStamped original_handle_location;

    // Lookup the current vector of the desired TF to the grippoint
    if(tf_listener_->waitForTransform(msg->header.frame_id, goal->handle_location_estimate.header.frame_id,
                                   msg->header.stamp, ros::Duration(1.0)))
    {
        try
        {
            tf_listener_->transformPoint(msg->header.frame_id, goal->handle_location_estimate, original_handle_location);
        }
        catch(tf::TransformException ex)
        {
            as_->setAborted();
            ROS_ERROR("%s",ex.what());
            return;
        }
    } else
    {
        as_->setAborted();
        ROS_ERROR("TF could not find transform");
        return;
    }

    // This crops the pointcloud to a bounding box of 25 cm around the original handle location
    double min_x = original_handle_location.point.x - (BOUNDING_BOX_SIZE/2.0);
    double min_z = original_handle_location.point.z - (BOUNDING_BOX_SIZE/2.0);
    double max_x = original_handle_location.point.x + (BOUNDING_BOX_SIZE/2.0);
    double max_z = original_handle_location.point.z + (BOUNDING_BOX_SIZE/2.0);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(min_x, max_x);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_z, max_z);
    pass.filter(*cloud_cropped);

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.01);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);

    uint nr_points = (int) cloud_cropped->points.size();
    while (cloud_cropped->points.size() > 0.8 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud(cloud_cropped);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0)
        {
            ROS_ERROR("Could not estimate a planar model for the given dataset.");
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_cropped);
        extract.setIndices(inliers);
        extract.setNegative(false);

        // Get the points associated with the planar surface
        extract.filter(*cloud_plane);

        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        extract.filter(*cloud_f);
        *cloud_cropped = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_cropped);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.01); // 1cm
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_cropped);
    ec.extract(cluster_indices);

    double min_error = MAX_ERROR; // initialized at MAX_ERROR but overwritten with found minimal error
    pcl::PointCloud<pcl::PointXYZ>::Ptr handle_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud_cropped->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_cluster, centroid);

        // Select the cluster with centroid closest to original estimate
        double error_x = std::abs(centroid(0) - original_handle_location.point.x);
        double error_y = std::abs(centroid(1) - original_handle_location.point.y);
        double error_z = std::abs(centroid(2) - original_handle_location.point.z);
        double measured_error = error_x + error_y + error_z;
        ROS_DEBUG_STREAM("Total error of the cluster: " << measured_error);
        // ToDo: could also filter based on number of points to improve selection (handle ~1000 points)
        if (measured_error < min_error)
        {
            *handle_cluster = *cloud_cluster;
            min_error = measured_error;
        }
        ROS_DEBUG_STREAM("PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points.");
    }

    // check that preempt has not been requested by the client
    if (as_->isPreemptRequested() || !ros::ok()) {
        // set the action state to preempted
        as_->setPreempted();
    }
    if (min_error < MAX_ERROR) {  // if the minimal error is sufficiently low, the corresponding cluster is found to be the handle
        Eigen::Vector4f min_point;
        Eigen::Vector4f max_point;
        pcl::getMinMax3D(*handle_cluster, min_point, max_point);
        ROS_INFO_STREAM("Handle cluster min_point: " << min_point << ", max_point: " << max_point);
        tue_msgs::LocateDoorHandleResult result;
        result.handle_edge_point1.header.frame_id = msg->header.frame_id;
        result.handle_edge_point1.point.x = min_point(0);
        result.handle_edge_point1.point.y = min_point(1);
        result.handle_edge_point1.point.z = min_point(2);
        result.handle_edge_point2.header.frame_id = msg->header.frame_id;
        result.handle_edge_point2.point.x = max_point(0);
        result.handle_edge_point2.point.y = max_point(1);
        result.handle_edge_point2.point.z = max_point(2);
        as_->setSucceeded(result);
    } else
    {
        as_->setAborted();
    }
}

int main(int argc, char** argv)
{

    ros::init(argc, argv, "locate_handle_server");
    ros::NodeHandle nh;

    tf_listener_ = std::unique_ptr<tf::TransformListener>(new tf::TransformListener());
    as_ = std::unique_ptr<actionlib::SimpleActionServer<tue_msgs::LocateDoorHandleAction> >(new actionlib::SimpleActionServer
                                            <tue_msgs::LocateDoorHandleAction>(nh, "locate_handle", &executeCB, false));

    as_->start();

    ROS_INFO("Handle locator server is active and spinning...");

    ros::spin();

    return 0;
}
