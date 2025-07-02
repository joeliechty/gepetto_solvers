#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <random>

using std::placeholders::_1;

void publish_tube_marker_array(
    const std::vector<Eigen::Isometry3d>& poses,
    const std::string& tube_name,
    const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& marker_pub,
    rclcpp::Time now)
{
    int num_poses = poses.size();
    if (num_poses < 2) return;

    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.reserve(num_poses - 1);
    int marker_id = 0;

    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.frame_id = "/ves/left/base";
    pose_array.header.stamp = now;

    for (const auto& pose : poses) {
        geometry_msgs::msg::Pose p;
        p.position.x = pose.translation().x();
        p.position.y = pose.translation().y();
        p.position.z = pose.translation().z();

        Eigen::Quaterniond q(pose.rotation());
        p.orientation.x = q.x();
        p.orientation.y = q.y();
        p.orientation.z = q.z();
        p.orientation.w = q.w();

        pose_array.poses.push_back(p);
    }

    for (int i = 0; i < num_poses - 1; ++i) {
        const auto& p1 = poses[i].translation();
        const auto& p2 = poses[i + 1].translation();
        Eigen::Vector3d dir = p2 - p1;
        double height = dir.norm();
        if (height < 1e-6) continue;

        Eigen::Vector3d midpoint = 0.5 * (p1 + p2);
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(
            Eigen::Vector3d::UnitZ(), dir.normalized());

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "/ves/left/base";
        marker.header.stamp = now;
        marker.ns = tube_name + "_tube";
        marker.id = marker_id++;
        marker.type = marker.CYLINDER;
        marker.action = marker.ADD;
        marker.pose.position.x = midpoint.x();
        marker.pose.position.y = midpoint.y();
        marker.pose.position.z = midpoint.z();
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        marker.scale.z = height;
        marker.color.a = 0.5f;
        marker.color.b = 0.2f;
        marker.lifetime = rclcpp::Duration::from_seconds(0.1);

        if (tube_name == "outer") {
            marker.scale.x = marker.scale.y = 0.0012;
            marker.color.r = 0.1f;
            marker.color.g = 0.8f;
        } else {
            marker.scale.x = marker.scale.y = 0.001;
            marker.color.r = 0.8f;
            marker.color.g = 0.1f;
        }

        marker_array.markers.push_back(marker);
    }

    marker_pub->publish(marker_array);
}

std::vector<Eigen::Isometry3d> deserialize_poses(const geometry_msgs::msg::PoseArray &msg) {
    std::vector<Eigen::Isometry3d> poses;
    poses.reserve(msg.poses.size());

    for (const auto &pose_msg : msg.poses) {
        Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();

        // Translation
        iso.translation() = Eigen::Vector3d(pose_msg.position.x, pose_msg.position.y, pose_msg.position.z);

        // Rotation (quaternion)
        Eigen::Quaterniond q(pose_msg.orientation.w,
                             pose_msg.orientation.x,
                             pose_msg.orientation.y,
                             pose_msg.orientation.z);
        iso.rotate(q);

        poses.push_back(iso);
    }
    return poses;
}

class VesVizNode : public rclcpp::Node {
public:
    VesVizNode() : Node("ves_viz_node") {
        outer_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "ves_model_gtsam/viz/outer_tube_marker", 10);
        
        inner_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "ves_model_gtsam/viz/inner_tube_marker", 10);

        outer_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "ves_model_gtsam/outer_tube_poses", 10,
            std::bind(&VesVizNode::outer_tube_callback, this, _1));

        inner_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "ves_model_gtsam/inner_tube_poses", 10,
            std::bind(&VesVizNode::inner_tube_callback, this, _1));
        
        tip_force_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/ves_model_gtsam/tip_force", 10,
            std::bind(&VesVizNode::tip_force_callback, this, std::placeholders::_1));
        
        tip_force_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "ves_model_gtsam/viz/tip_force_marker", 10);

        tip_pose_ = Eigen::Isometry3d::Identity();
    }

private:
    void outer_tube_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::vector<Eigen::Isometry3d> poses = deserialize_poses(*msg);
        publish_tube_marker_array(poses, "outer", outer_marker_pub_, this->now());
    }

    void inner_tube_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::vector<Eigen::Isometry3d> poses = deserialize_poses(*msg);
        publish_tube_marker_array(poses, "inner", inner_marker_pub_, this->now());

        tip_pose_ = poses.back();
    }
    
    void tip_force_callback(const geometry_msgs::msg::Vector3::SharedPtr msg){
        geometry_msgs::msg::Point start_point;
        start_point.x = tip_pose_.translation().x();
        start_point.y = tip_pose_.translation().y();
        start_point.z = tip_pose_.translation().z();

        double scale = 0.003;  // adjust to your liking

        // Convert geometry_msgs::msg::Vector3 to Eigen Vector3d
        Eigen::Vector3d tip_force_local(msg->x, msg->y, msg->z);

        // Rotate tip force into world frame:
        Eigen::Vector3d tip_force_global = tip_pose_.rotation() * tip_force_local;

        geometry_msgs::msg::Point end_point;
        end_point.x = start_point.x + scale * tip_force_global.x();
        end_point.y = start_point.y + scale * tip_force_global.y();
        end_point.z = start_point.z + scale * tip_force_global.z();

        // Create the marker
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "/ves/left/base";
        marker.header.stamp = this->now();
        marker.ns = "tip_force";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.points.push_back(start_point);
        marker.points.push_back(end_point);

        marker.scale.x = 0.0002;  // shaft diameter
        marker.scale.y = 0.0004;  // head diameter
        marker.scale.z = 0.0003;  // head length

        marker.color.r = 0.6;
        marker.color.g = 0.1;
        marker.color.b = 0.9;
        marker.color.a = 1.0;

        marker.lifetime = rclcpp::Duration::from_seconds(0.1);  // Optional: how long it stays

        tip_force_marker_pub_->publish(marker);
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr outer_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr inner_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr tip_force_sub_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr outer_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr inner_marker_pub_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tip_force_marker_pub_;
   
    Eigen::Isometry3d tip_pose_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VesVizNode>());
    rclcpp::shutdown();
    return 0;
}