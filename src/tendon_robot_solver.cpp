#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "tendon_robot_gtsam.h"

sensor_msgs::msg::PointCloud2 get_uncertainty_cloud(
    const std::vector<gtsam::Pose3>& poses,
    const std::vector<gtsam::Matrix6>& covariances,
    const int num_samples_per_pose,
    const std::string frame_id,
    const rclcpp::Time& now)
{
    using sensor_msgs::msg::PointCloud2;
    using sensor_msgs::PointCloud2Modifier;
    using sensor_msgs::PointCloud2Iterator;

    PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = now;
    cloud_msg.height = 1;
    cloud_msg.is_dense = false;

    size_t total_points = poses.size() * num_samples_per_pose;
    PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(total_points);

    PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < poses.size(); ++i) {
        const gtsam::Matrix6& cov = covariances[i];
        Eigen::SelfAdjointEigenSolver<gtsam::Matrix6> solver(cov);
        gtsam::Matrix6 sqrt_cov = solver.operatorSqrt();

        const gtsam::Pose3& pose = poses[i];

        for (int j = 0; j < num_samples_per_pose; ++j) {
            gtsam::Vector6 noise;
            for (int k = 0; k < 6; ++k)
                noise(k) = dist(gen);

            gtsam::Vector6 scaled_noise = sqrt_cov * noise;
            gtsam::Pose3 sample = pose.retract(scaled_noise);
            const gtsam::Point3& p = sample.translation();

            *iter_x = static_cast<float>(p.x());
            *iter_y = static_cast<float>(p.y());
            *iter_z = static_cast<float>(p.z());

            ++iter_x;
            ++iter_y;
            ++iter_z;
        }
    }

    return cloud_msg;
}

visualization_msgs::msg::MarkerArray get_disc_marker_array_msg(
    const std::vector<gtsam::Pose3>& poses,
    const gtsam::TendonDiscConfig& config,
    const std::string& frame_id,
    const rclcpp::Time& stamp)
{
    using visualization_msgs::msg::Marker;
    using geometry_msgs::msg::Point;
    visualization_msgs::msg::MarkerArray array_msg;
    int marker_id = 0;

    // === Draw disc cylinders ===
    for (size_t disc_idx = 0; disc_idx < config.num_discs; ++disc_idx) {
        
        int pose_idx = config.disc_pose_idx[disc_idx];

        const gtsam::Pose3& pose = poses[pose_idx];
        gtsam::Point3 center = pose.translation();
        gtsam::Quaternion q = pose.rotation().toQuaternion();

        Marker disc_marker;
        disc_marker.header.frame_id = frame_id;
        disc_marker.header.stamp = stamp;
        disc_marker.ns = "discs";
        disc_marker.id = marker_id++;
        disc_marker.type = Marker::CYLINDER;
        disc_marker.action = Marker::ADD;
        disc_marker.pose.position.x = center.x();
        disc_marker.pose.position.y = center.y();
        disc_marker.pose.position.z = center.z();
        disc_marker.pose.orientation.x = q.x();
        disc_marker.pose.orientation.y = q.y();
        disc_marker.pose.orientation.z = q.z();
        disc_marker.pose.orientation.w = q.w();
        disc_marker.scale.x = 1.1 * config.routing_radius * 2;
        disc_marker.scale.y = 1.1 * config.routing_radius * 2;
        disc_marker.scale.z = 0.1 * config.routing_radius;  // Disc thickness
        disc_marker.color.r = 0.6;
        disc_marker.color.g = 0.6;
        disc_marker.color.b = 0.7;
        disc_marker.color.a = 0.5;
        disc_marker.lifetime = rclcpp::Duration::from_seconds(0.0);

        // If its the base frame, draw a square instead
        if(disc_idx == 0){
            disc_marker.type = Marker::CUBE;
            disc_marker.scale.x = 4 * config.routing_radius * 2;
            disc_marker.scale.y = 4 * config.routing_radius * 2;
            disc_marker.scale.z = 0.7 * config.routing_radius;
            disc_marker.color.r = 0.6;
            disc_marker.color.g = 0.6;
            disc_marker.color.b = 0.6;
            disc_marker.color.a = 1.0;
        }

        array_msg.markers.push_back(disc_marker);
    }

    // === Draw tendons as line segments between holes ===
    for (int tendon_idx = 0; tendon_idx < config.num_tendons; ++tendon_idx) {
        for (size_t disc_idx = 0; disc_idx + 1 < config.num_discs; ++disc_idx) {
            const gtsam::Pose3& pose_1 = poses[config.disc_pose_idx[disc_idx]];
            const gtsam::Pose3& pose_2 = poses[config.disc_pose_idx[disc_idx + 1]];

            const gtsam::Vector3& local_hole_1 = config.local_hole_locations[disc_idx][tendon_idx];
            const gtsam::Vector3& local_hole_2 = config.local_hole_locations[disc_idx + 1][tendon_idx];

            gtsam::Point3 hole_1 = pose_1.transformFrom(local_hole_1);
            gtsam::Point3 hole_2 = pose_2.transformFrom(local_hole_2);

            // Compute center, direction, and length
            gtsam::Vector3 center = 0.5 * (hole_1 + hole_2);
            gtsam::Vector3 delta = hole_1 - hole_2;
            double length = delta.norm();
            gtsam::Vector3 z_axis = delta.normalized();

            // Orientation: align Z-axis to the direction vector
            Eigen::Vector3d up(0, 0, 1);
            Eigen::Quaterniond q;

            if ((z_axis - up).norm() < 1e-6) {
                q = Eigen::Quaterniond::Identity();
            } else if ((z_axis + up).norm() < 1e-6) {
                q = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
            } else {
                Eigen::Vector3d axis = up.cross(z_axis).normalized();
                double angle = acos(up.dot(z_axis));
                q = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
            }

            Marker cyl;
            cyl.header.frame_id = frame_id;
            cyl.header.stamp = stamp;
            cyl.ns = "tendons";
            cyl.id = marker_id++;
            cyl.type = Marker::CYLINDER;
            cyl.action = Marker::ADD;
            cyl.pose.position.x = center.x();
            cyl.pose.position.y = center.y();
            cyl.pose.position.z = center.z();
            cyl.pose.orientation.x = q.x();
            cyl.pose.orientation.y = q.y();
            cyl.pose.orientation.z = q.z();
            cyl.pose.orientation.w = q.w();
            cyl.scale.x = 0.1 * config.routing_radius;  // radius
            cyl.scale.y = 0.1 * config.routing_radius;
            cyl.scale.z = length; // height
            cyl.color.r = 0.0;
            cyl.color.g = 0.0;
            cyl.color.b = 0.0;
            cyl.color.a = 0.8;
            cyl.lifetime = rclcpp::Duration::from_seconds(0.0);

            if (tendon_idx == 0){
                cyl.color.r = 0.7;
                cyl.color.g = 0.1;
                cyl.color.b = 0.1;
            }

            if (tendon_idx == 1){
                cyl.color.r = 0.0;
                cyl.color.g = 0.5;
                cyl.color.b = 0.1;
            }

            if (tendon_idx == 2){
                cyl.color.r = 0.1;
                cyl.color.g = 0.3;
                cyl.color.b = 0.8;
            }

            if (tendon_idx == 3){
                cyl.color.r = 0.8;
                cyl.color.g = 0.6;
                cyl.color.b = 0.1;
            }
            
            array_msg.markers.push_back(cyl);
        }
    }

    // === Draw rod backbone ===
    for (size_t pose_idx = 0; pose_idx + 1 < poses.size(); ++pose_idx) {
        const gtsam::Pose3& pose_1 = poses[pose_idx];
        const gtsam::Pose3& pose_2 = poses[pose_idx + 1];

        gtsam::Point3 p_1 = pose_1.translation();
        gtsam::Point3 p_2 = pose_2.translation();

        // Compute center, direction, and length
        gtsam::Vector3 center = 0.5 * (p_1 + p_2);
        gtsam::Vector3 delta = p_1 - p_2;
        double length = delta.norm();
        gtsam::Vector3 z_axis = delta.normalized();

        // Orientation: align Z-axis to the direction vector
        Eigen::Vector3d up(0, 0, 1);
        Eigen::Quaterniond q;

        if ((z_axis - up).norm() < 1e-6) {
            q = Eigen::Quaterniond::Identity();
        } else if ((z_axis + up).norm() < 1e-6) {
            q = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
        } else {
            Eigen::Vector3d axis = up.cross(z_axis).normalized();
            double angle = acos(up.dot(z_axis));
            q = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
        }

        Marker cyl;
        cyl.header.frame_id = frame_id;
        cyl.header.stamp = stamp;
        cyl.ns = "backbone";
        cyl.id = marker_id++;
        cyl.type = Marker::CYLINDER;
        cyl.action = Marker::ADD;
        cyl.pose.position.x = center.x();
        cyl.pose.position.y = center.y();
        cyl.pose.position.z = center.z();
        cyl.pose.orientation.x = q.x();
        cyl.pose.orientation.y = q.y();
        cyl.pose.orientation.z = q.z();
        cyl.pose.orientation.w = q.w();
        cyl.scale.x = 0.3 * config.routing_radius;  // radius
        cyl.scale.y = 0.3 * config.routing_radius;
        cyl.scale.z = length; // height
        cyl.color.r = 0.1;
        cyl.color.g = 0.1;
        cyl.color.b = 0.1;
        cyl.color.a = 0.8;
        cyl.lifetime = rclcpp::Duration::from_seconds(0.0);

        array_msg.markers.push_back(cyl);
    }

    return array_msg;
}

geometry_msgs::msg::PoseArray get_pose_array_msg(
    const std::vector<gtsam::Pose3>& poses,
    const std::string& frame_id,
    const rclcpp::Time& stamp)
{
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp = stamp;
    pose_array.header.frame_id = frame_id;

    for (const auto& pose : poses) {
        geometry_msgs::msg::Pose ros_pose;

        // translation
        const gtsam::Point3& t = pose.translation();
        ros_pose.position.x = t.x();
        ros_pose.position.y = t.y();
        ros_pose.position.z = t.z();

        // rotation (quaternion)
        const gtsam::Rot3& R = pose.rotation();
        auto q = R.toQuaternion();  // returns gtsam::Quaternion
        ros_pose.orientation.x = q.x();
        ros_pose.orientation.y = q.y();
        ros_pose.orientation.z = q.z();
        ros_pose.orientation.w = q.w();

        pose_array.poses.push_back(ros_pose);
    }
    return pose_array;
}

class TendonRobotSolver : public rclcpp::Node {
public:
    TendonRobotSolver() : Node("ves_solver") {
        tip_force_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/tendon_robot/tip_force", 10,
            std::bind(&TendonRobotSolver::tip_force_callback, this, std::placeholders::_1));
        
        tensions_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/tendon_robot/tensions", 10,
            std::bind(&TendonRobotSolver::tensions_callback, this, std::placeholders::_1));
        

        backbone_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/tendon_robot/backbone_poses", 10);

        tip_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/tendon_robot/tip_pose", 10);
        
        uncertainty_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/tendon_robot/uncertainty_cloud", 1);
        
        disc_marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/tendon_robot/disc_marker_array", 10);

        solver_ = gtsam::TendonRobotGtsam();

        last_tip_force_ = gtsam::Vector3::Zero();
    }

private:

    void tensions_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        gtsam::Vector4 tensions;

        tensions << msg->position[0],
                    msg->position[1],
                    msg->position[2],
                    msg->position[3];

        gtsam::Vector6 tip_wrench;
        tip_wrench.head<3>() = gtsam::Vector3::Zero();
        tip_wrench.tail<3>() = last_tip_force_;

        gtsam::TendonRobotSolution solution = solver_.update(tip_wrench, tensions);

        RCLCPP_INFO(this->get_logger(), "GTSAM solve time (ms):  %.3f", solution.solve_time_ms);

        // Publish outer tube pose array
        auto pose_array_msg = get_pose_array_msg(solution.backbone_pose_mean, "world", this->now());
        backbone_pose_pub_->publish(pose_array_msg);

        // Publish disc markers 
        auto disc_marker_array_msg = get_disc_marker_array_msg(solution.backbone_pose_mean, solution.tendon_disc_config, "world", this->now());
        disc_marker_array_pub_->publish(disc_marker_array_msg);

        sensor_msgs::msg::PointCloud2 cloud_msg = get_uncertainty_cloud(
            solution.backbone_pose_mean,
            solution.backbone_pose_cov,
            100,
            "world",
            this->now());
        
        uncertainty_cloud_pub_->publish(cloud_msg);

        // publish_uncertainty_cloud(outer_poses, outer_covs, outer_cloud_pub_, this->now());


        // Publish tip pose with covariance
        const auto& tip_pose = solution.backbone_pose_mean.back();
        const auto& tip_cov = solution.backbone_pose_cov.back();

        geometry_msgs::msg::PoseWithCovarianceStamped tip_pose_msg;
        tip_pose_msg.header.stamp = this->now();
        tip_pose_msg.header.frame_id = "world";

        auto t = tip_pose.translation();
        auto q = tip_pose.rotation().toQuaternion();

        tip_pose_msg.pose.pose.position.x = t.x();
        tip_pose_msg.pose.pose.position.y = t.y();
        tip_pose_msg.pose.pose.position.z = t.z();

        tip_pose_msg.pose.pose.orientation.x = q.x();
        tip_pose_msg.pose.pose.orientation.y = q.y();
        tip_pose_msg.pose.pose.orientation.z = q.z();
        tip_pose_msg.pose.pose.orientation.w = q.w();
            
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                tip_pose_msg.pose.covariance[r * 6 + c] = tip_cov(r, c);

        tip_pose_pub_->publish(tip_pose_msg);
    }

    void tip_force_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        last_tip_force_ = gtsam::Vector3(msg->x, msg->y, msg->z);
    }

    gtsam::Vector3 last_tip_force_;

    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr tip_force_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr tensions_sub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr backbone_pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr uncertainty_cloud_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr tip_pose_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr disc_marker_array_pub_;

    gtsam::TendonRobotGtsam solver_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TendonRobotSolver>());
    rclcpp::shutdown();
    return 0;
}
