#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "tendon_robot_gtsam.h"

void publish_uncertainty_cloud(
    const std::vector<gtsam::Pose3>& poses,
    const std::vector<gtsam::Matrix6>& covariances,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& cloud_pub,
    const rclcpp::Time& now)
{
    using sensor_msgs::msg::PointCloud2;
    using sensor_msgs::PointCloud2Modifier;
    using sensor_msgs::PointCloud2Iterator;

    const int num_samples_per_pose = 25;
    const std::string frame_id = "world";

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

    cloud_pub->publish(cloud_msg);
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

        // for(int i = 0; i < solution.backbone_pose_mean.size(); i++)
        //     std::cout << solution.backbone_pose_mean[i] << std::endl;

        // Publish outer tube pose array and uncertainty cloud
        auto pose_array_msg = get_pose_array_msg(solution.backbone_pose_mean, "world", this->now());
        backbone_pose_pub_->publish(pose_array_msg);
        
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

    gtsam::TendonRobotGtsam solver_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TendonRobotSolver>());
    rclcpp::shutdown();
    return 0;
}
