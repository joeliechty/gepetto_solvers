#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <cmath>

class TipForcePublisher : public rclcpp::Node {
public:
  TipForcePublisher() : Node("tip_force_publisher") {
    dt_ = 0.05;

    force_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("/tendon_robot/tip_force", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(dt_ * 1000)),
      std::bind(&TipForcePublisher::publish, this)
    );
  }

private:
  void publish() {
    time_ += dt_;

    // Tip force message
    geometry_msgs::msg::Vector3 force_msg;
    double force_amp = 0.1;

    force_msg.x = force_amp * std::sin(0.10 * time_);
    force_msg.y = force_amp * std::sin(0.11 * time_);
    force_msg.z = force_amp * std::sin(0.12 * time_);
    force_pub_->publish(force_msg);

    RCLCPP_INFO(this->get_logger(),
      "Force: [%.2f, %.2f, %.2f]",
      force_msg.x, force_msg.y, force_msg.z);
  }

  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr force_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  double dt_;
  double time_;
  double force_amp_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TipForcePublisher>());
  rclcpp::shutdown();
  return 0;
}