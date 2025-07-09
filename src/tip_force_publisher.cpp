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

    // Smooth changing direction components
    double a = 0.10 * time_;
    double b = 0.11 * time_;
    double c = 0.12 * time_;

    double x = std::sin(a);
    double y = std::sin(b);
    double z = std::sin(c);

    // Normalize the direction vector
    double norm = std::sqrt(x * x + y * y + z * z);
    if (norm == 0.0) norm = 1.0;  // Avoid divide-by-zero

    x /= norm;
    y /= norm;
    z /= norm;

    // Oscillating magnitude between 0 and 0.1
    double force_amp = 0.1;
    double magnitude = force_amp * std::pow(0.5 * (1.0 + std::sin(2.0 * time_)), 5);

    // Final force vector
    geometry_msgs::msg::Vector3 force_msg;
    force_msg.x = magnitude * x;
    force_msg.y = magnitude * y;
    force_msg.z = magnitude * z;

    force_pub_->publish(force_msg);

    RCLCPP_INFO(this->get_logger(),
        "Force (mag=%.2f): [%.2f, %.2f, %.2f]",
        magnitude, force_msg.x, force_msg.y, force_msg.z);
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