#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <termios.h>
#include <unistd.h>
#include <iostream>


class TendonTensionPublisher : public rclcpp::Node {
public:
  TendonTensionPublisher()
  : Node("tendon_keyboard_control"), tensions_{0.0, 0.0, 0.0, 0.0}, dt_(0.05) {
    tension_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/tendon_robot/tensions", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(dt_ * 1000)),
      std::bind(&TendonTensionPublisher::publish_loop, this));

    set_terminal_raw_mode();
  }

  ~TendonTensionPublisher() {
    reset_terminal_mode();
  }

private:
  double dt_;
  std::array<double, 4> tensions_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr tension_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  struct termios orig_termios_;

  void publish_loop() {
    char c = read_key();

    // Map keys to tension changes
    switch (c) {
      case 'q': tensions_[0] += 0.01; break;
      case 'a': tensions_[0] -= 0.01; break;
      case 'w': tensions_[1] += 0.01; break;
      case 's': tensions_[1] -= 0.01; break;
      case 'e': tensions_[2] += 0.01; break;
      case 'd': tensions_[2] -= 0.01; break;
      case 'r': tensions_[3] += 0.01; break;
      case 'f': tensions_[3] -= 0.01; break;
      case 'z': rclcpp::shutdown(); return;
      default: break;
    }

    for (auto& t : tensions_) t = std::max(0.0, t);

    // Publish tensions
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = this->now();
    msg.name = {"tension_0", "tension_1", "tension_2", "tension_3"};
    msg.position = {tensions_[0], tensions_[1], tensions_[2], tensions_[3]};
    tension_pub_->publish(msg);

    RCLCPP_INFO(this->get_logger(), "Tensions: [%.3f, %.3f, %.3f, %.3f]",
                tensions_[0], tensions_[1], tensions_[2], tensions_[3]);
  }

  char read_key() {
    char c = 0;
    if (read(STDIN_FILENO, &c, 1) == 1)
      return c;
    return 0;
  }

  void set_terminal_raw_mode() {
    tcgetattr(STDIN_FILENO, &orig_termios_);
    struct termios raw = orig_termios_;
    raw.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
  }

  void reset_terminal_mode() {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios_);
  }
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TendonTensionPublisher>());
  rclcpp::shutdown();
  return 0;
}
