import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64
from nav_msgs.msg import Odometry
import math
from rclpy.qos import qos_profile_sensor_data

class STM32MockNode(Node):
    def __init__(self):
        super().__init__('stm32_mock_node')

        self.sub_base_rpm = self.create_subscription(Float32, '/cmd_rpm', self.base_rpm_callback, 10)
        self.sub_target_yaw = self.create_subscription(Float32, '/cmd_target_yaw', self.target_yaw_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/catamaran_usv/odom', self.odom_callback, qos_profile_sensor_data)

        self.pub_left_rpm = self.create_publisher(Float64, '/catamaran_usv/thruster_port/setpoint', 10)
        self.pub_right_rpm = self.create_publisher(Float64, '/catamaran_usv/thruster_stbd/setpoint', 10)

        self.base_rpm = 0.0
        self.target_yaw = 0.0
        self.current_yaw = 0.0
        
        # Căn chỉnh lại PID mượt mà hơn cho USV
        self.kp = 5.0    
        self.ki = 0.85
        self.kd = 1.2    
        
        self.integral_error = 0.0
        self.prev_yaw_deg = None # Đổi từ prev_error sang đo lường Yaw thực tế
        
        self.prev_sim_time = 0.0
        self.max_rpm = 1000.0
        self.min_rpm = -1000.0

    def base_rpm_callback(self, msg):
        self.base_rpm = msg.data

    def target_yaw_callback(self, msg):
        self.target_yaw = msg.data

    def odom_callback(self, msg):
        current_sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.prev_sim_time == 0.0:
            self.prev_sim_time = current_sim_time
            return
            
        dt = current_sim_time - self.prev_sim_time
        self.prev_sim_time = current_sim_time

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_ned = math.atan2(siny_cosp, cosy_cosp)
        yaw_enu = (math.pi / 2.0) - yaw_ned
        self.current_yaw = math.atan2(math.sin(yaw_enu), math.cos(yaw_enu))

        if 0.0 < dt < 0.5:
            self.pid_control_loop(dt)

    def pid_control_loop(self, dt):
        error_rad = self.target_yaw - self.current_yaw
        error_rad = math.atan2(math.sin(error_rad), math.cos(error_rad)) 
        error_deg = math.degrees(error_rad) 
        current_yaw_deg = math.degrees(self.current_yaw)

        # Khởi tạo prev_yaw nếu chạy lần đầu
        if self.prev_yaw_deg is None:
            self.prev_yaw_deg = current_yaw_deg

        p_term = self.kp * error_deg
        
        self.integral_error += error_deg * dt
        self.integral_error = max(min(self.integral_error, 50.0), -50.0)
        i_term = self.ki * self.integral_error
        
        # BÍ QUYẾT TRỊ SỐC ĐẠO HÀM (Derivative on Measurement)
        # Chỉ trừ hao khi TÀU xoay, bỏ qua sự giật lag của Target do AI xuất lệnh
        d_term = -self.kd * (current_yaw_deg - self.prev_yaw_deg) / dt

        pid_output = p_term + i_term + d_term

        left_rpm_cmd = self.base_rpm + pid_output
        right_rpm_cmd = self.base_rpm - pid_output

        left_rpm_cmd = max(min(left_rpm_cmd, self.max_rpm), self.min_rpm)
        right_rpm_cmd = max(min(right_rpm_cmd, self.max_rpm), self.min_rpm)

        msg_left = Float64()
        msg_left.data = float(left_rpm_cmd)
        self.pub_left_rpm.publish(msg_left)

        msg_right = Float64()
        msg_right.data = float(right_rpm_cmd)
        self.pub_right_rpm.publish(msg_right)

        # Lưu lại góc hiện tại cho vòng lặp sau
        self.prev_yaw_deg = current_yaw_deg

def main(args=None):
    rclpy.init(args=args)
    stm32_node = STM32MockNode()
    try:
        rclpy.spin(stm32_node)
    except KeyboardInterrupt:
        pass
    finally:
        stm32_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()