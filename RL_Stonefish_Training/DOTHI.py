import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import math

class LivePlotterNode(Node):
    """
    ROS 2 Node to subscribe to USV Odometry and Carrot position.
    Now extracts Yaw to draw an accurate directional arrow.
    """
    def __init__(self):
        super().__init__('usv_live_plotter')
        
        # Subscribe to USV Odometry
        self.sub_odom = self.create_subscription(
            Odometry, 
            '/catamaran_usv/odom', 
            self.odom_callback, 
            qos_profile_sensor_data
        )
        
        # Subscribe to Carrot Position
        self.sub_carrot = self.create_subscription(
            Point, 
            '/carrot_pos', 
            self.carrot_callback, 
            10
        )
        
        # Variables to store current coordinates and heading
        self.boat_x = 0.0
        self.boat_y = 0.0
        self.boat_yaw = 0.0  # MỚI: Thêm biến lưu góc Yaw
        self.carrot_x = 0.0
        self.carrot_y = 0.0
        
        # Store trajectory history
        self.path_x = []
        self.path_y = []
        self.max_path_length = 300

    def odom_callback(self, msg):
        """ Chuyển đổi tọa độ Stonefish (NED) sang chuẩn Bản đồ 2D (ENU) """
        # Trục X của Stonefish là Bắc -> Gán thành trục Y dọc của đồ thị
        # Trục Y của Stonefish là Đông -> Gán thành trục X ngang của đồ thị
        self.boat_x = msg.pose.pose.position.y
        self.boat_y = msg.pose.pose.position.x
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_ned = math.atan2(siny_cosp, cosy_cosp)
        
        # Góc xoay bản đồ = 90 độ - Góc xoay Hàng hải
        self.boat_yaw = (math.pi / 2.0) - yaw_ned
        
        self.path_x.append(self.boat_x)
        self.path_y.append(self.boat_y)
        if len(self.path_x) > self.max_path_length:
            self.path_x.pop(0)
            self.path_y.pop(0)

    def carrot_callback(self, msg):
        """ Nhận tọa độ chuẩn từ mạng AI, KHÔNG LẬT NỮA """
        self.carrot_x = msg.x
        self.carrot_y = msg.y

def main(args=None):
    rclpy.init(args=args)
    node = LivePlotterNode()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # --- MATPLOTLIB SETUP ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("ASV Continuous Carrot-Chasing Live Tracker", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax_limit = 20.0
    ax.set_xlim(-ax_limit, ax_limit)
    ax.set_ylim(-ax_limit, ax_limit)

    # Khởi tạo các thành phần vẽ
    carrot_scatter = ax.scatter([], [], c='orange', s=200, marker='*', label='Target (Carrot)', zorder=3)
    path_line, = ax.plot([], [], 'b--', alpha=0.4, label='Path', zorder=2)
    
    # MỚI: Dùng Quiver (Mũi tên) để thể hiện rõ Tọa độ + Góc xoay của tàu
    boat_arrow = ax.quiver(0, 0, 1, 0, color='blue', scale=25, width=0.015, zorder=4, label='ASV (Position & Heading)')
    
    ax.legend(loc='upper right')

    def update_plot(frame):
        """ Function called periodically by Matplotlib to update the screen. """
        # Cập nhật Vị trí VÀ Góc xoay cho Mũi tên
        u = math.cos(node.boat_yaw)
        v = math.sin(node.boat_yaw)
        boat_arrow.set_offsets([[node.boat_x, node.boat_y]])
        boat_arrow.set_UVC(u, v)
        
        # Cập nhật Cà rốt và Quỹ đạo
        carrot_scatter.set_offsets([[node.carrot_x, node.carrot_y]])
        path_line.set_data(node.path_x, node.path_y)
        
        return boat_arrow, carrot_scatter, path_line

    ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)
    plt.show()

    print("Shutting down live plotter...")
    rclpy.shutdown()
    spin_thread.join()

if __name__ == '__main__':
    main()