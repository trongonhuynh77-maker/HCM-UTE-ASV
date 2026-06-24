import rclpy
from rclpy.node import Node
import time
import math
import numpy as np
import os

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_SAVE_DIR = "./sac_usv_models3/"
MODEL_NAME = "sac_usv_carrot_chasing_fast"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
TENSORBOARD_LOG_DIR = "./sac_usv_tensorboard3/"
TOTAL_STEPS = 500_000

class PrintRemainingStepsCallback(BaseCallback):
    """ Custom callback for printing training progress cleanly to the terminal. """
    def __init__(self, total_timesteps, print_freq=100, verbose=0):
        super(PrintRemainingStepsCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            remaining = self.total_timesteps - self.num_timesteps
            percent = (self.num_timesteps / self.total_timesteps) * 100
            print(f"> [Progress] Steps: {self.num_timesteps}/{self.total_timesteps} ({percent:.2f}%) | Remaining: {remaining}")
        return True

class USVCarrotEnv(gym.Env, Node):
    """
    Custom Gymnasium Environment for USV Catamaran Continuous Carrot-Chasing.
    Single-threaded architecture utilizing spin_once for maximum FPS synchronization.
    """
    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, 'usv_rl_training_env')
        
        # Action: [Speed (-1 to 1), Yaw Offset (-1 to 1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Obs: [Distance, Heading Error, Yaw Rate]
        max_f = np.inf
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, -max_f], dtype=np.float32),
            high=np.array([max_f, np.pi, max_f], dtype=np.float32),
            dtype=np.float32
        )
        
        # --- PUBLISHERS ---
        self.pub_base_rpm = self.create_publisher(Float32, '/cmd_rpm', 10)
        self.pub_target_yaw = self.create_publisher(Float32, '/cmd_target_yaw', 10)
        self.pub_carrot = self.create_publisher(Point, '/carrot_pos', 10) 
        
        # --- SUBSCRIBERS ---
        self.sub_imu = self.create_subscription(Imu, '/catamaran_usv/imu', self._imu_callback, qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(Odometry, '/catamaran_usv/odom', self._odom_callback, qos_profile_sensor_data)
        
        # --- STATE VARIABLES ---
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.yaw_rate = 0.0
        self.carrot_x = 0.0
        self.carrot_y = 0.0
        self.prev_distance = 0.0
        self.step_counter = 0 
        
        # --- COMMAND TRACKING FOR SLEW RATE LIMITER & PENALTY ---
        self.actual_rpm_cmd = 0.0
        self.actual_yaw_offset = 0.0
        
        # --- SYNC FLAG & 30HZ TIMING CONTROL ---
        self.new_odom_received = False
        self.target_dt_ns = (1.0 / 30.0) * 1e9  # Exactly 33.33ms in nanoseconds for 30Hz loop
        self.last_step_time_ns = 0
        
        # Wait for initial data to ensure ROS is communicating
        print("[SYSTEM] Waiting for initial ROS 2 sensor data...")
        while self.current_x == 0.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.last_step_time_ns = self.get_clock().now().nanoseconds
        print("[SYSTEM] Sensor streams active! Locked to single-thread 30Hz.")

    def _imu_callback(self, msg):
        self.yaw_rate = -msg.angular_velocity.z

    def _odom_callback(self, msg):
        # Apply ENU conversion explicitly
        self.current_x = msg.pose.pose.position.y
        self.current_y = msg.pose.pose.position.x
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_ned = math.atan2(siny_cosp, cosy_cosp)
        
        yaw_enu = (math.pi / 2.0) - yaw_ned
        self.current_yaw = math.atan2(math.sin(yaw_enu), math.cos(yaw_enu))
        
        # Trigger the sync flag 
        self.new_odom_received = True

    def spawn_new_carrot(self):
        # Spawns targets dynamically between 1.0m and 3.5m
        distance = np.random.uniform(1.0, 3.5)
        # Limits spawn angle to a forward cone of +/- 35 degrees
        angle_offset = np.random.uniform(math.radians(-35), math.radians(35))
        
        target_angle = self.current_yaw + angle_offset
        
        self.carrot_x = self.current_x + distance * math.cos(target_angle)
        self.carrot_y = self.current_y + distance * math.sin(target_angle)
        
        self.prev_distance = self._calculate_distance()
        
        carrot_msg = Point()
        carrot_msg.x = float(self.carrot_x)
        carrot_msg.y = float(self.carrot_y)
        carrot_msg.z = 0.0
        self.pub_carrot.publish(carrot_msg)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.spawn_new_carrot()
        
        # Reset actuation states for hardware safety
        self.actual_rpm_cmd = 0.0
        self.actual_yaw_offset = 0.0
        
        # Force a synchronous wait to ensure physics are updated
        self.new_odom_received = False
        while not self.new_odom_received:
            rclpy.spin_once(self, timeout_sec=0.01)
            
        self.last_step_time_ns = self.get_clock().now().nanoseconds
        return self._get_obs(), {}

    def step(self, action):
        # 1. Map raw AI action to target physical values
        target_rpm = (action[0] + 1.0) / 2.0 * 300.0
        target_yaw_offset = action[1] * 1.57
        
        # 2. Calculate requested delta for Jerk Penalty
        delta_rpm = target_rpm - self.actual_rpm_cmd
        delta_yaw = target_yaw_offset - self.actual_yaw_offset
        
        # ========================================================
        # HARDWARE SLEW RATE LIMITER (Protects Motors)
        # Limits RPM change to max 15 per step. Yaw to 0.05 rad per step.
        # ========================================================
        self.actual_rpm_cmd += np.clip(delta_rpm, -15.0, 15.0)
        self.actual_yaw_offset += np.clip(delta_yaw, -0.05, 0.05)
        
        # Calculate final yaw based on limited offset
        target_yaw = self.current_yaw + self.actual_yaw_offset
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
        
        # Publish commands
        msg_rpm, msg_yaw = Float32(), Float32()
        msg_rpm.data, msg_yaw.data = float(self.actual_rpm_cmd), float(target_yaw)
        self.pub_base_rpm.publish(msg_rpm)
        self.pub_target_yaw.publish(msg_yaw)

        # Continually publish carrot position for visualization
        carrot_msg = Point()
        carrot_msg.x = float(self.carrot_x)
        carrot_msg.y = float(self.carrot_y)
        carrot_msg.z = 0.0
        self.pub_carrot.publish(carrot_msg)

        # ===============================================================
        # STRICT 30Hz SYNCHRONIZATION LOCK (ROS Time Aware)
        # Wait until exactly 1/30s of simulation time has passed since last step.
        # This prevents running at 50Hz just because Odometry publishes at 50Hz.
        # ===============================================================
        self.new_odom_received = False
        while True:
            rclpy.spin_once(self, timeout_sec=0.005)
            current_time_ns = self.get_clock().now().nanoseconds
            
            if (current_time_ns - self.last_step_time_ns) >= self.target_dt_ns:
                # Ensure we also have a fresh odom reading for state accuracy
                if self.new_odom_received:
                    self.last_step_time_ns = current_time_ns
                    break
        
        # Proceed with state observation
        current_distance = self._calculate_distance()
        heading_error = self._calculate_heading_error()
        obs = self._get_obs()
        
        heading_deg = math.degrees(abs(heading_error))
        
        # --- REWARD SHAPING ---
        r_time = -2.0
        r_progress = 110.0 * (self.prev_distance - current_distance)
        r_spin = -min(5.0 * (self.yaw_rate ** 2), 100.0)
        
        # ========================================================
        # ACTION SMOOTHNESS PENALTY (Jerk Penalty)
        # Punishes AI for requesting erratic, impossible commands
        # ========================================================
        r_jerk = -0.02 * abs(delta_rpm) - 1.0 * abs(delta_yaw)
        
        # ========================================================
        # PIECEWISE HEADING REWARD (Magnet Effect)
        # ========================================================
        if heading_deg <= 5.0:
            r_heading = 3.0 * (1.0 - (heading_deg / 5.0))
        elif heading_deg <= 35.0:
            r_heading = -10.0 * ((heading_deg - 5.0) / 30.0)
        else:
            r_heading = -10.0
            
        # Combine all rewards
        step_reward = r_time + r_progress + r_heading + r_spin + r_jerk
        
        # --- TERMINAL CONDITIONS ---
        terminated, truncated = False, False
        updated_carrot = False
        
        if current_distance < 0.2:
            step_reward += 2000.0
            self.spawn_new_carrot()
            updated_carrot = True
            print('\n[SUCCESS] Target Acquired! Spawning new carrot.')
            
        elif current_distance > 5.0:
            step_reward -= 1000.0
            terminated = True
            print('\n[FAIL] Boundary exceeded (> 5.0m)! Reset.')
            
        elif heading_deg > 35.0:
            step_reward -= 1000.0 
            terminated = True
            print(f'\n[PENALTY] Target slipped out of cone (Error: {heading_deg:.1f} deg). Reset.')
            
        if not updated_carrot:
            self.prev_distance = current_distance
            
        self.step_counter += 1
        # Print debug info approximately every 1 second (assumes ~30Hz execution)
        if self.step_counter % 30 == 0: 
            print(f"  -> Debug | Dist: {current_distance:.2f}m | Error: {heading_deg:.1f} deg | RPM: {self.actual_rpm_cmd:.0f} | R_Jerk: {r_jerk:.1f}")
            
        info = {"r_progress": r_progress, "r_heading": r_heading, "r_spin": r_spin, "r_jerk": r_jerk}
        
        return obs, step_reward, terminated, truncated, info

    def _get_obs(self):
        distance = self._calculate_distance()
        heading_error = self._calculate_heading_error()
        return np.array([distance, heading_error, self.yaw_rate], dtype=np.float32)

    def _calculate_distance(self):
        return math.hypot(self.carrot_x - self.current_x, self.carrot_y - self.current_y)

    def _calculate_heading_error(self):
        angle_to_target = math.atan2(self.carrot_y - self.current_y, self.carrot_x - self.current_x)
        error = angle_to_target - self.current_yaw
        return math.atan2(math.sin(error), math.cos(error))

# ==========================================
# MAIN EXECUTION
# ==========================================
def main(args=None):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    
    rclpy.init(args=args)
    env = USVCarrotEnv()
    
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"[INFO] Found existing model at {MODEL_PATH}. Loading weights to resume...")
        model = SAC.load(
            MODEL_PATH, 
            env=env, 
            tensorboard_log=TENSORBOARD_LOG_DIR,
            custom_objects={'tensorboard_log': TENSORBOARD_LOG_DIR}
        )
    else:
        print("[INFO] No existing model found. Starting fresh SAC Training...")
        model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, 
            buffer_size=100000, 
            tensorboard_log=TENSORBOARD_LOG_DIR,
            train_freq=8,         
            gradient_steps=8       
        )
        
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=MODEL_SAVE_DIR, name_prefix="rl_model_backup")
    progress_callback = PrintRemainingStepsCallback(total_timesteps=TOTAL_STEPS, print_freq=100)
    
    try:
        print(f"[INFO] Training initialized for {TOTAL_STEPS} steps. Execution synchronized strictly to 30Hz.")
        model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, progress_callback], reset_num_timesteps=False)
        model.save(MODEL_PATH)
        print("\n[SUCCESS] Training completed! Final model saved.")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user! Safely saving model weights...")
        model.save(MODEL_PATH)
        print(f"[SUCCESS] Emergency save complete: {MODEL_PATH}.")
        
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()