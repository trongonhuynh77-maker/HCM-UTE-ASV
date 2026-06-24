import numpy as np
import onnxruntime as ort
import math
from config import RL_MODEL_PATH

class RLAgent:
    def __init__(self):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1 
        sess_options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(RL_MODEL_PATH, providers=['CPUExecutionProvider'], sess_options=sess_options)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print("--> [RL] ONNX Hybrid Model Loaded (5 States -> 1 Action).")
        except Exception as e:
            print(f"--> [RL ERROR] Cannot load model: {e}")
            self.session = None

    def predict(self, closest_dist, best_target_yaw_err, ros2_yaw_rate, ros2_yaw):
        # [ENGLISH COMMENT]: Safe fallback if ONNX model fails to load
        if self.session is None:
            return 40.0

        yaw_rad = math.radians(ros2_yaw)
        
        # [ENGLISH COMMENT]: 5-state observation matching your working monolithic code
        obs_array = np.array([[
            closest_dist, 
            best_target_yaw_err, 
            math.radians(ros2_yaw_rate), 
            math.sin(yaw_rad), 
            math.cos(yaw_rad)
        ]], dtype=np.float32)
        
        action_output = self.session.run([self.output_name], {self.input_name: obs_array})[0]
        
        # [ENGLISH COMMENT]: Map single action output to RPM scale
        raw_rpm = ((action_output[0][0] + 1.0) / 2.0) * 300.0 
        raw_rpm = max(40.0, min(300.0, raw_rpm))
        pwm_offset = 40.0 + ((raw_rpm - 40.0) / (300.0 - 40.0)) * (150.0 - 40.0) 
        
        return pwm_offset, raw_rpm