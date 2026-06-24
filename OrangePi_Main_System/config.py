import math

# --- AI MODELS ---
RKNN_MODEL_PATH = 'best0906_int8_2.rknn' 
RL_MODEL_PATH = 'usv_actor_final.onnx' 

# --- VISION PARAMS ---
CAM_W, CAM_H = 640, 480
MODEL_W, MODEL_H = 640, 480
ROI_H = 240
ROI_START_Y = CAM_H - ROI_H  
CONF_THRES = 0.6
IOU_THRES = 0.45

# --- NAVIGATION PARAMS ---
MAX_TARGET_DIST = 2.5  
MEMORY_TIMEOUT = 1.5   
INTERCEPT_DIST = 0.35

# --- GEOFENCE BOUNDS ---
MIN_X, MAX_X = -1.0, 5.0
MIN_Y, MAX_Y = -5.0, 1.0
HOME_ZONE_X_MAX = 0.5
HOME_ZONE_Y_MIN = -0.5

# --- UTILS ---
def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle