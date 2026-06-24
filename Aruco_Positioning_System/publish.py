import cv2
import numpy as np
import socket
import struct
import time
import csv
from datetime import datetime

# --- UDP Setup ---
UDP_IP = "192.168.0.102" 
UDP_PORT = 5005
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- ArUco & Tracking Params ---
ORIGIN_MARKER_ID = 42  
REAL_MARKER_SIZE_CM = 28.5 

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshConstant = 10
aruco_params.minMarkerPerimeterRate = 0.1
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# HSV boundaries for tracking the orange boat
LOWER_ORANGE = np.array([0, 193, 54])
UPPER_ORANGE = np.array([179, 255, 255])

# Load Camera Calibration
try:
    with np.load('camera_calib.npz') as X:
        camera_matrix = X['mtx']
        dist_coeffs = X['dist']
except FileNotFoundError:
    print("[ERROR] 'camera_calib.npz' not found! Run calibration first.")
    exit()

# 3D points of the ArUco marker
half_size = REAL_MARKER_SIZE_CM / 2.0
marker_3d_points = np.array([
    [-half_size,  half_size, 0], [ half_size,  half_size, 0], 
    [ half_size, -half_size, 0], [-half_size, -half_size, 0]  
], dtype=np.float32)

def pixel_to_3d_plane(u, v, rvec, tvec, mtx, dist):
    """
    Casts a ray from the camera lens through pixel (u,v) down to the 
    3D floor plane defined by the origin ArUco marker.
    """
    pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
    pt_undist = cv2.undistortPoints(pt, mtx, dist, P=mtx)
    ux, uy = pt_undist[0][0]
    
    ray_dir = np.linalg.inv(mtx).dot([ux, uy, 1.0])
    R, _ = cv2.Rodrigues(rvec)
    normal_vector = R[:, 2] 
    point_on_plane = tvec.flatten() 
    
    numerator = np.dot(normal_vector, point_on_plane)
    denominator = np.dot(normal_vector, ray_dir)
    
    if denominator == 0: 
        return None 
        
    s = numerator / denominator
    P_cam = s * ray_dir 
    P_orig = R.T.dot(P_cam - point_on_plane)
    return P_orig

# --- Camera Setup ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0 

# --- Logging Files Setup (Video & CSV) ---
# Use the same timestamp for both video and CSV filenames
current_time = datetime.now().strftime("%Hh%Mm_%d-%m-%Y")

# Video setup
video_filename = f"tracking_record_{current_time}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# CSV/Excel setup matching video filename format
csv_filename = f"tracking_data_{current_time}.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write header with only X and Y coordinates
csv_writer.writerow(["X_cm", "Y_cm"])

print(f"[INFO] Server running: Optimized ArUco (5x5) + HSV Boat Tracking...")
print(f"[INFO] Streaming UDP coordinates to {UDP_IP}:{UDP_PORT}")
print(f"[INFO] Recording clean video to: {video_filename}")
print(f"[INFO] Saving coordinates to Excel-compatible file: {csv_filename}")

while True:
    ret, frame = cap.read()
    if not ret: break

    # IMPORTANT: Save the clean frame BEFORE drawing any overlays
    out.write(frame)

    rvec_O, tvec_O = None, None
    origin_found = False
    boat_u, boat_v = None, None
    largest_contour = None

    # HSV Color Tracking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] > 0:
            boat_u = int(M["m10"] / M["m00"])
            boat_v = int(M["m01"] / M["m00"])

    # ArUco Marker Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        ids_flat = ids.flatten()
        if ORIGIN_MARKER_ID in ids_flat:
            idx_origin = np.where(ids_flat == ORIGIN_MARKER_ID)[0][0]
            
            ret_O, rvec_O, tvec_O = cv2.solvePnP(
                marker_3d_points, corners[idx_origin][0], camera_matrix, dist_coeffs
            )
            if ret_O:
                origin_found = True

    # 3D Position Calculation & Logging
    if origin_found and boat_u is not None and boat_v is not None:
        boat_3d_pos = pixel_to_3d_plane(boat_u, boat_v, rvec_O, tvec_O, camera_matrix, dist_coeffs)
        
        if boat_3d_pos is not None:
            x_cm = round(boat_3d_pos[0], 2)
            y_cm = round(boat_3d_pos[1], 2)
            
            # Send data over UDP
            message = struct.pack('ff', float(x_cm), float(y_cm))
            udp_socket.sendto(message, (UDP_IP, UDP_PORT))
            
            # Log only X and Y to CSV file
            csv_writer.writerow([x_cm, y_cm])

    # --- Draw Overlays on Frame for Display Only ---
    if boat_u is not None and boat_v is not None:
        cv2.circle(frame, (boat_u, boat_v), 10, (0, 255, 255), -1)
        cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)
        
        if origin_found and boat_3d_pos is not None:
            info_text = f"Boat -> X: {x_cm}cm, Y: {y_cm}cm"
            cv2.putText(frame, info_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    if origin_found:
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_O, tvec_O, 15.0)
    if rejected is not None:
        cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))

    # Show Windows
    display_frame = cv2.resize(frame, (960, 540))
    display_mask = cv2.resize(mask, (480, 270))
    cv2.imshow("Indoor GPS (HSV + ArUco)", display_frame)
    cv2.imshow("Orange Mask", display_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Cleaning up resources...")
cap.release()
out.release() 
csv_file.close() 
cv2.destroyAllWindows()
udp_socket.close()