import numpy as np
import cv2
import glob
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Define the number of inner corners in X and Y directions
# Note: A 9x6 chessboard has 8x5 inner corners.
CHESSBOARD_SIZE = (8, 6) 

# Real-world size of a single square in millimeters (e.g., 65mm = 6.5cm)
SQUARE_SIZE_MM = 19.0  

# Criteria for sub-pixel corner refinement (max 30 iterations or epsilon 0.001)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on real-world dimensions: (0,0,0), (65,0,0), (130,0,0)...
# Z is set to 0 because the chessboard is flat.
obj_p = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
obj_p[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
obj_p = obj_p * SQUARE_SIZE_MM

# Arrays to store object points and image points from all the captured frames
obj_points = [] # 3D points in real world space
img_points = [] # 2D points in image plane

# ==========================================
# 2. CAPTURE IMAGES FOR CALIBRATION
# ==========================================
cap = cv2.VideoCapture(1)
# Set high resolution if supported by your camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("[INFO] Press 'c' to capture a good frame.")
print("[INFO] Press 'q' to stop capturing and start calibration.")

captured_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try to find the chessboard corners in the current frame
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    # Draw a visual copy for the user
    display_frame = frame.copy()
    if ret_corners:
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
        cv2.putText(display_frame, "Ready! Press 'c' to capture", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
         cv2.putText(display_frame, "Searching for chessboard...", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
    cv2.putText(display_frame, f"Captured: {captured_count}/20+", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
    cv2.imshow('Calibration Capture', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # If 'c' is pressed and corners are detected, save the points
    if key == ord('c') and ret_corners:
        obj_points.append(obj_p)
        
        # Refine the corner locations to sub-pixel accuracy before saving
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        img_points.append(corners_refined)
        
        captured_count += 1
        print(f"[INFO] Frame captured! Total: {captured_count}")
        
    # If 'q' is pressed, break the loop and proceed to calibration
    elif key == ord('q'):
        if captured_count < 10:
            print("[WARN] You should capture at least 15-20 images for good calibration.")
        else:
            break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 3. PERFORM CALIBRATION
# ==========================================
if len(obj_points) > 0:
    print("[INFO] Calculating camera matrix and distortion coefficients...")
    
    # Get image dimensions from the last captured grayscale frame
    h, w = gray.shape[::-1]
    
    # Run the calibration algorithm
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    
    print("\n--- Calibration Results ---")
    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)
    
    # Calculate Mean Reprojection Error to evaluate calibration quality
    # An error closer to 0 is better (ideally < 0.5 pixels for precision tasks)
    mean_error = 0
    for i in range(len(obj_points)):
        img_points_projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_projected, cv2.NORM_L2) / len(img_points_projected)
        mean_error += error
        
    print(f"\nTotal Reprojection Error: {mean_error/len(obj_points):.4f} pixels")
    
    # Save the parameters for future use (e.g., in your USV/ROV tracking script)
    np.savez('camera_calib.npz', mtx=camera_matrix, dist=dist_coeffs)
    print("[INFO] Calibration parameters saved to 'camera_calib.npz'")
    
else:
    print("[ERROR] No valid frames captured. Calibration aborted.")