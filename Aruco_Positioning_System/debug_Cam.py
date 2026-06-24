import cv2
import numpy as np

def nothing(x):
    pass

# ==========================================
# 1. SETUP TRACKBARS FOR HSV CALIBRATION
# ==========================================
# Create a window to hold the trackbars
cv2.namedWindow("Trackbars")

# Initialize trackbars with wider default values for a darker room
# H (Hue): Color type (Orange is usually 0-25)
# S (Saturation): Color intensity
# V (Value): Brightness
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 25, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# ==========================================
# 2. SETUP ARUCO DETECTOR
# ==========================================
# Try 4x4 first. If it doesn't work, comment it out and uncomment 5x5 or 6x6
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(1)

print("[INFO] Debugger is running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- A. ARUCO DEBUGGING ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Print detected IDs directly on the screen
        cv2.putText(frame, f"Found ArUco ID: {ids.flatten()}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No ArUco detected", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- B. HSV DEBUGGING ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Read the current positions of all trackbars
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    # Create lower and upper bounds based on trackbar values
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    
    # Generate the mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Display the results
    cv2.imshow("Main Camera", frame)
    cv2.imshow("Orange Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()