import cv2
import cv2.aruco as aruco

def main():
    cap = cv2.VideoCapture(0)
    
    # Thử đổi sang chuẩn 5x5 vì ảnh của bạn trông rất giống 5x5
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    parameters = aruco.DetectorParameters()
    
    # Tối ưu cho ảnh bị lóa và mờ
    parameters.adaptiveThreshConstant = 10
    parameters.minMarkerPerimeterRate = 0.03

    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Thử lật ngược màu nếu cần (tùy chọn)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                c = corners[i][0]
                cX = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                cY = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"Success ID:{ids[i][0]}", (cX, cY-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ các vùng bị từ chối màu đỏ (để bạn biết nó có đang 'thấy' nhưng không 'đọc' được không)
        if rejected is not None:
            aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))

        cv2.imshow('ARUCO DEBUG (Do: Rejected, Xanh: OK)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()