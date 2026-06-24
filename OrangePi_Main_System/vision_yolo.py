import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import queue
import threading
import math
from rknnlite.api import RKNNLite
from config import *

cv2.setNumThreads(0)

class ObjectTracker:
    def __init__(self):
        self.objects = {}; self.next_id = 1; self.max_disappeared = 5 
    def update(self, current_detections):
        new_objects = {}; matched_current_indices = set()
        for obj_id, obj_data in self.objects.items():
            prev_cx, prev_cy = obj_data['history'][-1]['center']
            prev_cls = obj_data['history'][-1]['class_id']
            best_match_idx, min_dist = None, 60
            for i, det in enumerate(current_detections):
                if i in matched_current_indices: continue
                dist = np.sqrt((det['center'][0] - prev_cx)**2 + (det['center'][1] - prev_cy)**2)
                if dist < min_dist and det['class_id'] == prev_cls:
                    min_dist, best_match_idx = dist, i
            if best_match_idx is not None:
                matched_current_indices.add(best_match_idx)
                hist = obj_data['history'].copy()
                hist.append(current_detections[best_match_idx])  
                if len(hist) > 10: hist.pop(0)
                new_objects[obj_id] = {'history': hist, 'disappeared': 0}
            else:
                disappeared = obj_data.get('disappeared', 0) + 1
                if disappeared <= self.max_disappeared:
                    new_objects[obj_id] = {'history': obj_data['history'], 'disappeared': disappeared}
        for i, det in enumerate(current_detections):
            if i not in matched_current_indices:
                new_objects[self.next_id] = {'history': [det], 'disappeared': 0}
                self.next_id += 1
        self.objects = new_objects
        return self.objects

def post_process(outputs):
    if np.max(outputs[0]) > 2.0: box_tensor, cls_tensor = outputs[0], outputs[1]
    else: box_tensor, cls_tensor = outputs[1], outputs[0]

    box_data, cls_data = np.squeeze(box_tensor).transpose(), np.squeeze(cls_tensor).transpose()
    scale_w, scale_h = CAM_W / float(MODEL_W), CAM_H / float(MODEL_H)
    max_scores, max_class_ids = np.max(cls_data, axis=1), np.argmax(cls_data, axis=1)

    valid_indices = np.where(max_scores > CONF_THRES)[0]
    boxes, scores, class_ids = [], [], []

    for i in valid_indices:
        cx, cy, w, h = box_data[i]
        x1, y1 = int((cx - w / 2.0) * scale_w), int((cy - h / 2.0) * scale_h)
        box_w, box_h = int(w * scale_w), int(h * scale_h)
        if box_w > 0 and box_h > 0:
            boxes.append([x1, y1, box_w, box_h]); scores.append(float(max_scores[i])); class_ids.append(int(max_class_ids[i]))

    results = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, IOU_THRES)
        if len(indices) > 0:
            for i in indices.flatten():
                x_top, y_top, w_box, h_box = boxes[i]
                x2, y2 = min(CAM_W - 1, x_top + w_box), min(CAM_H - 1, y_top + h_box)
                x_top, y_top = max(0, x_top), max(0, y_top)
                bx, by = (x_top + x2) / 2.0, (y_top + y2) / 2.0
                if by < ROI_START_Y: continue 
                target_u, target_v = int(bx), int((by + (y2 - y_top)/2.0) - (0.75 * (y2 - y_top)))
                results.append({'box': [int(x_top), int(y_top), int(x2), int(y2)],
                                'center': (int(bx), int(by)), 'target_pixel': (target_u, target_v),
                                'class_id': class_ids[i], 'score': scores[i]})
    return results

def internal_camera_thread(pipeline, align, raw_q, stop_event):
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            aligned = align.process(frames)
            c_frame, d_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not c_frame or not d_frame: continue
            bgr = np.require(np.asanyarray(c_frame.get_data()).copy(), requirements=['C'])
            d_array = np.require(np.asanyarray(d_frame.get_data()).copy(), requirements=['C'])
            if raw_q.full():
                try: raw_q.get_nowait()
                except queue.Empty: pass
            raw_q.put((bgr, d_array))
        except Exception: pass
def internal_video_thread(video_q, stop_event):
    current_time_str = time.strftime("%Y%m%d_%H%M%S")
    out_video = cv2.VideoWriter(f"USV_Debug_{current_time_str}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (CAM_W, CAM_H))
    
    # [ĐÃ SỬA]: Ép luồng này chạy cho đến khi hàng đợi KHÔNG CÒN ẢNH NÀO
    while not stop_event.is_set() or not video_q.empty():
        try:
            img_to_write = video_q.get(timeout=0.5)
            out_video.write(img_to_write)
        except queue.Empty: 
            continue
        except Exception: 
            pass
            
    out_video.release()
    print("--> [VISION] Video MP4 saved and closed safely.")

def vision_process_worker(out_q, stop_event):
    try: os.sched_setaffinity(0, {5, 6, 7})
    except: pass
    
    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL_PATH) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("--> [VISION] Failed to init RKNN!")
        return

    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
    config_rs.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, 30)
    profile = pipeline.start(config_rs)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    tracker = ObjectTracker()
    
    # [ĐÃ SỬA]: Chỉ khởi tạo Queue đúng 1 lần duy nhất
    raw_q, video_q = queue.Queue(maxsize=2), queue.Queue(maxsize=30)
    
    cam_thread = threading.Thread(target=internal_camera_thread, args=(pipeline, align, raw_q, stop_event), daemon=True)
    # [ĐÃ SỬA]: Bỏ daemon=True để luồng video không bị hệ điều hành "bóp cổ" đột ngột
    vid_thread = threading.Thread(target=internal_video_thread, args=(video_q, stop_event), daemon=False)
    
    cam_thread.start()
    vid_thread.start()
    
    print("--> [VISION] Hybrid Thread-Process Active!")
    
    # Đã XÓA dòng khởi tạo Queue thừa thãi ở đây
    
    while not stop_event.is_set():
        try:
            bgr, d_array = raw_q.get(timeout=1.0)
            if not video_q.full(): 
                video_q.put_nowait(bgr.copy())
            
            safe_rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            safe_rgb[0:ROI_START_Y, 0:CAM_W] = 0
            outputs = rknn.inference(inputs=[np.expand_dims(safe_rgb, axis=0)])
            dets = post_process(outputs) if outputs else []
            tracked_objects = tracker.update(dets)
            
            frame_targets = []
            cv2.line(bgr, (0, ROI_START_Y), (CAM_W, ROI_START_Y), (0, 0, 255), 2)
            
            for obj_id, data in tracked_objects.items():
                curr = data['history'][-1]
                tu, tv = curr['target_pixel']
                u, v = max(0, min(tu, CAM_W-1)), max(0, min(tv, CAM_H-1))
                valid_depths = d_array[max(0, v-2):min(CAM_H, v+3), max(0, u-2):min(CAM_W, u+3)]
                valid_depths = valid_depths[valid_depths > 0]
                z_m = np.median(valid_depths) * depth_scale if len(valid_depths) > 0 else 0.0
                
                if z_m > 0:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z_m)
                    distance = math.hypot(X, Z)
                    relative_yaw = -math.atan2(X, Z)
                    
                    is_swallowed, is_exiting = False, False
                    if len(data['history']) >= 3:
                        u_prev, v_prev = data['history'][-1 - min(3, len(data['history'])-1)]['target_pixel']
                        du, dv = u - u_prev, v - v_prev
                        if dv > 8 and v > (CAM_H - 120): is_swallowed = True
                        if (u < 200 and du < -2) or (u > CAM_W - 200 and du > 2) or (v < ROI_START_Y + 50 and dv < -2): is_exiting = True
                    if distance <= INTERCEPT_DIST: is_swallowed = True
                    
                    frame_targets.append({'id': obj_id, 'class_id': curr['class_id'], 'dist': distance, 'yaw': relative_yaw, 'is_swallowed': is_swallowed, 'is_exiting': is_exiting})
                    color = (0, 0, 255) if curr['class_id'] == 0 else (0, 255, 0)
                    cv2.rectangle(bgr, (curr['box'][0], curr['box'][1]), (curr['box'][2], curr['box'][3]), color, 2)
            
            if out_q.full():
                try: out_q.get_nowait()
                except queue.Empty: pass
            out_q.put_nowait((bgr, frame_targets))

        except queue.Empty: 
            pass
        except KeyboardInterrupt:
            # [ĐÃ SỬA]: Bắt lỗi Ctrl+C để nhảy xuống vòng giải phóng tài nguyên
            print("\n--> [VISION] Ctrl+C caught! Exiting loop safely...")
            break
        except Exception: 
            pass

    # ==========================================
    # QUY TRÌNH DỌN DẸP KHÔNG THỂ BỊ BỎ QUA
    # ==========================================
    print("--> [VISION] Flushing remaining video frames...")
    stop_event.set() # Ép các thread con dừng lại
    
    # [QUAN TRỌNG]: Chờ luồng video ghi nốt các frame đang kẹt trong hàng đợi
    if vid_thread.is_alive():
        vid_thread.join(timeout=5.0) 
        
    rknn.release()
    pipeline.stop()
    print("--> [VISION] Terminated cleanly.")