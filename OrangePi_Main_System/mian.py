import os
import time
import math
import multiprocessing as mp
import queue
import cv2

# Import Modules
from config import *
from web_server import web_server_process
from telemetry import udp_worker_process
from hardware_io import serial_worker_process
from vision_yolo import vision_process_worker
from rl_inference import RLAgent

def set_cpu_affinity(cores):
    try: os.sched_setaffinity(0, set(cores))
    except Exception: pass

def main():
    set_cpu_affinity([4])
    
    manager = mp.Manager()
    shared_dict = manager.dict({
        "boat_x": 0.0, "boat_y": 0.0,
        "usv_yaw": 0.0, "usv_yaw_rate": 0.0, "boat_yaw": 0.0,
        "target_yaw": 0.0,
        "state": "IDLE", "is_running": False,
        "active_trash": [], "web_cmd": ""
    })

    out_q = mp.Queue(maxsize=2)
    cmd_queue = mp.Queue(maxsize=10)
    stop_event = mp.Event()

    # Launch Microservices
    p_web = mp.Process(target=web_server_process, args=(shared_dict,))
    p_udp = mp.Process(target=udp_worker_process, args=(shared_dict, stop_event))
    p_serial = mp.Process(target=serial_worker_process, args=(shared_dict, cmd_queue, stop_event))
    p_vision = mp.Process(target=vision_process_worker, args=(out_q, stop_event))

    p_web.start(); p_udp.start(); p_serial.start(); p_vision.start()

    # Load RL Brain
    rl_agent = RLAgent()
    
    print("--> [SYSTEM] Main Orchestrator Active on Core 4.")

    is_running, is_finding = False, False
    usv_state = "SEARCHING" 
    mem_global_yaw, mem_distance, mem_timestamp = 0.0, 0.0, 0.0
    accumulated_yaw, prev_yaw_for_search = 0.0, 0.0
    last_cmd_str, last_cmd_time = "", 0.0
    
    try:
        while not stop_event.is_set():
            # --- 1. CHECK WEB COMMANDS ---
            web_cmd = shared_dict.get('web_cmd', '')
            if web_cmd:
                shared_dict['web_cmd'] = ""
                if web_cmd == 'RUN' and not is_running:
                    try: cmd_queue.put_nowait("START\n")
                    except queue.Full: pass
                    time.sleep(0.2) 
                    is_running = True
                elif web_cmd == 'STOP' and is_running:
                    is_running, is_finding = False, False
                    try:
                        cmd_queue.put_nowait("FMODE:0\n")
                        cmd_queue.put_nowait(f"RPM:0.0|TY:{shared_dict['usv_yaw']:.1f}\n")
                        cmd_queue.put_nowait("STOP\n")
                    except queue.Full: pass
                    last_cmd_str = "" # Xóa bộ nhớ lệnh để ép gửi lệnh mới
                elif web_cmd == 'RTL' and is_running:
                    usv_state, accumulated_yaw = "RTL", 0.0

            # --- 2. IDLE STATE ---
            if not is_running:
                try: 
                    bgr_img, _ = out_q.get(timeout=0.1)
                except queue.Empty: pass
                
                shared_dict["state"] = "IDLE"
                shared_dict["is_running"] = False
                shared_dict["boat_yaw"] = math.radians(-shared_dict["usv_yaw"]) 
                
                # [SAFETY HEARTBEAT]: Liên tục gửi lệnh dừng khi ở IDLE
                if time.time() - last_cmd_time > 0.5:
                    try:
                        cmd_queue.put_nowait("FMODE:0\n")
                        cmd_queue.put_nowait(f"RPM:0.0|TY:{shared_dict['usv_yaw']:.1f}\n")
                    except queue.Full: pass
                    last_cmd_time = time.time()
                    
                time.sleep(0.05)
                continue

            # --- 3. RUNNING STATE ---
            try:
                

                bgr_img, frame_targets = out_q.get(timeout=0.5)
                

                boat_x, boat_y = shared_dict['boat_x'], shared_dict['boat_y']
                usv_yaw, usv_yaw_rate = shared_dict['usv_yaw'], shared_dict['usv_yaw_rate']
                current_yaw_rad = math.radians(-usv_yaw)
                shared_dict['boat_yaw'] = current_yaw_rad 
                
                trash_list, obstacle_list = [], []
                
                for t in frame_targets:
                    abs_yaw = current_yaw_rad + t['yaw']
                    gx, gy = boat_x + t['dist'] * math.cos(abs_yaw), boat_y + t['dist'] * math.sin(abs_yaw)
                    if gx < MIN_X or gx > MAX_X or gy < MIN_Y or gy > MAX_Y: continue
                    
                    t['gx'], t['gy'] = gx, gy
                    if t['class_id'] == 0: obstacle_list.append(t)
                    else: trash_list.append(t)

                target_source, closest_dist, best_target_yaw_err = "NONE", 999.0, 0.0
                active_trash_coords = []

                # --- STATE MACHINE: RTL & APF ---
                if usv_state == "RTL" and trash_list:
                    usv_state, accumulated_yaw = "TRACKING", 0.0 
                    
                if usv_state == "RTL":
                    dist_to_home = math.hypot(0.0 - boat_x, 0.0 - boat_y)
                    abs_yaw_to_home = math.atan2(0.0 - boat_y, 0.0 - boat_x)
                    err_to_home = normalize_angle(abs_yaw_to_home - current_yaw_rad)
                    
                    if abs(err_to_home) > math.radians(60.0):
                        target_source = "NONE"
                    else:
                        target_source = "RTL"
                        blocking_obs = [obs for obs in obstacle_list if obs['dist'] < 1.0 and abs(normalize_angle(obs['yaw'] - err_to_home)) < math.radians(25)]
                                
                        if blocking_obs:
                            target_source = "AVOIDANCE"
                            closest_obs = min(blocking_obs, key=lambda x: x['dist'])
                            abs_obs_yaw = current_yaw_rad + closest_obs['yaw']
                            obs_global_x = boat_x + closest_obs['dist'] * math.cos(abs_obs_yaw)
                            obs_global_y = boat_y + closest_obs['dist'] * math.sin(abs_obs_yaw)
                            
                            avoidance_angle = abs_obs_yaw + math.copysign(math.pi/2, err_to_home - closest_obs['yaw']) 
                            virtual_wp_x = obs_global_x + 0.3 * math.cos(avoidance_angle)
                            virtual_wp_y = obs_global_y + 0.3 * math.sin(avoidance_angle)
                            
                            dist_to_vwp = math.hypot(virtual_wp_x - boat_x, virtual_wp_y - boat_y)
                            abs_yaw_to_vwp = math.atan2(virtual_wp_y - boat_y, virtual_wp_x - boat_x)
                            
                            closest_dist = min(0.4, dist_to_vwp)
                            best_target_yaw_err = normalize_angle(abs_yaw_to_vwp - current_yaw_rad)
                        else:
                            closest_dist = min(0.4, dist_to_home) 
                            best_target_yaw_err = err_to_home
                            
                    if (0.0 <= boat_x <= HOME_ZONE_X_MAX) and (HOME_ZONE_Y_MIN <= boat_y <= 0.0):
                        usv_state, target_source = "STANDBY", "STANDBY"

                # --- STATE MACHINE: TRACKING ---
                elif trash_list:
                    usv_state = "TRACKING"
                    best_trash = min(trash_list, key=lambda x: x['dist'])
                    closest_dist, best_target_yaw_err = best_trash['dist'], best_trash['yaw']
                    active_trash_coords.append({"x": best_trash['gx'], "y": best_trash['gy']})
                    
                    blocking_obs = [obs for obs in obstacle_list if obs['dist'] < closest_dist and abs(obs['yaw'] - best_target_yaw_err) < math.radians(25)]
                            
                    if blocking_obs:
                        target_source = "AVOIDANCE"
                        closest_obs = min(blocking_obs, key=lambda x: x['dist'])
                        obs_deg = math.degrees(closest_obs['yaw'])
                        virt_deg = -obs_deg
                        if abs(obs_deg) < 5.0: virt_deg = -25.0
                        elif abs(virt_deg) < 20.0: virt_deg = math.copysign(20.0, virt_deg)
                        best_target_yaw_err, closest_dist = math.radians(virt_deg), closest_obs['dist']
                    else:
                        target_source = "CAMERA"
                        if best_trash['is_swallowed']:
                            mem_timestamp = 0.0; target_source = "NONE"; active_trash_coords = [] 
                        elif best_trash['is_exiting']:
                            mem_global_yaw = (usv_yaw - math.degrees(best_target_yaw_err)) % 360.0
                            mem_distance, mem_timestamp = closest_dist, time.time()
                else:
                    if usv_state == "TRACKING":
                        if time.time() - mem_timestamp < MEMORY_TIMEOUT:
                            target_source = "MEMORY"
                            raw_err_deg = usv_yaw - mem_global_yaw
                            raw_err_deg = (raw_err_deg + 180) % 360 - 180
                            best_target_yaw_err = math.radians(raw_err_deg)
                            closest_dist = mem_distance 
                            
                            abs_mem_yaw = current_yaw_rad + best_target_yaw_err
                            active_trash_coords.append({"x": boat_x + mem_distance * math.cos(abs_mem_yaw), "y": boat_y + mem_distance * math.sin(abs_mem_yaw)})
                        else:
                            usv_state, prev_yaw_for_search, accumulated_yaw = "SEARCHING", current_yaw_rad, 0.0

                    if usv_state == "SEARCHING":
                        target_source = "NONE"
                        accumulated_yaw += normalize_angle(current_yaw_rad - prev_yaw_for_search)
                        prev_yaw_for_search = current_yaw_rad
                        if abs(accumulated_yaw) >= 4.0 * math.pi: usv_state = "RTL"
                
                # --- UPDATE DASHBOARD ---
                shared_dict["state"] = usv_state
                shared_dict["is_running"] = True
                shared_dict["active_trash"] = active_trash_coords
                if target_source not in ["CAMERA", "MEMORY", "RTL", "AVOIDANCE"]:
                    shared_dict['target_yaw'] = usv_yaw
                
                # --- HYBRID CONTROL MAPPING EXECUTION ---
                if target_source in ["CAMERA", "MEMORY", "RTL", "AVOIDANCE"]:
                    if is_finding:
                        try: cmd_queue.put_nowait("FMODE:0\n")
                        except queue.Full: pass
                        is_finding = False
                        
                    pwm_offset, target_yaw_deg = rl_agent.predict(closest_dist, best_target_yaw_err, math.radians(-usv_yaw_rate), usv_yaw)
                    
                    cmd = f"RPM:{pwm_offset:.1f}|TY:{target_yaw_deg:.1f}\n"
                    shared_dict['target_yaw'] = target_yaw_deg
                    
                    if cmd != last_cmd_str or (time.time() - last_cmd_time) > 0.1:
                        try: cmd_queue.put_nowait(cmd)
                        except queue.Full: pass
                        last_cmd_str, last_cmd_time = cmd, time.time()
                            
                elif target_source == "NONE":
                    if not is_finding:
                        try: cmd_queue.put_nowait("FMODE:1\n")
                        except queue.Full: pass
                        is_finding = True
                        last_cmd_str = "" 
                    
                    # [SAFETY HEARTBEAT]: Ép ga về 0.0 khi Searching để không bị phóng thẳng
                    if time.time() - last_cmd_time > 0.2:
                        cmd = f"RPM:0.0|TY:{usv_yaw:.1f}\n"
                        try:
                            cmd_queue.put_nowait("FMODE:1\n")
                            cmd_queue.put_nowait(cmd)
                        except queue.Full: pass
                        last_cmd_str, last_cmd_time = cmd, time.time()
                
                elif target_source == "STANDBY":
                    if is_finding:
                        try: cmd_queue.put_nowait("FMODE:0\n")
                        except queue.Full: pass
                        is_finding = False
                        last_cmd_str = ""
                    
                    # [SAFETY HEARTBEAT]: Neo thuyền tại bến
                    if time.time() - last_cmd_time > 0.5:
                        cmd = f"RPM:0.0|TY:{usv_yaw:.1f}\n"
                        try: cmd_queue.put_nowait(cmd)
                        except queue.Full: pass
                        last_cmd_str, last_cmd_time = cmd, time.time()

            except queue.Empty: continue
            except Exception as e: print(f"Main loop err: {e}")
                
    
    except KeyboardInterrupt:
        print("\n--> [SYSTEM] Ctrl+C Detected! Emergency Stop...")
        try:
            # Dùng put_nowait để KHÔNG BAO GIỜ bị kẹt lại đây
            cmd_queue.put_nowait("STOP\n")
        except queue.Full:
            pass
            
    finally:
        print("--> Cleaning up hardware and context resources safely...")
        stop_event.set()
        
        # [QUAN TRỌNG]: Cho tiến trình Vision 3 giây để lưu nốt video
        p_vision.join(timeout=3.0)
        
        # Nếu sau 3 giây mà Vision vẫn chưa chịu tắt thì mới được phép ép chết (terminate)
        if p_vision.is_alive():
            print("--> [WARN] Vision process timeout. Forcing termination.")
            p_vision.terminate()
            
        p_udp.terminate(); p_udp.join()
        p_serial.terminate(); p_serial.join()
        p_web.terminate(); p_web.join()
        print("--> System multiprocess cleanup complete.")

if __name__ == '__main__':
    main()