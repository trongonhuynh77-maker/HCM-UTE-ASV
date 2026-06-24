import os
import time
import serial
import queue

def serial_worker_process(shared_dict, cmd_queue, stop_event):
    try: os.sched_setaffinity(0, {5})
    except: pass
    
    ser = None
    is_connected = False

    while not stop_event.is_set():
        
        # ==========================================
        # 1. CƠ CHẾ TỰ ĐỘNG KẾT NỐI LẠI (AUTO-RECONNECT)
        # ==========================================
        if not is_connected:
            try:
                if ser is not None:
                    ser.close()
                ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
                ser.dtr = False; ser.rts = False 
                print("--> [STM32] Connected to /dev/ttyACM0")
                is_connected = True
            except Exception:
                # Nếu không kết nối được, ngủ 1 giây rồi thử lại, không làm chết tiến trình
                time.sleep(1.0)
                continue 

        # ==========================================
        # 2. XỬ LÝ ĐỌC/GHI DỮ LIỆU (READ/WRITE)
        # ==========================================
        try:
            # --- ĐỌC DỮ LIỆU (READ) ---
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("Y:"):
                    parts = line.split('|')
                    shared_dict['usv_yaw'] = float(parts[0].replace("Y:", ""))
                    if len(parts) > 1 and "V:" in parts[1]:
                        shared_dict['usv_yaw_rate'] = float(parts[1].replace("V:", ""))
            
            # --- GHI DỮ LIỆU (WRITE) ---
            while not cmd_queue.empty():
                cmd_str = cmd_queue.get_nowait()
                ser.write(cmd_str.encode('utf-8'))
                ser.flush()
                
        except queue.Empty: 
            pass
        except Exception as e:
            # Lỗi đứt cáp, nhiễu điện, hoặc STM32 bị crash reset
            print(f"--> [STM32 WARNING] Connection lost/Error: {e}")
            is_connected = False # Đánh dấu mất kết nối để vòng lặp sau tự động mở lại port
            time.sleep(0.5)
        
        # Ngủ ngắn để nhường CPU cho tiến trình khác
        time.sleep(0.005)
        
    # Dọn dẹp khi tắt hệ thống
    if ser is not None and ser.is_open:
        ser.close()
        print("--> [STM32] Port closed safely.")