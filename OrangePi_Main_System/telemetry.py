import socket
import struct
import os

def udp_worker_process(shared_dict, stop_event):
    try: os.sched_setaffinity(0, {6})
    except: pass
    
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("0.0.0.0", 5005))
    udp_socket.settimeout(1.0)
    print("--> [UDP] Listening on port 5005...")
    
    while not stop_event.is_set():
        try:
            data, _ = udp_socket.recvfrom(1024)
            if len(data) == 8: 
                x_cm, y_cm = struct.unpack('ff', data)
                shared_dict['boat_x'] = x_cm / 100.0
                shared_dict['boat_y'] = y_cm / 100.0
        except socket.timeout: continue
        except Exception: pass