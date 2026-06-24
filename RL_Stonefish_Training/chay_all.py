import subprocess
import sys

# Chạy thẳng tên file, không cần thông số rườm rà nữa
command_1 = [sys.executable, 'STM32.py']
command_2 = [sys.executable, 'train3.py']
command_3 = [sys.executable, 'DOTHI.py']

print("==================================================")
print("[HỆ THỐNG] Đang khởi động toàn bộ Node ROS 2...")
print("==================================================")

try:
    process1 = subprocess.Popen(command_1)
    process2 = subprocess.Popen(command_2)
    process3 = subprocess.Popen(command_3)
    
    process1.wait()
    process2.wait()
    process3.wait()
    print("All processes have successfully completed.")

except KeyboardInterrupt:
    print("\n[HỆ THỐNG] Phát hiện lệnh dừng khẩn cấp (Ctrl+C). Đang tắt an toàn...")
    process1.terminate()
    process2.terminate()
    process3.terminate()
    process1.wait()
    process2.wait()
    process3.wait()
    print("[HỆ THỐNG] Đã tắt sạch sẽ!")