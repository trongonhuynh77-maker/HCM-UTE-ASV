```markdown
# EcoASV: Edge AI & Reinforcement Learning Powered Autonomous Surface Vehicle for Aquatic Waste Collection

An autonomous, intelligent, and highly robust Cyber-Physical System (CPS) designed for real-time aquatic plastic waste monitoring and automated collection in urban inland waterways.

## 🚀 Project Overview
EcoASV is an autonomous surface vehicle featuring a catamaran hull design. The system eliminates cloud dependencies by executing high-level computer vision and dynamic path-planning entirely on the edge. It combines deep-learning-based perception with Reinforcement Learning (RL) control to adaptively navigate and collect surface waste under dynamic aquatic disturbances (e.g., water currents, wind tethers).

## 🛠️ Key Features
* **Edge AI Perception:** Real-time object detection using a customized **YOLO26n** model optimized for edge NPU hardware (Quantized to INT8, achieving 30-50 FPS with ~20ms latency).
* **Deep Reinforcement Learning (DRL) Control:** Dynamic path planning and obstacle avoidance driven by the **Soft Actor-Critic (SAC)** algorithm, bridging the Sim-to-Real gap smoothly.
* **Hybrid Global Positioning:** Utilizes an external **ArUco Marker** vision system for highly accurate global coordinate estimation in GPS-denied or testing environments, streaming state vectors ($X, Y, \theta$) via ultra-low-latency UDP protocols.
* **Distributed Software Architecture:** Powered by **ROS 2 Jazzy** on **Ubuntu 24.04** to manage concurrent real-time multi-processing workloads asynchronously.
* **Robust Mechanical Design:** Catamaran layout with a **Magnetic Coupling** thruster mechanism, ensuring 100% waterproof isolation for electronic enclosures and mechanical overload protection against debris entanglement.
* **Web-based HMI Dashboard:** Real-time telemetry visualization (Yaw response, spatial mapping) and RESTful API control panel.

## 📁 Repository Structure
```text
├── Aruco_Positioning_System/   # ArUco marker tracking, global coordinate estimation, and UDP broadcasting
├── RL_Stonefish_Training/      # DRL training scripts (SAC) and custom Stonefish physics environments
├── YOLO_Quantization/          # Model conversion scripts (PyTorch -> ONNX -> RKNN) & calibration datasets
├── OrangePi_Main_System/       # ROS 2 workspace (Nodes for inference, ZeroMQ/UDP, and HMI Webserver)
└── STM32_Hardware_Control/     # Low-level firmware (C/C++), PID control, IMU reading, and PWM generation

```

## 💻 Hardware Architecture

* **High-Level Compute (Edge AI):** Orange Pi 5 Plus (Rockchip RK3588, 6 TOPS NPU)
* **Low-Level Microcontroller:** STM32F407VET6
* **Perception Sensors:** Intel RealSense D435i Depth Camera (On-board) + External Shore Camera (ArUco tracking)
* **Navigation Sensors:** BNO055 9-axis IMU, Global ArUco System
* **Actuators:** Dual Brushless DC (BLDC) motors with Electronic Speed Controllers (ESC) for differential steering.

## ⚙️ Software Environment & Dependencies

* Operating System: Ubuntu 24.04 LTS
* Middleware: ROS 2 Jazzy Jalisco
* AI Inference Platform: Rockchip RKNN-Toolkit2 (INT8 Quantization)
* Positioning: OpenCV (ArUco module)
* Libraries: PyTorch, ZeroMQ, UDP Sockets, Flask/WebSockets (for HMI)
* Simulation Tool: Stonefish Simulator

## 👥 Contributors

* **Huỳnh Trọng Ơn** (Team Leader) - Control & Automation Engineering
* **Lương Trần Trung Đức** - Control & Automation Engineering
* **Nguyễn Tấn Hậu** - Control & Automation Engineering

**Academic Supervisor:** Dr. Dương Minh Thiện
*Department of Automatic Control, Faculty of Electrical and Electronics Engineering, Ho Chi Minh City University of Technology and Engineering (HCM-UTE).*
