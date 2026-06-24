from rknn.api import RKNN

# --- CONFIGURATION ---
ONNX_MODEL_PATH = 'best0906_int8.onnx'      # Your validated ONNX model  
RKNN_MODEL_PATH = 'best0906_int8_2.rknn' # Target output path for the new INT8 model
TARGET_PLATFORM = 'rk3588'             # Orange Pi 5 Plus SoC

# Path to a text file containing paths to calibration images. 
# This is MANDATORY for INT8 quantization.
DATASET_PATH = 'dataset.txt'           

def convert_onnx_to_rknn():
    print("--> Initializing RKNN compiler...")
    rknn = RKNN(verbose=True)

    # --- CONFIGURATION FOR QUANTIZATION ---
    # YOLOv8 expects pixels normalized to [0, 1].
    # By setting mean=[[0, 0, 0]] and std=[[255, 255, 255]], the NPU hardware
    # will automatically perform the (pixel - mean) / std operation.
    # quantized_dtype ensures asymmetric 8-bit quantization is used.
    # ...
    print(f"--> Configuring hardware normalization for {TARGET_PLATFORM}...")
    
    # Use 'mmse' algorithm which is highly recommended for YOLO models
    # It minimizes the precision loss during INT8 conversion
    rknn.config(
        target_platform=TARGET_PLATFORM,
        optimization_level=3,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype='asymmetric_quantized-8'
        #quantized_algorithm='mmse' # <-- ADD THIS LINE
    )
    # ...

    # Load ONNX Model
    print(f"--> Loading ONNX model: {ONNX_MODEL_PATH}")
    ret = rknn.load_onnx(model=ONNX_MODEL_PATH)
    if ret != 0:
        print("Error: Failed to load ONNX model.")
        return

    # --- THE CRITICAL FIX FOR INT8 ---
    # Set do_quantization=True and provide the calibration dataset.
    # The toolkit will run these images through the model to determine optimal INT8 scales.
    print("--> Building RKNN model with INT8 quantization...")
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
    if ret != 0:
        print("Error: Failed to build RKNN model.")
        return

    # Export to file
    print(f"--> Saving compiled model to: {RKNN_MODEL_PATH}")
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print("Error: Failed to export RKNN model.")
        return

    print("\n--> SUCCESS! INT8 model is compiled and optimized.")
    rknn.release()

if __name__ == '__main__':
    convert_onnx_to_rknn()