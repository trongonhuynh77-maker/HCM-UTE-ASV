import torch
import torch.nn as nn
from stable_baselines3 import SAC
import onnx
import os

# ==========================================
# ⚙️ CONFIGURATION FOR ONNX RUNTIME (C++)
# ==========================================
# Path to your trained SAC model (without .zip)
MODEL_NAME = "./sac_usv_models3/sac_usv_carrot_chasing_fast" 

# Desired output ONNX file name
ONNX_EXPORT_PATH = "usv_actor_final.onnx"

# Simplified Observation Space Dimension: [Distance, Heading Error, Yaw Rate]
OBS_DIM = 3

# ==========================================

class OnnxableActor(nn.Module):
    """
    Wrapper class to convert the SB3 SAC Actor into a pure PyTorch Module.
    Optimized for ONNX Runtime Execution.
    """
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation):
        # 1. Feature Extraction (Base Network)
        features = self.actor.extract_features(observation, self.actor.features_extractor)
        
        # 2. Latent Policy Network Processing
        latent_pi = self.actor.latent_pi(features)
        
        # 3. Deterministic Action (Mean value, no noise)
        mean_actions = self.actor.mu(latent_pi)
        
        # 4. Tanh Activation: Forces outputs strictly into [-1.0, 1.0] range
        actions = torch.tanh(mean_actions) 
        
        return actions

def main():
    print(f"[INFO] Initiating ONNX export for ONNX Runtime (C++)...")
    
    if not os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"[ERROR] Model file '{MODEL_NAME}.zip' not found!")
        return

    # Load model strictly on CPU for Edge execution
    print("[INFO] Loading SAC model weights...")
    model = SAC.load(MODEL_NAME, device='cpu') 

    # Wrap the core network
    onnxable_model = OnnxableActor(model.policy.actor)

    # We lock the batch size to exactly 1.
    # In C++ control loops, you evaluate one sensor reading at a time.
    # Removing dynamic_axes increases execution speed and simplifies C++ code.
    dummy_input = torch.randn(1, OBS_DIM, dtype=torch.float32)

    onnxable_model.eval()

    print(f"[INFO] Tracing and compiling PyTorch Graph...")
    
    # Export using modern Opset 18 (Best for ONNX Runtime)
    torch.onnx.export(
        onnxable_model,             
        dummy_input,                
        ONNX_EXPORT_PATH,           
        export_params=True,         
        opset_version=18,           # Optimal for ONNX Runtime. Avoids Relu warnings.
        do_constant_folding=True,   
        input_names=['observation'],# C++ Input Node Name
        output_names=['action']     # C++ Output Node Name
        # Note: 'dynamic_axes' is completely removed for maximum speed and Dynamo compatibility.
    )

    # Validation
    try:
        onnx_model = onnx.load(ONNX_EXPORT_PATH)
        onnx.checker.check_model(onnx_model)
        print(f"\n[SUCCESS] Model successfully exported: {ONNX_EXPORT_PATH}")
        print("[INFO] Ready for C++ integration using onnxruntime.dll / libonnxruntime.so")
    except Exception as e:
        print(f"\n[ERROR] ONNX validation failed: {e}")

if __name__ == "__main__":
    main()