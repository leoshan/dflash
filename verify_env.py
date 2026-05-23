import sglang
import torch
import transformers
import os

def verify():
    print(f"SGLang version: {sglang.__version__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"CUDA is available. Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is NOT available.")

    # 检查路径
    model_path = "/home/shanchuang/nvext/models/Qwen3.6-27B"
    if os.path.exists(model_path):
        print(f"Model path exists: {model_path}")
    else:
        print(f"Model path NOT found: {model_path}")

if __name__ == "__main__":
    verify()
