import os
import onnxruntime as ort

#  CẤU HÌNH MODEL 
MODEL_TYPE = os.getenv("MODEL_TYPE", "fp16")  
MODEL_PATH = f"models/edgeface_{MODEL_TYPE}.onnx"

#  CẤU HÌNH NHẬN DIỆN 
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))

# THƯ MỤC DỮ LIỆU 
EMB_DIR = "datasets/embeddings"
IMAGE_DIR = "datasets/images"
DEBUG_ALIGNED_ENROLL = "debug_aligned/enroll_img"
DEBUG_ALIGNED_TEST = "debug_aligned/test"

#  ONNX RUNTIME PROVIDERS 
available_providers = ort.get_available_providers()
preferred_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
CHOSEN_PROVIDERS = [p for p in preferred_providers if p in available_providers]

# CAMERA CONFIG 
CAMERA_INDEX = 1