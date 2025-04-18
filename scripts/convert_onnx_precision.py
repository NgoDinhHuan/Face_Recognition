import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16


# Đường dẫn model ONNX FP32 gốc
fp32_path = "models/edgeface_fp32.onnx"
fp16_path = "models/edgeface_fp16.onnx"
int8_path = "models/edgeface_int8.onnx"

# Bước 1: Chuyển sang FP
model_fp32 = onnx.load(fp32_path)
model_fp16 = convert_float_to_float16(model_fp32)
onnx.save(model_fp16, fp16_path)
print(f"Đã lưu model FP16 tại: {fp16_path}")

# Bước 2: Chuyển sang INT8
quantize_dynamic(
    model_input=fp32_path,
    model_output=int8_path,
    weight_type=QuantType.QInt8
)
print(f"Đã lưu model INT8 tại: {int8_path}")
