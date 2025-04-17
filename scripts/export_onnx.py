import torch
import os


model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', source='github', pretrained=True)
model.eval()
model.cpu()

# Tạo input mẫu
dummy_input = torch.randn(1, 3, 112, 112)


# Export mô hình sang ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/edgeface_fp32.onnx",
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=11
)

print(" Mô hình đã được export sang ONNX (FP32)")
