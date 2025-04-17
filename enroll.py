# enroll.py
import os
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms

sys.path.insert(0, './face_alignment')
from align import get_aligned_face

# LẤY THAM SỐ MODEL TYPE
MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "fp32"
MODEL_PATH = f"models/edgeface_{MODEL_TYPE}.onnx"
device = "cpu"

IMAGE_DIR = "datasets/images"
EMB_DIR = "datasets/embeddings"
DEBUG_ALIGNED_DIR = "debug_aligned/enroll_img"

# LOAD ONNX MODEL
print(f" Sử dụng mô hình ONNX: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# CHUẨN HOÁ ẢNH
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

def get_embedding(img_path, aligned_path=None):
    aligned_img = get_aligned_face(img_path)
    if aligned_img is None:
        print(f" Không tìm thấy khuôn mặt: {img_path}")
        return None

    if aligned_path:
        os.makedirs(os.path.dirname(aligned_path), exist_ok=True)
        aligned_img.save(aligned_path)

    tensor = transform(aligned_img).unsqueeze(0).numpy()  # CHUYỂN THÀNH numpy
    outputs = session.run(None, {input_name: tensor})
    emb = outputs[0][0]
    emb = emb / np.linalg.norm(emb)  # normalize
    return emb

def enroll_individual(person, img_path):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(EMB_DIR, person)
    os.makedirs(save_dir, exist_ok=True)

    aligned_debug_path = os.path.join(DEBUG_ALIGNED_DIR, person, f"{img_name}_aligned.jpg")
    emb_path = os.path.join(save_dir, f"{img_name}.npy")

    if os.path.exists(emb_path):
        print(f" Bỏ qua: {person}/{img_name}.npy (đã có)")
        return

    emb = get_embedding(img_path, aligned_debug_path)
    if emb is not None:
        np.save(emb_path, emb)
        print(f" Đã lưu: {person}/{img_name}.npy")

def enroll_all():
    os.makedirs(EMB_DIR, exist_ok=True)

    for person in os.listdir(IMAGE_DIR):
        person_dir = os.path.join(IMAGE_DIR, person)
        if not os.path.isdir(person_dir): continue

        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_dir, img_file)
            enroll_individual(person, img_path)

if __name__ == "__main__":
    enroll_all()
