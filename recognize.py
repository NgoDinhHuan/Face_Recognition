# recognize.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort
import sys

sys.path.insert(0, './face_alignment')
from align import get_aligned_face

#  CẤU HÌNH
THRESHOLD = 0.5
QUERY_IMG = "datasets/test/testok.png"
EMB_DIR = "datasets/embeddings"
DEBUG_ALIGNED_DIR = "debug_aligned/test"
os.makedirs(DEBUG_ALIGNED_DIR, exist_ok=True)

#  CHỌN MODEL TỪ TERMINAL
model_type = sys.argv[1] if len(sys.argv) > 1 else "fp32"
MODEL_PATH = f"models/edgeface_{model_type}.onnx"
print(f" Đang dùng mô hình: {MODEL_PATH}")

#  LOAD MODEL ONNX 
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

#  TRANSFORM ẢNH
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#  LẤY EMBEDDING 
def get_embedding(img_path, save_debug_path=None):
    aligned_img = get_aligned_face(img_path)
    if aligned_img is None:
        print(f"Không tìm thấy khuôn mặt trong: {img_path}")
        return None

    if save_debug_path:
        aligned_img.save(save_debug_path)

    tensor = transform(aligned_img).unsqueeze(0).numpy()
    outputs = session.run(None, {input_name: tensor})
    emb = outputs[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb

# RECOGNIZE 
def recognize():
    debug_name = os.path.splitext(os.path.basename(QUERY_IMG))[0] + "_aligned.jpg"
    debug_path = os.path.join(DEBUG_ALIGNED_DIR, debug_name)

    query_emb = get_embedding(QUERY_IMG, save_debug_path=debug_path)
    if query_emb is None:
        return

    best_score = -1
    best_match = "Người mới"

    for person in os.listdir(EMB_DIR):
        person_dir = os.path.join(EMB_DIR, person)
        if not os.path.isdir(person_dir): continue

        for emb_file in os.listdir(person_dir):
            if emb_file.endswith(".npy"):
                db_emb = np.load(os.path.join(person_dir, emb_file))
                score = cosine_similarity([query_emb], [db_emb])[0][0]
                print(f"🔍 So sánh với {person}/{emb_file}: score = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_match = person if score >= THRESHOLD else "Người mới"

    print(f"✅ Kết quả: {best_match} (score = {best_score:.4f})")

if __name__ == "__main__":
    recognize()
