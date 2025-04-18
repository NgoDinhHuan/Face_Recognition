# recognize.py
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import onnxruntime as ort
import sys

from config import MODEL_PATH, CHOSEN_PROVIDERS, EMB_DIR, DEBUG_ALIGNED_TEST, THRESHOLD

sys.path.insert(0, './face_alignment')
from align import get_aligned_face

session = ort.InferenceSession(MODEL_PATH, providers=CHOSEN_PROVIDERS)
input_name = session.get_inputs()[0].name
expected_dtype = session.get_inputs()[0].type

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

QUERY_IMG = "datasets/test/testok.png"

def get_embedding(img_path, save_debug_path=None):
    aligned_img = get_aligned_face(img_path)
    if aligned_img is None:
        print(f" Không tìm thấy khuôn mặt trong: {img_path}")
        return None
    if save_debug_path:
        os.makedirs(os.path.dirname(save_debug_path), exist_ok=True)
        aligned_img.save(save_debug_path)

    tensor = transform(aligned_img).unsqueeze(0).numpy()
    tensor = tensor.astype(np.float16) if "float16" in expected_dtype.lower() else tensor.astype(np.float32)
    outputs = session.run(None, {input_name: tensor})
    emb = outputs[0][0]
    return emb / np.linalg.norm(emb)

def recognize():
    debug_path = os.path.join(DEBUG_ALIGNED_TEST, "query_aligned.jpg")
    query_emb = get_embedding(QUERY_IMG, debug_path)
    if query_emb is None:
        return

    best_score = -1
    best_match = "New persion"

    for person in os.listdir(EMB_DIR):
        person_dir = os.path.join(EMB_DIR, person)
        if not os.path.isdir(person_dir): continue

        for emb_file in os.listdir(person_dir):
            if emb_file.endswith(".npy"):
                db_emb = np.load(os.path.join(person_dir, emb_file))
                score = cosine_similarity([query_emb], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = person if score >= THRESHOLD else "New persion"

    print(f" Kết quả: {best_match} (score = {best_score:.4f})")

if __name__ == "__main__":
    recognize()
