# recognize.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.insert(0, './face_alignment')
from align import get_aligned_face

# Load model từ torch.hub
model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', source='github', pretrained=True)
model.eval()
device = "cpu"
model = model.to(device)

EMB_DIR = "datasets/embeddings"
QUERY_IMG = "datasets/test/testok.png"
DEBUG_ALIGNED_DIR = "debug_aligned/test"
THRESHOLD = 0.5

os.makedirs(DEBUG_ALIGNED_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(img_path, save_debug_path=None):
    aligned_img = get_aligned_face(img_path)
    if aligned_img is None:
        print(f"\u274c Không tìm thấy khuôn mặt trong: {img_path}")
        return None

    if save_debug_path:
        aligned_img.save(save_debug_path)

    tensor = transform(aligned_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = F.normalize(emb)
    return emb.squeeze(0).cpu().numpy()

def recognize(query_path):
    debug_name = os.path.splitext(os.path.basename(query_path))[0] + "_aligned.jpg"
    debug_path = os.path.join(DEBUG_ALIGNED_DIR, debug_name)
    query_emb = get_embedding(query_path, save_debug_path=debug_path)
    if query_emb is None:
        return

    best_score = -1
    best_match = "new person"

    for person in os.listdir(EMB_DIR):
        person_dir = os.path.join(EMB_DIR, person)
        if not os.path.isdir(person_dir): continue

        for emb_file in os.listdir(person_dir):
            if emb_file.endswith(".npy"):
                db_emb = np.load(os.path.join(person_dir, emb_file))
                score = cosine_similarity([query_emb], [db_emb])[0][0]
                print(f"So sánh với {person}/{emb_file}: score = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_match = person if score >= THRESHOLD else "new person"

    print(f"Kết quả: {best_match} (score = {best_score:.4f})")

if __name__ == "__main__":
    recognize(QUERY_IMG)
