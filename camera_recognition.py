import cv2
import torch
import numpy as np
import os
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import sys

EMB_DIR = "datasets/embeddings"
THRESHOLD = 0.5
device = "cpu"

# Load model từ torch.hub
model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', source='github', pretrained=True)
model = model.to(device).eval()

# Load hàm align 
sys.path.insert(0, './face_alignment')
from mtcnn import MTCNN
mtcnn_detector = MTCNN(device=device, crop_size=(112, 112))

# Transform ảnh đầu vào
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding_from_pil(pil_img):
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = F.normalize(emb)
    return emb.squeeze(0).cpu().numpy()

def recognize_face(face_emb):
    best_score = -1
    best_match = "new person"
    for person in os.listdir(EMB_DIR):
        person_dir = os.path.join(EMB_DIR, person)
        if not os.path.isdir(person_dir): continue
        for emb_file in os.listdir(person_dir):
            if not emb_file.endswith(".npy"): continue
            db_emb = np.load(os.path.join(person_dir, emb_file))
            score = cosine_similarity([face_emb], [db_emb])[0][0]
            if score > best_score:
                best_score = score
                best_match = person if score >= THRESHOLD else "new person"
    return best_match, best_score

def main():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        try:
            bboxes, faces = mtcnn_detector.align_multi(pil_img, limit=10)
        except:
            bboxes, faces = [], []

        for box, aligned in zip(bboxes, faces):
            emb = get_embedding_from_pil(aligned)
            name, score = recognize_face(emb)

            # Vẽ bbox và hiển thị label
            x1, y1, x2, y2 = map(int, box[:4])  # Lấy 4 tọa độ đầu tiên
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Realtime Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
