
import cv2
import numpy as np
import sys
import os
from PIL import Image
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import torch

from config import MODEL_PATH, CHOSEN_PROVIDERS, EMB_DIR, THRESHOLD, CAMERA_INDEX
print(f" Đang sử dụng model: {MODEL_PATH}")

sys.path.insert(0, './face_alignment')
from mtcnn import MTCNN
mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu', crop_size=(112, 112))

session = ort.InferenceSession(MODEL_PATH, providers=CHOSEN_PROVIDERS)
input_name = session.get_inputs()[0].name
expected_dtype = session.get_inputs()[0].type

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding_from_pil(pil_img):
    tensor = transform(pil_img).unsqueeze(0).numpy()
    tensor = tensor.astype(np.float16) if "float16" in expected_dtype.lower() else tensor.astype(np.float32)
    emb = session.run(None, {input_name: tensor})[0]
    return emb[0] / np.linalg.norm(emb[0])

def recognize_face(emb):
    best_score = -1
    best_match = "new person"
    for person in os.listdir(EMB_DIR):
        person_dir = os.path.join(EMB_DIR, person)
        if not os.path.isdir(person_dir): continue
        for emb_file in os.listdir(person_dir):
            if emb_file.endswith(".npy"):
                db_emb = np.load(os.path.join(person_dir, emb_file))
                score = cosine_similarity([emb], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = person if score >= THRESHOLD else "new person"
    return best_match, best_score

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        try:
            bboxes, faces = mtcnn.align_multi(pil_img, limit=10)
        except:
            bboxes, faces = [], []

        for box, aligned in zip(bboxes, faces):
            emb = get_embedding_from_pil(aligned)
            name, score = recognize_face(emb)
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
