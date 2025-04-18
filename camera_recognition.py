import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import torch

# Face Alignment 
sys.path.insert(0, './face_alignment')
from mtcnn import MTCNN

# MTCNN dùng GPU, fallback về CPU
mtcnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn_detector = MTCNN(device=mtcnn_device, crop_size=(112, 112))

# 
EMB_DIR = "datasets/embeddings"
THRESHOLD = 0.5
model_type = sys.argv[1] if len(sys.argv) > 1 else "fp32"
model_path = f"models/edgeface_{model_type}.onnx"

#  Load ONNX Model 
available_providers = ort.get_available_providers()
preferred_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
chosen_providers = [p for p in preferred_providers if p in available_providers]

print(f" Đang dùng model: {model_path}")

session = ort.InferenceSession(model_path, providers=chosen_providers)
input_name = session.get_inputs()[0].name
expected_dtype = session.get_inputs()[0].type

#  Chuẩn hóa ảnh 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding_from_pil(pil_img):
    tensor = transform(pil_img).unsqueeze(0).numpy()
    tensor = tensor.astype(np.float16) if "float16" in expected_dtype.lower() else tensor.astype(np.float32)

    ort_inputs = {input_name: tensor}
    emb = session.run(None, ort_inputs)[0]
    return emb[0] / np.linalg.norm(emb[0])

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
    if not cap.isOpened():
        print(" Không thể mở camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Không đọc được frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        try:
            bboxes, faces = mtcnn_detector.align_multi(pil_img, limit=10)
        except:
            bboxes, faces = [], []

        for box, aligned in zip(bboxes, faces):
            emb = get_embedding_from_pil(aligned)
            name, score = recognize_face(emb)

            x1, y1, x2, y2 = map(int, box[:4])
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
