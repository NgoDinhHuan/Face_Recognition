import cv2
import torch
import numpy as np
import os
import argparse
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import sys

# CONFIG 
EMB_DIR = "datasets/embeddings"
THRESHOLD = 0.5
device = "cpu"

# ALIGN FACE
sys.path.insert(0, './face_alignment')
from mtcnn import MTCNN
mtcnn_detector = MTCNN(device=device, crop_size=(112, 112))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#  EMBEDDING 
def get_embedding_from_pil(pil_img, ort_session):
    tensor = transform(pil_img).unsqueeze(0).numpy()
    
    input_name = ort_session.get_inputs()[0].name
    expected_dtype = ort_session.get_inputs()[0].type

    if "float16" in expected_dtype.lower():
        tensor = tensor.astype(np.float16)
    else:
        tensor = tensor.astype(np.float32)

    ort_inputs = {input_name: tensor}
    emb = ort_session.run(None, ort_inputs)[0]
    norm_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return norm_emb.squeeze(0)

# RECOGNIZE 
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

#  MAIN 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Đường dẫn tới ONNX model (FP32, FP16, INT8)")
    args = parser.parse_args()

    print(f"Loading ONNX model: {args.model}")
    ort_session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])

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
            emb = get_embedding_from_pil(aligned, ort_session)
            name, score = recognize_face(emb)

            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Realtime Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# RUN 
if __name__ == "__main__":
    main()
