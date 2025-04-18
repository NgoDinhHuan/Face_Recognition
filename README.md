
# Face Recognition with EdgeFace + ONNX

A real-time face recognition system using the `EdgeFace` model converted to `ONNX`, supporting formats `fp32`, `fp16`, `int8`, and `int4`. It defaults to using **GPU**, falling back to **CPU** if GPU is unavailable.

## 📁 Directory Structure

```
.
├── models/                  # Chứa các model .onnx (fp16, int8...)
├── datasets/
│   ├── images/             # Ảnh gốc để enroll
│   ├── embeddings/         # Vector đã lưu
│   └── test/               # Ảnh test
├── debug_aligned/          # Lưu ảnh đã căn chỉnh
├── face_alignment/         # Thư viện MTCNN + align
├── config.py               # Cấu hình chung (model path, threshold...)
├── enroll.py               # Lưu khuôn mặt vào hệ thống
├── recognize.py            # Kiểm tra khuôn mặt từ ảnh tĩnh
├── camera_recognition.py   # Nhận diện realtime từ webcam
├── model_utils.py          # Hàm load ONNX, chuẩn hóa ảnh
└── requirements.txt        # Thư viện cần cài
```

## ⚙️ Installation

```bash
git clone https://github.com/NgoDinhHuan/Face_Recognition.git
cd Face_Recognition

# Recommended to use virtualenv or conda
pip install -r requirements.txt
```

## 🧪 ONNX Models

Available `.onnx` model files:

- `models/edgeface_fp32.onnx`
- `models/edgeface_fp16.onnx` ✅ **(Default)**
- `models/edgeface_int8.onnx`
- `models/edgeface_int4.onnx` *(if available)*

## ⚙️ Configuration in `config.py`

```python
# config.py
MODEL_TYPE = "fp16"  # "fp32", "fp16", "int8", "int4"
THRESHOLD = 0.5
EMB_DIR = "datasets/embeddings"
IMAGE_DIR = "datasets/images"
DEBUG_ALIGNED_DIR = "debug_aligned"
```

Just change `MODEL_TYPE` or `THRESHOLD` here without modifying individual files.

## 📌 Enroll - Register Faces to the System

```bash
python enroll.py
```

- Images should be placed under `datasets/images/{person_name}/`
- Embeddings will be saved to `datasets/embeddings/{person_name}/`

## 🔍 Recognize - Identify from Static Image

```bash
python recognize.py
```

- The test image is `datasets/test/testok.png`
- The result will print the best matched person

## 🎥 Realtime from Webcam

```bash
python camera_recognition.py
```

Press `Q` to exit.

##  Model Info

Model used from [otroshi/edgeface](https://github.com/otroshi/edgeface) converted to ONNX. Suitable for deployment on edge devices (Jetson, Raspberry Pi, etc.) and API integration.

## 📦 TODO

- [x] Convert ONNX to multiple formats
- [x] Auto-detect GPU if available
- [x] Unify shared configuration
- [ ] Integrate with FastAPI
- [ ] Benchmark performance per model

## ✨ Demo

(Insert demo image here if available)

## 🧑‍💻 Author

**Ngô Đình Huân**  
Project: `Face Recognition with EdgeFace + ONNX + Camera`  
Email: `ngodinhhuan07@gmail.com`
