import io
import time
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2

# ❗ убрали импорт YOLO сверху

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "yolov8n-seg.pt"
model = None  # ❗ теперь пусто


# -------------------------
# Lazy load модели
# -------------------------
def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        print("Loading YOLO model...")
        model = YOLO(MODEL_NAME)
    return model


# -------------------------
# Health
# -------------------------
@app.get("/")
def root():
    return {"message": "Привет"}


@app.get("/health")
def health():
    return {"status": "OK"}


# -------------------------
# Analyze
# -------------------------
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        start = time.time()

        # читаем картинку
        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img = np.array(pil_img)

        # ❗ грузим модель только здесь
        model = get_model()

        results = model.predict(img, verbose=False)
        result = results[0]

        processing_time = round((time.time() - start) * 1000, 2)

        return {
            "success": True,
            "objects": len(result.boxes) if result.boxes is not None else 0,
            "processing_time_ms": processing_time
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
