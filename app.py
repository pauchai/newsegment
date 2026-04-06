import io
import time
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODEL_NAME = "yolov8n-seg.pt"
model = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Lazy load
# -------------------------
def get_model():
    global model
    if model is None:
        print("🔥 Loading YOLO...")
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
    return {"status": "ok"}

# -------------------------
# Utils (БЕЗ cv2)
# -------------------------
def image_to_base64(img: Image.Image):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# -------------------------
# Main endpoint
# -------------------------
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        model = get_model()

        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")

        # YOLO принимает numpy
        img_np = np.array(pil_img)

        results = model.predict(img_np, imgsz=640, conf=0.25)
        result = results[0]

        # result.plot() возвращает numpy
        plotted = result.plot()

        # обратно в PIL
        output_img = Image.fromarray(plotted)

        return {
            "success": True,
            "image": image_to_base64(output_img)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
