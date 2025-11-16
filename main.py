import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import Response

from model_service import YOLOInferenceService

# Initialize FastAPI
app = FastAPI(
    title="YOLO Inference API",
    description="API for YOLO object detection with bounding boxes and attention visualization",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)

instrumentator.instrument(app)

# Custom Prometheus Metrics
INFERENCE_DURATION = Histogram(
    "yolo_inference_duration_seconds", "YOLO inference duration in seconds"
)

ATTENTION_DURATION = Histogram(
    "yolo_attention_duration_seconds", "Attention visualization duration in seconds"
)

DETECTIONS_COUNT = Counter("yolo_detections_total", "Total number of objects detected")

DETECTIONS_PER_CLASS = Counter(
    "yolo_detections_by_class", "Number of detections by class", ["class_name"]
)

MODEL_LOAD_STATUS = Gauge(
    "yolo_model_loaded", "Whether YOLO model is loaded (1=loaded, 0=not loaded)"
)

# Directories
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
ATTENTION_DIR = "gradcam_results"

for directory in [UPLOAD_DIR, RESULT_DIR, ATTENTION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static directories
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")
app.mount(
    "/gradcam_results", StaticFiles(directory=ATTENTION_DIR), name="gradcam_results"
)

# Initialize YOLO Service
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
yolo_service = YOLOInferenceService(MODEL_PATH)

# Update model load status
if yolo_service.model is not None:
    MODEL_LOAD_STATUS.set(1)
else:
    MODEL_LOAD_STATUS.set(0)


def draw_bounding_boxes(image_path, detections, save_dir):
    """
    วาด bounding boxes บนภาพและบันทึก
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    boxes = detections.get("boxes", [])
    labels = detections.get("labels", [])
    scores = detections.get("confidences", detections.get("scores", []))

    # สร้างสีสำหรับแต่ละ class
    np.random.seed(42)
    colors = {}
    for label in set(labels):
        colors[label] = tuple(map(int, np.random.randint(0, 255, 3)))

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (0, 255, 0))

        # วาด bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # เพิ่ม label และ confidence
        text = f"{label} {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # พื้นหลังสำหรับ text
        cv2.rectangle(
            img,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1,
        )

        # วาด text
        cv2.putText(
            img,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # บันทึกภาพ
    file_id = Path(image_path).stem
    output_path = os.path.join(save_dir, f"bbox_{file_id}.png")
    cv2.imwrite(output_path, img)

    return output_path


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "YOLO Inference API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    """Detailed health check"""
    model_loaded = yolo_service.model is not None
    MODEL_LOAD_STATUS.set(1 if model_loaded else 0)

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects และส่งกลับ JSON พร้อม URLs ของรูปภาพ
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    try:
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference with timing
        inference_start = time.time()
        detections = yolo_service.predict(upload_path)
        inference_time = time.time() - inference_start
        INFERENCE_DURATION.observe(inference_time)

        # Update metrics
        DETECTIONS_COUNT.inc(detections["num_detections"])
        for label in detections["labels"]:
            DETECTIONS_PER_CLASS.labels(class_name=label).inc()

        # Generate bounding box image
        bbox_start = time.time()
        bbox_image_path = draw_bounding_boxes(upload_path, detections, RESULT_DIR)
        bbox_time = time.time() - bbox_start

        # Generate attention map
        attention_start = time.time()
        attention_path = yolo_service.generate_attention(
            upload_path, save_dir=ATTENTION_DIR
        )
        attention_time = time.time() - attention_start
        ATTENTION_DURATION.observe(attention_time)

        # สร้าง URLs สำหรับดึงรูปภาพ
        bbox_filename = Path(bbox_image_path).name
        attention_filename = Path(attention_path).name

        bbox_url = f"/results/{bbox_filename}"
        attention_url = f"/gradcam_results/{attention_filename}"

        # Prepare response
        total_time = inference_time + bbox_time + attention_time
        response = {
            "status": "success",
            "file_id": file_id,
            "original_filename": file.filename,
            "detections": detections,
            "bbox_image_url": bbox_url,
            "attention_visualization_url": attention_url,
            "processing_time": {
                "inference_seconds": round(inference_time, 3),
                "bbox_seconds": round(bbox_time, 3),
                "attention_seconds": round(attention_time, 3),
                "total_seconds": round(total_time, 3),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bbox/{file_id}")
async def get_bbox_image(file_id: str):
    """Download bounding box image"""
    bbox_files = list(Path(RESULT_DIR).glob(f"bbox_*{file_id}*.*"))

    if not bbox_files:
        raise HTTPException(status_code=404, detail="Bounding box image not found")

    bbox_path = str(bbox_files[0])
    return FileResponse(
        bbox_path, media_type="image/png", filename=f"bbox_{file_id}.png"
    )


@app.get("/attention/{file_id}")
async def get_attention_image(file_id: str):
    """Download attention visualization image"""
    attention_files = list(Path(ATTENTION_DIR).glob(f"*{file_id}*.*"))

    if not attention_files:
        raise HTTPException(status_code=404, detail="Attention visualization not found")

    attention_path = str(attention_files[0])
    return FileResponse(
        attention_path, media_type="image/png", filename=f"attention_{file_id}.png"
    )


@app.delete("/cleanup")
async def cleanup_files():
    """Clean up temporary files"""
    try:
        for directory in [UPLOAD_DIR, RESULT_DIR, ATTENTION_DIR]:
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)

        return {"status": "success", "message": "All temporary files cleaned"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Starting YOLO Inference API Server")
    print("=" * 70)
    print(f"📍 Model: {MODEL_PATH}")
    print(f"🌐 Server: http://0.0.0.0:8000")
    print(f"📚 Docs: http://0.0.0.0:8000/docs")
    print(f"📊 Metrics: http://0.0.0.0:8000/metrics")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
