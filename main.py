import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

# Instrument app (ไม่ expose ที่นี่)
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

# Initialize YOLO Service
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
yolo_service = YOLOInferenceService(MODEL_PATH)

# Update model load status
if yolo_service.model is not None:
    MODEL_LOAD_STATUS.set(1)
else:
    MODEL_LOAD_STATUS.set(0)


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
    """Prometheus metrics endpoint - expose all registered metrics"""
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
    Predict objects in image and return both:
    1. Bounding boxes (JSON)
    2. Attention visualization (image path)
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

        print(f"📥 Received image: {file.filename}")

        # Run inference with timing
        inference_start = time.time()
        detections = yolo_service.predict(upload_path)
        inference_duration = time.time() - inference_start
        INFERENCE_DURATION.observe(inference_duration)

        # Update detection metrics
        num_detections = detections["num_detections"]
        DETECTIONS_COUNT.inc(num_detections)

        for label in detections["labels"]:
            DETECTIONS_PER_CLASS.labels(class_name=label).inc()

        print(f"✅ Detected {num_detections} objects in {inference_duration:.3f}s")

        # Generate attention visualization with timing
        attention_start = time.time()
        attention_path = yolo_service.generate_attention(
            upload_path, save_dir=ATTENTION_DIR
        )
        attention_duration = time.time() - attention_start
        ATTENTION_DURATION.observe(attention_duration)

        print(f"✅ Generated attention map in {attention_duration:.3f}s")

        # Prepare response
        response = {
            "status": "success",
            "file_id": file_id,
            "original_filename": file.filename,
            "detections": detections,
            "attention_visualization": attention_path,
            "processing_time": {
                "inference_seconds": round(inference_duration, 3),
                "attention_seconds": round(attention_duration, 3),
                "total_seconds": round(inference_duration + attention_duration, 3),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attention/{file_id}")
async def get_attention_image(file_id: str):
    """Download attention visualization image"""
    attention_files = list(Path(ATTENTION_DIR).glob(f"attention_*{file_id}*.png"))

    if not attention_files:
        raise HTTPException(status_code=404, detail="Attention visualization not found")

    attention_path = str(attention_files[0])

    return FileResponse(
        attention_path, media_type="image/png", filename=f"attention_{file_id}.png"
    )


@app.post("/predict-with-image")
async def predict_with_image(file: UploadFile = File(...)):
    """Alternative endpoint that returns attention image directly as bytes"""
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        inference_start = time.time()
        detections = yolo_service.predict(upload_path)
        INFERENCE_DURATION.observe(time.time() - inference_start)

        # Update metrics
        DETECTIONS_COUNT.inc(detections["num_detections"])
        for label in detections["labels"]:
            DETECTIONS_PER_CLASS.labels(class_name=label).inc()

        # Generate attention
        attention_start = time.time()
        attention_path = yolo_service.generate_attention(
            upload_path, save_dir=ATTENTION_DIR
        )
        ATTENTION_DURATION.observe(time.time() - attention_start)

        return FileResponse(
            attention_path,
            media_type="image/png",
            headers={
                "X-Detections": str(len(detections["boxes"])),
                "X-File-ID": file_id,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
