import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

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
    allow_origins=["*"],  # ปรับตามความต้องการ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
ATTENTION_DIR = "gradcam_results"

for directory in [UPLOAD_DIR, RESULT_DIR, ATTENTION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize YOLO Service
MODEL_PATH = "./weights/adamw_d1/best.pt"
yolo_service = YOLOInferenceService(MODEL_PATH)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "YOLO Inference API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects in image and return both:
    1. Bounding boxes (JSON)
    2. Attention visualization (image path)
    """

    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    try:
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📥 Received image: {file.filename}")

        # Run inference
        detections = yolo_service.predict(upload_path)

        # Generate attention visualization
        attention_path = yolo_service.generate_attention(
            upload_path, save_dir=ATTENTION_DIR
        )

        # Prepare response
        response = {
            "status": "success",
            "file_id": file_id,
            "original_filename": file.filename,
            "detections": detections,
            "attention_visualization": attention_path,
            "timestamp": datetime.now().isoformat(),
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Optional: Clean up uploaded file
        # os.remove(upload_path)
        pass


@app.get("/attention/{file_id}")
async def get_attention_image(file_id: str):
    """
    Download attention visualization image
    """
    # Find file in attention directory
    attention_files = list(Path(ATTENTION_DIR).glob(f"attention_*{file_id}*.png"))

    if not attention_files:
        raise HTTPException(status_code=404, detail="Attention visualization not found")

    attention_path = str(attention_files[0])

    return FileResponse(
        attention_path, media_type="image/png", filename=f"attention_{file_id}.png"
    )


@app.post("/predict-with-image")
async def predict_with_image(file: UploadFile = File(...)):
    """
    Alternative endpoint that returns attention image directly as bytes
    """
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    try:
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        detections = yolo_service.predict(upload_path)

        # Generate attention visualization
        attention_path = yolo_service.generate_attention(
            upload_path, save_dir=ATTENTION_DIR
        )

        # Return image file
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
    """
    Clean up temporary files
    """
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
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
