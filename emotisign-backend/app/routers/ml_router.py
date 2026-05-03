"""
ml_router.py - API endpoints for ML services (WLASL-100 model and environment validation).

Endpoints:
- POST /api/ml/sign-to-text - Upload video for ASL recognition
- POST /api/ml/validate-environment - Validate recording environment
- GET /api/ml/stats - Get ML service statistics
"""

import os
import tempfile
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from ..ml.wlasl_service import get_wlasl_service, WLASLModelService
from ..services.environment_validator import get_validator, EnvironmentValidator


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ML"])


# Dependency injection
async def get_ml_service_dep() -> WLASLModelService:
    """Dependency to get WLASL model service."""
    return await get_wlasl_service()


def get_validator_dep() -> EnvironmentValidator:
    """Dependency to get environment validator."""
    return get_validator()


@router.post("/sign-to-text")
async def sign_to_text_endpoint(
    video: UploadFile = File(..., description="Video file containing ASL signs"),
    use_tta: bool = Form(True, description="Enable Test-Time Augmentation"),
    ml_service: WLASLModelService = Depends(get_ml_service_dep)
):
    """
    Process uploaded video and return ASL translation.
    
    **Request:**
    - video: multipart/form-data video file (MP4, AVI, MOV, WEBM)
    - use_tta: optional boolean (default: true)
    
    **Response:**
    ```json
    {
        "recognized_text": "hello",
        "glosses": ["hello", "help", "home", "have", "hearing"],
        "confidence": 0.856,
        "top5_predictions": [[0, 0.856], [12, 0.042], ...],
        "frame_count": 87,
        "processing_time_ms": 342,
        "sign_language": "ASL"
    }
    ```
    
    **Errors:**
    - 400: Invalid video format or file too large
    - 500: Processing error
    """
    # Validate file format
    allowed_formats = {".mp4", ".avi", ".mov", ".webm"}
    file_ext = os.path.splitext(video.filename)[1].lower()
    
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {file_ext}. "
                   f"Allowed formats: {', '.join(allowed_formats)}"
        )
    
    # Validate file size (max 50MB)
    max_size_mb = int(os.getenv("MAX_VIDEO_SIZE_MB", "50"))
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await video.read()
            
            # Check file size
            if len(content) > max_size_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {max_size_mb}MB"
                )
            
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info(f"Processing video: {video.filename} ({len(content)} bytes)")
        
        # Process video
        try:
            result = await ml_service.sign_to_text(temp_path, use_tta=use_tta)
            return JSONResponse(content=result)
            
        except FileNotFoundError as e:
            logger.error(f"Video file not found: {e}")
            raise HTTPException(status_code=400, detail="Unable to process video file")
            
        except ValueError as e:
            logger.error(f"Video processing error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
            
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal error during video processing"
            )
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


@router.post("/validate-environment")
async def validate_environment_endpoint(
    frame: UploadFile = File(..., description="Webcam frame image"),
    validator: EnvironmentValidator = Depends(get_validator_dep)
):
    """
    Validate recording environment from webcam frame.
    
    **Request:**
    - frame: image file (JPEG, PNG)
    
    **Response:**
    ```json
    {
        "background": {
            "status": "clear",
            "score": 0.08,
            "message": "Background is clear - excellent!"
        },
        "body": {
            "status": "visible",
            "landmarks_detected": {
                "left_hand": true,
                "right_hand": true,
                "arms": true,
                "torso": true
            },
            "message": "All body parts visible - perfect!"
        },
        "distance": {
            "status": "optimal",
            "shoulder_width_px": 280,
            "message": "Distance is optimal - perfect!"
        },
        "overall": "ready"
    }
    ```
    
    **Errors:**
    - 400: Invalid image format
    - 500: Processing error
    """
    # Validate file format
    allowed_formats = {".jpg", ".jpeg", ".png"}
    file_ext = os.path.splitext(frame.filename)[1].lower()
    
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {file_ext}. "
                   f"Allowed formats: {', '.join(allowed_formats)}"
        )
    
    try:
        # Read image
        content = await frame.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")
        
        # Validate frame
        result = await validator.validate_frame(img)
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal error during environment validation"
        )


@router.get("/stats")
async def get_ml_stats(
    ml_service: WLASLModelService = Depends(get_ml_service_dep)
):
    """
    Get ML service statistics.
    
    **Response:**
    ```json
    {
        "total_predictions": 42,
        "average_inference_time_ms": 287.5,
        "average_confidence": 0.823,
        "model_loaded": true,
        "device": "cuda"
    }
    ```
    """
    stats = ml_service.get_stats()
    return JSONResponse(content=stats)


@router.get("/health")
async def health_check(
    ml_service: WLASLModelService = Depends(get_ml_service_dep)
):
    """
    Health check endpoint for ML service.
    
    **Response:**
    ```json
    {
        "status": "healthy",
        "model_loaded": true,
        "device": "cuda"
    }
    ```
    """
    stats = ml_service.get_stats()
    
    return JSONResponse(content={
        "status": "healthy" if stats["model_loaded"] else "unhealthy",
        "model_loaded": stats["model_loaded"],
        "device": stats["device"]
    })
