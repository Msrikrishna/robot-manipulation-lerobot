"""Training API endpoints for Qualia Studios integration."""

import asyncio

from fastapi import APIRouter, HTTPException

from backend.models.training import (
    GPUInstance,
    QualiaKeyRequest,
    QualiaKeyStatus,
    TrainingJobStatus,
    TrainingRequest,
    TrainingResponse,
)
from backend.services.qualia_service import qualia_service

router = APIRouter()


# The Qualia SDK uses synchronous HTTP. Wrapping every call in asyncio.to_thread
# keeps a slow or hanging Qualia request from blocking the FastAPI event loop —
# without this, status polling stalls every other endpoint (camera streams,
# inference start, etc.) for the duration of each Qualia round-trip.


@router.get("/key-status", response_model=QualiaKeyStatus)
async def get_key_status():
    """Check if a Qualia API key is configured and valid."""
    return await asyncio.to_thread(qualia_service.get_key_status)


@router.post("/validate-key", response_model=QualiaKeyStatus)
async def validate_key(request: QualiaKeyRequest):
    """Validate and save a Qualia API key."""
    return await asyncio.to_thread(qualia_service.validate_key, request.api_key)


@router.get("/instances", response_model=list[GPUInstance])
async def list_instances():
    """List available GPU instances for training."""
    try:
        return await asyncio.to_thread(qualia_service.list_instances)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list instances: {e}")


@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """Start a training job on Qualia."""
    try:
        return await asyncio.to_thread(
            qualia_service.start_training,
            dataset_id=request.dataset_id,
            vla_type=request.vla_type,
            instance_type=request.instance_type,
            batch_size=request.batch_size,
            hours=request.hours,
            output_model_name=request.output_model_name,
            job_description=request.job_description,
            camera_names=request.camera_names,
            model_id=request.model_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {e}")


@router.get("/status/{job_id}", response_model=TrainingJobStatus)
async def get_job_status(job_id: str):
    """Get training job status."""
    try:
        return await asyncio.to_thread(qualia_service.get_job_status, job_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}")


@router.post("/cancel/{job_id}", response_model=TrainingJobStatus)
async def cancel_job(job_id: str):
    """Cancel a training job."""
    try:
        return await asyncio.to_thread(qualia_service.cancel_job, job_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}")
