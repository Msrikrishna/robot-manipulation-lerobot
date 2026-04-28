"""Training API endpoints for Qualia Studios integration."""

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


@router.get("/key-status", response_model=QualiaKeyStatus)
async def get_key_status():
    """Check if a Qualia API key is configured and valid."""
    return qualia_service.get_key_status()


@router.post("/validate-key", response_model=QualiaKeyStatus)
async def validate_key(request: QualiaKeyRequest):
    """Validate and save a Qualia API key."""
    return qualia_service.validate_key(request.api_key)


@router.get("/instances", response_model=list[GPUInstance])
async def list_instances():
    """List available GPU instances for training."""
    try:
        return qualia_service.list_instances()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list instances: {e}")


@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """Start a training job on Qualia."""
    try:
        return qualia_service.start_training(
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
        return qualia_service.get_job_status(job_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}")


@router.post("/cancel/{job_id}", response_model=TrainingJobStatus)
async def cancel_job(job_id: str):
    """Cancel a training job."""
    try:
        return qualia_service.cancel_job(job_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}")
