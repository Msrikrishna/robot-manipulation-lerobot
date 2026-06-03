"""Training-related models for Qualia Studios integration."""

from typing import Optional

from pydantic import BaseModel, Field


class QualiaKeyRequest(BaseModel):
    """Request to validate/save a Qualia API key."""

    api_key: str = Field(..., description="Qualia Studios API key")


class QualiaKeyStatus(BaseModel):
    """Qualia API key status."""

    is_valid: bool = Field(..., description="Whether the API key is valid")
    message: str = Field("", description="Status message")


class InstanceSpecs(BaseModel):
    """GPU instance hardware specifications."""

    vcpus: int = 0
    memory_gib: float = 0
    storage_gib: float = 0
    gpu_count: int = 0
    gpu_type: str = ""


class GPUInstance(BaseModel):
    """Available GPU instance for training."""

    id: str = Field(..., description="Instance type ID (e.g. gpu_1x_a100)")
    name: str = Field(..., description="Display name (e.g. A100 80GB)")
    description: str = Field("", description="Instance description")
    gpu_description: str = Field("", description="GPU hardware details")
    credits_per_hour: float = Field(..., description="Cost in credits per hour")
    specs: InstanceSpecs = Field(default_factory=InstanceSpecs)
    regions: list[str] = Field(default_factory=list, description="Available region names")


class TrainingRequest(BaseModel):
    """Request to start a training job."""

    dataset_id: str = Field(..., description="HuggingFace dataset repo ID")
    vla_type: str = Field("act", description="Model type: act, smolvla, pi0, etc.")
    model_id: Optional[str] = Field(
        None, description="HF base model ID, required for smolvla/pi0/pi05"
    )
    instance_type: str = Field(..., description="GPU instance type ID")
    batch_size: int = Field(32, description="Training batch size")
    hours: float = Field(1.0, description="Training duration in hours (max 168)")
    output_model_name: str = Field(..., description="Name for the output model")
    job_description: str = Field("", description="Human-readable job description")
    camera_names: list[str] = Field(
        default_factory=list,
        description="Camera names from the dataset (e.g. front_cam, hand_cam, side_cam)",
    )


class TrainingJobStatus(BaseModel):
    """Training job status from Qualia."""

    job_id: str = Field(..., description="Qualia job ID")
    project_id: str = Field("", description="Qualia project ID")
    status: str = Field(..., description="Current job status")
    phase: str = Field("", description="Current job phase")
    message: str = Field("", description="Status message")
    output_model_id: Optional[str] = Field(
        None, description="HuggingFace repo ID of the trained model (available after completion)"
    )


class TrainingResponse(BaseModel):
    """Response after starting a training job."""

    job_id: str = Field(..., description="Qualia job ID")
    project_id: str = Field(..., description="Qualia project ID")
    message: str = Field(..., description="Status message")
