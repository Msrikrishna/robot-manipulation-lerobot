"""Inference-related models."""

from typing import Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Inference start request."""

    policy_path: str = Field(..., description="Path to the trained policy (local folder or HF repo)")
    repo_id: str = Field(..., description="HuggingFace repo ID for evaluation results (username/eval_dataset)")
    single_task: str = Field(..., description="Task description (should match training task)")
    num_episodes: int = Field(10, description="Number of evaluation episodes")
    episode_time_s: int = Field(50, description="Episode duration in seconds")
    display_data: bool = Field(True, description="Whether to show visualization")


class InferenceResponse(BaseModel):
    """Inference start response."""

    process_id: str = Field(..., description="Process identifier for tracking")
    message: str = Field(..., description="Status message")
