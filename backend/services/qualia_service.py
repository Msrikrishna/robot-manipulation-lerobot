"""Qualia Studios integration service for model training."""

import os
from pathlib import Path
from typing import Optional

from backend.models.training import (
    GPUInstance,
    InstanceSpecs,
    QualiaKeyStatus,
    TrainingJobStatus,
    TrainingResponse,
)

# Camera mapping: dataset image key suffix → Qualia camera view slot
# Dataset keys look like "observation.images.front_cam"
CAMERA_MAPPING = {
    "front_cam": "image_top",
    "hand_cam": "image_wrist",
    "side_cam": "image_side",
}

# Path to .env file for persisting the API key (gitignored)
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def _load_api_key() -> Optional[str]:
    """Load API key from environment or .env file."""
    key = os.environ.get("QUALIA_API_KEY")
    if key:
        return key
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("QUALIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def _save_api_key(api_key: str) -> None:
    """Persist API key to .env file (gitignored)."""
    lines: list[str] = []
    found = False
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            if line.strip().startswith("QUALIA_API_KEY="):
                lines.append(f"QUALIA_API_KEY={api_key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"QUALIA_API_KEY={api_key}")
    _ENV_PATH.write_text("\n".join(lines) + "\n")
    os.environ["QUALIA_API_KEY"] = api_key


class QualiaService:
    """Wraps the Qualia SDK for training operations."""

    def _get_client(self, api_key: Optional[str] = None):
        """Get an authenticated Qualia client."""
        try:
            from qualia import Qualia
        except ImportError:
            raise RuntimeError(
                "qualia-sdk is not installed. Run: pip install qualia-sdk"
            )
        key = api_key or _load_api_key()
        if not key:
            raise RuntimeError("No Qualia API key configured")
        return Qualia(api_key=key)

    def validate_key(self, api_key: str) -> QualiaKeyStatus:
        """Validate an API key by attempting to list instances."""
        try:
            client = self._get_client(api_key)
            # A simple API call to verify the key works
            client.instances.list()
            _save_api_key(api_key)
            return QualiaKeyStatus(is_valid=True, message="API key is valid")
        except RuntimeError as e:
            return QualiaKeyStatus(is_valid=False, message=str(e))
        except Exception as e:
            return QualiaKeyStatus(is_valid=False, message=f"Invalid API key: {e}")

    def get_key_status(self) -> QualiaKeyStatus:
        """Check if we have a saved valid API key."""
        key = _load_api_key()
        if not key:
            return QualiaKeyStatus(is_valid=False, message="No API key configured")
        try:
            client = self._get_client(key)
            client.instances.list()
            return QualiaKeyStatus(is_valid=True, message="API key is valid")
        except Exception:
            return QualiaKeyStatus(
                is_valid=False, message="Saved API key is no longer valid"
            )

    def list_instances(self) -> list[GPUInstance]:
        """List available GPU instances."""
        client = self._get_client()
        raw_instances = client.instances.list()
        result: list[GPUInstance] = []
        for inst in raw_instances:
            regions = []
            if hasattr(inst, "regions") and inst.regions:
                regions = [r.name if hasattr(r, "name") else str(r) for r in inst.regions]
            specs = InstanceSpecs()
            if hasattr(inst, "specs") and inst.specs:
                s = inst.specs
                specs = InstanceSpecs(
                    vcpus=getattr(s, "vcpus", 0),
                    memory_gib=getattr(s, "memory_gib", 0),
                    storage_gib=getattr(s, "storage_gib", 0),
                    gpu_count=getattr(s, "gpu_count", 0),
                    gpu_type=getattr(s, "gpu_type", ""),
                )
            result.append(
                GPUInstance(
                    id=inst.id,
                    name=getattr(inst, "name", inst.id),
                    description=getattr(inst, "description", ""),
                    gpu_description=getattr(inst, "gpu_description", ""),
                    credits_per_hour=getattr(inst, "credits_per_hour", 0),
                    specs=specs,
                    regions=regions,
                )
            )
        return result

    def start_training(
        self,
        dataset_id: str,
        vla_type: str,
        instance_type: str,
        batch_size: int,
        hours: float,
        output_model_name: str,
        job_description: str,
        camera_names: list[str],
        model_id: Optional[str] = None,
    ) -> TrainingResponse:
        """Create a project and start a training job."""
        client = self._get_client()

        # Build camera mappings: Qualia slot → dataset image key
        # camera_names are full keys from the dataset (e.g. "observation.images.front_cam")
        camera_mappings = {}
        for full_key in camera_names:
            # Extract the short name (e.g. "front_cam" from "observation.images.front_cam")
            short_name = full_key.rsplit(".", 1)[-1] if "." in full_key else full_key
            qualia_slot = CAMERA_MAPPING.get(short_name)
            if qualia_slot:
                camera_mappings[qualia_slot] = full_key

        # Create a new project for this training job
        project_name = output_model_name or f"training-{dataset_id.split('/')[-1]}"
        project = client.projects.create(
            name=project_name,
            description=job_description or f"Training {vla_type} on {dataset_id}",
        )
        project_id = str(project.project_id)

        # Build finetune create kwargs
        create_kwargs = {
            "project_id": project_id,
            "vla_type": vla_type,
            "dataset_id": dataset_id,
            "hours": hours,
            "camera_mappings": camera_mappings,
            "instance_type": instance_type,
            "batch_size": batch_size,
            "name": job_description or output_model_name,
        }
        # model_id is required for smolvla, pi0, pi05 — must NOT be sent for act, gr00t_n1_5
        if model_id:
            create_kwargs["model_id"] = model_id

        job = client.finetune.create(**create_kwargs)

        return TrainingResponse(
            job_id=str(job.job_id),
            project_id=project_id,
            message="Training job submitted successfully",
        )

    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        """Get the status of a training job."""
        client = self._get_client()
        status = client.finetune.get(job_id)

        # Extract output model ID from model_uploading phase completed event
        # Event message format: "Model successfully uploaded to qualia-robotics/act-xxx"
        output_model_id = None
        for phase in getattr(status, "phases", []):
            if phase.name == "model_uploading":
                for event in phase.events:
                    msg = event.message or ""
                    if "uploaded to " in msg:
                        output_model_id = msg.split("uploaded to ")[-1].strip()

        return TrainingJobStatus(
            job_id=job_id,
            status=getattr(status, "status", "unknown"),
            phase=getattr(status, "current_phase", ""),
            message=str(status),
            output_model_id=output_model_id,
        )

    def cancel_job(self, job_id: str) -> TrainingJobStatus:
        """Cancel a training job."""
        client = self._get_client()
        result = client.finetune.cancel(job_id)
        return TrainingJobStatus(
            job_id=job_id,
            status="cancelled",
            phase="",
            message=str(result),
        )


qualia_service = QualiaService()
