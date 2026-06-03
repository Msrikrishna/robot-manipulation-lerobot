"""HuggingFace service for authentication and repo management."""

import json
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download, list_datasets

from backend.models.recording import HFRepoInfo
from backend.models.system import HFLoginStatus


class HuggingFaceService:
    """Service for HuggingFace Hub integration."""

    def __init__(self):
        """Initialize HuggingFaceService."""
        self.api = HfApi()

    def check_login(self) -> HFLoginStatus:
        """Check if user is authenticated with HuggingFace Hub.

        Uses the huggingface_hub Python API (same method lerobot uses to upload
        datasets) rather than the CLI binary, so it works regardless of whether
        huggingface-cli is on PATH.

        Returns:
            HFLoginStatus with authentication information.
        """
        try:
            info = self.api.whoami()
            username = info.get("name") if isinstance(info, dict) else None
            return HFLoginStatus(is_logged_in=True, username=username)
        except Exception:
            return HFLoginStatus(is_logged_in=False, username=None)

    def list_repos(self, username: str) -> List[HFRepoInfo]:
        """List user's dataset repositories.

        Args:
            username: HuggingFace username.

        Returns:
            List of HFRepoInfo objects.
        """
        try:
            datasets = list_datasets(author=username)

            return [
                HFRepoInfo(
                    repo_id=dataset.id,
                    repo_type="dataset",
                    private=dataset.private,
                    url=f"https://huggingface.co/datasets/{dataset.id}",
                )
                for dataset in datasets
            ]

        except Exception as e:
            print(f"Error listing repos: {e}")
            return []

    def create_repo(self, username: str, repo_name: str, private: bool = False) -> Optional[HFRepoInfo]:
        """Create a new dataset repository.

        Args:
            username: HuggingFace username.
            repo_name: Repository name (without username prefix).
            private: Whether to create private repository.

        Returns:
            HFRepoInfo for created repo, or None if failed.
        """
        repo_id = f"{username}/{repo_name}"

        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            return HFRepoInfo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                url=f"https://huggingface.co/datasets/{repo_id}",
            )

        except Exception as e:
            print(f"Error creating repo: {e}")

        return None

    def get_dataset_image_keys(self, repo_id: str) -> List[str]:
        """Read image/video feature keys from a dataset's meta/info.json.

        Args:
            repo_id: HuggingFace dataset repo ID (e.g. "user/dataset").

        Returns:
            List of image feature keys (e.g. ["observation.images.front_cam"]).
        """
        try:
            path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
            with open(path) as f:
                info = json.load(f)
            features = info.get("features", {})
            return [
                key
                for key, val in features.items()
                if isinstance(val, dict) and val.get("dtype") == "video"
            ]
        except Exception as e:
            print(f"Error reading dataset image keys: {e}")
            return []

    def get_cache_path(self, repo_id: str) -> Optional[str]:
        """Get the local cache path for a dataset.

        Args:
            repo_id: HuggingFace repo ID (username/dataset_name).

        Returns:
            Path to cached dataset, or None if not exists.
        """
        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

        if cache_dir.exists():
            return str(cache_dir)

        return None
