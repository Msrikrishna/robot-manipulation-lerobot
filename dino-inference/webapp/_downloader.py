"""Stand-alone HF snapshot download target, run in its own process.

Kept deliberately tiny: it imports only `huggingface_hub`, never torch /
transformers, so the spawned process starts fast and stays light. The parent
(`app.py`) can terminate this process to cancel a download — HF leaves a
`*.incomplete` file behind, so a later attempt resumes instead of restarting.
"""
from __future__ import annotations


def download(repo: str, patterns: list[str], q) -> None:
    """Download `repo` (filtered to `patterns`) and report the outcome on `q`.

    Xet is disabled on purpose: the classic HTTP downloader writes `*.incomplete`
    files progressively into the cache `blobs/` dir, which gives the parent a
    real byte-level progress signal and a resumable partial if we cancel
    mid-download. (Xet stages chunks elsewhere and only materializes the file at
    the end — no visible progress, no classic resume.)
    """
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo, allow_patterns=patterns)
        q.put((True, None))
    except Exception as e:  # auth / gated / network -> surface to the parent
        q.put((False, str(e)))
