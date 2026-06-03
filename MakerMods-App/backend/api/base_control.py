"""REST API endpoints for base (LeKiwi) motor control."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.base_control import base_control_service
from backend.services.port_lock_manager import PortInUseError, port_lock_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class DetectRequest(BaseModel):
    ports: list[str]


class DetectResponse(BaseModel):
    detected_port: str | None
    message: str


class ConnectRequest(BaseModel):
    port: str


class StatusResponse(BaseModel):
    connected: bool
    port: str | None
    speed_index: int


@router.post("/detect", response_model=DetectResponse)
async def detect_base(req: DetectRequest):
    """Auto-detect which port has base motors (IDs 7, 8, 9)."""
    # Filter out ports that are currently in use
    available = [p for p in req.ports if not port_lock_manager.is_port_busy(p)[0]]
    port = await asyncio.to_thread(base_control_service.detect_base_port, available)
    if port:
        return DetectResponse(detected_port=port, message=f"Base motors found on {port}")
    return DetectResponse(detected_port=None, message="No base motors detected on any port")


@router.post("/connect", response_model=StatusResponse)
async def connect_base(req: ConnectRequest):
    """Connect to base motors on the specified port."""
    try:
        await port_lock_manager.acquire([req.port], owner="base_control", mode="direct")
    except PortInUseError as e:
        raise HTTPException(status_code=409, detail={"message": str(e), "owner": e.owner, "port": e.port})

    try:
        await asyncio.to_thread(base_control_service.connect, req.port)
        return StatusResponse(
            connected=True, port=req.port, speed_index=base_control_service.speed_index,
        )
    except Exception as e:
        await port_lock_manager.release([req.port])
        logger.error(f"Failed to connect base on {req.port}: {e}")
        raise


@router.post("/disconnect", response_model=StatusResponse)
async def disconnect_base():
    """Disconnect from base motors and stop wheels."""
    port = base_control_service.port
    await asyncio.to_thread(base_control_service.disconnect)
    if port:
        await port_lock_manager.release([port])
    return StatusResponse(connected=False, port=None, speed_index=0)


@router.get("/status", response_model=StatusResponse)
async def base_status():
    """Get current base control status."""
    return StatusResponse(
        connected=base_control_service.is_connected,
        port=base_control_service.port,
        speed_index=base_control_service.speed_index,
    )
