"""Pydantic models for debug/diagnostics endpoints."""

from typing import List, Optional

from pydantic import BaseModel


class MotorScanRequest(BaseModel):
    """Request to scan motors on a port."""

    port: str


class MotorStatus(BaseModel):
    """Status of a single motor."""

    id: int
    name: str
    responding: bool
    model_number: Optional[int] = None
    position: Optional[int] = None
    speed: Optional[int] = None
    load: Optional[int] = None
    voltage: Optional[float] = None
    temperature: Optional[int] = None
    move: Optional[int] = None


class MotorScanResponse(BaseModel):
    """Response from a motor scan."""

    port: str
    connected: bool
    baudrate: Optional[int] = None
    error: Optional[str] = None
    hint: Optional[str] = None
    motors: List[MotorStatus] = []
    log: List[str] = []
