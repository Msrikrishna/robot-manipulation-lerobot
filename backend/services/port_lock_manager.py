"""Port lock manager — prevents concurrent access to serial ports.

Tracks which serial ports are in use and by what feature, providing
acquire/release semantics so that teleoperation, calibration, recording,
etc. never fight over the same port.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PortInUseError(Exception):
    """Raised when a port is already in use by another feature."""

    def __init__(self, port: str, owner: str):
        self.port = port
        self.owner = owner
        super().__init__(f"Port {port} is currently in use by {owner}")


@dataclass
class PortLease:
    """Tracks ownership of a serial port."""

    port: str
    owner: str  # human-readable: "teleoperation", "manual_calibration", etc.
    mode: str  # "direct" (Python bus) or "subprocess" (lerobot CLI)
    acquired_at: datetime = field(default_factory=datetime.now)
    process_id: Optional[str] = None


class PortLockManager:
    """Manages exclusive access to serial ports.

    All-or-nothing acquisition: when acquiring multiple ports, either all
    succeed or none are acquired. Fails immediately with PortInUseError
    rather than blocking.
    """

    def __init__(self):
        self._leases: dict[str, PortLease] = {}  # normalized_port -> lease
        self._lock = asyncio.Lock()
        self._process_ports: dict[str, list[str]] = {}  # process_id -> [normalized_ports]

    @staticmethod
    def _normalize(port: str) -> str:
        """Normalize a port path to avoid double-locking the same device."""
        try:
            return os.path.realpath(port)
        except (OSError, ValueError):
            return port

    async def acquire(
        self,
        ports: list[str],
        owner: str,
        mode: str = "direct",
        process_id: Optional[str] = None,
    ) -> None:
        """Acquire exclusive access to one or more ports.

        Args:
            ports: Serial port paths to acquire.
            owner: Human-readable name of the feature (e.g. "teleoperation").
            mode: "direct" or "subprocess".
            process_id: Optional process ID for subprocess mode.

        Raises:
            PortInUseError: If any port is already in use.
        """
        normalized = [self._normalize(p) for p in ports]

        async with self._lock:
            # Check all ports first (all-or-nothing)
            for norm, raw in zip(normalized, ports):
                if norm in self._leases:
                    existing = self._leases[norm]
                    raise PortInUseError(raw, existing.owner)

            # All clear — acquire all
            for norm, raw in zip(normalized, ports):
                self._leases[norm] = PortLease(
                    port=raw, owner=owner, mode=mode, process_id=process_id,
                )

            if process_id:
                self._process_ports[process_id] = normalized

        logger.info(f"Port lock acquired: {ports} by {owner} ({mode})")

    async def register_process(self, process_id: str, ports: list[str]) -> None:
        """Associate a process ID with already-acquired ports.

        Called after subprocess start to enable release_for_process().
        """
        normalized = [self._normalize(p) for p in ports]

        async with self._lock:
            for norm in normalized:
                if norm in self._leases:
                    self._leases[norm].process_id = process_id
            self._process_ports[process_id] = normalized

    async def release(self, ports: list[str]) -> None:
        """Release one or more ports."""
        normalized = [self._normalize(p) for p in ports]

        async with self._lock:
            for norm in normalized:
                lease = self._leases.pop(norm, None)
                if lease and lease.process_id and lease.process_id in self._process_ports:
                    self._process_ports.pop(lease.process_id, None)

        logger.info(f"Port lock released: {ports}")

    async def release_for_process(self, process_id: str) -> None:
        """Release all ports held by a subprocess."""
        async with self._lock:
            normalized_ports = self._process_ports.pop(process_id, [])
            for norm in normalized_ports:
                self._leases.pop(norm, None)

        if normalized_ports:
            logger.info(f"Port locks released for process {process_id}")

    async def release_all(self) -> None:
        """Release all port locks (called on shutdown)."""
        async with self._lock:
            self._leases.clear()
            self._process_ports.clear()
        logger.info("All port locks released")

    def is_port_busy(self, port: str) -> tuple[bool, Optional[str]]:
        """Check if a port is busy. Non-async for use in sync contexts.

        Returns:
            (is_busy, owner_name) tuple.
        """
        norm = self._normalize(port)
        lease = self._leases.get(norm)
        if lease:
            return True, lease.owner
        return False, None

    def get_status(self) -> dict[str, dict]:
        """Return current port allocations for diagnostics."""
        return {
            lease.port: {
                "owner": lease.owner,
                "mode": lease.mode,
                "acquired_at": lease.acquired_at.isoformat(),
                "process_id": lease.process_id,
            }
            for lease in self._leases.values()
        }

    @asynccontextmanager
    async def hold(self, ports: list[str], owner: str):
        """Async context manager: acquire on enter, release on exit.

        Usage:
            async with port_lock_manager.hold(["/dev/ttyUSB0"], "wiggle"):
                do_stuff()
        """
        await self.acquire(ports, owner, mode="direct")
        try:
            yield
        finally:
            await self.release(ports)


# Global singleton
port_lock_manager = PortLockManager()
