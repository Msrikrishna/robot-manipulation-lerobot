"""Debug API endpoints for hardware diagnostics.

Uses the raw feetech-servo-sdk (scservo_sdk) to ping and read each servo
directly, without going through lerobot's abstractions.
"""

import asyncio
import logging
import traceback as tb

from fastapi import APIRouter

from backend.models.debug import MotorScanRequest, MotorScanResponse, MotorStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# SO-101 motor names by ID (1-indexed)
MOTOR_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}

# STS3215 control table register addresses and sizes
SCS_PRESENT_POSITION = 56  # 2 bytes
SCS_PRESENT_SPEED = 58     # 2 bytes
SCS_PRESENT_LOAD = 60      # 2 bytes
SCS_PRESENT_VOLTAGE = 62   # 1 byte
SCS_PRESENT_TEMPERATURE = 63  # 1 byte
SCS_MOVING = 66            # 1 byte

# Common baud rates for STS3215 servos (try in order)
BAUDRATES = [1_000_000, 500_000, 115_200, 57_600]
PROTOCOL_END = 0  # STS/SMS protocol


def _try_ping_any(port_handler, packet_handler, scs, baudrate: int) -> bool:
    """Try pinging motor ID 1 at a given baud rate. Returns True if any response."""
    if not port_handler.setBaudRate(baudrate):
        return False
    _, comm_result, _ = packet_handler.ping(port_handler, 1)
    return comm_result == scs.COMM_SUCCESS


def _read_motors(port_handler, packet_handler, scs, log: list[str]) -> list[MotorStatus]:
    """Ping and read diagnostics for all 6 motors at the current baud rate."""
    motors: list[MotorStatus] = []

    for motor_id in range(1, 7):
        name = MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")
        model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)

        if comm_result != scs.COMM_SUCCESS:
            log.append(
                f"Motor {motor_id} ({name}): no response "
                f"({packet_handler.getTxRxResult(comm_result)})"
            )
            motors.append(MotorStatus(id=motor_id, name=name, responding=False))
            continue

        if error != 0:
            log.append(
                f"Motor {motor_id} ({name}): ping OK but error flag "
                f"({packet_handler.getRxPacketError(error)})"
            )

        log.append(f"Motor {motor_id} ({name}): responding, model={model_number}")

        # Read present position (2 bytes)
        position = None
        pos_val, comm_result, _ = packet_handler.read2ByteTxRx(
            port_handler, motor_id, SCS_PRESENT_POSITION
        )
        if comm_result == scs.COMM_SUCCESS:
            position = scs.SCS_TOHOST(pos_val, 15)
            log.append(f"  Position: {position}")

        # Read present speed (2 bytes)
        speed = None
        spd_val, comm_result, _ = packet_handler.read2ByteTxRx(
            port_handler, motor_id, SCS_PRESENT_SPEED
        )
        if comm_result == scs.COMM_SUCCESS:
            speed = scs.SCS_TOHOST(spd_val, 15)

        # Read present load (2 bytes)
        load = None
        load_val, comm_result, _ = packet_handler.read2ByteTxRx(
            port_handler, motor_id, SCS_PRESENT_LOAD
        )
        if comm_result == scs.COMM_SUCCESS:
            load = scs.SCS_TOHOST(load_val, 10)

        # Read voltage (1 byte) — value is in 0.1V units
        voltage = None
        volt_val, comm_result, _ = packet_handler.read1ByteTxRx(
            port_handler, motor_id, SCS_PRESENT_VOLTAGE
        )
        if comm_result == scs.COMM_SUCCESS:
            voltage = round(volt_val / 10.0, 1)

        # Read temperature (1 byte) — degrees Celsius
        temperature = None
        temp_val, comm_result, _ = packet_handler.read1ByteTxRx(
            port_handler, motor_id, SCS_PRESENT_TEMPERATURE
        )
        if comm_result == scs.COMM_SUCCESS:
            temperature = temp_val

        # Read moving flag (1 byte)
        move = None
        move_val, comm_result, _ = packet_handler.read1ByteTxRx(
            port_handler, motor_id, SCS_MOVING
        )
        if comm_result == scs.COMM_SUCCESS:
            move = move_val

        motors.append(MotorStatus(
            id=motor_id,
            name=name,
            responding=True,
            model_number=model_number,
            position=position,
            speed=speed,
            load=load,
            voltage=voltage,
            temperature=temperature,
            move=move,
        ))

    return motors


def _scan_motors_sync(port: str) -> MotorScanResponse:
    """Scan motors on a port using the raw feetech servo SDK."""
    log: list[str] = []

    try:
        import scservo_sdk as scs
    except ImportError:
        return MotorScanResponse(
            port=port,
            connected=False,
            error="feetech-servo-sdk is not installed",
            hint="Install it with: pip install feetech-servo-sdk",
            motors=[
                MotorStatus(id=i, name=MOTOR_NAMES[i], responding=False)
                for i in range(1, 7)
            ],
            log=log,
        )

    try:
        port_handler = scs.PortHandler(port)
        packet_handler = scs.PacketHandler(PROTOCOL_END)

        if not port_handler.openPort():
            return MotorScanResponse(
                port=port,
                connected=False,
                error=f"Failed to open port {port}",
                hint="Check that the device is plugged in and no other process is using the port.",
                motors=[
                    MotorStatus(id=i, name=MOTOR_NAMES[i], responding=False)
                    for i in range(1, 7)
                ],
                log=log,
            )

        try:
            # Auto-detect baud rate by pinging motor 1 at each candidate
            detected_baud = None
            for baud in BAUDRATES:
                log.append(f"Trying {baud} baud...")
                if _try_ping_any(port_handler, packet_handler, scs, baud):
                    detected_baud = baud
                    log.append(f"Got response at {baud} baud")
                    break

            if detected_baud is None:
                log.append("No response at any baud rate")
                port_handler.closePort()
                log.append("Port closed")
                return MotorScanResponse(
                    port=port,
                    connected=True,
                    error="No motors responded at any baud rate",
                    hint="Check that: (1) the power supply is connected and turned on, "
                         "(2) the servo daisy-chain cable is plugged into the correct port on the driver board, "
                         "(3) at least one servo has its LED blinking on power-up.",
                    motors=[
                        MotorStatus(id=i, name=MOTOR_NAMES[i], responding=False)
                        for i in range(1, 7)
                    ],
                    log=log,
                )

            # Now scan all 6 motors at the detected baud rate
            log.append(f"Scanning all motors at {detected_baud} baud...")
            motors = _read_motors(port_handler, packet_handler, scs, log)

        finally:
            port_handler.closePort()
            log.append("Port closed")

        responding_count = sum(1 for m in motors if m.responding)
        log.append(f"Scan complete: {responding_count}/6 motors responding")

        return MotorScanResponse(
            port=port,
            connected=True,
            baudrate=detected_baud,
            motors=motors,
            log=log,
        )

    except Exception as e:
        log.append(f"Error: {e}")
        log.append(tb.format_exc())
        error_msg = str(e)
        hint = None
        if "Permission denied" in error_msg or "access" in error_msg.lower():
            hint = "Permission denied. Try: sudo chmod 666 " + port
        elif "No such file" in error_msg or "not found" in error_msg.lower():
            hint = "Port not found. Is the device plugged in?"
        elif "busy" in error_msg.lower():
            hint = "Port is busy. Another process may be using it."
        return MotorScanResponse(
            port=port,
            connected=False,
            error=error_msg,
            hint=hint,
            motors=[
                MotorStatus(id=i, name=MOTOR_NAMES[i], responding=False)
                for i in range(1, 7)
            ],
            log=log,
        )


@router.post("/scan-motors", response_model=MotorScanResponse)
async def scan_motors(request: MotorScanRequest):
    """Scan a port for responding servo motors."""
    from backend.services.port_lock_manager import PortInUseError, port_lock_manager

    try:
        async with port_lock_manager.hold([request.port], owner="motor_scan"):
            return await asyncio.wait_for(
                asyncio.to_thread(_scan_motors_sync, request.port),
                timeout=15.0,
            )
    except PortInUseError as e:
        return MotorScanResponse(
            port=request.port,
            connected=False,
            error=str(e),
            hint=f"Port is being used by {e.owner}. Stop it first.",
            motors=[
                MotorStatus(id=i, name=MOTOR_NAMES[i], responding=False)
                for i in range(1, 7)
            ],
            log=[str(e)],
        )
    except asyncio.TimeoutError:
        return MotorScanResponse(
            port=request.port,
            connected=False,
            error="Motor scan timed out after 15 seconds",
            hint="The arm may not be powered on or responding.",
            motors=[
                MotorStatus(id=i, name=MOTOR_NAMES[i], responding=False)
                for i in range(1, 7)
            ],
            log=["Timed out"],
        )
