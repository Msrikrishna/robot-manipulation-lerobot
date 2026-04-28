"""Base (LeKiwi 3-wheel omnidirectional) control service.

Drives motors 7/8/9 on a Feetech bus in velocity mode.
Accepts key-press state from the frontend via WebSocket and runs
a smooth acceleration/deceleration controller at ~50 Hz.
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

logger = logging.getLogger(__name__)

BASE_MOTOR_IDS = {"base_left_wheel": 7, "base_back_wheel": 8, "base_right_wheel": 9}

TELEOP_KEYS = {
    "forward": "i",
    "backward": "k",
    "left": "j",
    "right": "l",
    "rotate_left": "u",
    "rotate_right": "o",
    "speed_up": "n",
    "speed_down": "m",
}

SPEED_LEVELS = [
    {"xy": 0.1, "theta": 30},
    {"xy": 0.2, "theta": 60},
    {"xy": 0.3, "theta": 90},
]

# Smooth controller parameters
BASE_ACCELERATION_RATE = 10.0
BASE_DECELERATION_RATE = 10.0
BASE_MAX_SPEED = 6.0
MIN_VELOCITY_THRESHOLD = 0.02

# Wheel kinematics
WHEEL_RADIUS = 0.05
BASE_RADIUS = 0.125
WHEEL_ANGLES_DEG = [240, 0, 120]


def _degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    speed_int = int(round(degps * steps_per_deg))
    return max(-0x8000, min(0x7FFF, speed_int))


def _body_to_wheel_raw(x: float, y: float, theta: float, max_raw: int = 3000) -> dict:
    theta_rad = theta * (np.pi / 180.0)
    vel = np.array([x, y, theta_rad])
    angles = np.radians(np.array(WHEEL_ANGLES_DEG) - 90)
    m = np.array([[np.cos(a), np.sin(a), BASE_RADIUS] for a in angles])
    wheel_linear = m.dot(vel)
    wheel_radps = wheel_linear / WHEEL_RADIUS
    wheel_degps = wheel_radps * (180.0 / np.pi)
    steps_per_deg = 4096.0 / 360.0
    raw_floats = [abs(d) * steps_per_deg for d in wheel_degps]
    max_raw_computed = max(raw_floats) if raw_floats else 0
    if max_raw_computed > max_raw:
        wheel_degps = wheel_degps * (max_raw / max_raw_computed)
    wheel_raw = [_degps_to_raw(d) for d in wheel_degps]
    return {
        "base_left_wheel": wheel_raw[0],
        "base_back_wheel": wheel_raw[1],
        "base_right_wheel": wheel_raw[2],
    }


class BaseControlService:
    """Manages a single base-control session: connect, run control loop, disconnect."""

    def __init__(self):
        self.bus: Optional[FeetechMotorsBus] = None
        self.port: Optional[str] = None
        self._pressed_keys: set[str] = set()
        self._speed_index: int = 0
        self._loop_task: Optional[asyncio.Task] = None
        # Smooth controller state
        self._current_speed: float = 0.0
        self._last_time: float = 0.0
        self._last_direction: dict = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self._is_moving: bool = False

    @property
    def is_connected(self) -> bool:
        return self.bus is not None and self.bus.is_connected

    @property
    def speed_index(self) -> int:
        return self._speed_index

    def detect_base_port(self, ports: list[str]) -> Optional[str]:
        """Try each port and return the first one where motors 7/8/9 all respond."""
        for port in ports:
            try:
                bus = FeetechMotorsBus(
                    port=port,
                    motors={
                        name: Motor(mid, "sts3215", MotorNormMode.RANGE_M100_100)
                        for name, mid in BASE_MOTOR_IDS.items()
                    },
                )
                bus.connect()
                # Try reading position — if motors aren't there this will error
                bus.sync_read("Present_Position", list(BASE_MOTOR_IDS.keys()), normalize=False)
                bus.disconnect()
                logger.info(f"Base motors detected on {port}")
                return port
            except Exception as e:
                logger.debug(f"No base motors on {port}: {e}")
                try:
                    bus.disconnect()
                except Exception:
                    pass
        return None

    def connect(self, port: str) -> None:
        if self.is_connected:
            self.disconnect()

        self.bus = FeetechMotorsBus(
            port=port,
            motors={
                name: Motor(mid, "sts3215", MotorNormMode.RANGE_M100_100)
                for name, mid in BASE_MOTOR_IDS.items()
            },
        )
        self.bus.connect()

        # Write default calibration (velocity mode doesn't need real calibration)
        cal = {}
        for name, motor in self.bus.motors.items():
            cal[name] = MotorCalibration(
                id=motor.id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095,
            )
        self.bus.write_calibration(cal)

        # Set velocity mode
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in BASE_MOTOR_IDS:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        self.bus.enable_torque()

        self.port = port
        self._speed_index = 0
        self._current_speed = 0.0
        self._last_time = time.time()
        self._last_direction = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self._is_moving = False
        self._pressed_keys = set()
        logger.info(f"Base connected on {port}")

    def disconnect(self) -> None:
        self._stop_loop()
        if self.bus and self.bus.is_connected:
            try:
                self.bus.sync_write(
                    "Goal_Velocity",
                    {name: 0 for name in BASE_MOTOR_IDS},
                )
            except Exception:
                pass
            try:
                self.bus.disconnect()
            except Exception:
                pass
        self.bus = None
        self.port = None
        logger.info("Base disconnected")

    def update_keys(self, pressed: set[str]) -> None:
        """Update the set of currently pressed keys (called from WebSocket handler)."""
        # Handle speed changes on key-down edges
        if "n" in pressed and "n" not in self._pressed_keys:
            self._speed_index = min(self._speed_index + 1, len(SPEED_LEVELS) - 1)
        if "m" in pressed and "m" not in self._pressed_keys:
            self._speed_index = max(self._speed_index - 1, 0)
        self._pressed_keys = pressed

    def _compute_action(self) -> dict:
        """Run one tick of the smooth controller. Returns body-frame velocities."""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        direction_keys = [
            TELEOP_KEYS["forward"], TELEOP_KEYS["backward"],
            TELEOP_KEYS["left"], TELEOP_KEYS["right"],
            TELEOP_KEYS["rotate_left"], TELEOP_KEYS["rotate_right"],
        ]
        any_pressed = any(k in self._pressed_keys for k in direction_keys)
        action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

        if any_pressed:
            self._is_moving = True
            s = SPEED_LEVELS[self._speed_index]
            if TELEOP_KEYS["forward"] in self._pressed_keys:
                action["x.vel"] += s["xy"]
            if TELEOP_KEYS["backward"] in self._pressed_keys:
                action["x.vel"] -= s["xy"]
            if TELEOP_KEYS["left"] in self._pressed_keys:
                action["y.vel"] += s["xy"]
            if TELEOP_KEYS["right"] in self._pressed_keys:
                action["y.vel"] -= s["xy"]
            if TELEOP_KEYS["rotate_left"] in self._pressed_keys:
                action["theta.vel"] += s["theta"]
            if TELEOP_KEYS["rotate_right"] in self._pressed_keys:
                action["theta.vel"] -= s["theta"]
            self._last_direction = action.copy()
            self._current_speed = min(self._current_speed + BASE_ACCELERATION_RATE * dt, BASE_MAX_SPEED)
        else:
            self._is_moving = False
            if self._current_speed > 0.01 and self._last_direction:
                action = self._last_direction.copy()
            self._current_speed = max(self._current_speed - BASE_DECELERATION_RATE * dt, 0.0)

        # Apply speed scaling
        if action.get("x.vel") or action.get("y.vel") or action.get("theta.vel"):
            for key in action:
                orig = action[key]
                action[key] = orig * self._current_speed
                if self._current_speed > 0.01 and abs(action[key]) < MIN_VELOCITY_THRESHOLD and abs(orig) > 1e-6:
                    action[key] = MIN_VELOCITY_THRESHOLD if orig > 0 else -MIN_VELOCITY_THRESHOLD

        return action

    def _send_action(self, action: dict) -> None:
        if not self.is_connected:
            return
        wheel_cmds = _body_to_wheel_raw(
            action.get("x.vel", 0.0),
            action.get("y.vel", 0.0),
            action.get("theta.vel", 0.0),
        )
        self.bus.sync_write("Goal_Velocity", wheel_cmds)

    async def _control_loop(self) -> None:
        """Run at ~50 Hz, reading key state and sending motor commands."""
        logger.info("Base control loop started")
        try:
            while True:
                action = self._compute_action()
                await asyncio.to_thread(self._send_action, action)
                await asyncio.sleep(1.0 / 50)
        except asyncio.CancelledError:
            pass
        finally:
            # Stop wheels
            if self.is_connected:
                try:
                    await asyncio.to_thread(
                        self.bus.sync_write,
                        "Goal_Velocity",
                        {name: 0 for name in BASE_MOTOR_IDS},
                    )
                except Exception:
                    pass
            logger.info("Base control loop stopped")

    def start_loop(self) -> None:
        if self._loop_task and not self._loop_task.done():
            return
        self._last_time = time.time()
        self._loop_task = asyncio.get_event_loop().create_task(self._control_loop())

    def _stop_loop(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            self._loop_task = None


# Singleton instance
base_control_service = BaseControlService()
