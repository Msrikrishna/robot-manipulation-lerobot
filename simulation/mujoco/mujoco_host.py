#!/usr/bin/env python3
"""
XLeRobot MuJoCo Simulation Host with Remote Control

Bridges the web control server (ZeroMQ ports 5555/5556) to the MuJoCo simulation.
Run this alongside the web server so the browser UI can drive the MuJoCo sim.

Usage:
    cd simulation/mujoco
    python mujoco_host.py [--mjcf scene.xml]
"""

import argparse
import base64
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Always resolve paths relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

import cv2
import glfw
import mujoco
import mujoco_viewer
import numpy as np
import zmq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CMD_ENDPOINT = "tcp://*:5555"
DATA_ENDPOINT = "tcp://*:5556"
CMD_QUEUE_MAXSIZE = 100
RESPONSE_QUEUE_MAXSIZE = 50
ZMQ_TIMEOUT_MS = 100
STATE_PUSH_INTERVAL = 0.05   # 20 Hz
VIDEO_PUSH_INTERVAL = 0.1    # 10 Hz
CONTROL_LOOP_HZ = 60
NETWORK_IDLE_SLEEP = 0.001
OFFSCREEN_WIDTH = 640
OFFSCREEN_HEIGHT = 480


class XLeRobotMuJoCoHost:
    """MuJoCo simulation host with ZeroMQ remote control interface."""

    @staticmethod
    def _extract_command(command: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        if not command:
            return None, {}
        data = command.get("data", {})
        return command.get("command"), data if isinstance(data, dict) else {}

    def __init__(self, mjcf_path: str = "scene.xml"):
        self.mjcf_path = mjcf_path

        self.cmd_queue = queue.Queue(maxsize=CMD_QUEUE_MAXSIZE)
        self.response_queue = queue.Queue(maxsize=RESPONSE_QUEUE_MAXSIZE)
        self.shutdown_event = threading.Event()

        # MuJoCo objects — set in initialize_simulation()
        self.model = None
        self.data = None
        self.viewer = None

        # Control arrays — set in initialize_simulation()
        self.qCmd = None
        self.qdCmd = None
        self.qFb = None
        self.qdFb = None

        self.abs_vel = np.array([1.0, 1.0, 1.0])
        self.chassis_ref_vel = np.zeros(3)
        self.kp = 1
        self.render_freq = 60
        self.render_interval = 1.0 / self.render_freq
        self.last_render_time = time.time()

        self.network_thread = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("XLeRobot MuJoCo Host initialized")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    def initialize_simulation(self) -> bool:
        try:
            logger.info(f"Loading MuJoCo model from {self.mjcf_path}...")
            self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

            # Suppress viewer key callbacks for keys used by robot control
            _conflicting = {
                glfw.KEY_W, glfw.KEY_S, glfw.KEY_D, glfw.KEY_E,
                glfw.KEY_I, glfw.KEY_J, glfw.KEY_O,
                glfw.KEY_R, glfw.KEY_F, glfw.KEY_P,
            }
            _orig_key_callback = self.viewer._key_callback
            def _filtered_key_callback(window, key, scancode, action, mods):
                if key not in _conflicting:
                    _orig_key_callback(window, key, scancode, action, mods)
            glfw.set_key_callback(self.viewer.window, _filtered_key_callback)

            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.model.body("chassis").id
            self.camera.distance = 3.0
            self.camera.azimuth = 90.0
            self.camera.elevation = -30.0
            self.camera.lookat = np.array([0.0, 0.0, 0.0])

            # Renderer created lazily on first video frame request to avoid
            # OpenGL context conflicts with mujoco_viewer on macOS
            self.qCmd = np.zeros(self.model.nu)
            self.qdCmd = np.zeros(self.model.nu)
            self.qFb = np.zeros(self.model.nu)
            self.qdFb = np.zeros(self.model.nu)

            logger.info(f"MuJoCo model loaded: {self.model.nu} actuators, {self.model.nq} qpos")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize simulation: {e}")
            return False

    def update_feedback(self):
        self.qFb = self.data.qpos
        self.qdFb = self.data.qvel

    def update_reference(self):
        yaw = self.qFb[2]
        rotmz = np.array([
            [np.cos(yaw),  np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])
        chassis_vel = rotmz @ self.qdFb[0:3]

        k_p = 10
        k_p_rot = 100
        self.qdCmd[0] = (self.chassis_ref_vel[0] * np.cos(yaw) +
                         self.chassis_ref_vel[1] * np.cos(yaw + 1.5708) +
                         k_p * (self.chassis_ref_vel[0] - chassis_vel[0]) * np.cos(yaw) +
                         k_p * (self.chassis_ref_vel[1] - chassis_vel[1]) * np.cos(yaw + 1.5708))
        self.qdCmd[1] = (self.chassis_ref_vel[0] * np.sin(yaw) +
                         self.chassis_ref_vel[1] * np.sin(yaw + 1.5708) +
                         k_p * (self.chassis_ref_vel[0] - chassis_vel[0]) * np.sin(yaw) +
                         k_p * (self.chassis_ref_vel[1] - chassis_vel[1]) * np.sin(yaw + 1.5708))
        self.qdCmd[2] = self.chassis_ref_vel[2] + k_p_rot * (self.chassis_ref_vel[2] - chassis_vel[2])

        radius = 0.1
        vel2wheel_matrix = np.array([
            [0,                -np.sqrt(3) * 0.5,  np.sqrt(3) * 0.5],
            [1,                -0.5,               -0.5],
            [-radius,          -radius,             -radius],
        ]).T
        coe_vel_to_wheel = 20
        self.qCmd[15:18] = coe_vel_to_wheel * np.dot(vel2wheel_matrix, chassis_vel)
        self.qdCmd[2] = np.clip(self.qdCmd[2], -1.0, 1.0)

        # Keep wrist joints at zero
        self.qCmd[6:8] = 0.0
        self.qCmd[12:14] = 0.0

    def update_control(self):
        self.qdCmd[0:3] = self.kp * self.qdCmd[0:3]
        self.data.ctrl[:3] = self.qdCmd[:3]
        self.data.ctrl[3:] = self.qCmd[3:]

    def handle_move_command(self, direction: str, speed: float):
        speed = max(0.0, min(1.0, speed))
        if direction == "forward":
            self.chassis_ref_vel[0] = +self.abs_vel[0] * speed
            self.chassis_ref_vel[1] = 0.0
        elif direction == "backward":
            self.chassis_ref_vel[0] = -self.abs_vel[0] * speed
            self.chassis_ref_vel[1] = 0.0
        elif direction in ("left", "rotate_left"):
            self.chassis_ref_vel[1] = +self.abs_vel[1] * speed
            self.chassis_ref_vel[0] = 0.0
        elif direction in ("right", "rotate_right"):
            self.chassis_ref_vel[1] = -self.abs_vel[1] * speed
            self.chassis_ref_vel[0] = 0.0
        elif direction == "stop":
            self.chassis_ref_vel[:] = 0.0
        else:
            logger.warning(f"Unknown move direction: {direction}")

    def get_robot_state(self) -> Dict[str, Any]:
        return {
            "position": {"x": float(self.data.qpos[0]), "y": float(self.data.qpos[1]), "z": 0.0},
            "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": float(self.data.qpos[2])},
            "arm_joints": {
                "left":  self.qCmd[3:9].tolist(),
                "right": self.qCmd[9:15].tolist(),
            },
            "status": "connected",
            "timestamp": time.time(),
        }

    def get_video_frame(self) -> Optional[str]:
        try:
            h = self.viewer.viewport.height
            w = self.viewer.viewport.width
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, self.viewer.viewport, self.viewer.ctx)
            rgb = np.flipud(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            success, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                return base64.b64encode(buf).decode('utf-8')
        except Exception as e:
            logger.error(f"Video frame error: {e}")
        return None

    def reset_robot_state(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.qCmd[:] = 0.0
        self.qdCmd[:] = 0.0
        self.chassis_ref_vel[:] = 0.0
        logger.info("Robot state reset")

    def render_ui(self):
        current_time = time.time()
        if current_time - self.last_render_time >= self.render_interval:
            self.viewer.cam = self.camera
            self.viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPLEFT] = [
                f"Time: {self.data.time:.3f} sec",
                "",
            ]
            self.viewer._overlay[mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT] = [
                f"Chassis cmd: [{self.qdCmd[0]:.2f}, {self.qdCmd[1]:.2f}, {self.qdCmd[2]:.2f}]\n"
                f"Chassis fb:  [{self.qdFb[0]:.2f}, {self.qdFb[1]:.2f}, {self.qdFb[2]:.2f}]\n"
                f"Left Arm:    [{self.qCmd[3]:.2f}, {self.qCmd[4]:.2f}, {self.qCmd[5]:.2f}]\n"
                f"Right Arm:   [{self.qCmd[9]:.2f}, {self.qCmd[10]:.2f}, {self.qCmd[11]:.2f}]",
                "",
            ]
            self.viewer.render()
            self.last_render_time = current_time

    def send_response(self, response_type: str, data: Any):
        response = {
            "type": "response",
            "response": response_type,
            "data": data,
            "timestamp": time.time(),
        }
        try:
            self.response_queue.put_nowait(response)
        except queue.Full:
            # Evict oldest to make room — latest state/video is always preferred
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.response_queue.put_nowait(response)
            except queue.Full:
                pass  # extremely unlikely — silently drop

    def handle_command(self, command: Dict[str, Any]):
        cmd_type, data = self._extract_command(command)
        if not cmd_type:
            logger.warning("Received command without type")
            return
        try:
            if cmd_type == "move":
                self.handle_move_command(data.get("direction", "stop"), data.get("speed", 1.0))
            elif cmd_type == "stop":
                self.chassis_ref_vel[:] = 0.0
            elif cmd_type == "reset":
                self.reset_robot_state()
            elif cmd_type == "ping":
                self.send_response("pong", {"timestamp": time.time()})
            elif cmd_type == "get_state":
                self.send_response("state", self.get_robot_state())
            elif cmd_type == "set_arm_joint":
                arm = data.get("arm", "left")
                idx = int(data.get("joint_index", 0))
                angle = float(data.get("angle", 0.0))
                base = 3 if arm == "left" else 9
                if 0 <= idx < 6:
                    self.qCmd[base + idx] = angle
            elif cmd_type in ("reset_camera", "set_camera_position"):
                pass  # camera is always tracking chassis
            else:
                logger.warning(f"Unknown command: {cmd_type}")
        except Exception as e:
            logger.error(f"Error handling command {cmd_type}: {e}")

    def process_commands(self):
        while not self.cmd_queue.empty():
            try:
                command = self.cmd_queue.get_nowait()
                self.handle_command(command)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Command processing error: {e}")

    def network_communication_thread(self):
        logger.info("Starting network communication thread...")
        context = zmq.Context.instance()
        cmd_socket = None
        data_socket = None
        try:
            cmd_socket = context.socket(zmq.PULL)
            cmd_socket.setsockopt(zmq.CONFLATE, 1)
            cmd_socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
            cmd_socket.bind(CMD_ENDPOINT)

            data_socket = context.socket(zmq.PUSH)
            data_socket.setsockopt(zmq.CONFLATE, 1)
            data_socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)
            data_socket.bind(DATA_ENDPOINT)

            logger.info("Network sockets bound (%s commands, %s telemetry)", CMD_ENDPOINT, DATA_ENDPOINT)

            while not self.shutdown_event.is_set():
                # --- RECEIVE: poll for incoming commands from server (port 5555) ---
                try:
                    cmd_data = cmd_socket.recv_string(zmq.NOBLOCK)
                    command = json.loads(cmd_data)
                    self.cmd_queue.put_nowait(command)
                    cmd_type = command.get('command', 'unknown')
                    cmd_payload = command.get('data', {})
                    logger.info(f"[ZMQ ←5555] ACTION received: '{cmd_type}' | data={cmd_payload}")
                except zmq.Again:
                    pass  # no command waiting — normal
                except json.JSONDecodeError as e:
                    logger.warning(f"[ZMQ ←5555] Invalid JSON: {e}")
                except queue.Full:
                    logger.warning("[ZMQ ←5555] Command queue full, dropping command")

                # --- SEND: flush one response/frame to server (port 5556) ---
                try:
                    response = self.response_queue.get_nowait()
                    data_socket.send_string(json.dumps(response), zmq.NOBLOCK)
                    rtype = response.get('response', 'unknown')
                    if rtype == 'video':
                        frame_kb = len(response.get('data', {}).get('frame', '')) * 3 // 4 // 1024
                        logger.info(f"[ZMQ →5556] VIDEO  sent: {frame_kb} KB JPEG frame")
                    elif rtype == 'state':
                        pos = response.get('data', {}).get('position', {})
                        logger.info(f"[ZMQ →5556] STATE  sent: pos=({pos.get('x',0):.2f}, {pos.get('y',0):.2f})")
                    else:
                        logger.info(f"[ZMQ →5556] RESP   sent: '{rtype}'")
                except queue.Empty:
                    pass  # nothing to send — normal
                except zmq.Again:
                    # CONFLATE=1 means only latest matters — just drop, never re-queue
                    logger.debug("[ZMQ →5556] Send buffer full, dropping stale response")

                time.sleep(NETWORK_IDLE_SLEEP)

        except Exception as e:
            logger.error(f"Network communication error: {e}")
        finally:
            if cmd_socket is not None:
                cmd_socket.close(0)
            if data_socket is not None:
                data_socket.close(0)
            logger.info("Network communication thread stopped")

    def main_simulation_loop(self):
        logger.info("Starting main simulation loop...")
        last_state_time = 0.0
        last_video_time = time.time()  # don't capture video before viewer is warmed up
        loop_count = 0

        try:
            while not self.shutdown_event.is_set() and self.viewer.is_alive:
                self.update_feedback()
                self.process_commands()
                self.update_reference()
                self.update_control()
                mujoco.mj_step(self.model, self.data)
                self.render_ui()

                now = time.time()

                # Capture video right after render_ui() while GLFW context is current
                if now - last_video_time >= VIDEO_PUSH_INTERVAL:
                    frame = self.get_video_frame()
                    if frame:
                        self.send_response("video", {
                            "frame": frame,
                            "width": OFFSCREEN_WIDTH,
                            "height": OFFSCREEN_HEIGHT,
                            "quality": 80,
                            "camera_id": "main",
                            "format": "jpeg",
                        })
                    last_video_time = now

                if now - last_state_time >= STATE_PUSH_INTERVAL:
                    self.send_response("state", self.get_robot_state())
                    last_state_time = now

                time.sleep(0.002)

                loop_count += 1

        except Exception as e:
            logger.error(f"Main simulation loop error: {e}")
        finally:
            logger.info(f"Main simulation loop stopped after {loop_count} iterations")

    def run(self) -> int:
        logger.info("Starting XLeRobot MuJoCo Host...")
        try:
            if not self.initialize_simulation():
                logger.error("Failed to initialize simulation, exiting")
                return 1

            self.network_thread = threading.Thread(
                target=self.network_communication_thread,
                name="NetworkThread",
                daemon=True,
            )
            self.network_thread.start()
            logger.info("Network thread started")

            self.main_simulation_loop()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return 1
        finally:
            self.cleanup()

        return 0

    def cleanup(self):
        logger.info("Starting cleanup...")
        self.shutdown_event.set()

        if self.network_thread and self.network_thread.is_alive():
            self.network_thread.join(timeout=2.0)
            if self.network_thread.is_alive():
                logger.warning("Network thread did not shut down cleanly")

        if self.viewer:
            try:
                self.viewer.close()
            except Exception:
                pass

        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="XLeRobot MuJoCo ZeroMQ Host")
    parser.add_argument("--mjcf", default="scene.xml", help="Path to MJCF scene file")
    args = parser.parse_args()

    host = XLeRobotMuJoCoHost(mjcf_path=args.mjcf)
    return host.run()


if __name__ == "__main__":
    sys.exit(main())
