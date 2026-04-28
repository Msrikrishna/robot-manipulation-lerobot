"""WebSocket endpoint for real-time base keyboard control.

The frontend sends JSON messages with the set of currently pressed keys.
The backend runs a 50 Hz control loop that reads that key state and drives
the three omniwheels via the FeetechMotorsBus.
"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.base_control import base_control_service, SPEED_LEVELS

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/base-control")
async def websocket_base_control(websocket: WebSocket):
    await websocket.accept()

    if not base_control_service.is_connected:
        await websocket.send_json({"type": "error", "message": "Base not connected"})
        await websocket.close()
        return

    # Start the 50 Hz control loop
    base_control_service.start_loop()

    # Send initial status
    await websocket.send_json({
        "type": "status",
        "connected": True,
        "speed_index": base_control_service.speed_index,
        "speed_levels": len(SPEED_LEVELS),
    })

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "keys":
                pressed = set(msg.get("pressed", []))
                base_control_service.update_keys(pressed)
                # Send back speed index if it changed
                await websocket.send_json({
                    "type": "status",
                    "speed_index": base_control_service.speed_index,
                })

            elif msg.get("type") == "disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Base control WebSocket error: {e}")
    finally:
        base_control_service.update_keys(set())
        base_control_service._stop_loop()
