"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export type AutoCalPhase = "idle" | "running" | "stopped" | "error";

export interface AutoCalibrationState {
  phase: AutoCalPhase;
  processId: string | null;
  logs: string[];
  isConnected: boolean;
  error: string | null;
  /** Path where calibration was saved (set after completion) */
  savedPath: string | null;
}

const INITIAL_STATE: AutoCalibrationState = {
  phase: "idle",
  processId: null,
  logs: [],
  isConnected: false,
  error: null,
  savedPath: null,
};

export function useAutoCalibration() {
  const [state, setState] = useState<AutoCalibrationState>(INITIAL_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Clean up WebSocket and polling on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  // Connect WebSocket for log streaming when processId changes
  useEffect(() => {
    const pid = state.processId;
    if (!pid) return;

    const ws = new WebSocket(`${WS_BASE}/ws/logs/${pid}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setState((s) => ({ ...s, isConnected: true }));
    };

    ws.onmessage = (event) => {
      setState((s) => ({
        ...s,
        logs: [...s.logs.slice(-999), event.data],
      }));
    };

    ws.onclose = () => {
      setState((s) => ({ ...s, isConnected: false }));
      wsRef.current = null;
    };

    ws.onerror = () => {
      setState((s) => ({ ...s, isConnected: false }));
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [state.processId]);

  // Poll process status while running
  useEffect(() => {
    const pid = state.processId;
    if (!pid || state.phase !== "running") return;

    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/teleoperation/status/${pid}`);
        if (!res.ok) return;
        const status = await res.json();
        if (status.state === "stopped" || status.state === "error") {
          setState((s) => ({
            ...s,
            phase: status.state === "error" ? "error" : "stopped",
            error: status.state === "error" ? (status.error_message || "Process exited with error") : null,
          }));
        }
      } catch {
        // Ignore polling errors
      }
    }, 1500);

    pollRef.current = poll;
    return () => {
      clearInterval(poll);
      pollRef.current = null;
    };
  }, [state.processId, state.phase]);

  const start = useCallback(async (port: string, deviceId: string) => {
    setState({ ...INITIAL_STATE, phase: "running" });

    try {
      const res = await fetch(`${API_BASE}/api/calibration/auto/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ port, device_id: deviceId }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `API Error: ${res.status}`);
      }

      const data = await res.json();
      setState((s) => ({ ...s, processId: data.process_id }));
    } catch (e) {
      setState((s) => ({
        ...s,
        phase: "error",
        error: e instanceof Error ? e.message : String(e),
      }));
    }
  }, []);

  const stop = useCallback(async () => {
    if (!state.processId) return;
    try {
      await fetch(`${API_BASE}/api/calibration/auto/stop/${state.processId}`, {
        method: "POST",
      });
      setState((s) => ({ ...s, phase: "stopped" }));
    } catch {
      // Ignore
    }
  }, [state.processId]);

  const completeAndSave = useCallback(async (deviceId: string, category: string = "robots", robotType: string = "so101_follower") => {
    try {
      const res = await fetch(`${API_BASE}/api/calibration/auto/complete/${deviceId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, robot_type: robotType }),
      });
      if (res.ok) {
        const data = await res.json();
        setState((s) => ({ ...s, savedPath: data.path }));
      }
    } catch {
      // Non-critical: the so_follower file still exists
    }
  }, []);

  const reset = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setState(INITIAL_STATE);
  }, []);

  return { state, start, stop, completeAndSave, reset };
}
