"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Gamepad2,
  Loader2,
  PlugZap,
  Unplug,
  Wifi,
  WifiOff,
  SearchCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

const DIRECTION_KEYS = new Set(["i", "k", "j", "l", "u", "o", "n", "m"]);

type BaseState = "disconnected" | "connecting" | "connected" | "detecting";

interface BaseControlPanelProps {
  disabled?: boolean;
  disabledReason?: string;
  onConnectionChange?: (connected: boolean) => void;
}

export function BaseControlPanel({
  disabled,
  disabledReason,
  onConnectionChange,
}: BaseControlPanelProps) {
  const { state } = useWizard();
  const [baseState, setBaseState] = useState<BaseState>("disconnected");
  const [selectedPort, setSelectedPort] = useState<string>("");
  const [detectedPort, setDetectedPort] = useState<string | null>(null);
  const [detectMsg, setDetectMsg] = useState<string | null>(null);
  const [speedIndex, setSpeedIndex] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);
  const [activeKeys, setActiveKeys] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pressedRef = useRef<Set<string>>(new Set());

  // Gather available ports from wizard state
  const availablePorts = Object.values(state.portAssignments).filter(Boolean);

  // Notify parent of connection state changes
  const setBaseStateAndNotify = useCallback(
    (newState: BaseState) => {
      setBaseState(newState);
      onConnectionChange?.(newState === "connected");
    },
    [onConnectionChange]
  );

  // Auto-detect base port
  async function handleDetect() {
    if (availablePorts.length === 0) return;
    setBaseState("detecting");
    setDetectMsg(null);
    setError(null);
    try {
      const res = await services.detectBasePort(availablePorts);
      if (res.detected_port) {
        setDetectedPort(res.detected_port);
        setSelectedPort(res.detected_port);
        setDetectMsg(`Found base motors on ${res.detected_port.split(".").pop()}`);
      } else {
        setDetectMsg("No base motors (IDs 7, 8, 9) found on any port");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setBaseState("disconnected");
    }
  }

  // Connect to base
  async function handleConnect() {
    if (!selectedPort) return;
    setBaseState("connecting");
    setError(null);
    try {
      await services.connectBase(selectedPort);
      setBaseStateAndNotify("connected");
      connectWebSocket();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed");
      setBaseState("disconnected");
    }
  }

  // Disconnect
  async function handleDisconnect() {
    disconnectWebSocket();
    try {
      await services.disconnectBase();
    } catch {
      // ignore
    }
    setBaseStateAndNotify("disconnected");
    setActiveKeys(new Set());
  }

  // WebSocket connection for real-time key streaming
  function connectWebSocket() {
    if (wsRef.current) return;
    const ws = new WebSocket(`${WS_BASE}/ws/base-control`);
    wsRef.current = ws;

    ws.onopen = () => setWsConnected(true);
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "status" && msg.speed_index !== undefined) {
          setSpeedIndex(msg.speed_index);
        }
        if (msg.type === "error") {
          setError(msg.message);
          setBaseStateAndNotify("disconnected");
        }
      } catch {
        // ignore
      }
    };
    ws.onclose = () => {
      setWsConnected(false);
      wsRef.current = null;
    };
    ws.onerror = () => {
      setWsConnected(false);
      wsRef.current = null;
    };
  }

  function disconnectWebSocket() {
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ type: "disconnect" }));
      } catch {
        // ignore
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsConnected(false);
  }

  // Send current pressed keys to backend
  const sendKeys = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "keys",
          pressed: Array.from(pressedRef.current),
        })
      );
    }
  }, []);

  // Keyboard event handlers
  useEffect(() => {
    if (baseState !== "connected") return;

    function onKeyDown(e: KeyboardEvent) {
      const key = e.key.toLowerCase();
      if (!DIRECTION_KEYS.has(key)) return;
      e.preventDefault();
      if (!pressedRef.current.has(key)) {
        pressedRef.current.add(key);
        setActiveKeys(new Set(pressedRef.current));
        sendKeys();
      }
    }

    function onKeyUp(e: KeyboardEvent) {
      const key = e.key.toLowerCase();
      if (!DIRECTION_KEYS.has(key)) return;
      e.preventDefault();
      pressedRef.current.delete(key);
      setActiveKeys(new Set(pressedRef.current));
      sendKeys();
    }

    function onBlur() {
      pressedRef.current.clear();
      setActiveKeys(new Set());
      sendKeys();
    }

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [baseState, sendKeys]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectWebSocket();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isConnected = baseState === "connected";
  const speedLabels = ["Slow", "Medium", "Fast"];

  return (
    <div className="space-y-4 rounded-lg border p-4">
      <div className="flex items-center gap-2">
        <Gamepad2 className="h-4 w-4 text-muted-foreground" />
        <p className="text-sm font-medium">Base (Keyboard) Control</p>
        {isConnected && (
          <span className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
            {wsConnected ? (
              <Wifi className="h-3 w-3 text-green-500" />
            ) : (
              <WifiOff className="h-3 w-3 text-red-500" />
            )}
            {wsConnected ? "Live" : "Reconnecting..."}
          </span>
        )}
      </div>

      {/* Disabled overlay message */}
      {disabled && !isConnected && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {disabledReason || "Unavailable while arm teleoperation is running."}
        </p>
      )}

      {/* Port selection + auto-detect */}
      {!isConnected && (
        <div className="space-y-3">
          <div className="flex items-end gap-2">
            <div className="flex-1 space-y-1.5">
              <Label className="text-xs">Motor Controller Port</Label>
              <Select
                value={selectedPort}
                onValueChange={setSelectedPort}
                disabled={disabled}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue placeholder="Select port..." />
                </SelectTrigger>
                <SelectContent>
                  {availablePorts.map((port) => (
                    <SelectItem key={port} value={port} className="text-xs">
                      {port.split(".").pop() || port}
                      {port === detectedPort && " (detected)"}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-8"
              onClick={handleDetect}
              disabled={
                disabled ||
                baseState === "detecting" ||
                availablePorts.length === 0
              }
            >
              {baseState === "detecting" ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <SearchCheck className="mr-1.5 h-3.5 w-3.5" />
              )}
              Auto-Detect
            </Button>
          </div>

          {detectMsg && (
            <p
              className={`text-xs ${detectedPort ? "text-green-600 dark:text-green-400" : "text-muted-foreground"}`}
            >
              {detectMsg}
            </p>
          )}

          {error && (
            <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
          )}

          <Button
            size="sm"
            onClick={handleConnect}
            disabled={
              disabled ||
              !selectedPort ||
              baseState === "connecting" ||
              baseState === "detecting"
            }
          >
            {baseState === "connecting" ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <PlugZap className="mr-1.5 h-3.5 w-3.5" />
            )}
            {baseState === "connecting" ? "Connecting..." : "Connect Base"}
          </Button>
        </div>
      )}

      {/* Connected: keyboard control UI */}
      {isConnected && (
        <div className="space-y-4">
          {error && (
            <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
          )}

          <p className="text-xs text-muted-foreground">
            Click here and use the keyboard to drive the base. Keep this window
            focused.
          </p>

          {/* Speed indicator */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Speed:</span>
            <div className="flex gap-1">
              {speedLabels.map((label, idx) => (
                <span
                  key={label}
                  className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                    idx === speedIndex
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground"
                  }`}
                >
                  {label}
                </span>
              ))}
            </div>
            <span className="text-xs text-muted-foreground ml-1">
              (N/M to change)
            </span>
          </div>

          {/* Keyboard layout */}
          <div className="flex flex-col items-center gap-1.5">
            {/* Row 1: rotation + forward */}
            <div className="flex gap-1.5">
              <KeyCap keyChar="U" label="Rot L" active={activeKeys.has("u")} />
              <KeyCap keyChar="I" label="Fwd" active={activeKeys.has("i")} />
              <KeyCap keyChar="O" label="Rot R" active={activeKeys.has("o")} />
            </div>
            {/* Row 2: left, backward, right */}
            <div className="flex gap-1.5">
              <KeyCap keyChar="J" label="Left" active={activeKeys.has("j")} />
              <KeyCap keyChar="K" label="Back" active={activeKeys.has("k")} />
              <KeyCap keyChar="L" label="Right" active={activeKeys.has("l")} />
            </div>
            {/* Row 3: speed controls */}
            <div className="flex gap-1.5 mt-1">
              <KeyCap
                keyChar="N"
                label="Spd+"
                active={activeKeys.has("n")}
                small
              />
              <KeyCap
                keyChar="M"
                label="Spd-"
                active={activeKeys.has("m")}
                small
              />
            </div>
          </div>

          <Button variant="outline" size="sm" onClick={handleDisconnect}>
            <Unplug className="mr-1.5 h-3.5 w-3.5" />
            Disconnect Base
          </Button>
        </div>
      )}
    </div>
  );
}

function KeyCap({
  keyChar,
  label,
  active,
  small,
}: {
  keyChar: string;
  label: string;
  active: boolean;
  small?: boolean;
}) {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-lg border-2 transition-all select-none ${
        small ? "h-10 w-14" : "h-14 w-16"
      } ${
        active
          ? "border-primary bg-primary/10 shadow-inner scale-95"
          : "border-border bg-card hover:border-muted-foreground/30"
      }`}
    >
      <span
        className={`font-mono font-bold leading-none ${small ? "text-sm" : "text-base"} ${
          active ? "text-primary" : "text-foreground"
        }`}
      >
        {keyChar}
      </span>
      <span
        className={`text-muted-foreground leading-none mt-0.5 ${small ? "text-[9px]" : "text-[10px]"}`}
      >
        {label}
      </span>
    </div>
  );
}
