"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bug, RefreshCw, Wifi, WifiOff, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { LogViewer } from "@/components/common/log-viewer";
import { DevErrorPanel } from "@/components/common/dev-error-panel";
import { useWizard } from "../wizard-provider";
import { services } from "@/lib/services";

interface MotorStatus {
  id: number;
  name: string;
  responding: boolean;
  model_number: number | null;
  position: number | null;
  speed: number | null;
  load: number | null;
  voltage: number | null;
  temperature: number | null;
  move: number | null;
}

interface ScanResult {
  port: string;
  connected: boolean;
  baudrate: number | null;
  error: string | null;
  hint: string | null;
  motors: MotorStatus[];
  log: string[];
}

export function DebugPanel() {
  const { state, dispatch } = useWizard();
  const [selectedPort, setSelectedPort] = useState<string>("");
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [loadingPorts, setLoadingPorts] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load ports if not already detected
  const refreshPorts = useCallback(async () => {
    setLoadingPorts(true);
    try {
      const ports = await services.listPorts();
      dispatch({ type: "SET_DETECTED_PORTS", ports });
    } catch {
      // ignore
    } finally {
      setLoadingPorts(false);
    }
  }, [dispatch]);

  useEffect(() => {
    if (state.detectedPorts.length === 0) {
      refreshPorts();
    }
  }, [state.detectedPorts.length, refreshPorts]);

  const doScan = useCallback(async () => {
    if (!selectedPort || scanning) return;
    setScanning(true);
    setError(null);
    try {
      const result = await services.scanMotors(selectedPort);
      setScanResult(result);
      setLogs((prev) => [
        ...prev,
        `--- Scan at ${new Date().toLocaleTimeString()} ---`,
        ...result.log,
      ]);
      if (result.error) {
        const { DevError } = await import("@/lib/services");
        setError(new DevError(result.error, undefined, result.hint ?? undefined));
      }
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setScanning(false);
    }
  }, [selectedPort, scanning]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && selectedPort) {
      intervalRef.current = setInterval(doScan, 2000);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [autoRefresh, selectedPort, doScan]);

  const respondingCount = scanResult?.motors.filter((m) => m.responding).length ?? 0;
  const totalMotors = scanResult?.motors.length ?? 6;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <Bug className="h-5 w-5 text-muted-foreground" />
          <h2 className="text-lg font-semibold">Hardware Diagnostics</h2>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Scan a port to check which servo motors are responding and their current positions.
        </p>
      </div>

      {/* Controls */}
      <Card>
        <CardContent className="pt-6 space-y-3">
          <div className="flex items-end gap-2">
            <div className="flex-1 min-w-0">
              <Label className="mb-1.5 block text-sm">Port</Label>
              <Select value={selectedPort} onValueChange={setSelectedPort}>
                <SelectTrigger className="truncate">
                  <SelectValue placeholder="Select a port..." />
                </SelectTrigger>
                <SelectContent>
                  {state.detectedPorts.map((p) => (
                    <SelectItem key={p.port} value={p.port}>
                      {p.port}
                      {p.description ? ` — ${p.description}` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              onClick={refreshPorts}
              variant="outline"
              size="icon"
              disabled={loadingPorts}
              title="Refresh port list"
              className="shrink-0"
            >
              <RefreshCw className={`h-4 w-4 ${loadingPorts ? "animate-spin" : ""}`} />
            </Button>
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={doScan} disabled={!selectedPort || scanning}>
              {scanning ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 h-4 w-4" />
              )}
              Scan Motors
            </Button>
            <div className="flex items-center gap-2">
              <Switch
                id="auto-refresh"
                checked={autoRefresh}
                onCheckedChange={setAutoRefresh}
                disabled={!selectedPort}
              />
              <Label htmlFor="auto-refresh" className="text-sm">
                Auto-refresh
              </Label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error */}
      <DevErrorPanel error={error} />

      {/* Motor Grid */}
      {scanResult && (
        <>
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">
              {respondingCount}/{totalMotors} motors responding
              {scanResult.baudrate && (
                <span className="ml-2 font-normal text-muted-foreground">
                  @ {(scanResult.baudrate / 1000).toFixed(0)}k baud
                </span>
              )}
            </p>
            {scanResult.connected ? (
              <Badge variant="outline" className="gap-1 text-green-600 border-green-300">
                <Wifi className="h-3 w-3" /> Connected
              </Badge>
            ) : (
              <Badge variant="outline" className="gap-1 text-red-600 border-red-300">
                <WifiOff className="h-3 w-3" /> Disconnected
              </Badge>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {scanResult.motors.map((motor) => (
              <Card
                key={motor.id}
                className={
                  motor.responding
                    ? "border-green-200 bg-green-50"
                    : "border-red-200 bg-red-50"
                }
              >
                <CardHeader className="pb-2 pt-4 px-4">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">
                      {formatMotorName(motor.name)}
                    </CardTitle>
                    <Badge
                      variant={motor.responding ? "default" : "destructive"}
                      className="text-xs"
                    >
                      {motor.responding ? "OK" : "N/A"}
                    </Badge>
                  </div>
                  <CardDescription className="text-xs">
                    ID: {motor.id}
                    {motor.model_number != null && ` · Model: ${motor.model_number}`}
                  </CardDescription>
                </CardHeader>
                <CardContent className="px-4 pb-4">
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    <div>
                      Pos:{" "}
                      <span className="font-mono font-medium text-foreground">
                        {motor.position != null ? motor.position : "—"}
                      </span>
                    </div>
                    <div>
                      Speed:{" "}
                      <span className="font-mono font-medium text-foreground">
                        {motor.speed != null ? motor.speed : "—"}
                      </span>
                    </div>
                    <div>
                      Load:{" "}
                      <span className="font-mono font-medium text-foreground">
                        {motor.load != null ? motor.load : "—"}
                      </span>
                    </div>
                    <div>
                      Volt:{" "}
                      <span className="font-mono font-medium text-foreground">
                        {motor.voltage != null ? `${motor.voltage}V` : "—"}
                      </span>
                    </div>
                    <div>
                      Temp:{" "}
                      <span className="font-mono font-medium text-foreground">
                        {motor.temperature != null ? `${motor.temperature}°C` : "—"}
                      </span>
                    </div>
                    <div>
                      Moving:{" "}
                      <span className="font-mono font-medium text-foreground">
                        {motor.move != null ? (motor.move ? "Yes" : "No") : "—"}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* Log Viewer */}
      {logs.length > 0 && (
        <div>
          <Label className="mb-2 block text-sm">Diagnostic Log</Label>
          <LogViewer
            logs={logs}
            isConnected={autoRefresh}
            onClear={() => setLogs([])}
            maxHeight="200px"
          />
        </div>
      )}
    </div>
  );
}

function formatMotorName(name: string): string {
  return name
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
