"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import {
  Loader2,
  Check,
  CircleDot,
  Circle,
  Info,
  AlertTriangle,
  FolderOpen,
  ChevronDown,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getCalibrationPaths, validateBimanualCalibrationNames } from "@/lib/wizard-types";
import { services } from "@/lib/services";
import {
  useManualCalibration,
  type MotorValues,
} from "@/hooks/use-manual-calibration";
import { useAutoCalibration } from "@/hooks/use-auto-calibration";
import { LogViewer } from "@/components/common/log-viewer";
import { DevErrorPanel } from "@/components/common/dev-error-panel";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

const ROLE_LABELS: Record<string, string> = {
  follower: "Follower",
  leader: "Leader",
  left_follower: "Left Follower",
  right_follower: "Right Follower",
  left_leader: "Left Leader",
  right_leader: "Right Leader",
};

const BIMANUAL_PLACEHOLDERS: Record<string, string> = {
  left_follower: "e.g., bimanual_follower_left",
  right_follower: "e.g., bimanual_follower_right",
  left_leader: "e.g., bimanual_leader_left",
  right_leader: "e.g., bimanual_leader_right",
};

// ─── Encoder table component ────────────────────────────────────────────────

function EncoderTable({
  motors,
  positions,
}: {
  motors: string[];
  positions: Record<string, MotorValues>;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border bg-black/80">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="border-b border-white/10 text-green-400/70">
            <th className="px-3 py-2 text-left font-medium">Motor</th>
            <th className="px-3 py-2 text-right font-medium">Min</th>
            <th className="px-3 py-2 text-right font-medium">Current</th>
            <th className="px-3 py-2 text-right font-medium">Max</th>
          </tr>
        </thead>
        <tbody className="text-green-400">
          {motors.map((motor) => {
            const v = positions[motor];
            return (
              <tr key={motor} className="border-b border-white/5 last:border-0">
                <td className="px-3 py-1.5 text-green-300">{motor}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {v?.min ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums font-bold">
                  {v?.pos ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {v?.max ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Phase indicator ─────────────────────────────────────────────────────────

function PhaseIndicator({ phase }: { phase: number }) {
  const steps = ["Set Middle Position", "Record Range of Motion"];
  return (
    <div className="flex items-center gap-3">
      {steps.map((label, i) => {
        const done = i < phase;
        const active = i === phase;
        return (
          <div key={label} className="flex items-center gap-1.5 text-xs">
            {done ? (
              <Check className="h-3.5 w-3.5 text-green-500" />
            ) : active ? (
              <CircleDot className="h-3.5 w-3.5 text-primary" />
            ) : (
              <Circle className="h-3.5 w-3.5 text-muted-foreground/40" />
            )}
            <span
              className={
                active
                  ? "font-medium text-foreground"
                  : done
                    ? "text-green-600 dark:text-green-400"
                    : "text-muted-foreground/60"
              }
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className="ml-1 h-px w-6 bg-muted-foreground/20" />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── New calibration panel ───────────────────────────────────────────────────

function NewCalibrationPanel({
  role,
  category,
  robotType,
}: {
  role: string;
  category: string;
  robotType: string;
}) {
  const { state: wizState, dispatch } = useWizard();
  const cal = useManualCalibration();
  const { state: calState } = cal;

  const name = (wizState.newCalibrationNames[role] || "").trim();
  const port = wizState.portAssignments[role];

  // Keep a ref to calibrationFiles so the effect below doesn't go stale
  const calibrationFilesRef = useRef(wizState.calibrationFiles);
  calibrationFilesRef.current = wizState.calibrationFiles;

  // When calibration is saved, update the wizard selection to the actual filename
  useEffect(() => {
    if (calState.phase === "saved" && name) {
      const key = `${category}/${robotType}`;
      const currentFiles = calibrationFilesRef.current[key] || [];
      if (!currentFiles.includes(name)) {
        dispatch({ type: "SET_CALIBRATION_FILES", key, files: [...currentFiles, name] });
      }
      dispatch({ type: "SET_CALIBRATION_SELECTION", role, filename: name });
    }
  }, [calState.phase, name, role, category, robotType, dispatch]);

  const calPhase = calState.phase;
  const isIdle =
    calPhase === "disconnected" || calPhase === "error" || calPhase === "saved";

  // Determine which step we're in (0 = homing, 1 = recording)
  const uiPhase =
    calPhase === "homing_done" ||
    calPhase === "recording" ||
    calPhase === "saving" ||
    calPhase === "saved"
      ? 1
      : 0;

  function handleConnect() {
    if (!name || !port) return;
    const deviceType = category === "robots" ? "robot" : "teleoperator";
    cal.connect(port, deviceType, robotType, name);
  }

  function handleSetHoming() {
    cal.setHoming();
  }

  function handleStartRecording() {
    cal.startRecording();
  }

  function handleDone() {
    cal.stopAndSave();
  }

  function handleReset() {
    cal.reset();
  }

  return (
    <div className="mt-2 space-y-4 rounded-lg border bg-muted/30 p-4">
      {/* Name input */}
      <div className="space-y-1.5">
        <Label htmlFor={`cal-name-${role}`}>Calibration File Name</Label>
        <Input
          id={`cal-name-${role}`}
          placeholder={BIMANUAL_PLACEHOLDERS[role] || "e.g., my_follower"}
          value={wizState.newCalibrationNames[role] || ""}
          onChange={(e) =>
            dispatch({
              type: "SET_NEW_CALIBRATION_NAME",
              role,
              name: e.target.value,
            })
          }
          disabled={!isIdle}
        />
        <p className="text-xs text-muted-foreground">
          {BIMANUAL_PLACEHOLDERS[role]
            ? `Must end with ${role.startsWith("left") ? '"_left"' : '"_right"'} and share the same prefix as its pair.`
            : "This name will be used as the robot/teleoperator ID."}
        </p>
      </div>

      {!port && (
        <p className="text-xs text-destructive">
          No port assigned for this role. Go back to the Ports step to assign
          one.
        </p>
      )}

      {/* Show phase indicator once connected */}
      {!isIdle && calPhase !== "connecting" && <PhaseIndicator phase={uiPhase} />}

      {/* ── Step 1: Homing ── */}
      {(calPhase === "disconnected" ||
        calPhase === "connecting" ||
        calPhase === "connected" ||
        calPhase === "homing" ||
        calPhase === "error") && (
        <>
          {/* Reference image */}
          <div className="flex flex-col items-center gap-3 rounded-lg border bg-muted/30 p-4">
            <Image
              src="/lerobot-side.png"
              alt="SO-101 arm in middle position (side view)"
              width={800}
              height={640}
              className="h-auto w-full max-w-sm object-contain"
              priority
            />
            <p className="text-center text-sm text-muted-foreground">
              Move the arm to the <strong>middle</strong> of its range of
              motion, matching the reference above, before proceeding.
            </p>
          </div>

          {/* Connect + Set Homing button */}
          {calPhase === "disconnected" || calPhase === "error" ? (
            <Button
              className="w-full"
              disabled={!name || !port}
              onClick={handleConnect}
            >
              Connect &amp; Set Middle Position
            </Button>
          ) : calPhase === "connecting" ? (
            <Button className="w-full" disabled>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Connecting...
            </Button>
          ) : calPhase === "connected" ? (
            <Button className="w-full" onClick={handleSetHoming}>
              Set Middle Position
            </Button>
          ) : calPhase === "homing" ? (
            <Button className="w-full" disabled>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Setting homing offsets...
            </Button>
          ) : null}
        </>
      )}

      {/* ── Step 2: Range Recording ── */}
      {(calPhase === "homing_done" || calPhase === "recording") && (
        <>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Move all joints sequentially through their{" "}
              <strong>entire range of motion</strong>. The table below updates
              live. Click <strong>Done</strong> when finished.
            </p>

            <EncoderTable
              motors={calState.motors}
              positions={calState.positions}
            />
          </div>

          {calPhase === "homing_done" ? (
            <Button className="w-full" onClick={handleStartRecording}>
              Start Recording
            </Button>
          ) : (
            <Button className="w-full" onClick={handleDone}>
              <Check className="mr-2 h-4 w-4" />
              Done
            </Button>
          )}
        </>
      )}

      {/* ── Saving ── */}
      {calPhase === "saving" && (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          <span className="text-sm text-muted-foreground">
            Saving calibration...
          </span>
        </div>
      )}

      {/* ── Saved ── */}
      {calPhase === "saved" && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 rounded-lg border border-green-200 bg-green-50 p-3 dark:border-green-900 dark:bg-green-950">
            <Check className="h-5 w-5 text-green-600 dark:text-green-400" />
            <div>
              <p className="text-sm font-medium text-green-800 dark:text-green-200">
                Calibration saved successfully
              </p>
              {calState.savedPath && (
                <p className="text-xs text-green-600 dark:text-green-400 font-mono break-all">
                  {calState.savedPath}
                </p>
              )}
            </div>
          </div>
          <Button variant="outline" className="w-full" onClick={handleReset}>
            Calibrate Again
          </Button>
        </div>
      )}

      {/* ── Error ── */}
      {calState.error && (
        <div className="space-y-2">
          <DevErrorPanel error={new Error(calState.error)} />
          <p className="text-xs text-muted-foreground px-1">
            This error usually means the arm is not powered on. Make sure the
            motor power supply is connected and switched on, then try again.
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Auto calibration panel ─────────────────────────────────────────────────

function AutoCalibrationPanel({
  role,
  category,
  robotType,
}: {
  role: string;
  category: string;
  robotType: string;
}) {
  const { state: wizState, dispatch } = useWizard();
  const auto = useAutoCalibration();
  const { state: autoState } = auto;

  const name = (wizState.newCalibrationNames[role] || "").trim();
  const port = wizState.portAssignments[role];

  const calibrationFilesRef = useRef(wizState.calibrationFiles);
  calibrationFilesRef.current = wizState.calibrationFiles;

  const [logsOpen, setLogsOpen] = useState(false);
  const isRunning = autoState.phase === "running";
  const isDone = autoState.phase === "stopped";

  // When process completes, copy calibration file and update wizard
  useEffect(() => {
    if (isDone && name) {
      // Copy from so_follower/ to the correct category/robotType folder
      auto.completeAndSave(name, category, robotType);

      const key = `${category}/${robotType}`;
      const currentFiles = calibrationFilesRef.current[key] || [];
      if (!currentFiles.includes(name)) {
        dispatch({ type: "SET_CALIBRATION_FILES", key, files: [...currentFiles, name] });
      }
      dispatch({ type: "SET_CALIBRATION_SELECTION", role, filename: name });
    }
  }, [isDone, name, role, category, robotType, dispatch, auto]);

  function handleStart() {
    if (!name || !port) return;
    auto.start(port, name);
  }

  return (
    <div className="mt-2 space-y-4 rounded-lg border bg-muted/30 p-4">
      {/* Name input */}
      <div className="space-y-1.5">
        <Label htmlFor={`auto-cal-name-${role}`}>Calibration File Name</Label>
        <Input
          id={`auto-cal-name-${role}`}
          placeholder={BIMANUAL_PLACEHOLDERS[role] || "e.g., my_follower"}
          value={wizState.newCalibrationNames[role] || ""}
          onChange={(e) =>
            dispatch({
              type: "SET_NEW_CALIBRATION_NAME",
              role,
              name: e.target.value,
            })
          }
          disabled={isRunning}
        />
        <p className="text-xs text-muted-foreground">
          {BIMANUAL_PLACEHOLDERS[role]
            ? `Must end with ${role.startsWith("left") ? '"_left"' : '"_right"'} and share the same prefix as its pair.`
            : "This name will be used as the robot/teleoperator ID."}
        </p>
      </div>

      {!port && (
        <p className="text-xs text-destructive">
          No port assigned for this role. Go back to the Ports step to assign one.
        </p>
      )}

      {/* Start / Stop buttons */}
      {autoState.phase === "idle" && (
        <Button className="w-full" disabled={!name || !port} onClick={handleStart}>
          Start Auto Calibration
        </Button>
      )}

      {isRunning && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span className="text-sm font-medium">Calibrating all motors...</span>
            <Button
              size="sm"
              variant="destructive"
              className="ml-auto"
              onClick={() => auto.stop()}
            >
              Stop
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            The arm will move automatically. Keep the area clear. This takes about 2 minutes.
          </p>
        </div>
      )}

      {/* Log viewer — collapsible */}
      {autoState.logs.length > 0 && (
        <div className="rounded-lg border">
          <button
            type="button"
            className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setLogsOpen((o) => !o)}
          >
            <span>Logs ({autoState.logs.length})</span>
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${logsOpen ? "rotate-180" : ""}`} />
          </button>
          {logsOpen && (
            <div className="border-t">
              <LogViewer
                logs={autoState.logs}
                isConnected={autoState.isConnected}
                maxHeight="300px"
              />
            </div>
          )}
        </div>
      )}

      {/* Success */}
      {isDone && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 rounded-lg border border-green-200 bg-green-50 p-3 dark:border-green-900 dark:bg-green-950">
            <Check className="h-5 w-5 text-green-600 dark:text-green-400" />
            <div>
              <p className="text-sm font-medium text-green-800 dark:text-green-200">
                Auto-calibration completed successfully
              </p>
              {autoState.savedPath && (
                <p className="text-xs text-green-600 dark:text-green-400 font-mono break-all">
                  {autoState.savedPath}
                </p>
              )}
            </div>
          </div>
          <Button variant="outline" className="w-full" onClick={() => auto.reset()}>
            Calibrate Again
          </Button>
        </div>
      )}

      {/* Error */}
      {autoState.phase === "error" && (
        <div className="space-y-3">
          {autoState.error && <DevErrorPanel error={new Error(autoState.error)} />}
          <p className="text-xs text-muted-foreground px-1">
            Make sure the arm is powered on and the motor power supply is connected.
          </p>
          <Button variant="outline" className="w-full" onClick={() => auto.reset()}>
            Try Again
          </Button>
        </div>
      )}
    </div>
  );
}

// ─── Calibration method picker ──────────────────────────────────────────────

type CalibrationMethod = "manual" | "auto";

function CalibrationMethodPicker({
  role,
  category,
  robotType,
}: {
  role: string;
  category: string;
  robotType: string;
}) {
  const [method, setMethod] = useState<CalibrationMethod>("manual");

  return (
    <div className="mt-2 space-y-3">
      {/* Method toggle */}
      <div className="flex gap-2">
        <Button
          size="sm"
          variant={method === "manual" ? "default" : "outline"}
          onClick={() => setMethod("manual")}
          className="flex-1"
        >
          Manual Calibration
        </Button>
        <Button
          size="sm"
          variant={method === "auto" ? "default" : "outline"}
          onClick={() => setMethod("auto")}
          className="flex-1"
        >
          Auto Calibration
        </Button>
      </div>

      <p className="text-xs text-muted-foreground">
        {method === "manual"
          ? "Manually move the arm through its range of motion to record calibration."
          : "Automatically drive each servo to its physical limits to discover calibration values."}
      </p>

      {method === "manual" ? (
        <NewCalibrationPanel role={role} category={category} robotType={robotType} />
      ) : (
        <AutoCalibrationPanel role={role} category={category} robotType={robotType} />
      )}
    </div>
  );
}

// ─── Main step component ─────────────────────────────────────────────────────

export function CalibrationStep() {
  const { state, dispatch } = useWizard();
  const [loading, setLoading] = useState(false);

  const calPaths = state.robotMode ? getCalibrationPaths(state.robotMode) : [];
  const isBimanual = state.robotMode === "bimanual";

  const allSelected = calPaths.every((p) => {
    const sel = state.calibrationSelections[p.role];
    if (sel === undefined || sel === null) return false;
    if (sel === "new")
      return (state.newCalibrationNames[p.role] || "").trim() !== "";
    return true;
  });

  const validation = isBimanual
    ? validateBimanualCalibrationNames(state.calibrationSelections, state.newCalibrationNames)
    : null;
  const canProceed = allSelected && (validation ? validation.valid : true);

  // Load calibration files for each role on mount
  useEffect(() => {
    if (!state.robotMode) return;

    async function loadFiles() {
      setLoading(true);
      try {
        const paths = getCalibrationPaths(state.robotMode!);
        const seen = new Set<string>();
        for (const p of paths) {
          const key = `${p.category}/${p.robotType}`;
          if (seen.has(key)) continue;
          seen.add(key);
          const files = await services.listCalibrationFiles(
            p.category,
            p.robotType
          );
          dispatch({ type: "SET_CALIBRATION_FILES", key, files });
        }
      } finally {
        setLoading(false);
      }
    }

    loadFiles();
  }, [state.robotMode, dispatch]);

  return (
    <StepCard
      title="Calibration"
      description="Choose an existing calibration file or start a new calibration for each arm."
      nextDisabled={!canProceed}
    >
      <div className="space-y-5">
        {/* Bimanual naming info */}
        {isBimanual && (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              <p>
                Left and right arms of the same type must share the same base ID.
                For example, if the left follower is{" "}
                <code className="text-xs">my_robot_left</code>, the right follower must be{" "}
                <code className="text-xs">my_robot_right</code>.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">
              Loading calibration files...
            </span>
          </div>
        ) : (
          calPaths.map((p) => {
            const key = `${p.category}/${p.robotType}`;
            const files = state.calibrationFiles[key] || [];
            const selection = state.calibrationSelections[p.role];
            const isNew = selection === "new";

            return (
              <div key={p.role} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label>
                    {ROLE_LABELS[p.role]}
                    <span className="ml-2 text-xs font-normal text-muted-foreground">
                      ({p.robotType})
                    </span>
                  </Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 gap-1.5 text-xs text-muted-foreground"
                    onClick={() => services.openCalibrationFolder(p.category, p.robotType)}
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                    Open Folder
                  </Button>
                </div>
                <Select
                  value={selection || ""}
                  onValueChange={(val) =>
                    dispatch({
                      type: "SET_CALIBRATION_SELECTION",
                      role: p.role,
                      filename: val,
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select calibration..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="new">
                      <span className="font-medium">+ New Calibration</span>
                    </SelectItem>
                    {files.map((file) => (
                      <SelectItem key={file} value={file}>
                        {file}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {isNew && (
                  <CalibrationMethodPicker
                    role={p.role}
                    category={p.category}
                    robotType={p.robotType}
                  />
                )}
              </div>
            );
          })
        )}

        {/* Validation errors for bimanual naming */}
        {isBimanual && allSelected && validation && !validation.valid && validation.errors.length > 0 && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              <ul className="list-disc pl-4 space-y-1">
                {validation.errors.map((err) => (
                  <li key={err} className="text-xs">{err}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}
      </div>
    </StepCard>
  );
}
