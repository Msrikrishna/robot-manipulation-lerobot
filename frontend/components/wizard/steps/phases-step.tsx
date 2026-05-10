"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Clock,
  Home,
  Loader2,
  Pencil,
  Play,
  PlayCircle,
  Plus,
  Square,
  Trash2,
  XCircle,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  formatPhaseDuration,
  getPhaseTimeSeconds,
} from "@/lib/wizard-types";
import { services } from "@/lib/services";
import { useWebSocket } from "@/hooks/use-websocket";
import { LogViewer } from "@/components/common/log-viewer";
import { CameraFeedPanel } from "@/components/common/robot-display";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";
import { InferenceStep } from "./inference-step";

const PHASE_SETTLE_MS = 2500; // wait between sequenced phases for cameras to release cleanly
const RETRY_DELAY_MS = 3500; // extra wait before retrying after a camera fps error
const PHASE_MAX_RETRIES = 1; // auto-retries per phase on transient camera errors

const CAMERA_TRANSIENT_ERROR_PATTERNS = [
  /failed to set fps/i,
  /OpenCVCamera\(\d+\) failed to set fps/i,
];

function isCameraTransientError(logLines: string[]): boolean {
  return logLines.some((line) =>
    CAMERA_TRANSIENT_ERROR_PATTERNS.some((p) => p.test(line))
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function withEvalPrefix(repoId: string): string {
  const trimmed = repoId.trim();
  if (!trimmed) return trimmed;
  const slashIdx = trimmed.lastIndexOf("/");
  if (slashIdx === -1) {
    return trimmed.startsWith("eval_") ? trimmed : `eval_${trimmed}`;
  }
  const owner = trimmed.slice(0, slashIdx);
  const name = trimmed.slice(slashIdx + 1);
  if (!name) return trimmed;
  return name.startsWith("eval_") ? trimmed : `${owner}/eval_${name}`;
}

export function PhasesStep() {
  const { state, dispatch } = useWizard();
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [returningHome, setReturningHome] = useState(false);
  const [homeMsg, setHomeMsg] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const [sequenceMode, setSequenceMode] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runningIndex, setRunningIndex] = useState<number | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [lastFailedProcessId, setLastFailedProcessId] = useState<string | null>(null);

  // Stream logs from whichever inference process is active (running OR last failed).
  // Keeping the processId around after failure lets us still pull the buffered logs.
  const logsProcessId = state.inferenceProcessId ?? lastFailedProcessId;
  const { logs, isConnected, clearLogs } = useWebSocket(logsProcessId);

  const hardwareReady =
    state.completedSteps[0] &&
    state.completedSteps[1] &&
    state.completedSteps[2] &&
    state.completedSteps[3];

  const activePhase = state.phases.find((p) => p.id === state.activePhaseId) ?? null;
  const inferenceRunning = state.inferenceProcessId !== null;

  // Refs for use inside async callbacks (avoid stale closures)
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cancelRef = useRef(false); // true when the next process-end is user-initiated (stop / switch)
  const sequenceModeRef = useRef(false);
  const runningIndexRef = useRef<number>(-1);
  const prevProcessIdRef = useRef<string | null>(null);
  const retryCountRef = useRef(0); // retries used for current phase attempt
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const logsRef = useRef<string[]>([]);
  useEffect(() => {
    logsRef.current = logs;
  }, [logs]);

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  function cancelPendingRetry() {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }

  useEffect(
    () => () => {
      stopPolling();
      cancelPendingRetry();
    },
    []
  );

  // Sync the active phase's config into wizard.inferenceConfig whenever the
  // active phase changes (i.e. user picks a different phase). Edits flow back
  // to the phase when the modal closes.
  const lastSyncedPhaseId = useRef<string | null>(null);
  useEffect(() => {
    if (state.activePhaseId && state.activePhaseId !== lastSyncedPhaseId.current) {
      const phase = state.phases.find((p) => p.id === state.activePhaseId);
      if (phase) {
        dispatch({ type: "SET_INFERENCE_CONFIG", config: phase.config });
      }
      lastSyncedPhaseId.current = state.activePhaseId;
    } else if (!state.activePhaseId) {
      lastSyncedPhaseId.current = null;
    }
  }, [state.activePhaseId, state.phases, dispatch]);

  function pollProcess(processId: string) {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const status = await services.getInferenceStatus(processId);
        if (status.state === "error") {
          stopPolling();
          await services.stopInference(processId).catch(() => {});

          const transient = isCameraTransientError(logsRef.current);
          const canRetry =
            transient && retryCountRef.current < PHASE_MAX_RETRIES;
          const phaseIdx = runningIndexRef.current;

          if (canRetry && phaseIdx >= 0) {
            retryCountRef.current += 1;
            setRunError(
              `Camera didn't settle — retrying in ${Math.round(
                RETRY_DELAY_MS / 1000
              )}s (attempt ${retryCountRef.current + 1}/${PHASE_MAX_RETRIES + 1})`
            );
            // mark this end as intentional so the falling-edge effect doesn't advance
            cancelRef.current = true;
            dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
            cancelPendingRetry();
            retryTimerRef.current = setTimeout(() => {
              retryTimerRef.current = null;
              setRunError(null);
              startPhaseByIndex(phaseIdx);
            }, RETRY_DELAY_MS);
            return;
          }

          setLastFailedProcessId(processId);
          setShowLogs(true);
          setRunError(
            status.error_message ||
              (transient
                ? "Camera failed to initialize after retry. Try again or restart the app."
                : "Phase exited with an error")
          );
          sequenceModeRef.current = false;
          setSequenceMode(false);
          dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
        } else if (status.state === "stopped") {
          stopPolling();
          await services.stopInference(processId).catch(() => {});
          dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
        }
      } catch {
        stopPolling();
        setRunError("Lost connection to inference process");
        sequenceModeRef.current = false;
        setSequenceMode(false);
        dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
      }
    }, 2000);
  }

  async function stopRunningProcess(intentional: boolean) {
    if (intentional) cancelRef.current = true;
    stopPolling();
    const id = stateRef.current.inferenceProcessId;
    if (id) {
      try {
        await services.stopInference(id);
      } catch {}
      dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
    }
  }

  async function startPhaseByIndex(idx: number) {
    const phases = stateRef.current.phases;
    if (idx < 0 || idx >= phases.length) return;
    const phase = phases[idx];

    setLastFailedProcessId(null);
    setShowLogs(false);
    clearLogs();

    // Reset retry budget when starting a different phase (vs retrying same one).
    if (runningIndexRef.current !== idx) {
      retryCountRef.current = 0;
    }
    runningIndexRef.current = idx;
    setRunningIndex(idx);
    dispatch({ type: "SET_ACTIVE_PHASE", id: phase.id });
    dispatch({ type: "SET_INFERENCE_CONFIG", config: phase.config });

    const finalRepoId = withEvalPrefix(phase.config.repoId);
    const cfg = { ...phase.config, repoId: finalRepoId };
    try {
      await services.saveConfig({
        ...stateRef.current,
        inferenceConfig: phase.config,
      });
      await services.stopCameraStreams().catch(() => {});
      const res = await services.startInference(cfg);
      cancelRef.current = false;
      dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: res.process_id });
      pollProcess(res.process_id);
    } catch (err) {
      runningIndexRef.current = -1;
      setRunningIndex(null);
      sequenceModeRef.current = false;
      setSequenceMode(false);
      setRunError(
        err instanceof Error ? err.message : "Failed to start phase"
      );
    }
  }

  // React to the inference process ending. Decides whether to advance the
  // sequence, end it, or just stop. Works whether the end was detected by our
  // poller or by InferenceStep's poller (when the modal is open).
  useEffect(() => {
    const prev = prevProcessIdRef.current;
    const curr = state.inferenceProcessId;
    prevProcessIdRef.current = curr;
    if (!prev || curr) return; // only react to a falling edge: id -> null

    // Intentional stop (user pressed Stop, switched phase, or a retry is queued).
    // Don't advance / clear running state — retry path manages its own flow.
    if (cancelRef.current) {
      cancelRef.current = false;
      if (!retryTimerRef.current) {
        runningIndexRef.current = -1;
        setRunningIndex(null);
      }
      return;
    }

    // Natural completion. Reset retry budget for this phase.
    retryCountRef.current = 0;

    // Advance sequence if active.
    if (sequenceModeRef.current) {
      const next = runningIndexRef.current + 1;
      if (next < stateRef.current.phases.length) {
        // Settle delay between phases so cameras can release cleanly.
        cancelPendingRetry();
        retryTimerRef.current = setTimeout(() => {
          retryTimerRef.current = null;
          if (!sequenceModeRef.current) return; // cancelled during wait
          startPhaseByIndex(next);
        }, PHASE_SETTLE_MS);
        return;
      }
      sequenceModeRef.current = false;
      setSequenceMode(false);
    }
    runningIndexRef.current = -1;
    setRunningIndex(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.inferenceProcessId]);

  function handleAddPhase() {
    dispatch({ type: "ADD_PHASE" });
  }

  function handleRemovePhase(id: string) {
    dispatch({ type: "REMOVE_PHASE", id });
    if (state.activePhaseId === id) {
      setModalOpen(false);
    }
  }

  function handleOpenPhase(id: string) {
    if (
      state.activePhaseId &&
      state.activePhaseId !== id
    ) {
      dispatch({
        type: "UPDATE_PHASE_CONFIG",
        id: state.activePhaseId,
        config: state.inferenceConfig,
      });
    }
    dispatch({ type: "SET_ACTIVE_PHASE", id });
    setModalOpen(true);
  }

  function handleCloseModal(open: boolean) {
    if (open) return;
    if (inferenceRunning) return; // don't close while a process is running
    if (state.activePhaseId) {
      dispatch({
        type: "UPDATE_PHASE_CONFIG",
        id: state.activePhaseId,
        config: state.inferenceConfig,
      });
    }
    setModalOpen(false);
  }

  function startEditName(id: string, current: string) {
    setEditingNameId(id);
    setDraftName(current);
  }

  function commitEditName() {
    if (editingNameId) {
      dispatch({
        type: "UPDATE_PHASE",
        id: editingNameId,
        patch: { name: draftName.trim() || "Untitled phase" },
      });
    }
    setEditingNameId(null);
    setDraftName("");
  }

  async function handleReturnHome() {
    setHomeMsg(null);
    setReturningHome(true);
    try {
      // Placeholder: backend endpoint is not wired up yet.
      // When available, call the move-to-home service here.
      await new Promise((r) => setTimeout(r, 400));
      setHomeMsg("Sent return-to-home command.");
    } catch (err) {
      setHomeMsg(
        err instanceof Error ? err.message : "Failed to send return-to-home"
      );
    } finally {
      setReturningHome(false);
    }
  }

  async function handleRunAll() {
    if (!hardwareReady || state.phases.length === 0) return;
    setRunError(null);
    setActionPending(true);
    cancelPendingRetry();
    retryCountRef.current = 0;
    sequenceModeRef.current = true;
    setSequenceMode(true);
    await stopRunningProcess(true);
    await startPhaseByIndex(0);
    setActionPending(false);
  }

  async function handleRunPhase(idx: number) {
    if (!hardwareReady) return;
    setRunError(null);
    setActionPending(true);
    cancelPendingRetry();
    retryCountRef.current = 0;
    sequenceModeRef.current = false;
    setSequenceMode(false);
    await stopRunningProcess(true);
    await startPhaseByIndex(idx);
    setActionPending(false);
  }

  async function handleStopAll() {
    setActionPending(true);
    cancelPendingRetry();
    retryCountRef.current = 0;
    sequenceModeRef.current = false;
    setSequenceMode(false);
    await stopRunningProcess(true);
    setActionPending(false);
  }

  const totalSeconds = state.phases.reduce(
    (acc, p) => acc + getPhaseTimeSeconds(p),
    0
  );

  const anyRunning = inferenceRunning || actionPending;

  const selectedCameraFeeds = state.cameraSelections
    .filter((c) => c.included && c.name)
    .map((c) => ({ opencvIndex: c.opencvIndex, name: c.name }));

  return (
    <StepCard
      title="Phases"
      description="Define multiple inference phases. Run all in sequence, or click a phase to run just that one."
      showNext={false}
    >
      <div className="space-y-5">
        {!hardwareReady && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Robot type, ports, cameras, and calibration must be configured
              before running phases.
            </AlertDescription>
          </Alert>
        )}

        {/* Run controls */}
        <div className="flex items-center gap-2">
          <Button
            onClick={handleRunAll}
            disabled={!hardwareReady || state.phases.length === 0 || actionPending}
          >
            {sequenceMode ? (
              <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
            ) : (
              <PlayCircle className="mr-1.5 h-4 w-4" />
            )}
            Run All Phases
          </Button>
          {anyRunning && (
            <Button
              variant="outline"
              onClick={handleStopAll}
              disabled={actionPending && !inferenceRunning}
            >
              <Square className="mr-1.5 h-4 w-4" />
              Stop
            </Button>
          )}
          {sequenceMode && runningIndex !== null && (
            <span className="text-xs text-muted-foreground ml-2">
              {inferenceRunning
                ? `Running ${runningIndex + 1} of ${state.phases.length}`
                : `Settling between phases…`}
            </span>
          )}
        </div>

        {/* Error banner */}
        {runError && (
          <div className="space-y-2">
            <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950">
              <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0" />
              <div className="flex-1">
                <p className="text-sm font-medium text-red-800 dark:text-red-200">
                  Phase run failed
                </p>
                <p className="text-xs text-red-600 dark:text-red-400 mt-0.5">
                  {runError}
                </p>
              </div>
              {logs.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowLogs((v) => !v)}
                >
                  {showLogs ? "Hide Logs" : `Show Logs (${logs.length})`}
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setRunError(null);
                  setShowLogs(false);
                }}
              >
                Dismiss
              </Button>
            </div>
            {showLogs && logs.length > 0 && (
              <LogViewer
                logs={logs}
                isConnected={isConnected}
                onClear={clearLogs}
                maxHeight="300px"
              />
            )}
          </div>
        )}

        {/* Phase list */}
        <div className="space-y-2">
          {state.phases.length === 0 ? (
            <div className="rounded-lg border border-dashed p-6 text-center">
              <p className="text-sm text-muted-foreground">
                No phases yet. Add one to get started.
              </p>
            </div>
          ) : (
            state.phases.map((phase, idx) => {
              const seconds = getPhaseTimeSeconds(phase);
              const configured =
                phase.config.policyPath.trim() !== "" &&
                phase.config.task.trim() !== "";
              const isActive = state.activePhaseId === phase.id;
              const isRunningThis = runningIndex === idx && inferenceRunning;
              return (
                <div
                  key={phase.id}
                  className={
                    "flex items-center gap-3 rounded-lg border p-3 transition-colors " +
                    (isRunningThis
                      ? "border-blue-400 bg-blue-50/60 dark:bg-blue-950/40"
                      : isActive
                      ? "border-primary bg-primary/5"
                      : "hover:bg-muted/40")
                  }
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium">
                    {isRunningThis ? (
                      <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                    ) : (
                      idx + 1
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    {editingNameId === phase.id ? (
                      <Input
                        autoFocus
                        value={draftName}
                        onChange={(e) => setDraftName(e.target.value)}
                        onBlur={commitEditName}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitEditName();
                          if (e.key === "Escape") {
                            setEditingNameId(null);
                            setDraftName("");
                          }
                        }}
                        className="h-7 text-sm"
                      />
                    ) : (
                      <button
                        type="button"
                        onClick={() => startEditName(phase.id, phase.name)}
                        className="flex items-center gap-1.5 text-sm font-medium hover:underline"
                      >
                        {phase.name}
                        <Pencil className="h-3 w-3 text-muted-foreground" />
                      </button>
                    )}
                    <div className="mt-0.5 flex items-center gap-3 text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatPhaseDuration(seconds)}
                      </span>
                      <span>
                        {phase.config.numEpisodes} ep × {phase.config.episodeTimeS}s
                      </span>
                      {!configured && (
                        <span className="text-amber-600 dark:text-amber-400">
                          needs setup
                        </span>
                      )}
                      {isRunningThis && (
                        <span className="text-blue-600 dark:text-blue-400 font-medium">
                          running
                        </span>
                      )}
                      {isActive && !isRunningThis && (
                        <span className="text-primary font-medium">active</span>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="default"
                    size="sm"
                    onClick={() => handleRunPhase(idx)}
                    disabled={!hardwareReady || actionPending || !configured}
                    title={
                      inferenceRunning
                        ? "Stops current run and starts this phase"
                        : "Run this phase"
                    }
                  >
                    <Play className="mr-1.5 h-3.5 w-3.5" />
                    Run
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleOpenPhase(phase.id)}
                    disabled={!hardwareReady || actionPending}
                  >
                    {isActive ? "Resume" : "Open"}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => handleRemovePhase(phase.id)}
                    disabled={actionPending || isRunningThis}
                    title="Remove phase"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              );
            })
          )}
        </div>

        {/* Totals + add */}
        <div className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {state.phases.length} phase{state.phases.length === 1 ? "" : "s"}
            {state.phases.length > 0 && (
              <> · total {formatPhaseDuration(totalSeconds)}</>
            )}
          </div>
          <Button variant="outline" size="sm" onClick={handleAddPhase}>
            <Plus className="mr-1.5 h-4 w-4" />
            Add Phase
          </Button>
        </div>

        {/* Return-to-initial-position button (manual, between runs) */}
        <div className="rounded-lg border p-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium">Return to initial position</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                Send the robot back to its starting pose between phase runs.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleReturnHome}
              disabled={!hardwareReady || returningHome || inferenceRunning}
            >
              {returningHome ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Home className="mr-1.5 h-3.5 w-3.5" />
              )}
              {returningHome ? "Sending..." : "Return Home"}
            </Button>
          </div>
          {homeMsg && (
            <p className="mt-2 text-xs text-muted-foreground">{homeMsg}</p>
          )}
        </div>

        {/* Live camera feeds — visible whenever a phase is running */}
        {inferenceRunning && selectedCameraFeeds.length > 0 && (
          <CameraFeedPanel cameras={selectedCameraFeeds} />
        )}

        {/* Phase modal: reuses the InferenceStep UI bound to the active phase */}
        <Dialog open={modalOpen} onOpenChange={handleCloseModal}>
          <DialogContent
            className="max-w-3xl max-h-[90vh] overflow-y-auto sm:max-w-3xl"
            showCloseButton={!inferenceRunning}
            onPointerDownOutside={(e) => {
              if (inferenceRunning) e.preventDefault();
            }}
            onEscapeKeyDown={(e) => {
              if (inferenceRunning) e.preventDefault();
            }}
          >
            <DialogHeader>
              <DialogTitle>{activePhase?.name ?? "Phase"}</DialogTitle>
              <DialogDescription>
                Configure and run this phase. Changes are saved when you close.
              </DialogDescription>
            </DialogHeader>
            <div className="pt-2">
              <InferenceStep />
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => handleCloseModal(false)}
                disabled={inferenceRunning}
              >
                Done
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </StepCard>
  );
}
