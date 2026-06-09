"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clapperboard,
  Film,
  FolderOpen,
  Loader2,
  Play,
  RefreshCw,
  Square,
  StepForward,
  XCircle,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { LogViewer } from "@/components/common/log-viewer";
import {
  useMotorState,
  MotorPanel,
  CameraFeedPanel,
} from "@/components/common/robot-display";
import { useWebSocket } from "@/hooks/use-websocket";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

// ---------------------------------------------------------------------------
// Recording phase detection
// ---------------------------------------------------------------------------

type RecordPhase = "idle" | "recording" | "resetting" | "encoding" | "finalizing";

const EPISODE_RE = /Recording episode (\d+)/;
const RESET_RE = /Reset the environment/;
const ENCODING_RE = /[Ee]ncoding (?:video|videos|remaining)/;
const DONE_RE = /Stop recording|Exiting/;

function useRecordingPhase(logs: string[], isRunning: boolean) {
  const [phase, setPhase] = useState<RecordPhase>("idle");
  const [currentEpisode, setCurrentEpisode] = useState<number | null>(null);

  useEffect(() => {
    if (!isRunning) {
      setPhase("idle");
      return;
    }

    const tail = logs.slice(-30);
    for (let i = tail.length - 1; i >= 0; i--) {
      const line = tail[i];

      if (DONE_RE.test(line)) {
        setPhase("finalizing");
        return;
      }
      if (ENCODING_RE.test(line)) {
        setPhase("encoding");
        return;
      }
      if (RESET_RE.test(line)) {
        setPhase("resetting");
        return;
      }
      const epMatch = line.match(EPISODE_RE);
      if (epMatch) {
        setPhase("recording");
        setCurrentEpisode(parseInt(epMatch[1], 10));
        return;
      }
    }
  }, [logs, isRunning]);

  return { phase, currentEpisode };
}

// ---------------------------------------------------------------------------
// Repo ID duplicate tracking via localStorage
// ---------------------------------------------------------------------------

const STORAGE_KEY = "lerobot_saved_repo_ids";

function getSavedRepoIds(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function saveRepoId(repoId: string) {
  const existing = getSavedRepoIds();
  if (!existing.includes(repoId)) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...existing, repoId]));
  }
}

// ---------------------------------------------------------------------------
// Phase status card
// ---------------------------------------------------------------------------

type StatusCardPhase = RecordPhase | "done";

function RecordingStatusCard({
  phase,
  currentEpisode,
  numEpisodes,
  onStop,
  stopping,
}: {
  phase: StatusCardPhase;
  currentEpisode: number | null;
  numEpisodes: number;
  onStop: () => void;
  stopping: boolean;
}) {
  if (phase === "idle") return null;

  type PhaseConfig = {
    icon: React.ReactNode;
    title: string;
    subtitle: string;
    accent: string;
  };

  const configs: Record<StatusCardPhase, PhaseConfig> = {
    idle: { icon: null, title: "", subtitle: "", accent: "" },
    recording: {
      icon: <Clapperboard className="h-5 w-5 text-red-500 shrink-0" />,
      title:
        currentEpisode !== null
          ? `Recording episode ${currentEpisode} / ${numEpisodes}`
          : "Recording…",
      subtitle: "Perform the task now.",
      accent: "border-red-200 dark:border-red-900",
    },
    resetting: {
      icon: <RefreshCw className="h-5 w-5 text-amber-500 shrink-0" />,
      title: "Reset the environment",
      subtitle: "Prepare the scene for the next episode.",
      accent: "border-amber-200 dark:border-amber-900",
    },
    encoding: {
      icon: <Film className="h-5 w-5 text-blue-500 shrink-0" />,
      title: "Encoding videos…",
      subtitle: "Please wait while the episode is being saved.",
      accent: "border-blue-200 dark:border-blue-900",
    },
    finalizing: {
      icon: <Loader2 className="h-5 w-5 text-blue-500 shrink-0 animate-spin" />,
      title: "Uploading to HuggingFace…",
      subtitle: "All episodes recorded. Please wait for the upload to finish.",
      accent: "border-blue-200 dark:border-blue-900",
    },
    done: {
      icon: <CheckCircle2 className="h-5 w-5 text-emerald-500 shrink-0" />,
      title: "Recording complete",
      subtitle: "All episodes recorded and uploaded to HuggingFace.",
      accent: "border-emerald-200 dark:border-emerald-900",
    },
  };

  const cfg = configs[phase];

  return (
    <div className={`flex items-center gap-3 rounded-lg border p-4 ${cfg.accent}`}>
      {cfg.icon}
      <div className="flex-1">
        <p className="text-sm font-medium">{cfg.title}</p>
        <p className="text-xs text-muted-foreground mt-0.5">{cfg.subtitle}</p>
      </div>
      {phase !== "done" && phase !== "finalizing" && (
        <Button variant="outline" size="sm" onClick={onStop} disabled={stopping}>
          {stopping ? (
            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
          ) : (
            <Square className="mr-2 h-3.5 w-3.5" />
          )}
          Stop
        </Button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Countdown for the active phase (recording or resetting)
// ---------------------------------------------------------------------------

function usePhaseCountdown(
  phase: RecordPhase,
  currentEpisode: number | null,
) {
  const [elapsedMs, setElapsedMs] = useState(0);
  const startRef = useRef<number>(Date.now());

  useEffect(() => {
    startRef.current = Date.now();
    setElapsedMs(0);
    if (phase !== "recording" && phase !== "resetting") return;
    const id = setInterval(() => {
      setElapsedMs(Date.now() - startRef.current);
    }, 200);
    return () => clearInterval(id);
  }, [phase, currentEpisode]);

  return elapsedMs / 1000;
}

// ---------------------------------------------------------------------------
// Round / countdown progress card
// ---------------------------------------------------------------------------

function RoundProgressCard({
  phase,
  currentEpisode,
  numEpisodes,
  episodeTimeS,
  resetTimeS,
}: {
  phase: RecordPhase;
  currentEpisode: number | null;
  numEpisodes: number;
  episodeTimeS: number;
  resetTimeS: number;
}) {
  const elapsed = usePhaseCountdown(phase, currentEpisode);

  const isRecording = phase === "recording";
  const isResetting = phase === "resetting";
  if (!isRecording && !isResetting) return null;

  const totalS = isRecording ? episodeTimeS : resetTimeS;
  const remaining = Math.max(0, Math.ceil(totalS - elapsed));
  const progressPct =
    totalS > 0 ? Math.min(100, (elapsed / totalS) * 100) : 0;

  const phaseLabel = isRecording ? "Recording" : "Waiting — reset the scene";
  const phaseSub = isRecording
    ? "Perform the task now."
    : "Get the environment ready for the next episode.";
  const barColor = isRecording ? "bg-red-500" : "bg-amber-500";
  const numberColor = isRecording
    ? "text-red-600 dark:text-red-400"
    : "text-amber-600 dark:text-amber-400";

  const roundNum = currentEpisode !== null ? currentEpisode + 1 : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-base">
          <span>
            Round{" "}
            <span className="tabular-nums">
              {roundNum ?? "—"}
            </span>{" "}
            <span className="text-muted-foreground font-normal">
              / {numEpisodes}
            </span>
          </span>
          <span
            className={`text-xs font-medium uppercase tracking-wide ${
              isRecording
                ? "text-red-600 dark:text-red-400"
                : "text-amber-600 dark:text-amber-400"
            }`}
          >
            {phaseLabel}
          </span>
        </CardTitle>
        <CardDescription>{phaseSub}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2">
          <span
            className={`font-mono tabular-nums text-5xl font-semibold ${numberColor}`}
          >
            {remaining}
          </span>
          <span className="text-sm text-muted-foreground">
            s remaining / {totalS}s
          </span>
        </div>
        <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={`h-full rounded-full transition-[width] duration-200 ease-linear ${barColor}`}
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Track which episodes have finished recording (transitioned out of "recording")
// ---------------------------------------------------------------------------

function useCompletedEpisodes(
  phase: RecordPhase,
  currentEpisode: number | null,
  isRunning: boolean,
) {
  const [completed, setCompleted] = useState<Set<number>>(new Set());
  const prevPhaseRef = useRef<RecordPhase>("idle");
  const prevEpisodeRef = useRef<number | null>(null);

  // Reset accumulated state whenever a new run starts
  useEffect(() => {
    if (!isRunning) return;
    setCompleted(new Set());
    prevPhaseRef.current = "idle";
    prevEpisodeRef.current = null;
  }, [isRunning]);

  useEffect(() => {
    const prevPhase = prevPhaseRef.current;
    const prevEp = prevEpisodeRef.current;

    setCompleted((prev) => {
      const next = new Set(prev);
      // recording → non-recording: current episode just finished
      if (
        prevPhase === "recording" &&
        phase !== "recording" &&
        currentEpisode !== null
      ) {
        next.add(currentEpisode);
      }
      // recording episode N → recording episode N+1: previous episode finished
      if (
        prevPhase === "recording" &&
        phase === "recording" &&
        prevEp !== null &&
        currentEpisode !== null &&
        currentEpisode !== prevEp
      ) {
        next.add(prevEp);
      }
      return next.size === prev.size ? prev : next;
    });

    prevPhaseRef.current = phase;
    prevEpisodeRef.current = currentEpisode;
  }, [phase, currentEpisode]);

  const reset = useCallback(() => setCompleted(new Set()), []);
  return { completed, reset };
}

// ---------------------------------------------------------------------------
// Episode status circles — green by default, click to mark failed (red)
// ---------------------------------------------------------------------------

function EpisodeStatusList({
  numEpisodes,
  completed,
  failed,
  currentEpisode,
  phase,
  onToggleFailed,
}: {
  numEpisodes: number;
  completed: Set<number>;
  failed: Set<number>;
  currentEpisode: number | null;
  phase: RecordPhase;
  onToggleFailed: (episode: number) => void;
}) {
  if (completed.size === 0) return null;

  // Sort completed episodes ascending so the row reads left-to-right by index
  const items = Array.from(completed).sort((a, b) => a - b);
  const failedCount = items.filter((ep) => failed.has(ep)).length;
  const successCount = items.length - failedCount;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-base">
          <span>Episodes</span>
          <span className="text-xs font-normal text-muted-foreground">
            <span className="text-emerald-600 dark:text-emerald-400">
              {successCount} good
            </span>
            {" · "}
            <span className="text-red-600 dark:text-red-400">
              {failedCount} failed
            </span>
            {" · "}
            <span>{items.length} / {numEpisodes} recorded</span>
          </span>
        </CardTitle>
        <CardDescription>
          Click a circle to toggle it between good (green) and failed (red).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-1.5">
          {items.map((ep) => {
            const isFailed = failed.has(ep);
            return (
              <button
                key={ep}
                type="button"
                onClick={() => onToggleFailed(ep)}
                title={`Episode ${ep + 1} — ${isFailed ? "failed (click to mark good)" : "good (click to mark failed)"}`}
                className={`flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-medium text-white tabular-nums transition-all hover:scale-110 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background ${
                  isFailed
                    ? "bg-red-500 focus:ring-red-500"
                    : "bg-emerald-500 focus:ring-emerald-500"
                }`}
              >
                {ep + 1}
              </button>
            );
          })}
          {/* Show a placeholder for the currently-recording episode */}
          {phase === "recording" &&
            currentEpisode !== null &&
            !completed.has(currentEpisode) && (
              <div
                title={`Episode ${currentEpisode + 1} — recording…`}
                className="flex h-7 w-7 items-center justify-center rounded-full border-2 border-red-500 text-[11px] font-medium tabular-nums text-red-600 dark:text-red-400 animate-pulse"
              >
                {currentEpisode + 1}
              </div>
            )}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function RecordStep() {
  const { state, dispatch, allPriorStepsComplete } = useWizard();
  const [starting, setStarting] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [repoIdWarning, setRepoIdWarning] = useState(false);
  const [recordingSuccess, setRecordingSuccess] = useState(false);
  const [hfStatus, setHfStatus] = useState<{
    is_logged_in: boolean;
    username: string | null;
  } | null>(null);
  const [hfChecking, setHfChecking] = useState(true);
  const priorComplete = allPriorStepsComplete(5);
  const isRunning = state.recordProcessId !== null;

  const { logs, isConnected, clearLogs } = useWebSocket(state.recordProcessId);

  const config = state.recordingConfig;

  // Phase detection from log stream
  const { phase, currentEpisode } = useRecordingPhase(logs, isRunning);

  // Track which episodes have completed recording, plus user-marked failures
  const { completed: completedEpisodes, reset: resetCompleted } =
    useCompletedEpisodes(phase, currentEpisode, isRunning);
  const [failedEpisodes, setFailedEpisodes] = useState<Set<number>>(new Set());

  const toggleFailedEpisode = useCallback((episode: number) => {
    setFailedEpisodes((prev) => {
      const next = new Set(prev);
      if (next.has(episode)) next.delete(episode);
      else next.add(episode);
      return next;
    });
  }, []);

  // Motor + camera feeds (only when displayData is on)
  const { motors, motorOrder, frequency } = useMotorState(
    logs,
    isRunning && config.displayData
  );
  const selectedCameraFeeds = state.cameraSelections
    .filter((c) => c.included && c.name)
    .map((c) => ({ opencvIndex: c.opencvIndex, name: c.name }));

  // Process crash polling
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (processId: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const status = await services.getProcessStatus(processId);
          if (status.state === "error") {
            setErrorMsg(status.error_message || "Process exited with an error");
            setShowLogs(true);
            // Ensure port locks are released even if _collect_logs hasn't finished cleanup
            services.stopRecording(processId).catch(() => {});
            dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
            stopPolling();
          } else if (status.state === "stopped") {
            setRecordingSuccess(true);
            // Ensure port locks are released
            services.stopRecording(processId).catch(() => {});
            dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
            stopPolling();
          }
        } catch {
          setErrorMsg("Lost connection to process");
          setShowLogs(true);
          dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
          stopPolling();
        }
      }, 2000);
    },
    [stopPolling, dispatch]
  );

  // Resume polling if process was already running when component mounts
  useEffect(() => {
    if (state.recordProcessId) {
      startPolling(state.recordProcessId);
    }
    return stopPolling;
  }, [state.recordProcessId, startPolling, stopPolling]);

  // Check HuggingFace auth status on mount
  useEffect(() => {
    let cancelled = false;
    setHfChecking(true);
    services.checkHFStatus()
      .then((status) => { if (!cancelled) setHfStatus(status); })
      .catch(() => { if (!cancelled) setHfStatus({ is_logged_in: false, username: null }); })
      .finally(() => { if (!cancelled) setHfChecking(false); });
    return () => { cancelled = true; };
  }, []);

  // Check if repo ID was used before
  useEffect(() => {
    if (config.repoId.trim() === "") {
      setRepoIdWarning(false);
      return;
    }
    setRepoIdWarning(getSavedRepoIds().includes(config.repoId.trim()));
  }, [config.repoId]);

  const hfReady = hfStatus?.is_logged_in === true;

  const canStart =
    priorComplete &&
    hfReady &&
    config.repoId.trim() !== "" &&
    config.task.trim() !== "" &&
    config.numEpisodes > 0 &&
    config.episodeTimeS > 0;

  // `resume=false` starts a fresh recording (overwriting any existing dataset).
  // `resume=true` continues an existing dataset, appending new episodes after
  // the last recorded one — so it must NOT clear the cached dataset.
  async function handleStart(resume = false) {
    if (resume) setResuming(true);
    else setStarting(true);
    setErrorMsg(null);
    setShowLogs(false);
    setRecordingSuccess(false);
    setFailedEpisodes(new Set());
    resetCompleted();
    try {
      const suffix = config.repoId.trim();
      if (!hfStatus?.username) {
        throw new Error("Not logged in to HuggingFace");
      }
      const fullRepoId = `${hfStatus.username}/makermods_${suffix}`;
      // Save current wizard state (ports, cameras, calibration) to backend config
      await services.saveConfig(state);
      // Release any MJPEG camera streams so the recording subprocess can access them
      await services.stopCameraStreams().catch(() => {});
      if (!resume) {
        // Clear cached data so the new recording replaces any previous dataset.
        // Skipped when resuming so the existing episodes are kept and appended to.
        await services.clearCache(fullRepoId).catch(() => {});
      }
      const res = await services.startRecording(
        { ...config, repoId: fullRepoId },
        resume,
      );
      dispatch({ type: "SET_RECORD_PROCESS_ID", id: res.process_id });
      saveRepoId(suffix);
      setRepoIdWarning(false);
      startPolling(res.process_id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to start recording");
      setShowLogs(true);
    } finally {
      setStarting(false);
      setResuming(false);
    }
  }

  async function handleStop() {
    if (!state.recordProcessId) return;
    setStopping(true);
    stopPolling();
    try {
      await services.stopRecording(state.recordProcessId);
    } finally {
      dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
      setStopping(false);
    }
  }

  function updateConfig(partial: Partial<typeof config>) {
    dispatch({ type: "SET_RECORDING_CONFIG", config: partial });
  }

  // Show the log toggle once a process has produced any output
  const hasLogs = logs.length > 0;

  return (
    <StepCard
      title="Record Data"
      description="Configure and start data recording."
      showNext={false}
    >
      <div className="space-y-5">
        {!priorComplete && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Previous steps are not all completed. It is not recommended to
              proceed without completing them first.
            </AlertDescription>
          </Alert>
        )}

        {/* HuggingFace auth status */}
        {hfChecking && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Checking HuggingFace authentication…
          </div>
        )}

        {!hfChecking && hfStatus && !hfStatus.is_logged_in && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              <span className="font-medium">Not logged in to HuggingFace.</span>{" "}
              Recording uploads datasets to HuggingFace Hub. Log in by running:{" "}
              <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono">
                hf auth login
              </code>{" "}
              in your terminal, then refresh this page.
            </AlertDescription>
          </Alert>
        )}

        {!hfChecking && hfStatus && hfStatus.is_logged_in && hfStatus.username && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            Logged in to HuggingFace as <span className="font-medium text-foreground">{hfStatus.username}</span>
          </div>
        )}

        {/* Form fields */}
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="repo-id">Dataset Name</Label>
            <Input
              id="repo-id"
              placeholder="pick_cube"
              value={config.repoId}
              onChange={(e) => updateConfig({ repoId: e.target.value })}
              disabled={isRunning || !hfReady}
            />
            {repoIdWarning && (
              <p className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                A dataset with this repo ID was recorded before.{" "}
                <span className="font-medium">Start Recording</span> overwrites it;
                use <span className="font-medium">Continue from last episode</span>{" "}
                to append new episodes instead.
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="task">Task Description</Label>
            <Input
              id="task"
              placeholder="e.g. Pick up the cube and place it in the bin"
              value={config.task}
              onChange={(e) => updateConfig({ task: e.target.value })}
              disabled={isRunning}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="num-episodes">Episodes</Label>
              <Input
                id="num-episodes"
                type="number"
                min={1}
                value={config.numEpisodes || ""}
                onChange={(e) =>
                  updateConfig({ numEpisodes: parseInt(e.target.value) || 0 })
                }
                className={config.numEpisodes < 1 ? "border-red-500" : ""}
                disabled={isRunning}
              />
              {config.numEpisodes < 1 && (
                <p className="text-xs text-red-500">Must be at least 1</p>
              )}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="episode-time">Episode Time (s)</Label>
              <Input
                id="episode-time"
                type="number"
                min={1}
                value={config.episodeTimeS || ""}
                onChange={(e) =>
                  updateConfig({ episodeTimeS: parseInt(e.target.value) || 0 })
                }
                className={config.episodeTimeS < 1 ? "border-red-500" : ""}
                disabled={isRunning}
              />
              {config.episodeTimeS < 1 && (
                <p className="text-xs text-red-500">Must be at least 1</p>
              )}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="reset-time">Reset Time (s)</Label>
              <Input
                id="reset-time"
                type="number"
                min={0}
                value={config.resetTimeS ?? ""}
                onChange={(e) => {
                  const val = e.target.value;
                  updateConfig({ resetTimeS: val === "" ? 0 : parseInt(val) || 0 });
                }}
                disabled={isRunning}
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Switch
              id="display-data"
              checked={config.displayData}
              onCheckedChange={(checked) =>
                updateConfig({ displayData: checked })
              }
              disabled={isRunning}
            />
            <Label htmlFor="display-data">Display data while recording</Label>
          </div>

          {/* Advanced settings (collapsible) */}
          <div className="rounded-lg border">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex w-full items-center gap-2 p-3 text-sm font-medium hover:bg-muted/50 transition-colors"
            >
              {showAdvanced ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
              <Camera className="h-4 w-4 text-muted-foreground" />
              Advanced Settings
              <span className="text-xs text-muted-foreground font-normal ml-1">
                {config.cameraWidth}×{config.cameraHeight} @ {config.cameraFps}fps
              </span>
            </button>
            {showAdvanced && (
              <div className="space-y-3 border-t px-4 pb-4 pt-3">
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-1.5">
                    <Label htmlFor="camera-fps">FPS</Label>
                    <Select
                      value={String(config.cameraFps)}
                      onValueChange={(v) => updateConfig({ cameraFps: parseInt(v) })}
                      disabled={isRunning}
                    >
                      <SelectTrigger id="camera-fps">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="15">15</SelectItem>
                        <SelectItem value="24">24</SelectItem>
                        <SelectItem value="30">30 (recommended)</SelectItem>
                        <SelectItem value="60">60</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="camera-resolution">Resolution</Label>
                    <Select
                      value={`${config.cameraWidth}x${config.cameraHeight}`}
                      onValueChange={(v) => {
                        const [w, h] = v.split("x").map(Number);
                        updateConfig({ cameraWidth: w, cameraHeight: h });
                      }}
                      disabled={isRunning}
                    >
                      <SelectTrigger id="camera-resolution">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="320x240">320×240</SelectItem>
                        <SelectItem value="640x480">640×480 (recommended)</SelectItem>
                        <SelectItem value="1280x720">1280×720 (HD)</SelectItem>
                        <SelectItem value="1920x1080">1920×1080 (Full HD)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  Higher resolution and FPS produce better data but require more storage and processing power.
                </p>
              </div>
            )}
          </div>
        </div>

        <Separator />

        {/* Start button + folder button */}
        <div className="flex items-center gap-2">
          {!isRunning && !errorMsg && (
            <>
              <Button
                onClick={() => handleStart(false)}
                disabled={starting || resuming || !canStart}
              >
                {starting ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                {starting ? "Starting…" : "Start Recording"}
              </Button>
              {/* Resume only makes sense for a dataset that already exists. */}
              {repoIdWarning && (
                <Button
                  variant="outline"
                  onClick={() => handleStart(true)}
                  disabled={starting || resuming || !canStart}
                  title="Append new episodes to the existing dataset, continuing after the last recorded episode"
                >
                  {resuming ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <StepForward className="mr-2 h-4 w-4" />
                  )}
                  {resuming ? "Resuming…" : "Continue from last episode"}
                </Button>
              )}
            </>
          )}
          <Button
            variant="outline"
            size="icon"
            title="Open data folder (~/.cache/huggingface/lerobot)"
            onClick={() => services.openDataFolder().catch(() => {})}
          >
            <FolderOpen className="h-4 w-4" />
          </Button>
        </div>

        {/* Error state */}
        {errorMsg && (
          <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
            <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-800 dark:text-red-200">
                Recording failed
              </p>
              <p className="text-xs text-red-600 dark:text-red-400 mt-0.5">
                {errorMsg}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setErrorMsg(null);
                setShowLogs(false);
              }}
            >
              Dismiss
            </Button>
          </div>
        )}

        {/* Phase status banner (shown while process is running or after success) */}
        {(isRunning || recordingSuccess) && (
          <RecordingStatusCard
            phase={recordingSuccess ? "done" : phase}
            currentEpisode={currentEpisode}
            numEpisodes={config.numEpisodes}
            onStop={handleStop}
            stopping={stopping}
          />
        )}

        {/* Live camera + motor feeds (only when displayData is enabled) */}
        {isRunning && config.displayData && (
          <div className="space-y-3">
            {selectedCameraFeeds.length > 0 && (
              <CameraFeedPanel cameras={selectedCameraFeeds} />
            )}
            <MotorPanel
              motors={motors}
              motorOrder={motorOrder}
              frequency={frequency}
            />
          </div>
        )}

        {/* Collapsible terminal logs */}
        {hasLogs && (
          <div>
            <button
              type="button"
              onClick={() => setShowLogs(!showLogs)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showLogs ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
              {showLogs ? "Hide Logs" : "Show Logs"}
              <span className="text-muted-foreground/60">({logs.length} lines)</span>
            </button>
            {showLogs && (
              <div className="mt-2">
                <LogViewer
                  logs={logs}
                  isConnected={isConnected}
                  onClear={clearLogs}
                  maxHeight="300px"
                />
              </div>
            )}
          </div>
        )}

        {/* Round + countdown progress card */}
        {isRunning && (
          <RoundProgressCard
            phase={phase}
            currentEpisode={currentEpisode}
            numEpisodes={config.numEpisodes}
            episodeTimeS={config.episodeTimeS}
            resetTimeS={config.resetTimeS}
          />
        )}

        {/* Per-episode status circles (good / failed) */}
        <EpisodeStatusList
          numEpisodes={config.numEpisodes}
          completed={completedEpisodes}
          failed={failedEpisodes}
          currentEpisode={currentEpisode}
          phase={phase}
          onToggleFailed={toggleFailedEpisode}
        />
      </div>
    </StepCard>
  );
}
