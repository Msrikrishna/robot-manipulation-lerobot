"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Bot,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  Loader2,
  Play,
  Square,
  Trash2,
  XCircle,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { LogViewer } from "@/components/common/log-viewer";
import {
  useMotorState,
  MotorPanel,
  CameraFeedPanel,
} from "@/components/common/robot-display";
import { useWebSocket } from "@/hooks/use-websocket";
import { services } from "@/lib/services";
import { INFERENCE_MODELS } from "@/lib/wizard-types";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

// ---------------------------------------------------------------------------
// Persistent policy path storage (localStorage)
// ---------------------------------------------------------------------------

const POLICY_PATHS_KEY = "lerobot_trained_policy_paths";
const LAST_INFERENCE_REPO_KEY = "lerobot_last_inference_repo_id";

function getLastInferenceRepoId(): string | null {
  try {
    return localStorage.getItem(LAST_INFERENCE_REPO_KEY);
  } catch {
    return null;
  }
}

function saveLastInferenceRepoId(repoId: string) {
  try {
    localStorage.setItem(LAST_INFERENCE_REPO_KEY, repoId.trim());
  } catch {}
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

interface SavedPolicy {
  path: string;
  savedAt: string; // ISO date
}

function getSavedPolicies(): SavedPolicy[] {
  try {
    const raw = localStorage.getItem(POLICY_PATHS_KEY);
    return raw ? (JSON.parse(raw) as SavedPolicy[]) : [];
  } catch {
    return [];
  }
}

function savePolicyPath(path: string) {
  const trimmed = path.trim();
  if (!trimmed) return;
  const existing = getSavedPolicies();
  if (existing.some((p) => p.path === trimmed)) return;
  const updated = [{ path: trimmed, savedAt: new Date().toISOString() }, ...existing];
  localStorage.setItem(POLICY_PATHS_KEY, JSON.stringify(updated));
}

function removePolicyPath(path: string) {
  const existing = getSavedPolicies();
  localStorage.setItem(
    POLICY_PATHS_KEY,
    JSON.stringify(existing.filter((p) => p.path !== path))
  );
}

export function InferenceStep() {
  const { state, dispatch } = useWizard();
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [savedPolicies, setSavedPolicies] = useState<SavedPolicy[]>([]);
  const [showPolicyDropdown, setShowPolicyDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  // Inference only requires hardware setup (steps 0-3: robot type, ports, cameras, calibration)
  // It does NOT require teleoperate (step 4) or record (step 5) to have been visited
  const hardwareReady =
    state.completedSteps[0] &&
    state.completedSteps[1] &&
    state.completedSteps[2] &&
    state.completedSteps[3];
  const isRunning = state.inferenceProcessId !== null;

  const { logs, isConnected, clearLogs } = useWebSocket(
    state.inferenceProcessId
  );

  const config = state.inferenceConfig;

  // Load saved policies on mount
  useEffect(() => {
    setSavedPolicies(getSavedPolicies());
  }, []);

  // Pre-fill policy path from training step's output model and save it
  useEffect(() => {
    if (state.trainingOutputModelId) {
      savePolicyPath(state.trainingOutputModelId);
      setSavedPolicies(getSavedPolicies());
      if (!config.policyPath) {
        dispatch({
          type: "SET_INFERENCE_CONFIG",
          config: { policyPath: state.trainingOutputModelId },
        });
      }
    }
  }, [state.trainingOutputModelId, config.policyPath, dispatch]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowPolicyDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
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
          const status = await services.getInferenceStatus(processId);
          if (status.state === "error") {
            setErrorMsg(
              status.error_message || "Process exited with an error"
            );
            setShowLogs(true);
            // Ensure port locks are released even if _collect_logs hasn't finished cleanup
            services.stopInference(processId).catch(() => {});
            dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
            stopPolling();
          } else if (status.state === "stopped") {
            // Ensure port locks are released
            services.stopInference(processId).catch(() => {});
            dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
            stopPolling();
          }
        } catch {
          setErrorMsg("Lost connection to process");
          setShowLogs(true);
          dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
          stopPolling();
        }
      }, 2000);
    },
    [stopPolling, dispatch]
  );

  // Resume polling if process was already running when component mounts
  useEffect(() => {
    if (state.inferenceProcessId) {
      startPolling(state.inferenceProcessId);
    }
    return stopPolling;
  }, [state.inferenceProcessId, startPolling, stopPolling]);

  const canStart =
    hardwareReady &&
    config.policyPath.trim() !== "" &&
    config.repoId.trim() !== "" &&
    config.task.trim() !== "" &&
    config.numEpisodes > 0 &&
    config.episodeTimeS > 0 &&
    INFERENCE_MODELS.some((m) => m.value === config.modelType && m.supported);

  async function handleStart() {
    setStarting(true);
    setErrorMsg(null);
    setShowLogs(false);
    try {
      await services.saveConfig(state);
      await services.stopCameraStreams().catch(() => {});
      const finalRepoId = withEvalPrefix(config.repoId);
      const res = await services.startInference({ ...config, repoId: finalRepoId });
      saveLastInferenceRepoId(finalRepoId);
      dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: res.process_id });
      startPolling(res.process_id);
    } catch (err) {
      setErrorMsg(
        err instanceof Error ? err.message : "Failed to start inference"
      );
      setShowLogs(true);
    } finally {
      setStarting(false);
    }
  }

  async function handleStop() {
    if (!state.inferenceProcessId) return;
    setStopping(true);
    stopPolling();
    try {
      await services.stopInference(state.inferenceProcessId);
    } finally {
      dispatch({ type: "SET_INFERENCE_PROCESS_ID", id: null });
      setStopping(false);
    }
  }

  function updateConfig(partial: Partial<typeof config>) {
    dispatch({ type: "SET_INFERENCE_CONFIG", config: partial });
  }

  const hasLogs = logs.length > 0;

  return (
    <StepCard
      title="Inference"
      description="Run a trained policy to autonomously control the robot."
      showNext={false}
    >
      <div className="space-y-5">
        {!hardwareReady && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Robot type, ports, cameras, and calibration must be configured
              before running inference.
            </AlertDescription>
          </Alert>
        )}

        {/* Model selection */}
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="model-type">Policy Model</Label>
            <Select
              value={config.modelType}
              onValueChange={(v) => updateConfig({ modelType: v })}
              disabled={isRunning}
            >
              <SelectTrigger id="model-type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {INFERENCE_MODELS.map((model) => (
                  <SelectItem
                    key={model.value}
                    value={model.value}
                    disabled={!model.supported}
                  >
                    <span className="flex items-center gap-2">
                      {model.label}
                      {!model.supported && (
                        <Badge
                          variant="secondary"
                          className="text-[10px] px-1.5 py-0"
                        >
                          Coming Soon
                        </Badge>
                      )}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Policy path */}
          <div className="space-y-1.5">
            <Label htmlFor="policy-path">Policy Path</Label>
            <div className="relative" ref={dropdownRef}>
              <div className="flex gap-1.5">
                <Input
                  id="policy-path"
                  placeholder="e.g. /path/to/outputs/train/act_policy or username/act_policy"
                  value={config.policyPath}
                  onChange={(e) => updateConfig({ policyPath: e.target.value })}
                  onFocus={() => savedPolicies.length > 0 && setShowPolicyDropdown(true)}
                  disabled={isRunning}
                  className="flex-1"
                />
                {savedPolicies.length > 0 && (
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    className="shrink-0"
                    disabled={isRunning}
                    onClick={() => setShowPolicyDropdown(!showPolicyDropdown)}
                  >
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                )}
              </div>
              {showPolicyDropdown && savedPolicies.length > 0 && (
                <div className="absolute z-50 mt-1 w-full rounded-md border bg-popover shadow-lg">
                  <div className="p-1">
                    <p className="px-2 py-1 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                      Trained Policies
                    </p>
                    {savedPolicies.map((p) => (
                      <div
                        key={p.path}
                        className="flex items-center gap-1 rounded-sm hover:bg-accent group"
                      >
                        <button
                          type="button"
                          className="flex-1 text-left px-2 py-1.5 text-sm font-mono truncate"
                          onClick={() => {
                            updateConfig({ policyPath: p.path });
                            setShowPolicyDropdown(false);
                          }}
                        >
                          {p.path}
                        </button>
                        <button
                          type="button"
                          className="p-1 mr-1 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-opacity"
                          title="Remove from saved policies"
                          onClick={(e) => {
                            e.stopPropagation();
                            removePolicyPath(p.path);
                            setSavedPolicies(getSavedPolicies());
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Local folder path or HuggingFace repo ID of the trained policy.
            </p>
          </div>

          {/* Eval repo ID */}
          <div className="space-y-1.5">
            <Label htmlFor="eval-repo-id">Evaluation Repo ID</Label>
            <Input
              id="eval-repo-id"
              placeholder="username/dataset_name"
              value={config.repoId}
              onChange={(e) => updateConfig({ repoId: e.target.value })}
              disabled={isRunning}
            />
            <p className="text-xs text-muted-foreground">
              Evaluation results will be saved to this HuggingFace dataset. The
              dataset name is automatically prefixed with{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono">eval_</code>{" "}
              — this is required for evaluation runs.
            </p>
            {config.repoId.trim() !== "" && (
              <p className="text-xs text-muted-foreground">
                Will be saved as{" "}
                <code className="rounded bg-muted px-1 py-0.5 font-mono">
                  {withEvalPrefix(config.repoId)}
                </code>
              </p>
            )}
            {config.repoId.trim() !== "" &&
              withEvalPrefix(config.repoId) === getLastInferenceRepoId() && (
              <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                <AlertTriangle className="inline h-3 w-3 mr-1 -mt-0.5" />
                This repo ID was used in a previous inference run. The old local cache will be cleared automatically before starting.
              </p>
            )}
          </div>

          {/* Task description */}
          <div className="space-y-1.5">
            <Label htmlFor="inference-task">Task Description</Label>
            <Input
              id="inference-task"
              placeholder="Use the same task description as during training"
              value={config.task}
              onChange={(e) => updateConfig({ task: e.target.value })}
              disabled={isRunning}
            />
          </div>

          {/* Episodes and timing */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="inf-episodes">Episodes</Label>
              <Input
                id="inf-episodes"
                type="number"
                min={1}
                value={config.numEpisodes || ""}
                onChange={(e) =>
                  updateConfig({
                    numEpisodes: parseInt(e.target.value) || 0,
                  })
                }
                className={config.numEpisodes < 1 ? "border-red-500" : ""}
                disabled={isRunning}
              />
              {config.numEpisodes < 1 && (
                <p className="text-xs text-red-500">Must be at least 1</p>
              )}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="inf-episode-time">Episode Time (s)</Label>
              <Input
                id="inf-episode-time"
                type="number"
                min={1}
                value={config.episodeTimeS || ""}
                onChange={(e) =>
                  updateConfig({
                    episodeTimeS: parseInt(e.target.value) || 0,
                  })
                }
                className={config.episodeTimeS < 1 ? "border-red-500" : ""}
                disabled={isRunning}
              />
              {config.episodeTimeS < 1 && (
                <p className="text-xs text-red-500">Must be at least 1</p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Switch
              id="inf-display-data"
              checked={config.displayData}
              onCheckedChange={(checked) =>
                updateConfig({ displayData: checked })
              }
              disabled={isRunning}
            />
            <Label htmlFor="inf-display-data">
              Display data during inference
            </Label>
          </div>
        </div>

        <Separator />

        {/* Start / Stop */}
        <div className="flex items-center gap-2">
          {!isRunning && !errorMsg && (
            <Button onClick={handleStart} disabled={starting || !canStart}>
              {starting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              {starting ? "Starting..." : "Run Inference"}
            </Button>
          )}
          {isRunning && (
            <Button
              variant="outline"
              onClick={handleStop}
              disabled={stopping}
            >
              {stopping ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Square className="mr-2 h-4 w-4" />
              )}
              Stop
            </Button>
          )}
        </div>

        {/* Running status */}
        {isRunning && (
          <div className="flex items-center gap-3 rounded-lg border border-blue-200 dark:border-blue-900 p-4">
            <Bot className="h-5 w-5 text-blue-500 shrink-0 animate-pulse" />
            <div className="flex-1">
              <p className="text-sm font-medium">
                Policy is controlling the robot
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                The robot is running autonomously. Press Stop when done.
              </p>
            </div>
          </div>
        )}

        {/* Error state */}
        {errorMsg && (
          <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
            <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-800 dark:text-red-200">
                Inference failed
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

        {/* Live camera + motor feeds */}
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
              <span className="text-muted-foreground/60">
                ({logs.length} lines)
              </span>
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
      </div>
    </StepCard>
  );
}
