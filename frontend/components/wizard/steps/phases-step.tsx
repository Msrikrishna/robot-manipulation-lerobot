"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Clock,
  Home,
  Loader2,
  Pencil,
  Play,
  Plus,
  Trash2,
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
import { formatPhaseDuration, getPhaseTimeSeconds } from "@/lib/wizard-types";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";
import { InferenceStep } from "./inference-step";

export function PhasesStep() {
  const { state, dispatch } = useWizard();
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [returningHome, setReturningHome] = useState(false);
  const [homeMsg, setHomeMsg] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const hardwareReady =
    state.completedSteps[0] &&
    state.completedSteps[1] &&
    state.completedSteps[2] &&
    state.completedSteps[3];

  const activePhase = state.phases.find((p) => p.id === state.activePhaseId) ?? null;
  const inferenceRunning = state.inferenceProcessId !== null;

  // Load the active phase's config into wizard.inferenceConfig whenever the
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
    // If user clicked a different phase, persist current phase's edits first.
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
    // Persist edits back to the active phase, but keep the phase active so
    // we stay on it until the user explicitly picks another.
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

  const totalSeconds = state.phases.reduce(
    (acc, p) => acc + getPhaseTimeSeconds(p),
    0
  );

  return (
    <StepCard
      title="Phases"
      description="Define multiple inference phases. Open a phase to configure or run it."
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
              return (
                <div
                  key={phase.id}
                  className={
                    "flex items-center gap-3 rounded-lg border p-3 transition-colors " +
                    (isActive
                      ? "border-primary bg-primary/5"
                      : "hover:bg-muted/40")
                  }
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium">
                    {idx + 1}
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
                      {isActive && (
                        <span className="text-primary font-medium">active</span>
                      )}
                    </div>
                  </div>
                  <Button
                    variant={isActive ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleOpenPhase(phase.id)}
                    disabled={!hardwareReady}
                  >
                    <Play className="mr-1.5 h-3.5 w-3.5" />
                    {isActive ? "Resume" : "Open"}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => handleRemovePhase(phase.id)}
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
