"use client";

import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import type {
  WizardState,
  RobotMode,
  PortInfo,
  CameraInfo,
  CameraSelection,
  RecordingConfig,
  InferenceConfig,
  TrainingConfig,
  Phase,
} from "@/lib/wizard-types";
import {
  INITIAL_STATE,
  INITIAL_RECORDING_CONFIG,
  INITIAL_INFERENCE_CONFIG,
  INITIAL_TRAINING_CONFIG,
  SINGLE_PORT_ROLES,
  BIMANUAL_PORT_ROLES,
  validateBimanualCalibrationNames,
} from "@/lib/wizard-types";

// Actions
type Action =
  | { type: "GO_TO_STEP"; step: number }
  | { type: "SET_ROBOT_MODE"; mode: RobotMode }
  | { type: "SET_DETECTED_PORTS"; ports: PortInfo[] }
  | { type: "SET_PORT_ASSIGNMENT"; role: string; port: string }
  | { type: "SET_DETECTED_CAMERAS"; cameras: CameraInfo[] }
  | { type: "SET_CAMERA_SELECTIONS"; selections: CameraSelection[] }
  | { type: "TOGGLE_CAMERA"; opencvIndex: number; included: boolean }
  | { type: "SET_CAMERA_NAME"; opencvIndex: number; name: string }
  | { type: "SET_CALIBRATION_FILES"; key: string; files: string[] }
  | { type: "SET_CALIBRATION_SELECTION"; role: string; filename: string | null }
  | { type: "SET_NEW_CALIBRATION_NAME"; role: string; name: string }
  | { type: "SET_TELE_PROCESS_ID"; id: string | null }
  | { type: "SET_RECORDING_CONFIG"; config: Partial<RecordingConfig> }
  | { type: "SET_RECORD_PROCESS_ID"; id: string | null }
  | { type: "SET_TRAINING_CONFIG"; config: Partial<TrainingConfig> }
  | { type: "SET_TRAINING_JOB"; jobId: string; projectId: string }
  | { type: "SET_TRAINING_OUTPUT_MODEL"; modelId: string }
  | { type: "CLEAR_TRAINING_JOB" }
  | { type: "SET_INFERENCE_CONFIG"; config: Partial<InferenceConfig> }
  | { type: "SET_INFERENCE_PROCESS_ID"; id: string | null }
  | { type: "ADD_PHASE"; phase?: Partial<Phase> }
  | { type: "REMOVE_PHASE"; id: string }
  | { type: "UPDATE_PHASE"; id: string; patch: Partial<Phase> }
  | { type: "UPDATE_PHASE_CONFIG"; id: string; config: Partial<InferenceConfig> }
  | { type: "SET_ACTIVE_PHASE"; id: string | null }
  | { type: "TOGGLE_DEBUG_MODE" }
  | { type: "HYDRATE"; payload: PersistedWizardSettings }
  | { type: "CLEAR_ALL_VALUES" }
  | { type: "RESTART" };

// Step completion checker
function computeCompletedSteps(state: WizardState): boolean[] {
  const completed = [false, false, false, false, false, false, false, false, false];

  // Step 0: Robot Type
  completed[0] = state.robotMode !== null;

  // Step 1: Ports - all required roles assigned
  if (state.robotMode) {
    const roles =
      state.robotMode === "single" ? SINGLE_PORT_ROLES : BIMANUAL_PORT_ROLES;
    completed[1] = roles.every(
      (role) => state.portAssignments[role] && state.portAssignments[role] !== ""
    );
  }

  // Step 2: Cameras - optional, but must visit the step first
  const selectedCameras = state.cameraSelections.filter((c) => c.included);
  completed[2] = state.camerasStepVisited && selectedCameras.every((c) => c.name !== "");

  // Step 3: Calibration - all roles have a selection
  if (state.robotMode) {
    const calRoles =
      state.robotMode === "single"
        ? ["follower", "leader"]
        : ["left_follower", "right_follower", "left_leader", "right_leader"];
    const allSelected = calRoles.every((role) => {
      const sel = state.calibrationSelections[role];
      if (sel === undefined || sel === null) return false;
      if (sel === "new") return (state.newCalibrationNames[role] || "").trim() !== "";
      return true;
    });
    if (state.robotMode === "bimanual") {
      const validation = validateBimanualCalibrationNames(
        state.calibrationSelections,
        state.newCalibrationNames,
      );
      completed[3] = allSelected && validation.valid;
    } else {
      completed[3] = allSelected;
    }
  }

  // Steps 4-8: complete once the user has visited them
  completed[4] = state.teleStepVisited;
  completed[5] = state.recordStepVisited;
  completed[6] = state.trainingStepVisited;
  completed[7] = state.inferenceStepVisited;
  completed[8] = state.phasesStepVisited;

  return completed;
}

// Reset steps from a given index onwards
function resetStepsFrom(state: WizardState, fromStep: number): WizardState {
  let s = { ...state };

  if (fromStep <= 1) {
    s.detectedPorts = [];
    s.portAssignments = {};
  }
  if (fromStep <= 2) {
    s.camerasStepVisited = false;
    s.detectedCameras = [];
    s.cameraSelections = [];
  }
  if (fromStep <= 3) {
    s.calibrationFiles = {};
    s.calibrationSelections = {};
    s.newCalibrationNames = {};
  }
  if (fromStep <= 4) {
    s.teleStepVisited = false;
    s.teleProcessId = null;
  }
  if (fromStep <= 5) {
    s.recordStepVisited = false;
    s.recordingConfig = { ...INITIAL_RECORDING_CONFIG };
    s.recordProcessId = null;
  }
  if (fromStep <= 6) {
    s.trainingStepVisited = false;
    s.trainingConfig = { ...INITIAL_TRAINING_CONFIG };
    s.trainingJobId = null;
    s.trainingProjectId = null;
    s.trainingOutputModelId = null;
  }
  if (fromStep <= 7) {
    s.inferenceStepVisited = false;
    s.inferenceConfig = { ...INITIAL_INFERENCE_CONFIG };
    s.inferenceProcessId = null;
  }
  if (fromStep <= 8) {
    s.phasesStepVisited = false;
    s.phases = [];
    s.activePhaseId = null;
  }

  s.completedSteps = computeCompletedSteps(s);
  return s;
}

function reducer(state: WizardState, action: Action): WizardState {
  let next: WizardState;

  switch (action.type) {
    case "GO_TO_STEP":
      next = {
        ...state,
        currentStep: action.step,
        camerasStepVisited: state.camerasStepVisited || action.step === 2,
        teleStepVisited: state.teleStepVisited || action.step === 4,
        recordStepVisited: state.recordStepVisited || action.step === 5,
        trainingStepVisited: state.trainingStepVisited || action.step === 6,
        inferenceStepVisited: state.inferenceStepVisited || action.step === 7,
        phasesStepVisited: state.phasesStepVisited || action.step === 8,
      };
      break;

    case "SET_ROBOT_MODE": {
      // Changing robot type resets everything after step 0
      next = resetStepsFrom(
        { ...state, robotMode: action.mode },
        1
      );
      break;
    }

    case "SET_DETECTED_PORTS":
      next = { ...state, detectedPorts: action.ports };
      break;

    case "SET_PORT_ASSIGNMENT": {
      const newAssignments = { ...state.portAssignments };
      // If this port is already assigned to another role, swap them
      const previousPort = newAssignments[action.role] || "";
      for (const [otherRole, otherPort] of Object.entries(newAssignments)) {
        if (otherRole !== action.role && otherPort === action.port) {
          newAssignments[otherRole] = previousPort;
          break;
        }
      }
      newAssignments[action.role] = action.port;
      next = { ...state, portAssignments: newAssignments };
      break;
    }

    case "SET_DETECTED_CAMERAS":
      next = {
        ...state,
        detectedCameras: action.cameras,
        cameraSelections: action.cameras.map((c) => ({
          opencvIndex: c.opencvIndex,
          label: c.label,
          name: "",
          included: false,
        })),
      };
      break;

    case "SET_CAMERA_SELECTIONS":
      next = { ...state, cameraSelections: action.selections };
      break;

    case "TOGGLE_CAMERA":
      next = {
        ...state,
        cameraSelections: state.cameraSelections.map((c) =>
          c.opencvIndex === action.opencvIndex
            ? { ...c, included: action.included }
            : c
        ),
      };
      break;

    case "SET_CAMERA_NAME":
      next = {
        ...state,
        cameraSelections: state.cameraSelections.map((c) =>
          c.opencvIndex === action.opencvIndex ? { ...c, name: action.name } : c
        ),
      };
      break;

    case "SET_CALIBRATION_FILES":
      next = {
        ...state,
        calibrationFiles: {
          ...state.calibrationFiles,
          [action.key]: action.files,
        },
      };
      break;

    case "SET_CALIBRATION_SELECTION":
      next = {
        ...state,
        calibrationSelections: {
          ...state.calibrationSelections,
          [action.role]: action.filename,
        },
      };
      break;

    case "SET_NEW_CALIBRATION_NAME":
      next = {
        ...state,
        newCalibrationNames: {
          ...state.newCalibrationNames,
          [action.role]: action.name,
        },
      };
      break;

    case "SET_TELE_PROCESS_ID":
      next = { ...state, teleProcessId: action.id };
      break;

    case "SET_RECORDING_CONFIG":
      next = {
        ...state,
        recordingConfig: { ...state.recordingConfig, ...action.config },
      };
      break;

    case "SET_RECORD_PROCESS_ID":
      next = { ...state, recordProcessId: action.id };
      break;

    case "SET_TRAINING_CONFIG":
      next = {
        ...state,
        trainingConfig: { ...state.trainingConfig, ...action.config },
      };
      break;

    case "SET_TRAINING_JOB":
      next = {
        ...state,
        trainingJobId: action.jobId,
        trainingProjectId: action.projectId,
      };
      break;

    case "SET_TRAINING_OUTPUT_MODEL":
      next = {
        ...state,
        trainingOutputModelId: action.modelId,
      };
      break;

    case "CLEAR_TRAINING_JOB":
      next = {
        ...state,
        trainingJobId: null,
        trainingProjectId: null,
        trainingOutputModelId: null,
      };
      break;

    case "SET_INFERENCE_CONFIG":
      next = {
        ...state,
        inferenceConfig: { ...state.inferenceConfig, ...action.config },
      };
      break;

    case "SET_INFERENCE_PROCESS_ID":
      next = { ...state, inferenceProcessId: action.id };
      break;

    case "ADD_PHASE": {
      const id =
        action.phase?.id ??
        (typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `phase_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
      const baseConfig = action.phase?.config ?? {
        ...state.inferenceConfig,
      };
      const phase: Phase = {
        id,
        name: action.phase?.name ?? `Phase ${state.phases.length + 1}`,
        config: { ...baseConfig },
      };
      next = { ...state, phases: [...state.phases, phase] };
      break;
    }

    case "REMOVE_PHASE":
      next = {
        ...state,
        phases: state.phases.filter((p) => p.id !== action.id),
        activePhaseId:
          state.activePhaseId === action.id ? null : state.activePhaseId,
      };
      break;

    case "UPDATE_PHASE":
      next = {
        ...state,
        phases: state.phases.map((p) =>
          p.id === action.id ? { ...p, ...action.patch } : p
        ),
      };
      break;

    case "UPDATE_PHASE_CONFIG":
      next = {
        ...state,
        phases: state.phases.map((p) =>
          p.id === action.id
            ? { ...p, config: { ...p.config, ...action.config } }
            : p
        ),
      };
      break;

    case "SET_ACTIVE_PHASE":
      next = { ...state, activePhaseId: action.id };
      break;

    case "TOGGLE_DEBUG_MODE":
      next = { ...state, debugMode: !state.debugMode };
      break;

    case "HYDRATE": {
      // Apply persisted settings after mount. Kept out of the initial render
      // so the first client paint matches the server (avoids hydration errors).
      const saved = action.payload;
      next = {
        ...state,
        robotMode: saved.robotMode ?? state.robotMode,
        portAssignments: saved.portAssignments ?? state.portAssignments,
        cameraSelections: saved.cameraSelections ?? state.cameraSelections,
        calibrationSelections:
          saved.calibrationSelections ?? state.calibrationSelections,
        newCalibrationNames:
          saved.newCalibrationNames ?? state.newCalibrationNames,
        recordingConfig: {
          ...state.recordingConfig,
          ...(saved.recordingConfig ?? {}),
        },
        trainingConfig: {
          ...state.trainingConfig,
          ...(saved.trainingConfig ?? {}),
        },
        inferenceConfig: {
          ...INITIAL_INFERENCE_CONFIG,
          ...(saved.inferenceConfig ?? {}),
        },
        phases: saved.phases ?? state.phases,
        activePhaseId: saved.activePhaseId ?? state.activePhaseId,
      };
      break;
    }

    case "CLEAR_ALL_VALUES":
      next = { ...INITIAL_STATE, currentStep: state.currentStep };
      break;

    case "RESTART":
      next = { ...INITIAL_STATE };
      break;

    default:
      return state;
  }

  next.completedSteps = computeCompletedSteps(next);
  return next;
}

// Context
interface WizardContextValue {
  state: WizardState;
  dispatch: React.Dispatch<Action>;
  goToStep: (step: number) => void;
  goNext: () => void;
  clearAllValues: () => void;
  restart: () => void;
  allPriorStepsComplete: (step: number) => boolean;
}

const WizardContext = createContext<WizardContextValue | null>(null);

// Subset of WizardState that is safe to persist across sessions. Excludes
// runtime/detection state (process IDs, detected ports/cameras, current step,
// completion flags, debug toggle).
const PERSIST_KEY = "wizardSettings_v1";
const LEGACY_INFERENCE_KEY = "inferenceConfig";

type PersistedWizardSettings = Partial<
  Pick<
    WizardState,
    | "robotMode"
    | "portAssignments"
    | "cameraSelections"
    | "calibrationSelections"
    | "newCalibrationNames"
    | "recordingConfig"
    | "trainingConfig"
    | "inferenceConfig"
    | "phases"
    | "activePhaseId"
  >
>;

function readPersisted(): PersistedWizardSettings | null {
  try {
    const raw = localStorage.getItem(PERSIST_KEY);
    if (raw) return JSON.parse(raw) as PersistedWizardSettings;
    // Backwards compat: read legacy single-key inferenceConfig store
    const legacy = localStorage.getItem(LEGACY_INFERENCE_KEY);
    if (legacy) return { inferenceConfig: JSON.parse(legacy) };
  } catch {}
  return null;
}

function extractPersisted(state: WizardState): PersistedWizardSettings {
  return {
    robotMode: state.robotMode,
    portAssignments: state.portAssignments,
    cameraSelections: state.cameraSelections,
    calibrationSelections: state.calibrationSelections,
    newCalibrationNames: state.newCalibrationNames,
    recordingConfig: state.recordingConfig,
    trainingConfig: state.trainingConfig,
    inferenceConfig: state.inferenceConfig,
    phases: state.phases,
    activePhaseId: state.activePhaseId,
  };
}

export function WizardProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const hydratedRef = useRef(false);

  // Load persisted settings on the client after the first render. This keeps
  // the initial client render identical to the server's (both use
  // INITIAL_STATE), so reading from localStorage can't cause a hydration
  // mismatch.
  useEffect(() => {
    const saved = readPersisted();
    if (saved) dispatch({ type: "HYDRATE", payload: saved });
    hydratedRef.current = true;
  }, []);

  useEffect(() => {
    // Don't persist until after we've hydrated, otherwise the first run would
    // overwrite saved settings with INITIAL_STATE before HYDRATE applies them.
    if (!hydratedRef.current) return;
    try {
      localStorage.setItem(
        PERSIST_KEY,
        JSON.stringify(extractPersisted(state))
      );
    } catch {}
  }, [
    state.robotMode,
    state.portAssignments,
    state.cameraSelections,
    state.calibrationSelections,
    state.newCalibrationNames,
    state.recordingConfig,
    state.trainingConfig,
    state.inferenceConfig,
    state.phases,
    state.activePhaseId,
  ]);

  const goToStep = useCallback(
    (step: number) => dispatch({ type: "GO_TO_STEP", step }),
    []
  );

  const goNext = useCallback(
    () =>
      dispatch({
        type: "GO_TO_STEP",
        step: Math.min(state.currentStep + 1, 8),
      }),
    [state.currentStep]
  );

  const clearAllValues = useCallback(() => {
    try {
      localStorage.removeItem(PERSIST_KEY);
      localStorage.removeItem(LEGACY_INFERENCE_KEY);
    } catch {}
    dispatch({ type: "CLEAR_ALL_VALUES" });
  }, []);

  const restart = useCallback(() => dispatch({ type: "RESTART" }), []);

  const allPriorStepsComplete = useCallback(
    (step: number) => {
      for (let i = 0; i < step; i++) {
        if (!state.completedSteps[i]) return false;
      }
      return true;
    },
    [state.completedSteps]
  );

  return (
    <WizardContext.Provider
      value={{
        state,
        dispatch,
        goToStep,
        goNext,
        clearAllValues,
        restart,
        allPriorStepsComplete,
      }}
    >
      {children}
    </WizardContext.Provider>
  );
}

export function useWizard() {
  const ctx = useContext(WizardContext);
  if (!ctx) throw new Error("useWizard must be used within WizardProvider");
  return ctx;
}
