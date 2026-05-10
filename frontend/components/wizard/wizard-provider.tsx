"use client";

import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useEffect,
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

function getInitialState(): WizardState {
  if (typeof window === "undefined") return INITIAL_STATE;
  try {
    const saved = localStorage.getItem("inferenceConfig");
    if (saved) {
      return {
        ...INITIAL_STATE,
        inferenceConfig: { ...INITIAL_INFERENCE_CONFIG, ...JSON.parse(saved) },
      };
    }
  } catch {}
  return INITIAL_STATE;
}

export function WizardProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE, getInitialState);

  useEffect(() => {
    try {
      localStorage.setItem("inferenceConfig", JSON.stringify(state.inferenceConfig));
    } catch {}
  }, [state.inferenceConfig]);

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

  const clearAllValues = useCallback(
    () => dispatch({ type: "CLEAR_ALL_VALUES" }),
    []
  );

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
