// Robot mode
export type RobotMode = "single" | "bimanual";

// Port info from backend
export interface PortInfo {
  port: string;
  description: string | null;
  hwid: string | null;
}

// Camera info from backend OpenCV detection (ground truth indices)
export interface CameraInfo {
  opencvIndex: number; // OpenCV camera index (from backend, ground truth)
  label: string;       // Camera name from system_profiler or fallback
}

// Camera selection in wizard
export interface CameraSelection {
  opencvIndex: number; // OpenCV camera index (key, ground truth from backend)
  label: string;       // display label
  name: string;        // "front_cam" | "hand_cam" | "side_cam"
  included: boolean;
}

// Recording configuration
export interface RecordingConfig {
  repoId: string;
  task: string;
  numEpisodes: number;
  episodeTimeS: number;
  resetTimeS: number;
  displayData: boolean;
  cameraFps: number;
  cameraWidth: number;
  cameraHeight: number;
}

// Training configuration (Qualia Studios)
export interface TrainingConfig {
  datasetId: string;
  vlaType: string;
  modelId: string; // HF base model ID, required for smolvla/pi0/pi05
  instanceType: string;
  batchSize: number;
  hours: number;
  outputModelName: string;
  jobDescription: string;
  cameraNames: string[];
}

// VLA types that require a base model_id for Qualia
export const VLA_TYPES_REQUIRING_MODEL_ID = ["smolvla", "pi0", "pi05"] as const;

// Default base model IDs per VLA type
export const DEFAULT_MODEL_IDS: Record<string, string> = {
  smolvla: "lerobot/smolvla_base",
};

// Training model options
export const TRAINING_MODELS = [
  { value: "act", label: "ACT", supported: true },
  { value: "smolvla", label: "SmolVLA", supported: true },
  { value: "pi0", label: "Pi0", supported: true },
  { value: "pi05", label: "Pi0.5", supported: true },
  { value: "gr00t_n1_5", label: "GR00T N1.5", supported: true },
] as const;

// Camera mapping: our UI names → Qualia camera slots
export const QUALIA_CAMERA_MAPPING: Record<string, string> = {
  front_cam: "image_top",
  hand_cam: "image_wrist",
  side_cam: "image_side",
};

// Training job phases (in order)
export const TRAINING_PHASES = [
  "queuing",
  "credit_validation",
  "instance_booting",
  "instance_activation",
  "instance_setup",
  "dataset_preprocessing",
  "training_running",
  "model_uploading",
  "completed",
] as const;

// Inference configuration
export interface InferenceConfig {
  policyPath: string;
  repoId: string;
  task: string;
  numEpisodes: number;
  episodeTimeS: number;
  resetTimeS: number; // idle seconds between episodes (reposition the scene)
  displayData: boolean;
  modelType: string; // "act" | "smolvla" | "diffusion" etc.
}

// A phase is a saved inference configuration that can be run as part of a sequence
export interface Phase {
  id: string;
  name: string;
  config: InferenceConfig;
}

// Compute the total wall-clock time for a phase (in seconds)
export function getPhaseTimeSeconds(phase: Phase): number {
  return Math.max(0, phase.config.numEpisodes) * Math.max(0, phase.config.episodeTimeS);
}

// Format a duration in seconds as "Xm Ys" or "Ys"
export function formatPhaseDuration(seconds: number): string {
  if (seconds <= 0) return "0s";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m === 0) return `${s}s`;
  if (s === 0) return `${m}m`;
  return `${m}m ${s}s`;
}

// Supported inference model types
export const INFERENCE_MODELS = [
  { value: "act", label: "ACT", supported: true },
  { value: "smolvla", label: "SmolVLA", supported: true },
  { value: "diffusion", label: "Diffusion Policy", supported: true },
  { value: "tdmpc", label: "TD-MPC", supported: true },
  { value: "vqbet", label: "VQ-BeT", supported: true },
  { value: "dino_dt", label: "DINO Diffusion Transformer", supported: true },
] as const;

// API start response
export interface StartResponse {
  process_id: string;
  message: string;
}

// Wizard state
export interface WizardState {
  currentStep: number; // 0-8
  completedSteps: boolean[];
  debugMode: boolean;

  // Step 0: Robot Type
  robotMode: RobotMode | null;

  // Step 1: Ports
  detectedPorts: PortInfo[];
  portAssignments: Record<string, string>; // role → port path

  // Step 2: Cameras
  camerasStepVisited: boolean;
  detectedCameras: CameraInfo[];
  cameraSelections: CameraSelection[];

  // Step 3: Calibration
  calibrationFiles: Record<string, string[]>; // "robots/so101_follower" → filenames
  calibrationSelections: Record<string, string | null>; // role → filename or "new" or null
  newCalibrationNames: Record<string, string>; // role → calibration name (must follow {base}_left / {base}_right for bimanual)

  // Step 4: Teleoperation
  teleStepVisited: boolean;
  teleProcessId: string | null;

  // Step 5: Recording
  recordStepVisited: boolean;
  recordingConfig: RecordingConfig;
  recordProcessId: string | null;

  // Step 6: Training (Qualia)
  trainingStepVisited: boolean;
  trainingConfig: TrainingConfig;
  trainingJobId: string | null;
  trainingProjectId: string | null;
  trainingOutputModelId: string | null;

  // Step 7: Inference
  inferenceStepVisited: boolean;
  inferenceConfig: InferenceConfig;
  inferenceProcessId: string | null;

  // Step 8: Phases
  phasesStepVisited: boolean;
  phases: Phase[];
  activePhaseId: string | null;
}

// Port roles by mode
export const SINGLE_PORT_ROLES = ["follower", "leader"] as const;
export const BIMANUAL_PORT_ROLES = [
  "left_follower",
  "right_follower",
  "left_leader",
  "right_leader",
] as const;

// Camera name options
export const CAMERA_NAME_OPTIONS = [
  "front_cam",
  "hand_cam",
  "side_cam",
] as const;

// Calibration directory mapping
export function getCalibrationPaths(mode: RobotMode): { role: string; category: string; robotType: string }[] {
  if (mode === "single") {
    return [
      { role: "follower", category: "robots", robotType: "so101_follower" },
      { role: "leader", category: "teleoperators", robotType: "so101_leader" },
    ];
  }
  // Bimanual sub-arms are SO101Leader/SO101Follower instances that look for
  // calibration files under so101_follower / so101_leader (their own class name),
  // NOT bi_so101_follower / bi_so101_leader (the wrapper name).
  return [
    { role: "left_follower", category: "robots", robotType: "so101_follower" },
    { role: "right_follower", category: "robots", robotType: "so101_follower" },
    { role: "left_leader", category: "teleoperators", robotType: "so101_leader" },
    { role: "right_leader", category: "teleoperators", robotType: "so101_leader" },
  ];
}

// Step definitions
export const STEPS = [
  { label: "Robot Type", description: "Choose your robot arm configuration" },
  { label: "Ports", description: "Detect and assign USB device ports" },
  { label: "Cameras", description: "Select and name your cameras" },
  { label: "Calibration", description: "Choose calibration for each arm" },
  { label: "Teleoperate", description: "Test robot teleoperation" },
  { label: "Record", description: "Record training data" },
  { label: "Training", description: "Train a policy with Qualia Studios" },
  { label: "Inference", description: "Run trained policy on robot" },
  { label: "Phases", description: "Run multiple inference phases in sequence" },
] as const;

// Initial state
export const INITIAL_TRAINING_CONFIG: TrainingConfig = {
  datasetId: "",
  vlaType: "act",
  modelId: "",
  instanceType: "",
  batchSize: 32,
  hours: 1,
  outputModelName: "",
  jobDescription: "",
  cameraNames: [],
};

export const INITIAL_INFERENCE_CONFIG: InferenceConfig = {
  policyPath: "",
  repoId: "",
  task: "",
  numEpisodes: 10,
  episodeTimeS: 50,
  resetTimeS: 10,
  displayData: true,
  modelType: "act",
};

export const INITIAL_RECORDING_CONFIG: RecordingConfig = {
  repoId: "",
  task: "",
  numEpisodes: 10,
  episodeTimeS: 60,
  resetTimeS: 10,
  displayData: true,
  cameraFps: 30,
  cameraWidth: 640,
  cameraHeight: 480,
};

export const INITIAL_STATE: WizardState = {
  currentStep: 0,
  completedSteps: [false, false, false, false, false, false, false, false, false],
  debugMode: false,
  robotMode: null,
  detectedPorts: [],
  portAssignments: {},
  camerasStepVisited: false,
  detectedCameras: [],
  cameraSelections: [],
  calibrationFiles: {},
  calibrationSelections: {},
  newCalibrationNames: {},
  teleStepVisited: false,
  teleProcessId: null,
  recordStepVisited: false,
  recordingConfig: { ...INITIAL_RECORDING_CONFIG },
  recordProcessId: null,
  trainingStepVisited: false,
  trainingConfig: { ...INITIAL_TRAINING_CONFIG },
  trainingJobId: null,
  trainingProjectId: null,
  trainingOutputModelId: null,
  inferenceStepVisited: false,
  inferenceConfig: { ...INITIAL_INFERENCE_CONFIG },
  inferenceProcessId: null,
  phasesStepVisited: false,
  phases: [],
  activePhaseId: null,
};

// ─── Bimanual calibration naming validation ─────────────────────────────────

export interface BimanualValidationResult {
  valid: boolean;
  followerBaseId: string | null;
  leaderBaseId: string | null;
  errors: string[];
}

/**
 * Resolve the effective calibration name for a role.
 * Returns null if the role has no selection yet.
 */
function resolveCalName(
  role: string,
  selections: Record<string, string | null>,
  newNames: Record<string, string>,
): string | null {
  const sel = selections[role];
  if (sel === undefined || sel === null) return null;
  if (sel === "new") {
    const name = (newNames[role] || "").trim();
    return name || null;
  }
  return sel.replace(/\.json$/, "");
}

/**
 * Validate that bimanual left/right calibration names share a common prefix
 * and use the correct _left / _right suffixes.
 *
 * Returns early with valid=false and empty errors when selections are incomplete
 * (user hasn't filled everything yet — no premature error messages).
 */
export function validateBimanualCalibrationNames(
  selections: Record<string, string | null>,
  newNames: Record<string, string>,
): BimanualValidationResult {
  const result: BimanualValidationResult = {
    valid: false,
    followerBaseId: null,
    leaderBaseId: null,
    errors: [],
  };

  const pairs: Array<{
    label: string;
    leftRole: string;
    rightRole: string;
    setBase: (id: string) => void;
  }> = [
    {
      label: "Follower",
      leftRole: "left_follower",
      rightRole: "right_follower",
      setBase: (id) => { result.followerBaseId = id; },
    },
    {
      label: "Leader",
      leftRole: "left_leader",
      rightRole: "right_leader",
      setBase: (id) => { result.leaderBaseId = id; },
    },
  ];

  for (const pair of pairs) {
    const leftName = resolveCalName(pair.leftRole, selections, newNames);
    const rightName = resolveCalName(pair.rightRole, selections, newNames);

    // Not ready yet — no errors, just not valid
    if (!leftName || !rightName) return result;

    if (!leftName.endsWith("_left")) {
      result.errors.push(
        `Left ${pair.label} calibration name "${leftName}" must end with "_left" (e.g. "my_robot_left").`
      );
    }
    if (!rightName.endsWith("_right")) {
      result.errors.push(
        `Right ${pair.label} calibration name "${rightName}" must end with "_right" (e.g. "my_robot_right").`
      );
    }

    if (result.errors.length > 0) continue;

    const leftPrefix = leftName.slice(0, -"_left".length);
    const rightPrefix = rightName.slice(0, -"_right".length);

    if (leftPrefix !== rightPrefix) {
      result.errors.push(
        `${pair.label} calibration names must share the same base prefix — got "${leftPrefix}" (left) vs "${rightPrefix}" (right).`
      );
    } else {
      pair.setBase(leftPrefix);
    }
  }

  result.valid = result.errors.length === 0 && result.followerBaseId !== null && result.leaderBaseId !== null;
  return result;
}
