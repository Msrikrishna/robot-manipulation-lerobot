# Book Grasping on the Real SO101 — Modular Detector + IK Pipeline

## Context
We want the SO101 arm to pick books off a table. Today all book-picking is **end-to-end**
(ACT / ACT-DINO / DINO-DT / MolmoAct2 map RGB straight to joint actions) with no explicit
perception, which is hard to debug and gives no control over *which* book is grabbed.

Decision (confirmed with user): build a **modular** pipeline on the **real SO101** first —
detect book bounding boxes -> compute a 3D grasp pose -> execute via **inverse kinematics**.
Depth comes from **Depth Anything V2** (metric `da2-indoor` head).

The payoff: an interpretable, inspectable grasp loop where each stage (detection, pose,
IK, execution) can be validated independently, and where we can deliberately select a book
out of many in the frame.

## Pipeline (per grasp)
RGB frame from the fixed front camera (640x480) ->
1. **Detect** book boxes (open-vocab "book").
2. **Select** target book by a clearance/occlusion/topmost score.
3. **Orient** — segment the chosen box, PCA on the mask -> book long-axis angle (gripper yaw).
4. **Depth** — Depth Anything V2 metric depth at the grasp pixel (mask centroid).
5. **Back-project** pixel + depth via camera intrinsics -> camera-frame XYZ.
6. **Transform** to robot base frame via the hand-eye extrinsic T_base_cam.
7. **Build** a 4x4 desired EE pose: position = grasp point, orientation = top-down with yaw
   aligned to the book axis. Add a hover offset above for pre-grasp.
8. **IK** -> joint degrees via `RobotKinematics.inverse_kinematics`.
9. **Execute** state machine: hover -> descend -> close gripper -> lift -> place.
10. Optional closed loop: re-detect after each grasp (scene changes).

## Critical prerequisites (do these FIRST — geometry must be right before detection matters)
- **SO101 URDF**: none in the repo. Source from the lerobot / SO-ARM100 upstream, drop into the
  repo, and point IK at it (`robot.urdf_path=...`, frame `gripper_frame_link`). Requires the
  Placo library. *Ask before installing into the conda env.*
- **Camera intrinsics**: calibrate the front camera with an OpenCV checkerboard
  (`cv2.calibrateCamera`) -> fx, fy, cx, cy. None exist today.
- **Hand-eye extrinsic (eye-to-hand, fixed camera)**: attach an ArUco/checkerboard to the
  gripper, move the arm to N known joint poses, run `cv2.calibrateHandEye` -> T_base_cam.
  This is the highest-risk, fiddliest step and gates everything downstream.
- **Depth scale check**: metric Depth Anything is approximate. Anchor it with a small linear
  scale/offset correction against the known table height + a couple of measured points.

## New files (suggest a new `grasp/` dir)
- `grasp/calibrate_intrinsics.py` — checkerboard intrinsics capture/solve.
- `grasp/calibrate_handeye.py` — drive arm to poses, collect marker detections, solve T_base_cam.
- `grasp/detect_books.py` — run the detector, return boxes (+ select target, + mask PCA axis).
- `grasp/grasp_pose.py` — box + depth + intrinsics + extrinsics -> 4x4 base-frame grasp pose.
- `grasp/run_grasp.py` — orchestrator: capture -> detect -> pose -> IK -> execute state machine.

## Reuse (do not rewrite)
- IK/FK: `lerobot-MakerMods/src/lerobot/model/kinematics.py` (`RobotKinematics`).
- Arm + gripper commands: `lerobot-MakerMods/src/lerobot/robots/so101_follower/so101_follower.py`
  (`send_action`, degrees mode, gripper 0=open..100=close).
- EE-control pattern: `lerobot-MakerMods/examples/so100_to_so100_EE/record.py` and
  `.../so100_follower/robot_kinematic_processor.py` (fork for SO101).
- Depth: `dino-inference/depth_estimation.py` (`da2-indoor`, metric meters, MPS-capable).
- Camera capture: SO101 follower's OpenCV camera path (`cam.async_read()`), or
  `MakerMods-App/backend/services/camera_scanner.py` for device discovery.

## Detector choice (recommended default)
Start with an **open-vocab detector** (Grounding DINO or YOLO-World) prompted with "book" +
**FastSAM/SAM2** for the mask -> PCA gives orientation without training an oriented-box model.
If detection quality on stacked/leaning books is poor, switch to a fine-tuned **YOLOv8-OBB**
(oriented boxes give yaw directly). Requires installing the detector package — *ask first.*

## Grasping notes specific to books
- Leaning/standing books are far more graspable with a parallel gripper than flat books on a
  table. Prefer those targets; for flat books, plan a pre-grasp (tip-up or push-to-edge) later.
- Selection score per box: low IoU overlap with neighbors (occlusion), finger clearance on
  both sides of the spine, topmost-in-stack, reachability from the arm base.

## Phasing & verification

### Step 1 — Perception pipeline [DONE]
Built a full book detection + segmentation pipeline in `grasp/`:

**Detection approach: Grounding DINO + SAM2**
- Grounding DINO (`IDEA-Research/grounding-dino-base`) — open-vocab text-prompted detector.
  Returns N candidate boxes for prompt `"book."`. Runs on CPU (MPS unsupported due to `cummax`).
- SAM2 (`facebook/sam2-hiera-*`) — segments the dominant object inside each GDINO box.
  Masks upsampled from 256x256 via `F.interpolate` to original image size. Runs on CPU.
- Mask-IoU dedup — greedy NMS on pixel masks (not boxes) to collapse duplicate detections
  of the same physical book. Also suppresses masks >85% contained in a larger mask.
- `cv2.minAreaRect` on each surviving mask -> centroid + spine angle (gripper yaw) + oriented box.

**Key files:**
- `grasp/detect_books_grounded_sam.py` — CLI script, outputs annotated image + JSON per book
- `grasp/webapp/app.py` — FastAPI tuning UI at port 8009 with three tabs:
  - *DINO Boxes*: raw GDINO detections, fast (~3s), good for threshold tuning
  - *SAM Raw*: full-image point-grid segmentation (no GDINO), shows what SAM2 sees independently
  - *SAM + Dedup*: full pipeline with oriented boxes, the final per-book result

**Why this approach over alternatives:**
- YOLO-World: faster but boxes too loose for overlapping/leaning books
- Box-IoU NMS: can't distinguish "same book (duplicate)" from "adjacent book" at any threshold
- Mask-IoU NMS: pixel masks of adjacent books have low overlap even when their boxes heavily overlap
- DINOv3 PCA: visually similar books don't separate well in feature space — not useful here

**Output per book:** `{box, obb, centroid, angle_deg, area_px, score}`

---

- **Phase 0 — IK ready:** add SO101 URDF + Placo; verify FK->IK round-trips on a known joint
  pose (IK of FK(pose) returns the same joints).
- **Phase 1 — geometry sanity:** intrinsics + hand-eye done. Build a "click a table pixel ->
  arm touches that point" tool. *If this is accurate (~1cm) the hard part is done.*
- **Phase 2 — perception:** detector + selection + mask-PCA orientation; overlay boxes/axis on
  the live frame and eyeball correctness.
- **Phase 3 — 3D pose:** depth + back-projection + base transform; print/visualize the computed
  base-frame grasp point and compare to a tape-measured ground truth.
- **Phase 4 — execution:** hover -> descend -> close -> lift -> place state machine via IK +
  `send_action`; dry-run hover-only first (no gripper close) to confirm reach before full grasp.
- **End-to-end:** place several books, run `grasp/run_grasp.py`, confirm it selects a sensible
  book, aligns to its axis, grasps, lifts, and re-detects.

## Open risks
- Hand-eye accuracy dominates success; budget iteration here.
- Metric depth drift -> table-anchored correction is likely necessary.
- SO101 URDF availability/accuracy upstream; FK must be validated before trusting IK.
- Env installs (Placo, detector, SAM) need approval per project policy.
