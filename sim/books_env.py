"""ManiSkill environment: SO101 (SO100 model for now) on a tabletop with books.

Recreates the real desktop-manipulation scene (see sample_book_stack_image.png):
a dark table with a few books -- one paperback lying flat on the left, and a
cluster on the right with a thick hardcover standing on its edge and two thinner
books leaning against it. Every book is a *dynamic* rigid body, so the physics is
fully simulated (they rest, stack, topple, and can be pushed or grasped).

Run it with sim/demo_books.py.

NOTE: ManiSkill has no SO101 agent yet, so we use the kinematically-equivalent
`so100` arm. Swap `ROBOT_UID` once an SO101 agent is registered.
"""

from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

ROBOT_UID = "so100"

# Workspace is centered where the SO100 can reach on the table (matches the
# stock PickCubeSO100 config: robot at x=-0.615, objects around x=-0.46).
WORKSPACE_CENTER = np.array([-0.46, 0.0, 0.0])

# The table surface sits at z=0; everything rests relative to this. We recolor
# the wooden table mesh itself to black (see _paint_black) instead of laying a
# plate on top, so the whole table -- top, sides, legs -- reads black.
DESK_TOP = 0.0

# Off-white used for the page block of every book.
PAGES_COLOR = [0.90, 0.88, 0.82, 1.0]
# Thickness (half-extent) of the cover/spine slabs wrapped around the pages.
COVER_T = 0.0018

# A small "stand" / bookend on the right that the upright books lean against.
# Static body: a low base with a vertical back wall (the books rest on the wall).
STAND_COLOR = [0.12, 0.12, 0.14, 1.0]
STAND_BASE_HALF = [0.028, 0.072, 0.003]   # flat base on the desk
STAND_WALL_HALF = [0.006, 0.066, 0.056]   # vertical back wall (thin in x)
STAND_POS = [-0.53, -0.3875]             # (x, y) of the stand (absolute world)

# Friction for the leaning books so they don't slide out from under each other.
BOOK_FRICTION = 1.0

# The stand + the two leaning books form one assembly. We rotate the whole
# assembly CCW about z, around its centroid, so it can face a different way
# without breaking the books-leaning-on-the-stand geometry.
CLUSTER_CENTER = np.array([-0.2325, -0.3875])  # xy centroid (absolute world)
CLUSTER_YAW = np.pi / 2                    # 90 deg anticlockwise about z

# Book layout, matching sample_book_stack_image.png:
#   LEFT  -> two books stacked flat (pink "HyperFocus" on top of a darker book)
#   RIGHT -> two books standing nearly upright, leaning back against the stand
#
# Each book is a single dynamic rigid body that looks like a real closed book:
# an off-white page block, colored covers on top and bottom, and a colored spine
# down one long edge. `half_sizes` are the page-block [x, y, z] half-extents
# (x = long side, z = thickness). `pos` is [x, y, z] in ABSOLUTE world coords
# (table top is z=0, x in ~[-0.74, 0.47], y in ~[-1.21, 1.20], robot base near
# x=-0.61); `rpy` is (roll, pitch, yaw) in radians.
# A pitch near -pi/2 stands a book on its short edge with the long side vertical;
# easing it back toward -1.2 makes it lean against the stand (top toward +x).
BOOKS = [
    # --- LEFT stack: darker book on the bottom, lying flat ---
    dict(
        name="book_stack_bottom",
        half_sizes=[0.062, 0.046, 0.012],
        cover_color=[0.16, 0.34, 0.46, 1.0],
        # rests on the desk: center = full half-height (hz + 2c)
        pos=[-0.46, -0.04, 0.012 + 2 * COVER_T],
        rpy=[0.0, 0.0, -np.pi/2],
    ),
    # --- LEFT stack: pink "HyperFocus" resting on top of the bottom book ---
    dict(
        name="book_pink_top",
        half_sizes=[0.057, 0.041, 0.009],
        cover_color=[0.86, 0.16, 0.42, 1.0],
        # rests on the bottom book: 2*(bottom full half) + this book's full half
        pos=[-0.49, -0.05, 2 * (0.012 + 2 * COVER_T) + 0.009 + 2 * COVER_T],
        rpy=[0.0, 0.0, -np.pi/2],
    ),
    # --- RIGHT: big dark hardcover slanted back against the stand wall ---
    # Kinematic so the slant holds exactly. `cluster` = part of the rotated
    # stand assembly. Same row (y) as the red book; the red leans on its face.
    dict(
        name="book_dark_upright",
        half_sizes=[0.072, 0.055, 0.014],
        cover_color=[0.06, 0.06, 0.07, 1.0],
        pos=[-0.4625, -0.3875, 0.066],
        # pitch = vertical (-pi/2) leaned back by pi/8 (~22.5 deg)
        rpy=[0.0, -np.pi / 2 - np.pi / 8, 0.0],
        body_type="kinematic",
        cluster=True,
    ),
    # --- RIGHT: red book leaning only on the dark book, like books on a shelf ---
    # Placed just in front of the dark book, same slant, so it rests flush
    # against the dark book's exposed face.
    dict(
        name="book_red_upright",
        half_sizes=[0.062, 0.046, 0.011],
        cover_color=[0.52, 0.10, 0.10, 1.0],
        pos=[-0.4255, -0.3875, 0.060],
        # pitch = vertical (-pi/2) leaned back by pi/8 (~22.5 deg), same as dark
        rpy=[0.0, -np.pi / 2 - np.pi / 8, 0.0],
        body_type="kinematic",
        cluster=True,
    ),
]


def _rot_z(xy, center, angle):
    """Rotate the 2D point `xy` about `center` by `angle` radians (CCW)."""
    d = np.asarray(xy, dtype=float) - center
    c, s = np.cos(angle), np.sin(angle)
    return center + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _paint_black(actor, color=(0.02, 0.02, 0.02, 1.0)):
    """Recolor every visual material of an actor (e.g. the table mesh) to black.

    Walks each parallel-scene Entity -> RenderBodyComponent -> render shapes ->
    mesh parts and sets the PBR base color. The wood look comes from a base-color
    texture that overrides the color factor, so we clear that texture too.
    """
    for obj in actor._objs:
        rb = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if rb is None:
            continue
        for shape in rb.render_shapes:
            parts = shape.parts if hasattr(shape, "parts") else [shape]
            for part in parts:
                mat = part.material
                # Drop the wood textures so the base color is what shows.
                mat.set_base_color_texture(None)
                mat.set_diffuse_texture(None)
                mat.set_base_color(list(color))
                mat.set_metallic(0.0)
                mat.set_roughness(0.9)


def _build_book(
    scene, name, half_sizes, cover_color, initial_pose, body_type="dynamic", friction=None
):
    """Build one closed book as a single dynamic rigid body.

    Off-white page block with colored covers on the top/bottom faces and a
    colored spine down one long (-y) edge, so from above you see the cover and
    around the sides you see white page edges -- like the photo.
    """
    hx, hy, hz = half_sizes
    c = COVER_T
    cover_mat = sapien.render.RenderMaterial(base_color=cover_color)
    pages_mat = sapien.render.RenderMaterial(base_color=PAGES_COLOR)

    builder = scene.create_actor_builder()
    # Single collision box spanning the full OUTER extent, covers included. The
    # covers stick out c beyond the page block on the top/bottom faces, so the
    # collision half-height must be hz + 2c (not hz + c) or stacked books' covers
    # would visually sink into each other.
    phys_mat = None
    if friction is not None:
        phys_mat = sapien.physx.PhysxMaterial(friction, friction, 0.0)
    builder.add_box_collision(half_size=[hx + c, hy + c, hz + 2 * c], material=phys_mat)
    # Page block.
    builder.add_box_visual(half_size=[hx, hy, hz], material=pages_mat)
    # Top + bottom covers.
    for z in (hz + c, -(hz + c)):
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, z]),
            half_size=[hx + c, hy + c, c],
            material=cover_mat,
        )
    # Spine down the -y long edge.
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, -(hy + c), 0]),
        half_size=[hx + c, c, hz + c],
        material=cover_mat,
    )
    builder.set_initial_pose(initial_pose)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    return builder.build(name=name)


@register_env("BooksSO101-v1", max_episode_steps=200)
class BooksSO101Env(BaseEnv):
    """SO101/SO100 arm at a table with a physics-simulated cluster of books."""

    SUPPORTED_ROBOTS = ["so100"]
    agent: SO100

    def __init__(
        self,
        *args,
        robot_uids=ROBOT_UID,
        robot_init_qpos_noise=0.02,
        render_view="front",
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # which human-render camera to use: "front", "top", or "both"
        self.render_view = render_view
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _hand_camera_config(self, name, width, height):
        """Wrist / ego camera mounted on the gripper, like the SO101's gripper cam.

        Mounted on the Fixed_Jaw link; the pose is in that link's local frame.
        The gripper approaches along the link's local -y axis (toward the TCP),
        so we sit just above/behind the jaw and look down that approach axis at
        the gripping point. Built with look_at in local coords -> local pose.
        """
        mount = self.agent.robot.links_map["Fixed_Jaw"]
        local_pose = sapien_utils.look_at(
            eye=[-0.03, 0.04, -0.03],     # above + slightly behind the jaw
            target=[-0.03, -0.10, 0.02],  # the gripping point (near the TCP)
        )
        return CameraConfig(
            name, local_pose, width, height, np.pi / 2, 0.01, 100, mount=mount
        )

    # --- cameras: reuse the SO100 PickCube viewpoints so books are framed ---
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.27, 0, 0.4], target=[-0.56, 0, -0.25])
        # `front_cam` (640x480) is what the ACT policy sees. Pose/fovy copied from
        # the SAPIEN viewer (the camera previously labelled "top_camera").
        front_cam_pose = sapien.Pose(
            [-0.673923, -0.273742, 0.379082],
            [0.907975, -0.0489157, 0.401161, 0.110714],
        )
        return [
            CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100),
            # ego / wrist camera the policy sees, mounted near the gripper
            self._hand_camera_config("hand_camera", 128, 128),
            # front camera for policy inference (width=640, height=480, 90 deg vertical FOV)
            CameraConfig("front_cam", front_cam_pose, 640, 480, np.pi / 3.5, 0.1, 1000),
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Angled view from one side, matching the reference photo.
        front_pose = sapien_utils.look_at(
            eye=[-0.18, 0.30, 0.46], target=[-0.44, 0.03, 0.0]
        )
        front = CameraConfig("render_camera", front_pose, 512, 512, 1, 0.01, 100)
        # High-res version of the wrist cam, for previewing what the arm sees.
        wrist = self._hand_camera_config("wrist_camera", 512, 512)
        # NOTE: the policy's "front_cam" (the old top view) is a sensor camera, so
        # it already shows up in the viewer's camera dropdown in human mode.

        # In the interactive viewer, expose every render camera so they show up in
        # the viewer's camera dropdown. For offscreen --save, return just the one
        # picked by render_view (so --cam still selects a single image).
        if self.render_mode == "human":
            return [front, wrist]
        if self.render_view == "wrist":
            return wrist
        if self.render_view == "both":
            return [front, wrist]
        return front

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Paint the whole table black instead of its default wood texture.
        _paint_black(self.table_scene.table)

        # Static stand (base + vertical back wall) for the upright books. The
        # whole stand is rotated with the cluster (CLUSTER_YAW about z).
        sx, sy = STAND_POS
        stand_quat = euler2quat(0, 0, CLUSTER_YAW)
        base_xy = _rot_z([sx, sy], CLUSTER_CENTER, CLUSTER_YAW)
        base_pos = np.array(
            [base_xy[0], base_xy[1], DESK_TOP + STAND_BASE_HALF[2]]
        )
        actors.build_box(
            self.scene,
            half_sizes=STAND_BASE_HALF,
            color=STAND_COLOR,
            name="stand_base",
            body_type="static",
            initial_pose=sapien.Pose(p=base_pos, q=stand_quat),
        )
        wall_x = sx + STAND_BASE_HALF[0] - STAND_WALL_HALF[0]
        wall_xy = _rot_z([wall_x, sy], CLUSTER_CENTER, CLUSTER_YAW)
        wall_pos = np.array(
            [wall_xy[0], wall_xy[1], DESK_TOP + 2 * STAND_BASE_HALF[2] + STAND_WALL_HALF[2]]
        )
        actors.build_box(
            self.scene,
            half_sizes=STAND_WALL_HALF,
            color=STAND_COLOR,
            name="stand_wall",
            body_type="static",
            initial_pose=sapien.Pose(p=wall_pos, q=stand_quat),
        )

        self.books = []
        self._book_base_pose = []  # (pos_world, quat_wxyz) per book, for reset
        self._book_cluster = []    # whether each book is part of the stand cluster
        for cfg in BOOKS:
            # pos is [x, y, z] in ABSOLUTE world coords; z is above the desk top
            bx, by, bz = cfg["pos"]
            rpy = list(cfg["rpy"])
            is_cluster = cfg.get("cluster", False)
            # the leaning books rotate with the stand as one cluster
            if is_cluster:
                bx, by = _rot_z([bx, by], CLUSTER_CENTER, CLUSTER_YAW)
                rpy[2] += CLUSTER_YAW  # extra yaw = rotate orientation CCW about z
            pos = np.array([bx, by, bz + DESK_TOP])
            quat = euler2quat(*rpy)  # wxyz
            book = _build_book(
                self.scene,
                name=cfg["name"],
                half_sizes=cfg["half_sizes"],
                cover_color=cfg["cover_color"],
                initial_pose=sapien.Pose(p=pos, q=quat),
                body_type=cfg.get("body_type", "dynamic"),
                friction=BOOK_FRICTION if is_cluster else None,
            )
            self.books.append(book)
            self._book_base_pose.append((pos, quat))
            self._book_cluster.append(is_cluster)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            for book, (pos, quat), is_cluster in zip(
                self.books, self._book_base_pose, self._book_cluster
            ):
                p = torch.tensor(pos, dtype=torch.float32).repeat(b, 1)
                # small xy jitter so episodes aren't identical; skip the leaning
                # cluster books so the delicate lean settles consistently
                if not is_cluster:
                    p[:, :2] += (torch.rand((b, 2)) - 0.5) * 0.01
                q = torch.tensor(quat, dtype=torch.float32).repeat(b, 1)
                book.set_pose(Pose.create_from_pq(p, q))

    # --- minimal task plumbing (this is a scene/eval sandbox, no goal yet) ---
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        if "state" in self.obs_mode:
            # (num_envs, n_books * 7) flattened so the state obs stays 2D
            obs["book_poses"] = torch.cat(
                [bk.pose.raw_pose for bk in self.books], dim=1
            )
        return obs

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, dtype=bool, device=self.device),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)
