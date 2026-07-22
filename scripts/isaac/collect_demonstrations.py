#!/usr/bin/env python3
"""
Collect OpenArm grasping demonstrations.
Motion planning is handled by MoveIt2 (ROS2 side); this script manages
the Isaac Lab simulation, episode resets, data recording, and HDF5 output.

Setup — 共需要 3 個終端機（跟原本測試 platform_grasp 的 2 terminal 架構相同）：

  Terminal 1  (ROS2 all-in-one — ros2_control + move_group + merger):
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    ros2 launch openarm_bimanual_moveit_config demo.launch.py \\
        use_isaac_sim:=true launch_rviz:=false

  Terminal 2  (episode controller — 自動呼叫 platform_grasp):
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    python3 scripts/ros2_episode_controller.py

  Terminal 3  (this script — Isaac Lab simulation):
    conda run -n env_isaaclab TERM=xterm \\
        ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/collect_demonstrations.py \\
        --headless --enable_cameras --num_episodes 50

  或使用 tmux 一鍵啟動 Terminal 1+2：
    bash scripts/launch_ros2_stack.sh

Episode coordination file: /tmp/openarm_episode.json
  Isaac Lab writes:  {"state": "ready",   "cube": [x, y, z]}
  ROS2 ctrl writes:  {"state": "running"}
  ROS2 ctrl writes:  {"state": "done",    "success": true|false}
"""

# ---------------------------------------------------------------------------
# AppLauncher MUST be created before any omni / isaaclab imports
# ---------------------------------------------------------------------------
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect OpenArm grasping demonstrations (MoveIt2 backend).")
parser.add_argument("--num_episodes",          type=int,   default=50)
parser.add_argument("--output_dir",            type=str,   default="./datasets")
parser.add_argument("--max_steps_per_episode", type=int,   default=3000,
                    help="Timeout (sim steps) per episode. MoveIt2 pick-and-place ~150 s = ~9000 steps.")
parser.add_argument("--image_width",           type=int,   default=320)
parser.add_argument("--image_height",          type=int,   default=240)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (after app is up)
# ---------------------------------------------------------------------------
import json
import os
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

import omni.usd
import omni.graph.core as og
import omni.replicator.core as rep
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from pxr import UsdPhysics, UsdShade
import usdrt.Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ---------------------------------------------------------------------------
# Scene constants (verified from USD inspection)
# ---------------------------------------------------------------------------
USD_PATH = os.path.expanduser("~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env_test.usd")

ROBOT_PRIM       = "/World/openarm"
CUBE_PRIM        = "/World/Cube"
PLATFORM_PRIM    = "/World/plateform"
CAM_MAIN_PRIM    = "/CameraMain"
CAM_WRIST_L_PRIM = "/World/openarm/openarm_left_link7/openarm_left_hand/CameraWristLeft"
CAM_WRIST_R_PRIM = "/World/openarm/openarm_right_link7/openarm_right_hand/CameraWristRight"

PLATFORM_TOP_Z = 0.193
CUBE_HALF_SIDE = 0.020
CUBE_RESTING_Z = 0.21344  # measured: platform convex-hull top (0.19344) + cube half (0.020)

# Cube randomisation range (within platform surface, within left-arm reach)
CUBE_RAND_X = (0.20, 0.32)   # platform near edge ≈ 0.179; +cube half (0.02) → safe min 0.20
CUBE_RAND_Y = (0.10, 0.22)

SIM_DT         = 1.0 / 60.0
SETTLE_STEPS   = 90   # sim steps to let cube settle after teleport

# Episode coordination file
EPISODE_FILE = "/tmp/openarm_episode.json"

# Success criterion: cube must move at least 5 cm horizontally from its start position
SUCCESS_XY_DIST = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "left_arm":      ImplicitActuatorCfg(joint_names_expr=["openarm_left_joint[1-7]"],     stiffness=400.0, damping=40.0),
            "left_gripper":  ImplicitActuatorCfg(joint_names_expr=["openarm_left_finger_joint1"],  stiffness=10000.0, damping=150.0),
            "right_arm":     ImplicitActuatorCfg(joint_names_expr=["openarm_right_joint[1-7]"],    stiffness=400.0,   damping=40.0),
            "right_gripper": ImplicitActuatorCfg(joint_names_expr=["openarm_right_finger_joint1"], stiffness=10000.0, damping=150.0),
        },
    )


def make_cube_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(prim_path=CUBE_PRIM, spawn=None, init_state=RigidObjectCfg.InitialStateCfg())


def setup_cameras(img_w: int, img_h: int):
    rp_main = rep.create.render_product(CAM_MAIN_PRIM,    (img_w, img_h))
    rp_wl   = rep.create.render_product(CAM_WRIST_L_PRIM, (img_w, img_h))
    rp_wr   = rep.create.render_product(CAM_WRIST_R_PRIM, (img_w, img_h))

    def _ann(rp):
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rp])
        return a

    return _ann(rp_main), _ann(rp_wl), _ann(rp_wr)


def capture_rgb(annotator) -> np.ndarray | None:
    data = annotator.get_data()
    if data is None or len(data) == 0:
        return None
    img = data.get("data", None) if isinstance(data, dict) else data
    if img is None:
        return None
    return np.asarray(img)[..., :3]


def random_cube_pose(device: str) -> torch.Tensor:
    x = np.random.uniform(*CUBE_RAND_X)
    y = np.random.uniform(*CUBE_RAND_Y)
    return torch.tensor([[x, y, CUBE_RESTING_Z, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)


def set_platform_static(stage, prim_path: str):
    """Make the platform a static collider by removing RigidBodyAPI.

    The USD has the platform as a dynamic rigid body (kinematicEnabled=False),
    which means gravity would pull it down.  Setting kinematicEnabled=True does
    NOT fix cube collision (kinematic body + convexHull does not stop dynamic
    objects in this USD setup).  Removing RigidBodyAPI entirely makes it a
    pure static collider — PhysX static colliders support convexHull shapes and
    are immovable, which is exactly what we need.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print(f"[INFO] Removed RigidBodyAPI from '{prim_path}' → static collider.")


def apply_friction_materials(stage, static_friction: float = 1.0, dynamic_friction: float = 0.8):
    """Bind a high-friction physics material to finger collision prims and the cube.

    Targets any prim that has CollisionAPI and whose path contains 'finger',
    plus the Cube prim itself.
    """
    mat_path = "/World/PhysicsMaterials/HighFriction"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.CreateStaticFrictionAttr().Set(static_friction)
    phys_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    mat = UsdShade.Material(mat_prim)

    bound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        is_finger = "finger" in path.lower()
        is_cube   = path == CUBE_PRIM or path.startswith(CUBE_PRIM + "/")
        if (is_finger or is_cube) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, materialPurpose="physics"
            )
            bound += 1

    print(f"[INFO] Applied high-friction material (μs={static_friction}, μd={dynamic_friction}) to {bound} collision prims.")


def clear_baked_graphs(stage):
    """Remove pre-baked OmniGraph prims from USD (saved from GUI) to avoid null-prim crash."""
    to_delete = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "OmniGraph"]
    for path in to_delete:
        stage.RemovePrim(path)
    if to_delete:
        print(f"[INFO] Removed {len(to_delete)} pre-baked OmniGraph prims.")


def enable_ros2_bridge():
    """Enable the isaacsim.ros2.bridge extension (required for OmniGraph ROS2 nodes)."""
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        print("[INFO] Enabled isaacsim.ros2.bridge extension.")
    else:
        print("[INFO] isaacsim.ros2.bridge already enabled.")


def setup_ros2_action_graph(robot_prim: str):
    """
    Create an OmniGraph that:
      - Publishes robot joint states → /Isaac/isaac_joint_states
      - Subscribes joint commands   ← /Isaac/isaac_joint_command  (merged by isaac_joint_command_merger)
      - Applies commands to the robot articulation
    """
    og.Controller.edit(
        {"graph_path": "/ROS2ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick",      "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime",         "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context",             "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState",   "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationCtrl",    "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock",        "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick",      "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",      "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",      "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",      "ArticulationCtrl.inputs:execIn"),
                ("Context.outputs:context",          "PublishJointState.inputs:context"),
                ("Context.outputs:context",          "SubscribeJointState.inputs:context"),
                ("Context.outputs:context",          "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime","PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime","PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames",      "ArticulationCtrl.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationCtrl.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationCtrl.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand",   "ArticulationCtrl.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                # Match the topics defined in openarm_robot.xacro (use_isaac_sim=true)
                ("PublishJointState.inputs:topicName",   "Isaac/isaac_joint_states"),
                ("SubscribeJointState.inputs:topicName", "Isaac/isaac_joint_command"),
                # ArticulationRootAPI is on root_joint, not the parent Xform
                ("PublishJointState.inputs:targetPrim",  [usdrt.Sdf.Path(robot_prim + "/root_joint")]),
                ("ArticulationCtrl.inputs:robotPath",     robot_prim + "/root_joint"),
            ],
        },
    )
    print("[INFO] ROS2 OmniGraph created:")
    print(f"         publish  → /Isaac/isaac_joint_states")
    print(f"         subscribe← /Isaac/isaac_joint_command")
    print(f"         robot    → {robot_prim}/root_joint")


# ---------------------------------------------------------------------------
# Episode coordination (file-based IPC with ROS2 episode controller)
# ---------------------------------------------------------------------------

def write_episode(data: dict):
    with open(EPISODE_FILE, "w") as f:
        json.dump(data, f)


def read_episode() -> dict:
    try:
        with open(EPISODE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


def wait_for_ros2_ready(timeout_s: float = 60.0) -> bool:
    """Block until the ROS2 episode controller acknowledges by writing 'idle'."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ctrl = read_episode()
        if ctrl.get("state") in ("idle", "done"):
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # OmniGraph ROS2 bridge (ROS2PublishJointState / ArticulationCtrl) uses
    # CPU-based PhysX tensor API.  Using cuda:0 causes device-mismatch errors.
    # For single-robot demo collection at 60 Hz, CPU physics is sufficient.
    device = "cpu"

    # ------------------------------------------------------------------
    # 1. Open USD stage and wait for async loading to complete
    # ------------------------------------------------------------------
    usd_abs = str(Path(USD_PATH).resolve())
    print(f"[INFO] Opening stage: {usd_abs}")
    stage_utils.open_stage(usd_abs)

    # open_stage() is asynchronous — wait until the stage is fully loaded
    # before SimulationContext initialises the physics scene.
    while stage_utils.is_stage_loading():
        simulation_app.update()
    print("[INFO] Stage loaded.")

    # ------------------------------------------------------------------
    # 2. Platform → static collider (must be BEFORE SimulationContext so
    #    PhysX initialises the platform without RigidBodyAPI)
    # ------------------------------------------------------------------
    stage = omni.usd.get_context().get_stage()
    clear_baked_graphs(stage)
    set_platform_static(stage, PLATFORM_PRIM)
    apply_friction_materials(stage)

    # ------------------------------------------------------------------
    # 3. Simulation context
    # ------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # ------------------------------------------------------------------
    # 4. Asset wrappers
    # ------------------------------------------------------------------
    robot = Articulation(make_robot_cfg())
    cube  = RigidObject(make_cube_cfg())

    # ------------------------------------------------------------------
    # 5. Cameras
    # ------------------------------------------------------------------
    ann_main, ann_wl, ann_wr = setup_cameras(args_cli.image_width, args_cli.image_height)

    # ------------------------------------------------------------------
    # 6. Enable ROS2 bridge extension
    # ------------------------------------------------------------------
    enable_ros2_bridge()

    # ------------------------------------------------------------------
    # 7. Start simulation (sim.reset() must come before OmniGraph setup)
    # ------------------------------------------------------------------
    sim.reset()
    print("[INFO] Simulation started.")

    # ------------------------------------------------------------------
    # 8. Set up OmniGraph for ROS2 joint state pub/sub
    # ------------------------------------------------------------------
    setup_ros2_action_graph(ROBOT_PRIM)

    # Need a few steps for OmniGraph to initialise
    for _ in range(10):
        sim.step()
        sim.render()

    # Snapshot default joint state for resets
    robot.update(SIM_DT)
    cube.update(SIM_DT)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    n_joints = robot.data.joint_pos.shape[-1]
    print(f"[INFO] Total robot joints: {n_joints}")

    # ------------------------------------------------------------------
    # 9. Initialise episode coordination file
    # ------------------------------------------------------------------
    write_episode({"state": "idle"})
    print(f"[INFO] Episode control file: {EPISODE_FILE}")
    print("[INFO] Waiting for ROS2 episode controller to be ready...")
    print("[INFO]   → Run in another terminal:")
    print("[INFO]       source ~/openarm_isaaclab/ros2_ws/install/setup.bash")
    print("[INFO]       python3 scripts/ros2_episode_controller.py")
    print()

    # ------------------------------------------------------------------
    # 10. Output directory（依日期時間建立子資料夾）
    # ------------------------------------------------------------------
    run_dir = os.path.join(args_cli.output_dir,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving episodes to: {run_dir}")

    # ------------------------------------------------------------------
    # 11. Episode loop
    # ------------------------------------------------------------------
    total_success = 0

    for ep_idx in range(args_cli.num_episodes):
        print(f"\n[EP {ep_idx:03d}] ─────────────────────────────")

        # ---- Reset robot to default pose (direct sim write, bypasses OmniGraph) ----
        # reset() 先清除內部緩衝區，再 write 以避免 reset 覆蓋掉 teleport 指令
        robot.reset()
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)

        # ---- Randomise cube ----
        # 同上：先 reset 再 write，確保 teleport 指令不被清除
        cube.reset()
        cube_pose = random_cube_pose(device)
        cube.write_root_pose_to_sim(cube_pose)
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        # ---- Settle ----
        # Do NOT call cube.write_data_to_sim() here — that writes the stale
        # internal buffer (USD initial pose) back to PhysX, overriding the
        # randomised position we just set.
        for _ in range(SETTLE_STEPS):
            robot.write_data_to_sim()
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            cube.update(SIM_DT)

        # Read actual physics position (not the requested tensor)
        cube.update(SIM_DT)
        cx = cube.data.root_pose_w[0, 0].item()
        cy = cube.data.root_pose_w[0, 1].item()
        cz = cube.data.root_pose_w[0, 2].item()
        print(f"[EP {ep_idx:03d}] Cube at ({cx:.3f}, {cy:.3f}, {cz:.3f})")

        # ---- Signal ROS2 episode controller ----
        write_episode({
            "state": "ready",
            "cube": [cx, cy, cz],
            "episode": ep_idx,
        })
        print(f"[EP {ep_idx:03d}] Waiting for ROS2 grasp execution...")

        # ---- Run sim loop while ROS2 executes grasp ----
        ep_joint_pos  = []
        ep_joint_vel  = []
        ep_ee_pos     = []
        ep_cube_pos   = []
        ep_cube_quat  = []
        ep_cam_main   = []
        ep_cam_wl     = []
        ep_cam_wr     = []
        ep_actions    = []  # joint positions at each step (executed trajectory)

        success = False
        step = 0
        last_state = "ready"

        while step < args_cli.max_steps_per_episode:
            # Step simulation.
            # Do NOT call robot.write_data_to_sim() here — that would overwrite
            # the drive targets that OmniGraph's ArticulationController just set
            # from the ROS2 joint command.  The robot is driven by ROS2/MoveIt2
            # during the episode; Isaac Lab is read-only (just recording state).
            cube.write_data_to_sim()
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            cube.update(SIM_DT)

            # Read observations
            cur_joint_pos  = robot.data.joint_pos.clone()
            cur_joint_vel  = robot.data.joint_vel.clone()
            cur_cube_state = cube.data.root_pose_w

            # Find left EE body (link7) index
            # (cached on first call; we use find_bodies each time for simplicity)
            # Record: use joint_pos as "action" (what the robot is actually doing)
            ep_joint_pos.append(cur_joint_pos[0].cpu().numpy())
            ep_joint_vel.append(cur_joint_vel[0].cpu().numpy())
            ep_cube_pos.append(cur_cube_state[0, :3].cpu().numpy())
            ep_cube_quat.append(cur_cube_state[0, 3:7].cpu().numpy())
            ep_actions.append(cur_joint_pos[0].cpu().numpy())

            img_m  = capture_rgb(ann_main)
            img_wl = capture_rgb(ann_wl)
            img_wr = capture_rgb(ann_wr)
            if img_m is not None:
                ep_cam_main.append(img_m)
                ep_cam_wl.append(img_wl)
                ep_cam_wr.append(img_wr)

            step += 1

            # Check episode control file every 30 steps (~0.5 s)
            if step % 30 == 0:
                ctrl = read_episode()
                state = ctrl.get("state", "idle")

                if state != last_state:
                    print(f"[EP {ep_idx:03d}] ROS2 state: {last_state} → {state}  (step {step})")
                    last_state = state

                if state == "done":
                    # Physical success check: cube moved >= 5 cm from start position
                    cube_x_now = cur_cube_state[0, 0].item()
                    cube_y_now = cur_cube_state[0, 1].item()
                    xy_dist = ((cube_x_now - cx) ** 2 + (cube_y_now - cy) ** 2) ** 0.5
                    success = xy_dist >= SUCCESS_XY_DIST
                    print(f"[EP {ep_idx:03d}] Done. cube_XY_dist={xy_dist:.3f}  → {'SUCCESS' if success else 'FAIL'}")
                    break

        else:
            print(f"[EP {ep_idx:03d}] TIMEOUT ({args_cli.max_steps_per_episode} steps)")
            write_episode({"state": "idle"})

        # ---- Save episode (獨立 HDF5 檔案) ----
        ep_path = os.path.join(run_dir, f"demo_{ep_idx:04d}.hdf5")
        with h5py.File(ep_path, "w") as ep_file:
            obs_grp = ep_file.create_group("obs")

            def _save(name, lst, grp):
                if lst:
                    grp.create_dataset(name, data=np.stack(lst, axis=0), compression="gzip")

            _save("joint_pos",       ep_joint_pos,  obs_grp)
            _save("joint_vel",       ep_joint_vel,  obs_grp)
            _save("cube_pos",        ep_cube_pos,   obs_grp)
            _save("cube_quat",       ep_cube_quat,  obs_grp)
            _save("cam_main",        ep_cam_main,   obs_grp)
            _save("cam_wrist_left",  ep_cam_wl,     obs_grp)
            _save("cam_wrist_right", ep_cam_wr,     obs_grp)
            _save("actions",         ep_actions,    ep_file)

            ep_file.attrs["success"]   = bool(success)
            ep_file.attrs["num_steps"] = step
            ep_file.attrs["cube_xyz"]  = [cx, cy, cz]

        if success:
            total_success += 1
        print(f"[EP {ep_idx:03d}] Saved  steps={step}  success={success}  "
              f"(total {total_success}/{ep_idx + 1})")

        # Reset coordination file for next episode
        write_episode({"state": "idle"})

    # ------------------------------------------------------------------
    # 12. Finalise
    # ------------------------------------------------------------------
    print(f"\n[DONE] Dataset saved to: {run_dir}")
    print(f"       Success rate: {total_success}/{args_cli.num_episodes} "
          f"({100 * total_success / max(1, args_cli.num_episodes):.1f}%)")


if __name__ == "__main__":
    main()
    simulation_app.close()
