#!/usr/bin/env python3
"""
收集 OpenArm pick-and-place 示範資料（方塊 → 籃子）。

資料格式：lerobot/high_quality_folding 風格
  - state/action 16D：[R1..R7, R_grip, L1..L7, L_grip]（丟棄 mimic finger_joint2）
  - gripper：連續 float（finger_joint1 position，0~0.044 m）
  - 30 FPS 錄影（sim 仍跑 60 Hz，每 2 sim step 記錄 1 frame）
  - HDF5 attrs：fps, action_format, joint_names 方便日後轉 LeRobotDataset

與 collect_demonstrations.py 的差異：
  - USD 場景：openarm_bimanual_env.usd（含 /World/small_KLT 籃子）
  - 成功判斷：方塊落入籃子 XY 範圍內（而非移動 5cm）
  - action 從 OmniGraph 讀 ROS2 controller 真正發的 joint_command（非 cur_joint_pos）

Setup — 共需要 3 個終端機：

  Terminal 1  (ROS2 all-in-one):
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    ros2 launch openarm_bimanual_moveit_config demo.launch.py \
        use_isaac_sim:=true launch_rviz:=false

  Terminal 2  (episode controller):
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    python3 scripts/ros2/ros2_episode_controller.py

  Terminal 3  (this script，開 GUI 觀察夾爪/方塊；不要 --headless):
    conda run -n env_isaaclab TERM=xterm \
        ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/isaac/collect_basket_demos.py \
        --enable_cameras --num_episodes 50
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes",          type=int, default=50)
parser.add_argument("--output_dir",            type=str, default="./datasets_basket")
parser.add_argument("--max_steps_per_episode", type=int, default=3000)
parser.add_argument("--image_width",           type=int, default=320)
parser.add_argument("--image_height",          type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import json
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
# Scene constants
# ---------------------------------------------------------------------------
USD_PATH = os.path.expanduser(
    "~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd"
)

ROBOT_PRIM       = "/World/openarm"
CUBE_PRIM        = "/World/Cube"
PLATFORM_PRIM    = "/World/plateform"
BASKET_PRIM      = "/World/small_KLT"
CAM_MAIN_PRIM    = "/CameraMain"
CAM_WRIST_L_PRIM = "/World/openarm/openarm_left_link7/openarm_left_hand/CameraWristLeft"
CAM_WRIST_R_PRIM = "/World/openarm/openarm_right_link7/openarm_right_hand/CameraWristRight"

CUBE_HALF_SIDE = 0.020
CUBE_RESTING_Z = 0.21344

# 方塊隨機範圍：平台上，遠離籃子（籃子在 x≈0.298, y≈0）
# 籃子壁在 y=0.070，夾爪寬 4.4cm → y≥0.15 時有 8cm 淨距離（安全）
# x 限制在 0.20~0.28 避免 cube 直接生在籃子正前方
CUBE_RAND_X = (0.20, 0.28)
CUBE_RAND_Y = (0.20, 0.28)

# 籃子中心與範圍（從 USD 量測：translate=(0.298, 0, 0.246), scale=0.7）
# small_KLT 原尺寸約 30×20cm，0.7× 後約 21×14cm → 半徑 0.105, 0.07
BASKET_CENTER_X = 0.298
BASKET_CENTER_Y = 0.000
BASKET_HALF_X   = 0.105
BASKET_HALF_Y   = 0.070
BASKET_TOP_Z    = 0.246   # 籃子頂部約與此高度齊

SIM_DT        = 1.0 / 60.0   # 物理模擬 60 Hz（保持夾爪接觸穩定）
RECORD_EVERY  = 2             # 每 2 個 sim step 記錄 1 frame → 30 FPS（匹配 lerobot/folding）
RECORD_FPS    = 30
SETTLE_STEPS  = 90
EPISODE_FILE  = "/tmp/openarm_episode.json"

# lerobot/high_quality_folding 16D 順序：[R1..R7, R_grip, L1..L7, L_grip]
FOLDING_16D_NAMES = [
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos", "right_joint_4.pos",
    "right_joint_5.pos", "right_joint_6.pos", "right_joint_7.pos", "right_gripper.pos",
    "left_joint_1.pos",  "left_joint_2.pos",  "left_joint_3.pos",  "left_joint_4.pos",
    "left_joint_5.pos",  "left_joint_6.pos",  "left_joint_7.pos",  "left_gripper.pos",
]


# ---------------------------------------------------------------------------
# Helpers（與 collect_demonstrations.py 相同）
# ---------------------------------------------------------------------------

def make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "left_arm":      ImplicitActuatorCfg(joint_names_expr=["openarm_left_joint[1-7]"],     stiffness=400.0, damping=40.0),
            "left_gripper":  ImplicitActuatorCfg(joint_names_expr=["openarm_left_finger_joint1"],  stiffness=10000.0, damping=150.0),
            "right_arm":     ImplicitActuatorCfg(joint_names_expr=["openarm_right_joint[1-7]"],    stiffness=400.0, damping=40.0),
            "right_gripper": ImplicitActuatorCfg(joint_names_expr=["openarm_right_finger_joint1"], stiffness=10000.0, damping=150.0),
        },
    )


def make_cube_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(prim_path=CUBE_PRIM, spawn=None, init_state=RigidObjectCfg.InitialStateCfg())


def make_basket_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(prim_path=BASKET_PRIM, spawn=None, init_state=RigidObjectCfg.InitialStateCfg())


def setup_cameras(img_w: int, img_h: int):
    def _ann(prim):
        rp = rep.create.render_product(prim, (img_w, img_h))
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rp])
        return a
    return _ann(CAM_MAIN_PRIM), _ann(CAM_WRIST_L_PRIM), _ann(CAM_WRIST_R_PRIM)


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


def cube_in_basket(cx: float, cy: float, cz: float) -> bool:
    """方塊是否落入籃子 XY 範圍內且高於籃子頂部。"""
    in_x = abs(cx - BASKET_CENTER_X) <= BASKET_HALF_X
    in_y = abs(cy - BASKET_CENTER_Y) <= BASKET_HALF_Y
    return in_x and in_y


def set_platform_static(stage):
    prim = stage.GetPrimAtPath(PLATFORM_PRIM)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print(f"[INFO] Platform set to static collider.")


def apply_friction_materials(stage):
    mat_path = "/World/PhysicsMaterials/HighFriction"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.CreateStaticFrictionAttr().Set(1.0)
    phys_mat.CreateDynamicFrictionAttr().Set(0.8)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    mat = UsdShade.Material(mat_prim)
    bound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if ("finger" in path.lower() or path == CUBE_PRIM or path.startswith(CUBE_PRIM + "/")) \
                and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
            bound += 1
    print(f"[INFO] Applied high-friction material to {bound} collision prims.")


def clear_baked_graphs(stage):
    to_delete = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "OmniGraph"]
    for path in to_delete:
        stage.RemovePrim(path)
    if to_delete:
        print(f"[INFO] Removed {len(to_delete)} pre-baked OmniGraph prims.")


def enable_ros2_bridge():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)


def setup_ros2_action_graph():
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
                ("OnPlaybackTick.outputs:tick",       "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",       "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",       "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",       "ArticulationCtrl.inputs:execIn"),
                ("Context.outputs:context",           "PublishJointState.inputs:context"),
                ("Context.outputs:context",           "SubscribeJointState.inputs:context"),
                ("Context.outputs:context",           "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime","PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime","PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames",      "ArticulationCtrl.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationCtrl.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationCtrl.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand",   "ArticulationCtrl.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName",   "Isaac/isaac_joint_states"),
                ("SubscribeJointState.inputs:topicName", "Isaac/isaac_joint_command"),
                ("PublishJointState.inputs:targetPrim",  [usdrt.Sdf.Path(ROBOT_PRIM + "/root_joint")]),
                ("ArticulationCtrl.inputs:robotPath",     ROBOT_PRIM + "/root_joint"),
            ],
        },
    )


def get_joint_command_reader(robot_joint_names: list[str]):
    """從 OmniGraph SubscribeJointState 直接讀 positionCommand。
    回傳 (read_fn, ros_to_robot_idx)：
      - read_fn(): 回傳 18D ndarray（按 robot articulation 順序）；若 ROS2 還沒發過 command 則回傳 None
      - ros_to_robot_idx: ROS2 jointNames → robot.data.joint_names 的 index 映射
    """
    pos_cmd_attr = og.Controller.attribute(
        "/ROS2ActionGraph/SubscribeJointState.outputs:positionCommand"
    )
    name_attr = og.Controller.attribute(
        "/ROS2ActionGraph/SubscribeJointState.outputs:jointNames"
    )

    # joint name 對應（一次性算）；如果 ROS2 還沒發過，jointNames 會是空，延後算
    ros_to_robot_idx = {"computed": False, "idx": None, "ros_names": None}

    def read_fn():
        cmd = og.Controller.get(pos_cmd_attr)
        if cmd is None or len(cmd) == 0:
            return None

        if not ros_to_robot_idx["computed"]:
            ros_names = list(og.Controller.get(name_attr))
            if not ros_names:
                return None
            # 為每個 robot joint 找對應 ROS index；找不到的（mimic finger_joint2）用 -1
            # 之後 fallback 用 finger_joint1 的 command（mimic ratio=1）
            idx = []
            for n in robot_joint_names:
                if n in ros_names:
                    idx.append(ros_names.index(n))
                elif n.endswith("_finger_joint2"):
                    # mimic：用對應的 finger_joint1 command
                    sibling = n.replace("_finger_joint2", "_finger_joint1")
                    idx.append(ros_names.index(sibling) if sibling in ros_names else -1)
                else:
                    idx.append(-1)
            if any(i < 0 for i in idx):
                missing = [n for n, i in zip(robot_joint_names, idx) if i < 0]
                print(f"[WARN] joint_command 缺以下 joints: {missing}")
            ros_to_robot_idx["ros_names"] = ros_names
            ros_to_robot_idx["idx"] = idx
            ros_to_robot_idx["computed"] = True
            print(f"[INFO] joint_command name mapping established.")
            print(f"       ROS order ({len(ros_names)}): {ros_names}")
            print(f"       Robot order ({len(robot_joint_names)}): {robot_joint_names}")
            print(f"       idx mapping: {idx}")

        idx = ros_to_robot_idx["idx"]
        cmd_arr = np.asarray(cmd, dtype=np.float32)
        out = np.zeros(len(idx), dtype=np.float32)
        for i, ridx in enumerate(idx):
            if ridx >= 0:
                out[i] = cmd_arr[ridx]
        return out

    return read_fn, ros_to_robot_idx


def make_folding_indices(robot_joint_names: list[str]) -> list[int]:
    """產生 lerobot/folding 16D 順序對應 articulation 18D 的 index。
    順序：[R1..R7, R_gripper, L1..L7, L_gripper]，丟棄 mimic finger_joint2。
    """
    def find(n: str) -> int:
        return robot_joint_names.index(n)
    idx: list[int] = []
    for i in range(1, 8):
        idx.append(find(f"openarm_right_joint{i}"))
    idx.append(find("openarm_right_finger_joint1"))
    for i in range(1, 8):
        idx.append(find(f"openarm_left_joint{i}"))
    idx.append(find("openarm_left_finger_joint1"))
    return idx


def write_episode(data: dict):
    with open(EPISODE_FILE, "w") as f:
        json.dump(data, f)


def read_episode() -> dict:
    try:
        with open(EPISODE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cpu"

    usd_abs = str(Path(USD_PATH).resolve())
    print(f"[INFO] Opening stage: {usd_abs}")
    stage_utils.open_stage(usd_abs)
    while stage_utils.is_stage_loading():
        simulation_app.update()
    print("[INFO] Stage loaded.")

    stage = omni.usd.get_context().get_stage()
    clear_baked_graphs(stage)
    set_platform_static(stage)
    apply_friction_materials(stage)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=device))
    robot  = Articulation(make_robot_cfg())
    cube   = RigidObject(make_cube_cfg())
    basket = RigidObject(make_basket_cfg())
    ann_main, ann_wl, ann_wr = setup_cameras(args_cli.image_width, args_cli.image_height)

    enable_ros2_bridge()
    sim.reset()
    setup_ros2_action_graph()

    for _ in range(10):
        sim.step()
        sim.render()

    robot.update(SIM_DT)
    cube.update(SIM_DT)
    basket.update(SIM_DT)
    default_joint_pos  = robot.data.default_joint_pos.clone()
    default_joint_vel  = robot.data.default_joint_vel.clone()
    default_basket_pose = basket.data.root_pose_w.clone()
    default_basket_vel  = torch.zeros(1, 6, device=device)

    print(f"[INFO] Basket center: ({BASKET_CENTER_X}, {BASKET_CENTER_Y})  "
          f"half-size: ({BASKET_HALF_X}, {BASKET_HALF_Y})")

    write_episode({"state": "idle"})
    print("[INFO] Waiting for ROS2 episode controller...")

    # 從 OmniGraph 讀真正的 control target（ROS2 controller 發的 joint_command）
    robot_joint_names = list(robot.data.joint_names)
    print(f"[INFO] Robot articulation joint order: {robot_joint_names}")
    read_joint_command, _cmd_state = get_joint_command_reader(robot_joint_names)

    # 16D folding-style index：把 18D articulation array → 16D [R1..R7, Rg, L1..L7, Lg]
    folding_idx = make_folding_indices(robot_joint_names)
    print(f"[INFO] Folding 16D index mapping (articulation→16D): {folding_idx}")

    run_dir = os.path.join(args_cli.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Saving to: {run_dir}")

    total_success = 0

    for ep_idx in range(args_cli.num_episodes):
        print(f"\n[EP {ep_idx:03d}] ─────────────────────────────")

        robot.reset()
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        cube.reset()
        cube_pose = random_cube_pose(device)
        cube.write_root_pose_to_sim(cube_pose)
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        basket.reset()
        basket.write_root_pose_to_sim(default_basket_pose)
        basket.write_root_velocity_to_sim(default_basket_vel)

        for _ in range(SETTLE_STEPS):
            robot.write_data_to_sim()
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            cube.update(SIM_DT)

        cube.update(SIM_DT)
        cx = cube.data.root_pose_w[0, 0].item()
        cy = cube.data.root_pose_w[0, 1].item()
        cz = cube.data.root_pose_w[0, 2].item()
        print(f"[EP {ep_idx:03d}] Cube start: ({cx:.3f}, {cy:.3f}, {cz:.3f})")

        write_episode({"state": "ready", "cube": [cx, cy, cz], "episode": ep_idx})

        ep_joint_pos, ep_joint_vel = [], []
        ep_cube_pos, ep_cube_quat  = [], []
        ep_cam_main, ep_cam_wl, ep_cam_wr = [], [], []
        ep_actions = []

        success = False
        step = 0
        last_state = "ready"

        while step < args_cli.max_steps_per_episode:
            cube.write_data_to_sim()
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            cube.update(SIM_DT)

            step += 1

            cur_joint_pos18 = robot.data.joint_pos[0].cpu().numpy()
            cur_joint_vel18 = robot.data.joint_vel[0].cpu().numpy()
            cur_cube_state  = cube.data.root_pose_w[0].cpu().numpy()

            # 每 RECORD_EVERY 個 sim step 記錄一次 → 30 FPS
            if step % RECORD_EVERY == 0:
                cmd18 = read_joint_command()
                if cmd18 is None:
                    cmd18 = cur_joint_pos18

                # 18D articulation → 16D lerobot/folding 順序
                ep_joint_pos.append(cur_joint_pos18[folding_idx])
                ep_joint_vel.append(cur_joint_vel18[folding_idx])
                ep_actions.append(cmd18[folding_idx])
                ep_cube_pos.append(cur_cube_state[:3])
                ep_cube_quat.append(cur_cube_state[3:7])

                img_m  = capture_rgb(ann_main)
                img_wl = capture_rgb(ann_wl)
                img_wr = capture_rgb(ann_wr)
                if img_m is not None:
                    ep_cam_main.append(img_m)
                    ep_cam_wl.append(img_wl)
                    ep_cam_wr.append(img_wr)

            if step % 30 == 0:
                ctrl = read_episode()
                state = ctrl.get("state", "idle")
                if state != last_state:
                    print(f"[EP {ep_idx:03d}] state: {last_state} → {state}  (step {step})")
                    last_state = state
                if state == "done":
                    cube_x_now = float(cur_cube_state[0])
                    cube_y_now = float(cur_cube_state[1])
                    cube_z_now = float(cur_cube_state[2])
                    success = cube_in_basket(cube_x_now, cube_y_now, cube_z_now)
                    print(f"[EP {ep_idx:03d}] Done. cube=({cube_x_now:.3f}, {cube_y_now:.3f}, {cube_z_now:.3f}) "
                          f"→ {'SUCCESS (in basket)' if success else 'FAIL'}")
                    break
        else:
            print(f"[EP {ep_idx:03d}] TIMEOUT")
            write_episode({"state": "idle"})

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

            ep_file.attrs["success"]       = bool(success)
            ep_file.attrs["num_sim_steps"] = step
            ep_file.attrs["num_frames"]    = len(ep_joint_pos)
            ep_file.attrs["cube_xyz"]      = [cx, cy, cz]
            ep_file.attrs["fps"]           = RECORD_FPS
            ep_file.attrs["action_format"] = "lerobot_folding_16d"
            ep_file.attrs["state_dim"]     = 16
            ep_file.attrs["action_dim"]    = 16
            ep_file.attrs["joint_names"]   = FOLDING_16D_NAMES

        if success:
            total_success += 1
        print(f"[EP {ep_idx:03d}] Saved  steps={step}  success={success}  "
              f"(total {total_success}/{ep_idx + 1})")

        write_episode({"state": "idle"})

    print(f"\n[DONE] Saved to: {run_dir}")
    print(f"       Success rate: {total_success}/{args_cli.num_episodes} "
          f"({100 * total_success / max(1, args_cli.num_episodes):.1f}%)")


if __name__ == "__main__":
    main()
    simulation_app.close()
