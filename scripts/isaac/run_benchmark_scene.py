#!/usr/bin/env python3
"""
Isaac Sim benchmark scene：在 run_inference_scene.py 基礎上加上：
  - cube + basket RigidObject 引用
  - 每幀寫 cube 座標到 /tmp/openarm_episode.json（給 orchestrator 讀）
  - 偵測 orchestrator 寫入的 {"state": "reset"} → 隨機 cube X/Y + robot 回 home
  - reset 完成寫回 {"state": "ready", "episode": N, "cube": [x,y,z]}

跟 run_inference_scene.py 共用 USD / camera graphs / joint graphs。

啟動：
  bash scripts/isaac/run_benchmark_scene.sh

跟 orchestrator + bridge 一起用：
  Terminal 1: bash scripts/isaac/run_benchmark_scene.sh
  Terminal 2: bash policy_bridges/run_smolvla_rtc_bridge.sh   (或 run_act_bridge.sh)
  Terminal 3: python3 scripts/tools/benchmark_inference.py --policy smolvla --n_episodes 20
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--image_width",  type=int, default=320)
parser.add_argument("--image_height", type=int, default=240)
parser.add_argument("--seed", type=int, default=42,
                    help="cube position 種子，ep N 用 RandomState(seed + N) 產生 cube X/Y。"
                         "同 seed 下 ep N 永遠在同個位置，benchmark 可重現。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import omni.usd
import omni.graph.core as og
import omni.kit.app
import isaacsim.core.utils.stage as stage_utils
import usdrt.Sdf
from pathlib import Path
from pxr import UsdPhysics, UsdShade, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ── 設定 ────────────────────────────────────────────────────────────────────

USD_PATH      = os.path.expanduser("~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd")
ROBOT_PRIM    = "/World/openarm"
PLATFORM_PRIM = "/World/plateform"
CUBE_PRIM     = "/World/Cube"
BASKET_PRIM   = "/World/small_KLT"
SIM_DT        = 1.0 / 60.0

# Cube 隨機範圍（跟 collect_basket_demos.py 一致）
CUBE_RAND_X    = (0.20, 0.28)
CUBE_RAND_Y    = (0.20, 0.28)
CUBE_RESTING_Z = 0.21344

# Basket default pose / velocity 在啟動 settle 後 snapshot（不要 hardcode，USD origin 可能不在中心）

# Robot home pose: 16D 模型輸出順序，但 Isaac articulation 自己排
# 都設 0；gripper 0.04 (open)
ROBOT_HOME_JOINTS = {  # 對映 USD 解析後的 joint 名稱 → 目標角度
    "openarm_left_joint1": 0.0, "openarm_left_joint2": 0.0, "openarm_left_joint3": 0.0,
    "openarm_left_joint4": 0.0, "openarm_left_joint5": 0.0, "openarm_left_joint6": 0.0,
    "openarm_left_joint7": 0.0,
    "openarm_right_joint1": 0.0, "openarm_right_joint2": 0.0, "openarm_right_joint3": 0.0,
    "openarm_right_joint4": 0.0, "openarm_right_joint5": 0.0, "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
    "openarm_left_finger_joint1": 0.04, "openarm_left_finger_joint2": 0.04,
    "openarm_right_finger_joint1": 0.04, "openarm_right_finger_joint2": 0.04,
}

# IPC file（跟 bridges + orchestrator 共用）
EPISODE_FILE = "/tmp/openarm_episode.json"

# 物理 settle 步數（reset 後讓 cube 落到平台、機器人到 home 穩定）
RESET_SETTLE_STEPS = 90  # ~1.5s @ 60Hz

CAMERAS = [
    {"graph_path": "/Graph/CameraOverhead", "camera_prim": "/CameraMain",
     "namespace": "camera/overhead", "topic": "image_raw", "frame_id": "Camera_Main"},
    {"graph_path": "/Graph/CameraLeft",
     "camera_prim": "/World/openarm/openarm_left_link7/openarm_left_hand/CameraWristLeft",
     "namespace": "camera/left", "topic": "image_raw", "frame_id": "CameraWristLeft"},
    {"graph_path": "/Graph/CameraRight",
     "camera_prim": "/World/openarm/openarm_right_link7/openarm_right_hand/CameraWristRight",
     "namespace": "camera/right", "topic": "image_raw", "frame_id": "CameraWristRight"},
]


# ── Helpers（直接 copy/adapt from run_inference_scene.py）────────────────────

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


def set_platform_static(stage):
    prim = stage.GetPrimAtPath(PLATFORM_PRIM)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print(f"[INFO] Platform set to static collider.")


# Basket 物理調整：加重 + 高摩擦力，避免 cube 進籃時被推走（benchmark 純度提升）
BASKET_MASS_KG       = 5.0   # 原本 ~0.05kg，加 100×：cube ~0.02kg 動量打不動
BASKET_STATIC_FRIC   = 1.5
BASKET_DYN_FRIC      = 1.2


def stiffen_basket(stage):
    """把 basket 加重 + 給高摩擦表面，避免被 cube/手撞開。

    說明：訓練資料中 basket 是輕物件（會微微移動），但 benchmark 評估時
    希望 basket 固定，這樣失敗 = policy 把 cube 放錯位置，而不是 policy
    成功放進去但 basket 剛好被擠走。
    """
    basket = stage.GetPrimAtPath(BASKET_PRIM)
    if not basket.IsValid():
        print(f"[WARN] Basket prim {BASKET_PRIM} not found, skip stiffening.")
        return

    # 1. Mass
    mass_api = UsdPhysics.MassAPI.Apply(basket)
    mass_api.CreateMassAttr().Set(BASKET_MASS_KG)
    print(f"[INFO] Basket mass set to {BASKET_MASS_KG} kg")

    # 2. High-friction material on all basket collision prims
    mat_path = "/World/PhysicsMaterials/BasketHighFriction"
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim.IsValid():
        mat_prim = stage.DefinePrim(mat_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        phys_mat.CreateStaticFrictionAttr().Set(BASKET_STATIC_FRIC)
        phys_mat.CreateDynamicFrictionAttr().Set(BASKET_DYN_FRIC)
        phys_mat.CreateRestitutionAttr().Set(0.0)
    mat = UsdShade.Material(mat_prim)
    bound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(BASKET_PRIM) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
            bound += 1
    print(f"[INFO] Applied high-friction (μ={BASKET_STATIC_FRIC}/{BASKET_DYN_FRIC}) to "
          f"{bound} basket collision prims")


def enable_ros2_bridge():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)


def setup_joint_graph(robot_prim: str):
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
                ("PublishJointState.inputs:targetPrim",  [usdrt.Sdf.Path(robot_prim + "/root_joint")]),
                ("ArticulationCtrl.inputs:robotPath",     robot_prim + "/root_joint"),
            ],
        },
    )


def setup_camera_graphs(stage, img_w: int, img_h: int):
    """跟 run_inference_scene.py 同結構（已在生產驗證過）。"""
    keys = og.Controller.Keys
    for cam in CAMERAS:
        if stage.GetPrimAtPath(cam["graph_path"]).IsValid():
            stage.RemovePrim(cam["graph_path"])
        og.Controller.edit(
            {"graph_path": cam["graph_path"], "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick",      "omni.graph.action.OnPlaybackTick"),
                    ("Context",             "isaacsim.ros2.bridge.ROS2Context"),
                    ("RunOneFrame",         "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("CameraHelper",        "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                keys.SET_VALUES: [
                    ("CameraHelper.inputs:nodeNamespace",    cam["namespace"]),
                    ("CameraHelper.inputs:topicName",        cam["topic"]),
                    ("CameraHelper.inputs:type",             "rgb"),
                    ("CameraHelper.inputs:frameId",          cam["frame_id"]),
                    ("CreateRenderProduct.inputs:width",     img_w),
                    ("CreateRenderProduct.inputs:height",    img_h),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick",                   "RunOneFrame.inputs:execIn"),
                    ("RunOneFrame.outputs:step",                      "CreateRenderProduct.inputs:execIn"),
                    ("CreateRenderProduct.outputs:renderProductPath", "CameraHelper.inputs:renderProductPath"),
                    ("Context.outputs:context",                       "CameraHelper.inputs:context"),
                    ("OnPlaybackTick.outputs:tick",                   "CameraHelper.inputs:execIn"),
                ],
            },
        )
        rp_prim = stage.GetPrimAtPath(f"{cam['graph_path']}/CreateRenderProduct")
        rel = rp_prim.GetRelationship("inputs:cameraPrim")
        if rel:
            rel.SetTargets([Sdf.Path(cam["camera_prim"])])
        print(f"[INFO] Camera graph created: /{cam['namespace']}/{cam['topic']}")


# ── 多 episode：reset / IPC ───────────────────────────────────────────────

def write_episode(payload: dict):
    tmp = EPISODE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, EPISODE_FILE)


def read_episode() -> dict:
    try:
        with open(EPISODE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


def random_cube_pose(device: str, seed_base: int, ep_id: int) -> torch.Tensor:
    """用 RandomState(seed_base + ep_id) 產 cube 位置，確保 ep N 永遠在同個位置。"""
    rng = np.random.RandomState(seed_base + ep_id)
    x = rng.uniform(*CUBE_RAND_X)
    y = rng.uniform(*CUBE_RAND_Y)
    return torch.tensor([[x, y, CUBE_RESTING_Z, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)


def reset_episode(sim, robot, cube, basket, episode_id: int, device: str,
                  default_basket_pose, default_basket_vel, seed_base: int = 42):
    """重設機器人到 home、cube 隨機放置、basket 回 default（用啟動時 snapshot 的 pose）。

    時序：
      1. 寫 "resetting" → bridge 看到 → 進入 settle + publish HOME（與 Isaac 同調）
      2. 寫 home joint state + target，cube/basket 重設
      3. 物理 settle RESET_SETTLE_STEPS（bridge 持續 publish HOME，不會打架）
      4. 寫 "ready" + 實際 cube pose → orchestrator 開始計時
    """
    log_prefix = f"[EP {episode_id:03d}]"
    print(f"{log_prefix} Resetting...")

    # 0. 先告訴 bridge「正在 reset」→ bridge 進入 settle + publish HOME
    #    讓 bridge 先 1-2 frame 停止吐 RTC chunk 的舊指令
    write_episode({"state": "resetting", "episode": episode_id})

    # 1. Robot：寫 home joint pos + target
    joint_names = robot.data.joint_names
    home_state = torch.zeros((1, len(joint_names)), device=device, dtype=torch.float32)
    for i, jn in enumerate(joint_names):
        home_state[0, i] = ROBOT_HOME_JOINTS.get(jn, 0.0)
    zero_vel = torch.zeros_like(home_state)
    robot.write_joint_state_to_sim(home_state, zero_vel)
    robot.set_joint_position_target(home_state)
    robot.reset()

    # 2. Cube：seeded X/Y + 速度歸零（同 seed + ep_id 永遠同位置）
    cube.reset()
    cube_pose = random_cube_pose(device, seed_base, episode_id)
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    # 3. Basket：回到啟動時 snapshot 的 pose + 速度歸零（防夾爪撞歪）
    basket.reset()
    basket.write_root_pose_to_sim(default_basket_pose)
    basket.write_root_velocity_to_sim(default_basket_vel)

    # 4. Physics settle（bridge 在此期間持續 publish HOME → 不打架）
    for _ in range(RESET_SETTLE_STEPS):
        sim.step()
        sim.render()
        robot.update(SIM_DT)
        cube.update(SIM_DT)
        basket.update(SIM_DT)

    # 5. 抓 cube 實際 settle 位置（可能受平台 collision 微調）
    cx = float(cube.data.root_pose_w[0, 0].item())
    cy = float(cube.data.root_pose_w[0, 1].item())
    cz = float(cube.data.root_pose_w[0, 2].item())
    print(f"{log_prefix} Reset complete. Cube at ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    write_episode({"state": "ready", "episode": episode_id, "cube": [cx, cy, cz]})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Cube seed_base = {args_cli.seed} (ep N → RandomState({args_cli.seed} + N))")
    usd_abs = str(Path(USD_PATH).resolve())
    print(f"[INFO] Opening stage: {usd_abs}")
    stage_utils.open_stage(usd_abs)
    while stage_utils.is_stage_loading():
        simulation_app.update()
    print("[INFO] Stage loaded.")

    stage = omni.usd.get_context().get_stage()
    set_platform_static(stage)
    stiffen_basket(stage)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cpu"))
    device = sim.device

    robot  = Articulation(make_robot_cfg())
    cube   = RigidObject(make_cube_cfg())
    basket = RigidObject(make_basket_cfg())

    enable_ros2_bridge()
    sim.reset()
    print("[INFO] Simulation started.")

    setup_joint_graph(ROBOT_PRIM)
    setup_camera_graphs(stage, args_cli.image_width, args_cli.image_height)

    # Warm up OmniGraph + 讓 basket 落到 USD 原本位置穩定
    for _ in range(30):
        sim.step()
        sim.render()
        robot.update(SIM_DT)
        cube.update(SIM_DT)
        basket.update(SIM_DT)

    # ★ Snapshot basket default pose / velocity（USD origin 可能不在我們以為的位置）
    default_basket_pose = basket.data.root_pose_w.clone()
    default_basket_vel  = basket.data.root_vel_w.clone()
    bx, by, bz = (default_basket_pose[0, :3].cpu().numpy().tolist())
    print(f"[INFO] Basket default pose snapshot: ({bx:.3f}, {by:.3f}, {bz:.3f}) "
          f"quat={default_basket_pose[0, 3:].cpu().numpy().round(3).tolist()}")

    # 初始狀態：idle，等 orchestrator 寫 reset
    write_episode({"state": "idle"})

    print("[INFO] Benchmark scene ready. Topics publishing:")
    print("         /Isaac/isaac_joint_states / isaac_joint_command")
    print("         /camera/overhead /camera/left /camera/right /image_raw")
    print(f"[INFO] Waiting for episode reset signal via {EPISODE_FILE}")

    # 主迴圈：每幀 step + render；每 N 幀檢查 reset / 更新 cube pose
    POLL_EVERY_N = 30  # 0.5s @ 60Hz
    step_count = 0
    last_published_state = None

    while simulation_app.is_running():
        sim.step()
        sim.render()
        robot.update(SIM_DT)
        cube.update(SIM_DT)
        basket.update(SIM_DT)
        step_count += 1

        if step_count % POLL_EVERY_N == 0:
            data = read_episode()
            state = data.get("state", "idle")

            # 1. orchestrator 觸發 reset
            if state == "reset":
                ep_id = int(data.get("episode", 0))
                reset_episode(sim, robot, cube, basket, ep_id, device,
                              default_basket_pose, default_basket_vel,
                              seed_base=args_cli.seed)
                last_published_state = "ready"
                continue

            # 2. running 狀態：每 0.5s 寫 cube + basket 當下位置 + 左臂 joint state
            #    basket 位置會被 orchestrator 用來重新計算「cube 是否在籃子裡」，避免籃子被撞開時誤判
            if state in ("ready", "running"):
                ep_id = int(data.get("episode", 0))
                cx = float(cube.data.root_pose_w[0, 0].item())
                cy = float(cube.data.root_pose_w[0, 1].item())
                cz = float(cube.data.root_pose_w[0, 2].item())
                bx = float(basket.data.root_pose_w[0, 0].item())
                by = float(basket.data.root_pose_w[0, 1].item())
                bz = float(basket.data.root_pose_w[0, 2].item())
                # 左臂 joint state（轉成 dict by name 避免順序問題）
                jp = robot.data.joint_pos[0].cpu().numpy()
                joints_by_name = {jn: float(jp[i]) for i, jn in enumerate(robot.data.joint_names)
                                   if jn.startswith("openarm_left_")}
                write_episode({
                    "state": "running",
                    "episode": ep_id,
                    "cube":   [cx, cy, cz],
                    "basket": [bx, by, bz],
                    "left_joints": joints_by_name,
                })
                last_published_state = "running"


if __name__ == "__main__":
    main()
    simulation_app.close()
