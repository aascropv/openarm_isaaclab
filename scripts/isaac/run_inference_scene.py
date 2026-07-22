#!/usr/bin/env python3
"""
Isaac Sim 推理場景：啟動場景 + 發布 ROS2 topics，供 pi05_bridge_node 使用。

發布的 topics：
  /Isaac/isaac_joint_states      ← 機器人關節狀態
  /Isaac/isaac_joint_command     → 接收關節指令並驅動機器人
  /camera/overhead/image_raw     ← 俯視攝影機
  /camera/left/image_raw         ← 左腕攝影機
  /camera/right/image_raw        ← 右腕攝影機

啟動方式（Terminal 1）：
  conda run -n env_isaaclab TERM=xterm \
      ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/run_inference_scene.py \
      --headless --enable_cameras

啟動方式（Terminal 2）：
  bash policy_bridges/run_pi05_bridge.sh
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--image_width",  type=int, default=320)
parser.add_argument("--image_height", type=int, default=240)
# 冷啟動驗證：啟動時把機器人寫到 dataset 中「動作已開始」的 pose
# 預先用 lerobot env 把 state dump 成 .npy（避免 isaaclab env 缺 pyarrow）
# 候選：init_states/ep0_frame{0200,0350,0400,0450,0500}.npy
parser.add_argument("--init_state_npy", type=str, default="",
                    help="若給檔案路徑，啟動時把該 .npy 的 18D state 寫入機器人")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# NOTE: 警告抑制改在 shell 端做（用 stdbuf + grep -vE）。
# 原本嘗試在這裡用 carb.settings.set("/log/channels/...") 過濾，但 carb
# 把路徑中的 `.` 當作 dict 分隔符，會把 `isaacsim.core.simulation_manager.plugin`
# 拆成巢狀 dict 並覆蓋掉原本的 string 設定，反而引入一堆
# `getStringRawInternal: item ... is not a string` errors。
# Python sys.stderr filter 也擋不到 Isaac Sim 原生 C++ 直接 write 到 fd 2 的訊息。
# 啟動指令請見 docs/pi05_pipeline.md「任務啟動流程」。
# ---------------------------------------------------------------------------
import omni.usd
import omni.graph.core as og
import omni.kit.app
import isaacsim.core.utils.stage as stage_utils
import usdrt.Sdf
from pathlib import Path
from pxr import UsdPhysics, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ---------------------------------------------------------------------------

USD_PATH     = os.path.expanduser("~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd")
ROBOT_PRIM   = "/World/openarm"
PLATFORM_PRIM = "/World/plateform"
CUBE_PRIM     = "/World/Cube"
SIM_DT       = 1.0 / 60.0   # 與訓練時一致（collect_basket_demos.py + dataset fps=60）

# 啟動時把 cube 固定放在 fixed-grid 訓練分佈中心 (0.24, 0.24)
# 5 個訓練點：(0.20,0.20), (0.20,0.28), (0.28,0.20), (0.28,0.28), (0.24,0.24)
CUBE_INIT_X = 0.24
CUBE_INIT_Y = 0.24
CUBE_RESTING_Z = 0.21344

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


def load_init_state_npy(path: str):
    """讀預先 dump 的 18D joint state .npy（interleaved order）。"""
    import numpy as np
    state = np.load(path).astype(np.float32)
    print(f"[INFO] Loaded init state from {path}: "
          f"shape={state.shape}, max|state|={np.abs(state).max():.4f}")
    return state


def set_platform_static(stage):
    prim = stage.GetPrimAtPath(PLATFORM_PRIM)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print(f"[INFO] Platform set to static collider.")


def enable_ros2_bridge():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    print("[INFO] ROS2 bridge enabled.")


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
    print("[INFO] Joint state graph ready.")


def setup_camera_graphs(stage, img_w: int, img_h: int):
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


def main():
    # 1. Load stage
    usd_abs = str(Path(USD_PATH).resolve())
    print(f"[INFO] Opening stage: {usd_abs}")
    stage_utils.open_stage(usd_abs)
    while stage_utils.is_stage_loading():
        simulation_app.update()
    print("[INFO] Stage loaded.")

    # 2. Physics setup
    stage = omni.usd.get_context().get_stage()
    set_platform_static(stage)

    # 3. Simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cpu"))

    # 4. Robot articulation
    robot = Articulation(make_robot_cfg())

    # 4b. Cube RigidObject（要在 sim.reset() 前建好，sim.reset 會初始化它）
    cube = RigidObject(RigidObjectCfg(
        prim_path=CUBE_PRIM, spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(),
    ))

    # 5. ROS2 bridge + graphs
    enable_ros2_bridge()
    sim.reset()
    print("[INFO] Simulation started.")

    # 5z. Cube 放到 fixed-grid 中心 (0.24, 0.24) → 對齊訓練分佈
    import torch as _torch
    cube_pose = _torch.tensor(
        [[CUBE_INIT_X, CUBE_INIT_Y, CUBE_RESTING_Z, 1.0, 0.0, 0.0, 0.0]],
        dtype=_torch.float32, device=sim.device,
    )
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(_torch.zeros((1, 6), dtype=_torch.float32, device=sim.device))
    print(f"[INFO] Cube reset to fixed-grid center ({CUBE_INIT_X}, {CUBE_INIT_Y}, {CUBE_RESTING_Z}).")

    # 5a. （可選）冷啟動驗證：把機器人寫到 dataset 中「動作已開始」的 pose
    if args_cli.init_state_npy:
        import torch
        init_state_np = load_init_state_npy(args_cli.init_state_npy)
        # robot.data.joint_pos 順序 = articulation 解析 USD 後的順序，
        # 與 dataset (interleaved: left1, right1, left2, right2, ...) 一致
        n_joints = robot.data.joint_pos.shape[1]
        if init_state_np.shape[0] != n_joints:
            print(f"[WARN] dataset state dim {init_state_np.shape[0]} != robot joints {n_joints}, "
                  f"truncate/pad to {n_joints}")
        init_state = torch.zeros((1, n_joints), device=sim.device, dtype=torch.float32)
        init_state[0, :min(n_joints, init_state_np.shape[0])] = torch.from_numpy(
            init_state_np[:n_joints]
        )
        zero_vel = torch.zeros_like(init_state)
        robot.write_joint_state_to_sim(init_state, zero_vel)
        # 把 PD 目標也設到同一個位置，避免 controller 把它拉回原處
        robot.set_joint_position_target(init_state)
        robot.write_data_to_sim()
        # settle 30 步讓物理穩定
        for _ in range(30):
            sim.step()
            sim.render()
            robot.update(SIM_DT)
        max_pos = robot.data.joint_pos.abs().max().item()
        print(f"[INFO] Robot initialized from {args_cli.init_state_npy}.")
        print(f"       max|joint_pos| now = {max_pos:.4f}")
        # 同時寫到固定路徑，避免被 OmniGraph warning 洗板蓋掉
        with open("/tmp/init_pose.log", "w") as f:
            f.write(f"init_state_npy: {args_cli.init_state_npy}\n")
            f.write(f"target max|state|: {abs(init_state_np).max():.6f}\n")
            f.write(f"actual max|joint_pos| after settle: {max_pos:.6f}\n")
            f.write(f"joint_pos: {robot.data.joint_pos[0].cpu().numpy().tolist()}\n")

    setup_joint_graph(ROBOT_PRIM)
    setup_camera_graphs(stage, args_cli.image_width, args_cli.image_height)

    # Warm up OmniGraph
    for _ in range(10):
        sim.step()
        sim.render()

    print("[INFO] Scene ready. Topics publishing:")
    print("         /Isaac/isaac_joint_states")
    print("         /Isaac/isaac_joint_command")
    print("         /camera/overhead/image_raw")
    print("         /camera/left/image_raw")
    print("         /camera/right/image_raw")
    print("[INFO] Press Ctrl+C to stop.")

    # 6. Run forever
    # 不呼叫 robot.write_data_to_sim()，避免覆蓋 OmniGraph ArticulationCtrl 寫入的指令
    while simulation_app.is_running():
        sim.step()
        sim.render()
        robot.update(SIM_DT)


if __name__ == "__main__":
    main()
    simulation_app.close()
