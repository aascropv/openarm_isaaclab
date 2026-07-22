#!/usr/bin/env python3
"""
run_scene_bridge.py ─ 載入 phase2 場景 + 開 ROS2 bridge + 空轉，給 MoveIt 手動控制用。

用途：bimanual_handover.py / platform_grasp.py 這類 MoveIt 控制器需要 Isaac Sim 端
把機器人接上 ROS2（發 /Isaac/isaac_joint_states、收 /Isaac/isaac_joint_command）。
這段 bridge 原本只寫在 collect_basket_demos.py 的收資料迴圈裡（綁 json handshake +
錄影），沒有乾淨的 standalone runner。本腳本抽出那段「已驗證」的 setup，不錄影、不跑
episode 迴圈，只把場景接上 bridge 後 sim.step() 空轉。

機器人完全由 OmniGraph ArticulationController 驅動（讀 /Isaac/isaac_joint_command），
本腳本從不寫 robot 目標——與 collect_basket_demos 控制期間的行為一致。建立 robot
Articulation 只為了在 reset 時把夾爪 drive stiffness 覆寫成 10000（USD 預設 ~1511 偏軟）。

Setup（3 終端）：
  Terminal 1 (本腳本，開 GUI 觀察；不要 --headless):
    conda run -n env_isaaclab ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/isaac/run_scene_bridge.py
  Terminal 2 (ros2_control + move_group + isaac_joint_command_merger):
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    ros2 launch openarm_bimanual_moveit_config demo.launch.py use_isaac_sim:=true launch_rviz:=false
  Terminal 3 (控制器):
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    python3 scripts/ros2/bimanual_handover.py
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load phase2 scene + ROS2 bridge, idle for MoveIt control")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 以下需在 AppLauncher 之後 import ---
import omni.usd
import omni.graph.core as og
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from pxr import UsdPhysics, UsdShade
import usdrt.Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH = Path(
    "~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd"
).expanduser()

ROBOT_PRIM    = "/World/openarm"
PLATFORM_PRIM = "/World/plateform"
GRASP_PRIMS   = ["/World/apple"]   # 套高摩擦力的抓取目標（phase2 先只做 apple）
SIM_DT        = 1.0 / 60.0          # 60 Hz，保持夾爪接觸物理穩定


def make_robot_cfg() -> ArticulationCfg:
    # 與 collect_basket_demos.make_robot_cfg 相同：夾爪 stiffness=10000（已驗證夾得起）
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "left_arm":      ImplicitActuatorCfg(joint_names_expr=["openarm_left_joint[1-7]"],     stiffness=400.0,   damping=40.0),
            "left_gripper":  ImplicitActuatorCfg(joint_names_expr=["openarm_left_finger_joint1"],  stiffness=10000.0, damping=150.0),
            "right_arm":     ImplicitActuatorCfg(joint_names_expr=["openarm_right_joint[1-7]"],    stiffness=400.0,   damping=40.0),
            "right_gripper": ImplicitActuatorCfg(joint_names_expr=["openarm_right_finger_joint1"], stiffness=10000.0, damping=150.0),
        },
    )


def set_platform_static(stage):
    prim = stage.GetPrimAtPath(PLATFORM_PRIM)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print("[INFO] Platform set to static collider.")


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
        is_finger = "finger" in path.lower()
        is_grasp  = any(path == g or path.startswith(g + "/") for g in GRASP_PRIMS)
        if (is_finger or is_grasp) and prim.HasAPI(UsdPhysics.CollisionAPI):
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
    # 與 collect_basket_demos.setup_ros2_action_graph 完全相同
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
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
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


def main():
    device = "cpu"
    usd_abs = str(USD_PATH.resolve())
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
    robot = Articulation(make_robot_cfg())   # 只為覆寫夾爪 stiffness，不寫目標

    enable_ros2_bridge()
    sim.reset()
    setup_ros2_action_graph()

    for _ in range(10):
        sim.step()
        sim.render()
    robot.update(SIM_DT)

    print("[INFO] ROS2 bridge active. Robot driven by /Isaac/isaac_joint_command.")
    print("[INFO] Idling — 在 Terminal 2/3 啟動 move_group 與控制器。Ctrl+C 結束。")

    while simulation_app.is_running():
        sim.step()
        sim.render()


if __name__ == "__main__":
    main()
    simulation_app.close()
