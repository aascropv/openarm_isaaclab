#!/usr/bin/env python3
"""
run_handover_benchmark.py ─ 雙臂手遞手可靠度 benchmark（Isaac 端）

跑 N 個 episode：每集把 apple 重置回固定起點 + 機器人回 default → 寫 json "ready"
觸發 ROS2 端 run_handover_episodes.py 跑一次 bimanual_handover → 等 "done" →
判定 apple 是否落入 basket_red。全程記錄雙臂 18D 關節 + apple pose 時間序列到 HDF5，
供事後分析失敗模式。

固定 apple 起點：可靠度測的是交接機制本身（右臂 joint2 變異與位置無關）。

3 終端（與 run_scene_bridge 相同的 stack，只是 T1/T3 換成 benchmark 版）：
  T1 (本腳本): conda run -n env_isaaclab ~/IsaacLab-2.3.0/isaaclab.sh \
                 -p scripts/isaac/run_handover_benchmark.py --num_episodes 20
  T2: ros2 launch openarm_bimanual_moveit_config demo.launch.py use_isaac_sim:=true launch_rviz:=false
  T3: source ros2_ws/install/setup.bash; python3 scripts/ros2/run_handover_episodes.py
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--basket", choices=["red", "yellow"], default="red")
parser.add_argument("--fruit", choices=["apple", "strawberry", "banana"], default="apple")
parser.add_argument("--max_steps_per_episode", type=int, default=15000)  # ~250s，需蓋過每集重啟 move_group 的 overhead
parser.add_argument("--output_dir", type=str, default="./results_handover_bench")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import h5py
import omni.usd
import omni.graph.core as og
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from pxr import UsdPhysics, UsdShade
import usdrt.Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH = Path(
    "~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd"
).expanduser()

ROBOT_PRIM    = "/World/openarm"
PLATFORM_PRIM = "/World/plateform"
GRASP_PRIMS   = ["/World/apple", "/World/strawberry", "/World/banana"]

# 每集要重置的動態剛體（避免籃子被撞歪/水果被掃到跨集漂移）
OBJ_PRIMS = {
    "apple":         "/World/apple",
    "strawberry":    "/World/strawberry",
    "banana":        "/World/banana",
    "basket_red":    "/World/basket_red",
    "basket_yellow": "/World/basket_yellow",
}

# basket bbox（USD 量測）→ 判定 apple 是否入籃（XY 落在 footprint 內）
BASKET_BBOX = {
    "red":    dict(xmin=0.1983, xmax=0.3486, ymin=-0.1154, ymax=-0.0306),
    "yellow": dict(xmin=0.1983, xmax=0.3486, ymin=-0.2210, ymax=-0.1361),
}

SIM_DT       = 1.0 / 60.0
RECORD_EVERY = 2       # 30 FPS 記錄
SETTLE_STEPS = 90
EPISODE_FILE = "/tmp/openarm_episode.json"


def make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=ROBOT_PRIM, spawn=None, init_state=ArticulationCfg.InitialStateCfg(),
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
    mat_prim = stage.DefinePrim("/World/PhysicsMaterials/HighFriction", "Material")
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


def write_episode(data):
    with open(EPISODE_FILE, "w") as f:
        json.dump(data, f)


def read_episode():
    try:
        with open(EPISODE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


def apple_in_basket(x, y, basket) -> bool:
    b = BASKET_BBOX[basket]
    return b["xmin"] <= x <= b["xmax"] and b["ymin"] <= y <= b["ymax"]


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
    robot = Articulation(make_robot_cfg())
    objects = {n: RigidObject(RigidObjectCfg(prim_path=p, spawn=None,
                                             init_state=RigidObjectCfg.InitialStateCfg()))
               for n, p in OBJ_PRIMS.items()}
    apple = objects[args_cli.fruit]   # 追蹤選定水果（變數沿用 apple 名，dataset/CSV 欄位亦同）

    enable_ros2_bridge()
    sim.reset()
    setup_ros2_action_graph()
    for _ in range(60):   # 多跑幾步讓所有物件落穩，再擷取 default pose
        sim.step()
        sim.render()
    robot.update(SIM_DT)
    for o in objects.values():
        o.update(SIM_DT)

    joint_names = list(robot.data.joint_names)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    default_obj_pose = {n: o.data.root_pose_w.clone() for n, o in objects.items()}
    zero_vel = torch.zeros(1, 6, device=device)
    ap0 = default_obj_pose["apple"][0].cpu().numpy()
    print(f"[INFO] apple default pose: ({ap0[0]:.4f}, {ap0[1]:.4f}, {ap0[2]:.4f})")

    run_dir = os.path.join(args_cli.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    h5_path = os.path.join(run_dir, "run.hdf5")
    print(f"[INFO] Logging to: {h5_path}")
    print(f"[INFO] Robot joint order (18D): {joint_names}")

    write_episode({"state": "idle"})
    print("[INFO] Waiting for ROS2 episode controller (run_handover_episodes.py)...")

    results = []
    h5 = h5py.File(h5_path, "w")
    h5.attrs["joint_names"] = joint_names
    h5.attrs["basket"] = args_cli.basket
    h5.attrs["apple_start"] = [float(v) for v in ap0]
    h5.attrs["fps"] = 60 // RECORD_EVERY

    for ep in range(args_cli.num_episodes):
        print(f"\n[EP {ep:03d}] ─────────────────────────────")
        # reset robot + 所有動態剛體（apple/strawberry/兩籃子）回 default
        robot.reset()
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        for name, o in objects.items():
            o.reset()
            o.write_root_pose_to_sim(default_obj_pose[name])
            o.write_root_velocity_to_sim(zero_vel)
        for _ in range(SETTLE_STEPS):
            robot.write_data_to_sim()
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            for o in objects.values():
                o.update(SIM_DT)

        ep_joint_pos, ep_joint_vel, ep_apple = [], [], []
        write_episode({"state": "ready", "episode": ep})
        print(f"[EP {ep:03d}] ready → waiting for handover...")

        step = 0
        done = False
        while step < args_cli.max_steps_per_episode and not done:
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            apple.update(SIM_DT)
            step += 1
            if step % RECORD_EVERY == 0:
                ep_joint_pos.append(robot.data.joint_pos[0].cpu().numpy().copy())
                ep_joint_vel.append(robot.data.joint_vel[0].cpu().numpy().copy())
                ep_apple.append(apple.data.root_pose_w[0].cpu().numpy().copy())
            if step % 30 == 0:
                if read_episode().get("state") == "done":
                    done = True

        ap = apple.data.root_pose_w[0].cpu().numpy()
        success = apple_in_basket(float(ap[0]), float(ap[1]), args_cli.basket)
        timed_out = not done
        print(f"[EP {ep:03d}] {'TIMEOUT ' if timed_out else ''}apple=({ap[0]:.3f},{ap[1]:.3f},{ap[2]:.3f}) "
              f"→ {'SUCCESS' if success else 'FAIL'}")
        results.append((ep, success, timed_out, float(ap[0]), float(ap[1]), float(ap[2])))

        g = h5.create_group(f"ep_{ep:03d}")
        g.create_dataset("joint_pos", data=np.stack(ep_joint_pos), compression="gzip")
        g.create_dataset("joint_vel", data=np.stack(ep_joint_vel), compression="gzip")
        g.create_dataset("apple_pose", data=np.stack(ep_apple), compression="gzip")
        g.attrs["success"] = bool(success)
        g.attrs["timed_out"] = bool(timed_out)
        g.attrs["num_frames"] = len(ep_joint_pos)
        g.attrs["apple_final"] = [float(ap[0]), float(ap[1]), float(ap[2])]
        h5.flush()

        write_episode({"state": "idle", "episode": ep})

    h5.close()
    write_episode({"state": "all_done"})

    n_succ = sum(1 for r in results if r[1])
    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"  success: {n_succ}/{len(results)}  ({100*n_succ/max(len(results),1):.0f}%)")
    for ep, s, to, x, y, z in results:
        tag = "OK  " if s else ("TIMEOUT" if to else "FAIL")
        print(f"   ep{ep:03d}  {tag}  apple=({x:.3f},{y:.3f},{z:.3f})")
    # 寫 summary.csv
    with open(os.path.join(run_dir, "summary.csv"), "w") as f:
        f.write("episode,success,timed_out,apple_x,apple_y,apple_z\n")
        for ep, s, to, x, y, z in results:
            f.write(f"{ep},{int(s)},{int(to)},{x:.4f},{y:.4f},{z:.4f}\n")
    print(f"  HDF5: {h5_path}")
    print(f"  CSV : {os.path.join(run_dir, 'summary.csv')}")
    print("=======================================")


if __name__ == "__main__":
    main()
    simulation_app.close()
