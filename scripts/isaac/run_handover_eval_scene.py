#!/usr/bin/env python3
"""
run_handover_eval_scene.py — handover policy 評測場景（Isaac 端，policy 驅動）

= run_benchmark_scene.py（basket）改成 handover：3 水果 + 2 籃子、**固定位置**（訓練即固定位置重現）。
發布三相機 + 關節 state，訂閱 policy（bridge）吐的關節指令。每集 reset 回 home + 所有物件回 default。

跟收集（run_handover_benchmark.py，MoveIt 驅動）不同：這裡是 **policy 直接驅動**（不經 MoveIt），
故沿用 run_benchmark_scene 的 camera graph（發布 /camera/*）+ joint graph（state/command）+ episode IPC。

三終端：
  T1 (本腳本): bash scripts/isaac/run_handover_eval_scene.sh
  T2 (bridge): bash policy_bridges/run_smolvla_rtc_bridge.sh --ckpt <handover_ckpt> --task "<task 字串>" --disable_gripper_snap
  T3 (orch):   python3 scripts/tools/benchmark_handover.py --fruit banana --basket yellow --n_episodes 10
"""

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--image_width",  type=int, default=320)
parser.add_argument("--image_height", type=int, default=240)
parser.add_argument("--replay_npz", type=str, default="",
                    help="診斷用：逗號分隔的 npz(joints 16D + names) 路徑；設了就『replay demo 關節』"
                         "而非等 policy，配 EEF_LOG 錄 demo 回程弧線。每個 npz = CSV 一個 episode。")
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
from pxr import UsdPhysics, UsdShade, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH      = os.path.expanduser("~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd")
ROBOT_PRIM    = "/World/openarm"
PLATFORM_PRIM = "/World/plateform"
GRASP_PRIMS   = ["/World/apple", "/World/strawberry", "/World/banana"]
SIM_DT        = 1.0 / 60.0
# opt-in 診斷：設 EEF_LOG=<csv> 就把每幀（左右臂 body 世界座標 + 16 關節 + 三水果位置 + episode/frame）寫檔。
# 供逐幀 viewer notebook 用（選集+拉 frame 看關節/水果/成功/回home）。預設空=不影響正常 eval。
EEF_LOG_FILE  = os.environ.get("EEF_LOG", "")
# folding 16D 關節順序（跟 observation.state / 訓練資料一致）
JOINT_ORDER_16 = [
    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4",
    "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7", "openarm_right_finger_joint1",
    "openarm_left_joint1",  "openarm_left_joint2",  "openarm_left_joint3",  "openarm_left_joint4",
    "openarm_left_joint5",  "openarm_left_joint6",  "openarm_left_joint7",  "openarm_left_finger_joint1",
]

OBJ_PRIMS = {
    "apple":         "/World/apple",
    "strawberry":    "/World/strawberry",
    "banana":        "/World/banana",
    "basket_red":    "/World/basket_red",
    "basket_yellow": "/World/basket_yellow",
}
BASKET_NAMES = ["basket_red", "basket_yellow"]
FRUIT_NAMES  = ["apple", "strawberry", "banana"]

ROBOT_HOME_JOINTS = {
    "openarm_left_joint1": 0.0, "openarm_left_joint2": 0.0, "openarm_left_joint3": 0.0,
    "openarm_left_joint4": 0.0, "openarm_left_joint5": 0.0, "openarm_left_joint6": 0.0,
    "openarm_left_joint7": 0.0,
    "openarm_right_joint1": 0.0, "openarm_right_joint2": 0.0, "openarm_right_joint3": 0.0,
    "openarm_right_joint4": 0.0, "openarm_right_joint5": 0.0, "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
    "openarm_left_finger_joint1": 0.04, "openarm_left_finger_joint2": 0.04,
    # 雙夾爪 home = 開(0.04)。統一 train/eval home：收集端 handover 腳本也改成 home 就開雙爪
    # （bimanual_handover.py step1b）。原本收集閉(0.0)/eval開(0.04) 不一致 → policy 起步崩 home。
    "openarm_right_finger_joint1": 0.04, "openarm_right_finger_joint2": 0.04,
}

EPISODE_FILE       = "/tmp/openarm_episode.json"
RESET_SETTLE_STEPS = 90

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

BASKET_MASS_KG     = 5.0
BASKET_STATIC_FRIC = 1.5
BASKET_DYN_FRIC    = 1.2


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
    """高摩擦給夾爪指 + 三水果（與收集時 collect_handover_demos 一致，保持抓取物理相同）。"""
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


def stiffen_basket(stage, basket_prim: str):
    """單一 basket 加重 + 高摩擦，避免被水果/手撞開（評測純度：失敗=放錯而非籃子被擠走）。"""
    basket = stage.GetPrimAtPath(basket_prim)
    if not basket.IsValid():
        print(f"[WARN] Basket prim {basket_prim} not found, skip stiffening.")
        return
    mass_api = UsdPhysics.MassAPI.Apply(basket)
    mass_api.CreateMassAttr().Set(BASKET_MASS_KG)
    mat_path = f"/World/PhysicsMaterials/BasketHighFriction"
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
        if path.startswith(basket_prim) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
            bound += 1
    print(f"[INFO] Stiffened {basket_prim}: mass={BASKET_MASS_KG}kg, μ={BASKET_STATIC_FRIC}, {bound} prims")


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


def _xyz(obj) -> list:
    p = obj.data.root_pose_w[0, :3].cpu().numpy()
    return [float(p[0]), float(p[1]), float(p[2])]


def reset_episode(sim, robot, objects, episode_id, device, default_obj_pose, log_fn=None):
    """機器人回 home + 所有水果/籃子回固定 default（無隨機）。log_fn(ep,state) 有給則 reset settle 就錄。"""
    print(f"[EP {episode_id:03d}] Resetting (fixed positions)...")
    write_episode({"state": "resetting", "episode": episode_id})

    joint_names = robot.data.joint_names
    home_state = torch.zeros((1, len(joint_names)), device=device, dtype=torch.float32)
    for i, jn in enumerate(joint_names):
        home_state[0, i] = ROBOT_HOME_JOINTS.get(jn, 0.0)
    robot.write_joint_state_to_sim(home_state, torch.zeros_like(home_state))
    robot.set_joint_position_target(home_state)
    robot.reset()

    zero_vel = torch.zeros(1, 6, device=device)
    for name, o in objects.items():
        o.reset()
        o.write_root_pose_to_sim(default_obj_pose[name])
        o.write_root_velocity_to_sim(zero_vel)

    for _ in range(RESET_SETTLE_STEPS):
        sim.step(); sim.render()
        robot.update(SIM_DT)
        for o in objects.values():
            o.update(SIM_DT)
        if log_fn is not None:   # reset settle 就開始錄（手臂剛強制回 home）→ viewer 從真正起點開始
            log_fn(episode_id, "ready")

    fruits = {n: _xyz(objects[n]) for n in FRUIT_NAMES}
    print(f"[EP {episode_id:03d}] Reset complete. fruits={ {k:[round(v,3) for v in p] for k,p in fruits.items()} }")
    write_episode({"state": "ready", "episode": episode_id, "fruits": fruits})


def _log_frame_row(eef_writer, eef_log, robot, objects, log_body_idx, log_joint_idx, t0, episode, frame, state):
    """寫一幀：左右臂 body 世界座標 + 16 關節 + 三水果位置（供逐幀 viewer）。
    wall=絕對時間戳（跟三相機錄影對齊用）。"""
    bpw = robot.data.body_pos_w[0].cpu().numpy()
    jp = robot.data.joint_pos[0].cpu().numpy()
    row = [round(time.time(), 3), round(time.time() - t0, 3), episode, frame, state]
    for i in log_body_idx:
        row += [round(float(bpw[i, 0]), 4), round(float(bpw[i, 1]), 4), round(float(bpw[i, 2]), 4)]
    for i in log_joint_idx:
        row += [round(float(jp[i]), 4)]
    for fr in FRUIT_NAMES:
        p = objects[fr].data.root_pose_w[0, :3].cpu().numpy()
        row += [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
    eef_writer.writerow(row); eef_log.flush()


def run_replay(sim, robot, objects, device, default_obj_pose, npz_paths,
               eef_writer, eef_log, log_body_idx, log_joint_idx):
    """診斷：把 demo 16D 關節當 position target 播放，錄右臂 body 世界軌跡（= demo『正確』回程弧線）。
    水果是否被夾/放不影響手臂 EEF 弧線，故不管物理抓取。每個 npz = CSV 一個 episode。"""
    import time as _t
    sim_names = list(robot.data.joint_names)
    def idx(nm):
        return sim_names.index(nm) if nm in sim_names else None
    t0 = _t.time()
    for ep, path in enumerate(npz_paths):
        d = np.load(path, allow_pickle=True)
        joints = d["joints"]; names = [str(x) for x in d["names"]]
        print(f"[REPLAY] ep{ep} {path}: {joints.shape[0]} frames")
        reset_episode(sim, robot, objects, ep, device, default_obj_pose)
        for tstep in range(joints.shape[0]):
            target = robot.data.joint_pos.clone()
            for ci, nm in enumerate(names):
                val = float(joints[tstep, ci])
                if nm.endswith("gripper"):
                    side = "right" if nm.startswith("right") else "left"
                    for fj in (f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"):
                        j = idx(fj)
                        if j is not None: target[0, j] = val
                else:  # e.g. right_joint_3
                    side, k = nm.split("_joint_")
                    j = idx(f"openarm_{side}_joint{k}")
                    if j is not None: target[0, j] = val
            robot.set_joint_position_target(target)
            for _ in range(2):   # 2 substeps/frame（demo num_sim_steps=2×frames）
                robot.write_data_to_sim()   # ★ 把 target 推到 PhysX，否則手臂不動
                sim.step(); sim.render(); robot.update(SIM_DT)
                for o in objects.values(): o.update(SIM_DT)
            if eef_writer is not None:
                _log_frame_row(eef_writer, eef_log, robot, objects, log_body_idx, log_joint_idx, t0, ep, tstep, "replay")
    print("[REPLAY] all done")


def main():
    usd_abs = str(Path(USD_PATH).resolve())
    print(f"[INFO] Opening stage: {usd_abs}")
    stage_utils.open_stage(usd_abs)
    while stage_utils.is_stage_loading():
        simulation_app.update()
    print("[INFO] Stage loaded.")

    stage = omni.usd.get_context().get_stage()
    set_platform_static(stage)
    apply_friction_materials(stage)
    for bn in BASKET_NAMES:
        stiffen_basket(stage, OBJ_PRIMS[bn])

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cpu"))
    device = sim.device
    robot = Articulation(make_robot_cfg())
    objects = {n: RigidObject(RigidObjectCfg(prim_path=p, spawn=None,
                                             init_state=RigidObjectCfg.InitialStateCfg()))
               for n, p in OBJ_PRIMS.items()}

    enable_ros2_bridge()
    sim.reset()
    print("[INFO] Simulation started.")
    if not args_cli.replay_npz:   # replay 診斷模式不需 ROS 控制圖/相機（直接 set_joint_position_target）
        setup_joint_graph(ROBOT_PRIM)
        setup_camera_graphs(stage, args_cli.image_width, args_cli.image_height)

    for _ in range(30):
        sim.step(); sim.render()
        robot.update(SIM_DT)
        for o in objects.values():
            o.update(SIM_DT)

    default_obj_pose = {n: o.data.root_pose_w.clone() for n, o in objects.items()}
    for n in FRUIT_NAMES:
        print(f"[INFO] {n} default pose: {[round(v,3) for v in _xyz(objects[n])]}")

    # opt-in EEF 診斷 logger：右臂各 body 世界座標
    eef_log = eef_writer = None
    log_body_idx, log_body_names, log_joint_idx = [], [], []
    if EEF_LOG_FILE:
        import csv as _csv
        bn = list(robot.data.body_names)
        log_body_idx = [i for i, n in enumerate(bn) if ("link" in n or "finger" in n)]  # 左右兩臂
        log_body_names = [bn[i] for i in log_body_idx]
        sim_jn = list(robot.data.joint_names)
        log_joint_idx = [sim_jn.index(j) for j in JOINT_ORDER_16 if j in sim_jn]
        log_joint_names = [j for j in JOINT_ORDER_16 if j in sim_jn]
        eef_log = open(EEF_LOG_FILE, "w", newline="")
        eef_writer = _csv.writer(eef_log)
        header = ["wall", "t", "episode", "frame", "state"]
        header += [f"{n}_{ax}" for n in log_body_names for ax in "xyz"]
        header += log_joint_names                                    # 16 關節
        header += [f"{fr}_{ax}" for fr in FRUIT_NAMES for ax in "xyz"]  # 三水果位置
        eef_writer.writerow(header)
        eef_log.flush()
        print(f"[EEF_LOG] {len(log_body_names)} bodies + {len(log_joint_names)} joints + {len(FRUIT_NAMES)} fruits -> {EEF_LOG_FILE}")

    replay_list = [p for p in args_cli.replay_npz.split(",") if p.strip()]
    if replay_list:
        run_replay(sim, robot, objects, device, default_obj_pose, replay_list,
                   eef_writer, eef_log, log_body_idx, log_joint_idx)
        if eef_log:
            eef_log.close()
        return

    write_episode({"state": "idle"})
    print("[INFO] Handover eval scene ready. Publishing /camera/* + /Isaac/isaac_joint_states.")
    print(f"[INFO] Waiting for episode reset signal via {EPISODE_FILE}")

    POLL_EVERY_N = 30
    step_count = 0
    cur_state, cur_ep = "idle", -1
    logstate = {"prev_ep": -1, "frame": 0}
    t0 = time.time()
    def log_now(ep, state):   # reset settle + main loop 共用，frame 每集連續
        if eef_writer is None:
            return
        if ep != logstate["prev_ep"]:
            logstate["prev_ep"], logstate["frame"] = ep, 0
        _log_frame_row(eef_writer, eef_log, robot, objects, log_body_idx, log_joint_idx, t0, ep, logstate["frame"], state)
        logstate["frame"] += 1
    while simulation_app.is_running():
        sim.step(); sim.render()
        robot.update(SIM_DT)
        for o in objects.values():
            o.update(SIM_DT)
        step_count += 1

        if cur_state in ("ready", "running"):
            log_now(cur_ep, cur_state)

        if step_count % POLL_EVERY_N == 0:
            data = read_episode()
            state = data.get("state", "idle")
            cur_state, cur_ep = state, int(data.get("episode", cur_ep))
            if state == "reset":
                reset_episode(sim, robot, objects, int(data.get("episode", 0)), device, default_obj_pose, log_fn=log_now)
                continue
            if state in ("ready", "running"):
                jp = robot.data.joint_pos[0].cpu().numpy()
                joints = {jn: float(jp[i]) for i, jn in enumerate(robot.data.joint_names)
                          if jn.startswith("openarm_")}
                write_episode({
                    "state":   "running",
                    "episode": int(data.get("episode", 0)),
                    "fruits":  {n: _xyz(objects[n]) for n in FRUIT_NAMES},
                    "baskets": {bn.replace("basket_", ""): _xyz(objects[bn]) for bn in BASKET_NAMES},
                    "joints":  joints,
                })


if __name__ == "__main__":
    main()
    simulation_app.close()
