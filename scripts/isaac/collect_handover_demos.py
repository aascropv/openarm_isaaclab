#!/usr/bin/env python3
"""
collect_handover_demos.py ─ 雙臂手遞手「SmolVLA 訓練資料」收集（Isaac 端）

= run_handover_benchmark.py 的場景/ROS2 graph/握手 + collect_basket_demos.py 的
  相機 + 16D state/action（從 OmniGraph 讀真實 positionCommand）+ 每集存 HDF5。

與 benchmark 的差異：
  1. 加 3 相機（main + 雙腕）+ 16D folding 格式 + action，存成 lerobot/folding 風格 HDF5。
  2. **只錄 executing→done**：run_handover_episodes.py 在 move_group 就緒後才寫 "executing"，
     故跳過每集 ~30-75s 的 move_group 開機 idle（VLA 不要那段呆站 home 的幀）。
  3. **只存成功 demo，收到 --num_success 筆為止**（banana 80%→會多跑幾集補足）。
  4. 固定 spawn（重現腳本動作，不泛化）；每集物體重置回 default pose。
  5. 每集附 task 語言標籤（多任務 SmolVLA 用）。

3 終端：
  T1 (本腳本): conda run -n env_isaaclab ~/IsaacLab-2.3.0/isaaclab.sh \
                 -p scripts/isaac/collect_handover_demos.py --enable_cameras \
                 --fruit banana --basket yellow --num_success 80
  T2: ros2 launch openarm_bimanual_moveit_config demo.launch.py use_isaac_sim:=true launch_rviz:=false  ← 不用，T3 自管
  T3: source ros2_ws/install/setup.bash; python3 scripts/ros2/run_handover_episodes.py --fruit banana --basket yellow
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--fruit", choices=["apple", "strawberry", "banana"], default="apple")
parser.add_argument("--basket", choices=["red", "yellow"], default="red")
parser.add_argument("--num_success", type=int, default=80, help="目標成功 demo 數")
parser.add_argument("--max_attempts", type=int, default=0, help="最多跑幾集（0=自動 num_success*1.6+5）")
parser.add_argument("--max_steps_per_episode", type=int, default=15000)
parser.add_argument("--output_dir", type=str, default="./datasets_handover")
parser.add_argument("--image_width", type=int, default=320)
parser.add_argument("--image_height", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import h5py
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

USD_PATH = Path(
    "~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd"
).expanduser()

ROBOT_PRIM    = "/World/openarm"
PLATFORM_PRIM = "/World/plateform"
GRASP_PRIMS   = ["/World/apple", "/World/strawberry", "/World/banana"]

CAM_MAIN_PRIM    = "/CameraMain"
CAM_WRIST_L_PRIM = "/World/openarm/openarm_left_link7/openarm_left_hand/CameraWristLeft"
CAM_WRIST_R_PRIM = "/World/openarm/openarm_right_link7/openarm_right_hand/CameraWristRight"

OBJ_PRIMS = {
    "apple":         "/World/apple",
    "strawberry":    "/World/strawberry",
    "banana":        "/World/banana",
    "basket_red":    "/World/basket_red",
    "basket_yellow": "/World/basket_yellow",
}

BASKET_BBOX = {
    "red":    dict(xmin=0.1983, xmax=0.3486, ymin=-0.1154, ymax=-0.0306),
    "yellow": dict(xmin=0.1983, xmax=0.3486, ymin=-0.2210, ymax=-0.1361),
}

# 多任務 SmolVLA 語言標籤（每水果一個固定字串；訓練/推論都要完全一致）
TASK_DESC = {
    "apple":      "pick up the apple and place it into the red basket",
    "strawberry": "pick up the strawberry and place it into the red basket",
    "banana":     "pick up the banana and place it into the yellow basket",
}

SIM_DT       = 1.0 / 60.0
RECORD_EVERY = 2
RECORD_FPS   = 30
SETTLE_STEPS = 90
EPISODE_FILE = "/tmp/openarm_episode.json"
WAIT_EXEC_STEPS = 7200   # 等 "executing" 上限（~120s，蓋過 move_group 開機）

FOLDING_16D_NAMES = [
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos", "right_joint_4.pos",
    "right_joint_5.pos", "right_joint_6.pos", "right_joint_7.pos", "right_gripper.pos",
    "left_joint_1.pos",  "left_joint_2.pos",  "left_joint_3.pos",  "left_joint_4.pos",
    "left_joint_5.pos",  "left_joint_6.pos",  "left_joint_7.pos",  "left_gripper.pos",
]


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


def setup_cameras(img_w: int, img_h: int):
    def _ann(prim):
        rp = rep.create.render_product(prim, (img_w, img_h))
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rp])
        return a
    return _ann(CAM_MAIN_PRIM), _ann(CAM_WRIST_L_PRIM), _ann(CAM_WRIST_R_PRIM)


def capture_rgb(annotator):
    data = annotator.get_data()
    if data is None or len(data) == 0:
        return None
    img = data.get("data", None) if isinstance(data, dict) else data
    if img is None:
        return None
    return np.asarray(img)[..., :3]


def get_joint_command_reader(robot_joint_names):
    """從 OmniGraph SubscribeJointState 讀真實 positionCommand（=ROS2 controller target=action）。"""
    pos_cmd_attr = og.Controller.attribute("/ROS2ActionGraph/SubscribeJointState.outputs:positionCommand")
    name_attr    = og.Controller.attribute("/ROS2ActionGraph/SubscribeJointState.outputs:jointNames")
    state = {"computed": False, "idx": None}

    def read_fn():
        cmd = og.Controller.get(pos_cmd_attr)
        if cmd is None or len(cmd) == 0:
            return None
        if not state["computed"]:
            ros_names = list(og.Controller.get(name_attr))
            if not ros_names:
                return None
            idx = []
            for n in robot_joint_names:
                if n in ros_names:
                    idx.append(ros_names.index(n))
                elif n.endswith("_finger_joint2"):
                    sib = n.replace("_finger_joint2", "_finger_joint1")
                    idx.append(ros_names.index(sib) if sib in ros_names else -1)
                else:
                    idx.append(-1)
            state["idx"] = idx
            state["computed"] = True
            print(f"[INFO] joint_command name mapping established. idx={idx}")
        idx = state["idx"]
        cmd_arr = np.asarray(cmd, dtype=np.float32)
        out = np.zeros(len(idx), dtype=np.float32)
        for i, ridx in enumerate(idx):
            if ridx >= 0:
                out[i] = cmd_arr[ridx]
        return out

    return read_fn


def make_folding_indices(robot_joint_names):
    """18D articulation → 16D [R1..R7, Rg, L1..L7, Lg]（丟棄 mimic finger_joint2）。"""
    def find(n):
        return robot_joint_names.index(n)
    idx = []
    for i in range(1, 8):
        idx.append(find(f"openarm_right_joint{i}"))
    idx.append(find("openarm_right_finger_joint1"))
    for i in range(1, 8):
        idx.append(find(f"openarm_left_joint{i}"))
    idx.append(find("openarm_left_finger_joint1"))
    return idx


def write_episode(data):
    with open(EPISODE_FILE, "w") as f:
        json.dump(data, f)


def read_episode():
    try:
        with open(EPISODE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


def obj_in_basket(x, y, basket) -> bool:
    b = BASKET_BBOX[basket]
    return b["xmin"] <= x <= b["xmax"] and b["ymin"] <= y <= b["ymax"]


# 16D 順序: 0-6=右臂7關節, 7=右夾爪, 8-14=左臂7關節, 15=左夾爪。手臂關節=非 gripper。
ARM_IDX_16D = list(range(0, 7)) + list(range(8, 15))


def arms_at_home(jp16, tol=0.15) -> bool:
    """末幀雙臂 7 關節是否都回到 home(≈0)。擋掉『水果進籃但手臂回程卡住』的壞 demo
    （實測好 demo 末幀手臂 max=0.000、卡住的 =1.57）。"""
    return max(abs(float(jp16[i])) for i in ARM_IDX_16D) < tol


def main():
    device = "cpu"
    fruit, basket = args_cli.fruit, args_cli.basket
    target = args_cli.num_success
    max_attempts = args_cli.max_attempts or int(target * 1.6) + 5

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
    obj = objects[fruit]   # 追蹤選定水果（success 判定 + ground-truth）
    ann_main, ann_wl, ann_wr = setup_cameras(args_cli.image_width, args_cli.image_height)

    enable_ros2_bridge()
    sim.reset()
    setup_ros2_action_graph()
    for _ in range(60):
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

    read_joint_command = get_joint_command_reader(joint_names)
    folding_idx = make_folding_indices(joint_names)
    print(f"[INFO] Folding 16D index mapping (articulation→16D): {folding_idx}")

    run_dir = os.path.join(args_cli.output_dir,
                           f"{fruit}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Task: '{TASK_DESC[fruit]}'")
    print(f"[INFO] Saving SUCCESS demos to: {run_dir}  (target {target}, max_attempts {max_attempts})")

    write_episode({"state": "idle"})
    print("[INFO] Waiting for ROS2 episode controller (run_handover_episodes.py)...")

    saved = 0
    attempt = 0
    while saved < target and attempt < max_attempts:
        print(f"\n[ATTEMPT {attempt:03d}] (saved {saved}/{target}) ─────────────")
        # reset 機器人 + 所有動態剛體回固定 default
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

        write_episode({"state": "ready", "episode": attempt})
        print(f"[ATTEMPT {attempt:03d}] ready → 等 move_group 就緒(executing)...")

        # 等 "executing"（move_group 開機期間持續 step sim 讓 ROS2 bridge/clock 活著，但不錄）
        got_exec = False
        for w in range(WAIT_EXEC_STEPS):
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            obj.update(SIM_DT)
            if w % 30 == 0:
                st = read_episode().get("state")
                if st == "executing":
                    got_exec = True
                    break
                if st in ("done", "idle", "all_done"):
                    break   # move_group 開機失敗/跳過此集
        if not got_exec:
            print(f"[ATTEMPT {attempt:03d}] 未進入 executing（move_group 開機失敗?），跳過不錄")
            write_episode({"state": "idle", "episode": attempt})
            attempt += 1
            continue

        # 開始錄（executing→done）
        ep_jp, ep_jv, ep_act = [], [], []
        ep_op, ep_oq = [], []
        ep_cm, ep_cl, ep_cr = [], [], []
        step = 0
        done = False
        while step < args_cli.max_steps_per_episode and not done:
            sim.step()
            sim.render()
            robot.update(SIM_DT)
            obj.update(SIM_DT)
            step += 1
            if step % RECORD_EVERY == 0:
                jp18 = robot.data.joint_pos[0].cpu().numpy()
                jv18 = robot.data.joint_vel[0].cpu().numpy()
                cmd18 = read_joint_command()
                if cmd18 is None:
                    cmd18 = jp18
                op = obj.data.root_pose_w[0].cpu().numpy()
                ep_jp.append(jp18[folding_idx])
                ep_jv.append(jv18[folding_idx])
                ep_act.append(cmd18[folding_idx])
                ep_op.append(op[:3].copy())
                ep_oq.append(op[3:7].copy())
                img_m, img_l, img_r = capture_rgb(ann_main), capture_rgb(ann_wl), capture_rgb(ann_wr)
                if img_m is not None:
                    ep_cm.append(img_m)
                    ep_cl.append(img_l)
                    ep_cr.append(img_r)
            if step % 30 == 0:
                if read_episode().get("state") == "done":
                    done = True

        op = obj.data.root_pose_w[0].cpu().numpy()
        in_basket = obj_in_basket(float(op[0]), float(op[1]), basket)
        home_ok = bool(ep_jp) and arms_at_home(ep_jp[-1])   # 手臂要回 home 才算成功（擋回程卡住的壞 demo）
        success = in_basket and home_ok
        reason = "" if success else (" [不在籃]" if not in_basket else "") + ("" if home_ok else " [手臂沒回home]")
        print(f"[ATTEMPT {attempt:03d}] {'TIMEOUT ' if not done else ''}"
              f"obj=({op[0]:.3f},{op[1]:.3f},{op[2]:.3f}) frames={len(ep_jp)} "
              f"cams={len(ep_cm)} → {'SUCCESS' if success else 'FAIL（不存）' + reason}")

        if success and len(ep_jp) > 0:
            ep_path = os.path.join(run_dir, f"demo_{saved:04d}.hdf5")
            with h5py.File(ep_path, "w") as ef:
                og_grp = ef.create_group("obs")

                def _save(name, lst, grp):
                    if lst:
                        grp.create_dataset(name, data=np.stack(lst, axis=0), compression="gzip")

                _save("joint_pos",       ep_jp,  og_grp)
                _save("joint_vel",       ep_jv,  og_grp)
                _save("cube_pos",        ep_op,  og_grp)   # legacy key=追蹤水果 pose（eval用,訓練不用）
                _save("cube_quat",       ep_oq,  og_grp)
                _save("cam_main",        ep_cm,  og_grp)
                _save("cam_wrist_left",  ep_cl,  og_grp)
                _save("cam_wrist_right", ep_cr,  og_grp)
                _save("actions",         ep_act, ef)
                ef.attrs["success"]       = True
                ef.attrs["num_sim_steps"] = step
                ef.attrs["num_frames"]    = len(ep_jp)
                ef.attrs["fps"]           = RECORD_FPS
                ef.attrs["action_format"] = "lerobot_folding_16d"
                ef.attrs["state_dim"]     = 16
                ef.attrs["action_dim"]    = 16
                ef.attrs["joint_names"]   = FOLDING_16D_NAMES
                ef.attrs["task"]          = TASK_DESC[fruit]
                ef.attrs["fruit"]         = fruit
                ef.attrs["basket"]        = basket
                ef.attrs["obj_final"]     = [float(op[0]), float(op[1]), float(op[2])]
            saved += 1
            print(f"[ATTEMPT {attempt:03d}] SAVED demo_{saved-1:04d}.hdf5  (total {saved}/{target})")

        write_episode({"state": "idle", "episode": attempt})
        attempt += 1

    write_episode({"state": "all_done"})
    print(f"\n[DONE] fruit={fruit}  saved {saved}/{target} success demos in {attempt} attempts")
    print(f"       → {run_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
