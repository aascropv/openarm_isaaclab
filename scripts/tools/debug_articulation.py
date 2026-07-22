"""
診斷腳本：確認 /World/openarm articulation 能否被 Isaac Lab 初始化。
用法：
  conda run -n env_isaaclab TERM=xterm \
      ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/debug_articulation.py --headless
"""
import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after app launch ---
import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH   = os.path.expanduser("~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env_test.usd")
ROBOT_PRIM = "/World/openarm"

def main():
    print(f"\n[DEBUG] USD path: {USD_PATH}")
    print(f"[DEBUG] USD exists: {os.path.exists(USD_PATH)}")

    # Step 1: open stage
    print("\n[DEBUG] Opening stage...")
    ok = stage_utils.open_stage(USD_PATH)
    print(f"[DEBUG] open_stage returned: {ok}")

    # Step 2: wait for async load
    import time
    waited = 0
    while stage_utils.is_stage_loading():
        simulation_app.update()
        waited += 1
    print(f"[DEBUG] Stage loaded (waited {waited} update cycles)")

    # Step 3: inspect prim via USD Python API
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(ROBOT_PRIM)
    print(f"\n[DEBUG] Prim at '{ROBOT_PRIM}': valid={prim.IsValid()}, type={prim.GetTypeName()}")
    has_art_api = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    print(f"[DEBUG] ArticulationRootAPI on '{ROBOT_PRIM}': {has_art_api}")

    # Step 4: search child prims for ArticulationRootAPI (correct API)
    print(f"[DEBUG] Searching for ArticulationRootAPI under '{ROBOT_PRIM}'...")
    found = []
    for child in stage.Traverse():
        path = child.GetPath().pathString
        if path.startswith(ROBOT_PRIM) and child.HasAPI(UsdPhysics.ArticulationRootAPI):
            found.append(path)
    print(f"[DEBUG]   Found ArticulationRootAPI at: {found}")
    # Also print direct API check on root prim
    print(f"[DEBUG] '{ROBOT_PRIM}'.HasAPI(ArticulationRootAPI): {prim.HasAPI(UsdPhysics.ArticulationRootAPI)}")

    # Step 5: create SimulationContext
    print("\n[DEBUG] Creating SimulationContext...")
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    print("[DEBUG] SimulationContext created.")

    # Step 6: create Articulation wrapper
    print(f"\n[DEBUG] Creating Articulation wrapper for '{ROBOT_PRIM}'...")
    robot_cfg = ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "left_arm":  ImplicitActuatorCfg(joint_names_expr=["openarm_left_joint[1-7]"],  stiffness=400, damping=40),
            "right_arm": ImplicitActuatorCfg(joint_names_expr=["openarm_right_joint[1-7]"], stiffness=400, damping=40),
        },
    )
    robot = Articulation(robot_cfg)

    # Step 7: sim.reset() — this triggers _initialize_impl
    print("\n[DEBUG] Calling sim.reset()...")
    try:
        sim.reset()
        print("[DEBUG] sim.reset() OK")
    except Exception as e:
        print(f"[DEBUG] sim.reset() raised: {e}")
        simulation_app.close()
        return

    # Step 8: try robot.update()
    print("\n[DEBUG] Calling robot.update()...")
    try:
        robot.update(1/60)
        n_joints = robot.data.joint_pos.shape[-1]
        print(f"[DEBUG] robot.update() OK — num_joints={n_joints}")
        print(f"[DEBUG] joint_pos: {robot.data.joint_pos}")
    except Exception as e:
        print(f"[DEBUG] robot.update() raised: {e}")

    print("\n[DEBUG] Done.")
    simulation_app.close()

if __name__ == "__main__":
    main()
