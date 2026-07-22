#!/usr/bin/env python3
"""查 banana（及 apple/basket_yellow 參照）的世界位置、朝向、尺寸。香蕉 payload 是遠端 YCB 資產，
需 Isaac 的 asset resolver 才載得到。"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app
print(">>> app launched", flush=True)

import isaacsim.core.utils.stage as stage_utils
from pxr import Usd, UsdGeom, Gf

USD = Path("~/openarm_isaaclab/assets/Collected_openarm_bimanual_env_test/openarm_bimanual_env.usd").expanduser()
print(">>> opening stage", flush=True)
stage_utils.open_stage(str(USD.resolve()))
for i in range(400):
    simulation_app.update()
    if not stage_utils.is_stage_loading():
        print(f">>> stage loaded at update {i}", flush=True)
        break
for _ in range(120):
    simulation_app.update()
print(">>> querying", flush=True)

import omni.usd
stage = omni.usd.get_context().get_stage()
purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
xc = UsdGeom.XformCache(Usd.TimeCode.Default())

print("\n========== RESULT ==========")
for name in ["banana", "apple", "basket_yellow", "basket_red"]:
    p = stage.GetPrimAtPath(f"/World/{name}")
    if not p or not p.IsValid():
        print(f"{name}: INVALID"); continue
    M = xc.GetLocalToWorldTransform(p)
    T = M.ExtractTranslation()
    R = M.ExtractRotationMatrix()
    wb = bc.ComputeWorldBound(p).ComputeAlignedRange()
    lb = bc.ComputeLocalBound(p).ComputeAlignedRange()
    wmn, wmx = wb.GetMin(), wb.GetMax()
    lsz = lb.GetMax() - lb.GetMin()
    wsz = wmx - wmn
    print(f"\n[{name}]")
    print(f"  world translate = ({T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f})")
    print(f"  world AABB: min=({wmn[0]:.4f},{wmn[1]:.4f},{wmn[2]:.4f}) max=({wmx[0]:.4f},{wmx[1]:.4f},{wmx[2]:.4f}) top_z={wmx[2]:.4f}")
    print(f"  world AABB size = ({wsz[0]:.4f}, {wsz[1]:.4f}, {wsz[2]:.4f})")
    print(f"  local size (真實尺寸) = ({lsz[0]:.4f}, {lsz[1]:.4f}, {lsz[2]:.4f})")
    print(f"  rot rows: {tuple(round(v,3) for v in R.GetRow(0))} {tuple(round(v,3) for v in R.GetRow(1))} {tuple(round(v,3) for v in R.GetRow(2))}")
print("========== END ==========\n")
simulation_app.close()
