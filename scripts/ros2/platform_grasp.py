#!/usr/bin/env python3
"""
platform_grasp.py  ─  MoveIt2 Python 版本的 platform_grasp.cpp

使用 ROS2 action/service 直接與 move_group 通訊，不需要 C++ executable。
邏輯與 openarm_manipulation/platform_grasp.cpp 完全對應。

前置條件：
  Terminal 1: ros2 launch openarm_bringup openarm.bimanual.launch.py \\
                  use_isaac_sim:=true robot_controller:=joint_trajectory_controller
  Terminal 2: ros2 launch openarm_bimanual_moveit_config demo.launch.py

使用方式：
  source ~/openarm_isaaclab/ros2_ws/install/setup.bash
  python3 scripts/platform_grasp.py \\
      --cube_x 0.22321 --cube_y 0.16423 --cube_z 0.21344 \\
      [--place_x 0.25 --place_y 0.1 --place_z 0.25]
"""

import sys
import math
import time
import threading
import argparse

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, Quaternion, Point
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints,
    PositionConstraint, OrientationConstraint,
    JointConstraint, BoundingVolume,
    CollisionObject, PlanningScene,
    MoveItErrorCodes, PlanningOptions,
)
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import tf2_ros
from tf2_ros import TransformException

# ─── 常數 ───────────────────────────────────────────────────────────────────

ARM_GROUP     = "left_arm"
GRIPPER_GROUP = "left_gripper"
EEF_LINK      = "openarm_left_hand_tcp"
PLANNING_FRAME = "world"

GRIPPER_OPEN   = 0.044
GRIPPER_CLOSED = 0.005  # 方塊寬4cm，手指碰到方塊時 q≈0.014m；目標設0.005讓 deflection=9mm，增大夾持力

LEFT_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT   = "openarm_left_finger_joint1"

# 平台預設值（Isaac Sim USD）
_PLAT = dict(x=0.38591, y=0.0, z=0.10435, sx=0.41433, sy=0.8, sz=0.17818)

# 籃子（/World/small_KLT）— 外框尺寸用來阻擋手臂穿牆；下降至籃內前移除
_BASKET = dict(x=0.298, y=0.000, z=0.243, sx=0.210, sy=0.140, sz=0.096)


# ─── 工具函式 ────────────────────────────────────────────────────────────────

def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def _wait(future, timeout: float = 60.0):
    """Block until future is done; return result or None on timeout."""
    start = time.monotonic()
    while not future.done():
        if time.monotonic() - start > timeout:
            return None
        time.sleep(0.05)
    return future.result()


def _joint_constraints(names, values, tol=0.01) -> Constraints:
    c = Constraints()
    for name, val in zip(names, values):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = float(val)
        jc.tolerance_above = tol
        jc.tolerance_below = tol
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    return c


def _time_parameterize(traj_msg, vel_scale: float):
    """
    Cartesian 軌跡沒有時間戳，這裡以每步最大關節位移 / (vel_scale * 3.14 rad/s)
    計算各點的 time_from_start，讓 JointTrajectoryController 能正常執行。
    """
    pts = traj_msg.joint_trajectory.points
    n = len(pts)
    if n < 2:
        return traj_msg

    max_joint_vel = 3.14 * max(vel_scale, 0.05)  # rad/s
    t = 0.0
    for i, pt in enumerate(pts):
        if i > 0:
            prev = pts[i - 1].positions
            curr = pt.positions
            max_disp = max(abs(a - b) for a, b in zip(prev, curr))
            dt = max_disp / max_joint_vel if max_disp > 1e-6 else 0.01
            t += dt
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        pt.velocities = [0.0] * len(pt.positions)
        pt.accelerations = [0.0] * len(pt.positions)

    return traj_msg


# ─── 主類別 ─────────────────────────────────────────────────────────────────

class PlatformGrasp:

    def __init__(self, node: Node, args):
        self.node = node
        self.log  = node.get_logger()

        # 夾取 / 放置參數
        self.cx, self.cy, self.cz = args.cube_x, args.cube_y, args.cube_z
        self.cs = args.cube_size
        self.px, self.py, self.pz = args.place_x, args.place_y, args.place_z
        self.offset    = args.pre_grasp_offset
        self.grasp_zoff = args.grasp_z_offset
        self.grasp_xoff = args.grasp_x_offset  # 補償下降時 X 漂移
        self.grasp_yoff = args.grasp_y_offset  # 補償下降時 Y 漂移
        self.grasp_q   = _rpy_to_quaternion(args.grasp_roll, args.grasp_pitch, args.grasp_yaw)

        # 平台參數
        self.plat = dict(
            x=args.platform_x, y=args.platform_y, z=args.platform_z,
            sx=args.platform_size_x, sy=args.platform_size_y, sz=args.platform_size_z,
        )

        # Isaac Sim mimic joint publisher（finger_joint2 不在 ros2_control 控制內）
        self._finger2_pub = node.create_publisher(
            JointState, '/Isaac/left_joint_command', 1)

        # Action / Service clients
        self._move_ac  = ActionClient(node, MoveGroup,         '/move_action')
        self._exec_ac  = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        self._cart_sc  = node.create_client(GetCartesianPath,  '/compute_cartesian_path')
        self._scene_sc = node.create_client(ApplyPlanningScene, '/apply_planning_scene')

        # TF
        self._tf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf, node)

        # Wait for servers
        self.log.info("Waiting for move_group servers...")
        assert self._move_ac.wait_for_server(timeout_sec=15.0),  "/move_action unavailable"
        assert self._exec_ac.wait_for_server(timeout_sec=10.0),  "/execute_trajectory unavailable"
        assert self._cart_sc.wait_for_service(timeout_sec=10.0), "/compute_cartesian_path unavailable"
        assert self._scene_sc.wait_for_service(timeout_sec=10.0), "/apply_planning_scene unavailable"
        self.log.info("All servers ready.")

    # ─── 主流程（對應 C++ PlatformGraspNode::run）───────────────────────────

    def run(self):
        cx, cy, cz = self.cx, self.cy, self.cz
        cs = self.cs
        px, py, pz = self.px, self.py, self.pz
        offset      = self.offset
        grasp_zoff  = self.grasp_zoff
        top_cube    = cz + cs / 2.0
        top_place   = pz + cs / 2.0

        # ── 0  清除上一次 episode 遺留的碰撞物件 ─────────────────────────────
        self._remove_box("basket")
        self._remove_box("cube")
        self._remove_box("platform")

        # ── 1  先回到 home，確保起始狀態安全 ────────────────────────────────
        self._log_step(1, 12, "Move to home")

        # 前一個 episode 在 Step 7/8 失敗時，手臂會停在 z≈0.30（平台正上方）。
        # 若不加入平台碰撞，OMPL 會規劃穿過平台的路徑，物理上被平台擋在 z≈0.21。
        # 但若手臂在 USD default 位置（z≈0.207，links 在平台碰撞框內），
        # 加入平台會觸發 -10 START_STATE_IN_COLLISION，所以只在 z>0.25 時才加入。
        eef_init = self._current_eef_pose()
        platform_added_for_home = False
        if eef_init is not None and eef_init.position.z > 0.25:
            self._add_platform()
            platform_added_for_home = True

        for attempt in range(5):
            home_ok = self._move_home(vel=0.3, accel=0.1)
            time.sleep(1.5)  # let TF and joint_states settle after trajectory
            eef = self._current_eef_pose()
            # 門檻 0.12：MoveIt home 的 EEF z≈0.082 可通過
            if eef is not None and eef.position.z < 0.12:
                if not home_ok and attempt == 0:
                    # 手臂已在 home 但 move_group 拒絕請求：move_group 不響應
                    self.log.error("move_group unresponsive (home rejected, arm already at home), aborting episode")
                    return
                break
            eef_z = f"{eef.position.z:.3f}" if eef is not None else "N/A"
            self.log.warning(
                f"Home not reached (EEF z={eef_z}), retrying ({attempt + 1}/5)..."
            )
            time.sleep(1.0)
        else:
            # OMPL home × 5 失敗：手臂可能卡在奇異配置（如籃子正上方）。
            # 先移除 platform，用 Cartesian 逃脫到安全位置，再重試 OMPL home。
            self.log.warning("OMPL home × 5 failed, attempting Cartesian escape...")
            if platform_added_for_home:
                self._remove_box("platform")
                platform_added_for_home = False
            eef_stuck = self._current_eef_pose()
            escaped = False
            if eef_stuck is not None and eef_stuck.position.z > 0.20:
                # 垂直上升至 z=0.45
                _esc_wp = Pose()
                _esc_wp.position.x  = eef_stuck.position.x
                _esc_wp.position.y  = eef_stuck.position.y
                _esc_wp.position.z  = 0.45
                _esc_wp.orientation = eef_stuck.orientation
                self._run_cartesian([_esc_wp], vel=0.10, accel=0.05, min_fraction=0.3)
                # 若靠近籃子（|y| < 0.20），水平橫移遠離
                eef_risen = self._current_eef_pose()
                if eef_risen is not None and abs(eef_risen.position.y) < 0.20:
                    _lat_wp = Pose()
                    _lat_wp.position.x  = eef_risen.position.x
                    _lat_wp.position.y  = 0.30
                    _lat_wp.position.z  = eef_risen.position.z
                    _lat_wp.orientation = eef_risen.orientation
                    self._run_cartesian([_lat_wp], vel=0.10, accel=0.05, min_fraction=0.3)
                # 重試 OMPL home × 3
                self._add_platform()
                for _ in range(3):
                    self._move_home(vel=0.3, accel=0.1)
                    time.sleep(1.5)
                    eef = self._current_eef_pose()
                    if eef is not None and eef.position.z < 0.12:
                        escaped = True
                        break
                self._remove_box("platform")
            if not escaped:
                self.log.error("Cartesian escape also failed, aborting episode")
                return

        if platform_added_for_home:
            self._remove_box("platform")

        # ── 2  加入方塊、平台、籃子至規劃場景 ────────────────────────────────
        self._log_step(2, 12, "Add cube and platform to planning scene")
        self._add_box("cube", cx, cy, cz, cs)
        self._add_platform()
        self._add_basket()

        eef = self._current_eef_pose()
        if eef:
            self.log.info(
                f"Current EEF: pos=({eef.position.x:.4f}, {eef.position.y:.4f}, "
                f"{eef.position.z:.4f})")

        # ── 3  開夾爪 ─────────────────────────────────────────────────────────
        self._log_step(3, 12, "Open gripper")
        self._control_gripper(GRIPPER_OPEN)

        # ── 4  預夾取（OMPL 直接規劃到 pre-grasp，避免中間 Cartesian 失敗）────
        self._log_step(4, 12, "Pre-grasp")
        grasp_xoff = self.grasp_xoff  # 補償 Cartesian 下降時的 XY 漂移
        grasp_yoff = self.grasp_yoff
        if not self._move_pose(cx + grasp_xoff, cy + grasp_yoff, top_cube + offset, self.grasp_q,
                               vel=0.3, accel=0.1):
            self._cleanup_and_home(); return

        # ── 5  下降夾取 ───────────────────────────────────────────────────────
        self._log_step(5, 12, "Descend to grasp")
        self._remove_box("cube")
        self._remove_box("platform")
        self._remove_box("basket")
        target_z = top_cube + 0.01 + grasp_zoff
        if not self._cartesian_z(target_z, vel=0.1, accel=0.05):
            self._cleanup_and_home(); return

        # GRASP DIAGNOSIS
        self._grasp_diagnosis(cx, cy, cz, cs, top_cube, target_z)

        # ── 6  關夾爪 ─────────────────────────────────────────────────────────
        self._log_step(6, 12, "Close gripper")
        # 先快速移到接近方塊位置，再慢速夾緊，避免高速碰撞方塊
        self._control_gripper(0.018, vel=0.3, accel=0.2)
        self._control_gripper(GRIPPER_CLOSED, vel=0.08, accel=0.04)
        time.sleep(1.5)  # 等物理模擬讓夾爪確實壓緊方塊

        # ── 7  Cartesian 垂直上升至籃子上方安全高度 ──────────────────────────────
        # 夾取後用 Cartesian（Jacobian IK）直線上升，維持夾爪朝下方向。
        # 上升目標 _safe_z=0.37 > 籃子頂部(0.291)+8cm；不超過 0.40 避免奇異配置。
        self._log_step(7, 12, "Cartesian lift after grasp")
        _safe_z = 0.37
        _basket_top = _BASKET['z'] + _BASKET['sz'] / 2.0  # 0.291
        _eef_grasp = self._current_eef_pose()
        if _eef_grasp is None:
            self._cleanup_and_home(); return
        _lift_wp = Pose()
        _lift_wp.position.x  = _eef_grasp.position.x
        _lift_wp.position.y  = _eef_grasp.position.y
        _lift_wp.position.z  = _safe_z
        _lift_wp.orientation = _eef_grasp.orientation  # 保持夾取時方向，不額外修正
        if not self._run_cartesian([_lift_wp], vel=0.10, accel=0.05, min_fraction=0.5):
            # Cartesian 失敗時改用 OMPL（不限方向）
            self.log.warning("Cartesian lift failed, falling back to OMPL")
            self._add_basket()
            if not self._move_pose(
                    _eef_grasp.position.x, _eef_grasp.position.y, _safe_z,
                    q=None, vel=0.15, accel=0.08):
                self._remove_box("basket")
                self._cleanup_and_home(); return
            self._remove_box("basket")

        # ── 8  OMPL 橫移至籃子正上方（軟性方向約束，夾爪盡量朝下）────────────────
        # ori_tol=0.8 rad（≈46°）：給 OMPL 足夠自由度找到路徑，
        # 同時防止夾爪偏朝上（朝上需偏轉 >90°，遠超 tolerance）。
        # Cartesian lift 後關節配置安全，不會再 crash (-26)。
        self._log_step(8, 12, "Swing to pre-place above basket")
        self._add_basket()
        if not self._move_pose(px, py, _safe_z, q=self.grasp_q,
                               vel=0.15, accel=0.08, ori_tol=0.8):
            self._remove_box("basket")
            # OMPL 失敗時改用 Cartesian 橫移（保持當前方向，無籃子碰撞）
            self.log.warning("OMPL swing failed, falling back to Cartesian sweep")
            _eef_cur = self._current_eef_pose()
            if _eef_cur is None:
                self._cleanup_and_home(); return
            _swing_wp = Pose()
            _swing_wp.position.x = px
            _swing_wp.position.y = py
            _swing_wp.position.z = _safe_z
            _swing_wp.orientation = _eef_cur.orientation
            if not self._run_cartesian([_swing_wp], vel=0.08, accel=0.04, min_fraction=0.5):
                self._cleanup_and_home(); return
        else:
            self._remove_box("basket")
        time.sleep(0.5)

        # ── 9  Cartesian 小幅下降至放置高度（帶籃子碰撞）───────────────────────
        # 加入籃子碰撞：若 Step 8 OMPL 後手臂仍有角度偏斜，Cartesian 規劃
        # 偵測到潛在碰撞後早停（而不是實體碰撞）。就算只下降部分也無妨，
        # 只要 TCP 在籃口以上就能安全釋放方塊。
        self._log_step(9, 12, "Lower to release height")
        _eef_pre_lower = self._current_eef_pose()
        if _eef_pre_lower is None:
            self._cleanup_and_home(); return
        _release_z = _basket_top + 0.04  # ≈ 0.331，TCP 在籃口(0.291)上方 4cm
        _lower_wp = Pose()
        _lower_wp.position.x  = px
        _lower_wp.position.y  = py
        _lower_wp.position.z  = _release_z
        _lower_wp.orientation = _eef_pre_lower.orientation
        self._add_basket()
        # min_fraction=0.3：只要下降了 30% 就接受（arm 仍在籃口以上）
        if not self._run_cartesian([_lower_wp], vel=0.05, accel=0.03, min_fraction=0.3):
            self.log.warning("Descent blocked by basket collision, releasing at current height")
        self._remove_box("basket")

        # ── 10  開夾爪放下方塊 ────────────────────────────────────────────────
        self._log_step(10, 12, "Release")
        self._remove_box("platform")
        self._remove_box("cube")
        self._control_gripper(GRIPPER_OPEN)
        time.sleep(0.5)

        # ── 11  純 Cartesian 逃脫籃子區域（不用 OMPL，避免 -26 crash）──────────
        # 在籃子正上方位置使用 OMPL + basket 碰撞 → IK 採樣 crash (-26)。
        # 解法：全程 Cartesian：先垂直上升，再水平橫移離開籃子，完全避開 OMPL。
        self._log_step(11, 12, "Cartesian escape from basket area")
        _eef_placed = self._current_eef_pose()
        if _eef_placed is not None:
            # 1. 垂直上升至 z=0.45（比 _safe_z 高，給 OMPL home 更多空間）
            _rise_wp = Pose()
            _rise_wp.position.x  = _eef_placed.position.x
            _rise_wp.position.y  = _eef_placed.position.y
            _rise_wp.position.z  = 0.45
            _rise_wp.orientation = _eef_placed.orientation
            self._run_cartesian([_rise_wp], vel=0.10, accel=0.05, min_fraction=0.3)
            # 2. 水平橫移離開籃子（+Y 方向，遠離籃子中心）
            _eef_risen = self._current_eef_pose()
            if _eef_risen is not None:
                _lat_wp = Pose()
                _lat_wp.position.x  = _eef_risen.position.x
                _lat_wp.position.y  = py + 0.25  # 遠離籃子
                _lat_wp.position.z  = _eef_risen.position.z
                _lat_wp.orientation = _eef_risen.orientation
                self._run_cartesian([_lat_wp], vel=0.10, accel=0.05, min_fraction=0.3)

        # ── 12  回到 home（帶 platform + basket 碰撞，防止路徑穿過障礙物）────
        # Step 11 Cartesian 橫移後，手臂已離開籃子區域，加入 basket 碰撞安全。
        self._log_step(12, 12, "Return to home")
        time.sleep(0.5)
        self._add_platform()
        self._add_basket()
        for attempt in range(5):
            self._move_home(vel=0.3, accel=0.1)
            time.sleep(1.5)
            eef = self._current_eef_pose()
            if eef is not None and eef.position.z < 0.12:
                break
            eef_z = f"{eef.position.z:.3f}" if eef is not None else "N/A"
            self.log.warning(
                f"Home not reached (EEF z={eef_z}), retrying ({attempt + 1}/5)..."
            )
            time.sleep(1.0)
        self._remove_box("platform")
        self._remove_box("basket")

        self.log.info("Pick-and-place complete!")

    # ─── MoveGroup action 輔助 ──────────────────────────────────────────────

    def _send_move(self, req: MotionPlanRequest, vel=0.3, accel=0.1,
                   timeout=90.0) -> bool:
        req.max_velocity_scaling_factor    = vel
        req.max_acceleration_scaling_factor = accel

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False

        gh = _wait(self._move_ac.send_goal_async(goal), timeout=3.0)
        if gh is None or not gh.accepted:
            self.log.error("MoveGroup goal rejected")
            return False

        r = _wait(gh.get_result_async(), timeout=timeout)
        if r is None:
            self.log.error("MoveGroup result timed out")
            self._last_error_code = 0
            return False

        self._last_error_code = r.result.error_code.val
        ok = r.result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.log.error(f"MoveGroup failed: error_code={r.result.error_code.val}")
        return ok

    def _move_home(self, vel=0.3, accel=0.1) -> bool:
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.allowed_planning_time = 15.0
        req.num_planning_attempts = 10
        req.goal_constraints.append(
            _joint_constraints(LEFT_ARM_JOINTS, [0.0] * 7, tol=0.01)
        )
        return self._send_move(req, vel, accel)

    def _move_pose(self, x, y, z, q: Quaternion | None = None,
                   vel=0.3, accel=0.1,
                   pos_tol=0.01, ori_tol=0.3) -> bool:
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.allowed_planning_time = 15.0
        req.num_planning_attempts = 10

        c = Constraints()

        pos = PositionConstraint()
        pos.header.frame_id = PLANNING_FRAME
        pos.link_name = EEF_LINK
        pos.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[pos_tol])
        )
        pos.constraint_region.primitive_poses.append(
            Pose(position=Point(x=x, y=y, z=z),
                 orientation=Quaternion(w=1.0))
        )
        pos.weight = 1.0
        c.position_constraints.append(pos)

        if q is not None:
            ori = OrientationConstraint()
            ori.header.frame_id = PLANNING_FRAME
            ori.link_name = EEF_LINK
            ori.orientation = q
            ori.absolute_x_axis_tolerance = ori_tol
            ori.absolute_y_axis_tolerance = ori_tol
            ori.absolute_z_axis_tolerance = ori_tol
            ori.weight = 1.0
            c.orientation_constraints.append(ori)

        req.goal_constraints.append(c)

        self._last_error_code = 0
        for attempt in range(1, 6):
            if self._send_move(req, vel, accel):
                return True
            self.log.warning(f"Attempt {attempt}/5 failed for pose ({x:.3f}, {y:.3f}, {z:.3f}), retrying...")
            if self._last_error_code == -26:
                self.log.warning("MoveGroup CRASH (-26) detected, waiting 5s for recovery...")
                time.sleep(5.0)
        self.log.error(f"Planning failed after 5 attempts for pose ({x:.3f}, {y:.3f}, {z:.3f})")
        return False

    def _control_gripper(self, position: float, vel: float = 0.5, accel: float = 0.3):
        req = MotionPlanRequest()
        req.group_name = GRIPPER_GROUP
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 3
        req.goal_constraints.append(
            _joint_constraints([GRIPPER_JOINT], [position], tol=0.005)
        )
        label = "open" if position > 0.02 else "close"
        self.log.info(f"Gripper {label} ({position:.4f} m, vel={vel})...")
        self._send_move(req, vel=vel, accel=accel)
        # Isaac Sim 不自動執行 URDF mimic：finger_joint2 需要另外發指令
        self._mirror_finger2(position)

    def _mirror_finger2(self, position: float):
        """將 finger_joint2（mimic joint）的指令直接送到 Isaac Sim 的合併 topic。"""
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ["openarm_left_finger_joint2"]
        msg.position = [position]
        msg.velocity = [0.0]
        msg.effort = [0.0]
        # 多發幾次確保 isaac_joint_command_merger 的 dict 被更新
        for _ in range(5):
            self._finger2_pub.publish(msg)
            time.sleep(0.02)

    # ─── Cartesian path 輔助（GetCartesianPath + ExecuteTrajectory）────────

    def _current_eef_pose(self) -> Pose | None:
        try:
            t = self._tf.lookup_transform(
                PLANNING_FRAME, EEF_LINK,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            p = Pose()
            p.position.x    = t.transform.translation.x
            p.position.y    = t.transform.translation.y
            p.position.z    = t.transform.translation.z
            p.orientation.x = t.transform.rotation.x
            p.orientation.y = t.transform.rotation.y
            p.orientation.z = t.transform.rotation.z
            p.orientation.w = t.transform.rotation.w
            return p
        except TransformException as e:
            self.log.error(f"TF lookup failed: {e}")
            return None

    def _run_cartesian(self, waypoints: list,
                       vel=0.1, accel=0.05,
                       min_fraction=0.9,
                       avoid_collisions=True) -> bool:
        if not waypoints:
            return True

        req = GetCartesianPath.Request()
        req.header.frame_id = PLANNING_FRAME
        req.group_name = ARM_GROUP
        req.link_name  = EEF_LINK
        req.waypoints  = waypoints
        req.max_step   = 0.005
        req.jump_threshold  = 1.5
        req.avoid_collisions = avoid_collisions

        resp = _wait(self._cart_sc.call_async(req), timeout=15.0)
        if resp is None:
            self.log.error("GetCartesianPath timed out")
            return False
        if resp.fraction < min_fraction:
            self.log.error(
                f"Cartesian path {resp.fraction*100:.0f}% complete "
                f"(min={min_fraction*100:.0f}%)"
            )
            return False
        if resp.fraction < 1.0:
            self.log.warning(f"Partial Cartesian path: {resp.fraction*100:.0f}%")

        # 加入時間戳（GetCartesianPath 不提供）
        traj_msg = resp.solution
        _time_parameterize(traj_msg, vel)

        # 執行
        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = traj_msg

        gh = _wait(self._exec_ac.send_goal_async(exec_goal), timeout=10.0)
        if gh is None or not gh.accepted:
            self.log.error("ExecuteTrajectory goal rejected")
            return False

        r = _wait(gh.get_result_async(), timeout=60.0)
        if r is None:
            self.log.error("ExecuteTrajectory timed out")
            return False

        ok = r.result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.log.error(f"ExecuteTrajectory failed: {r.result.error_code.val}")
        return ok

    def _cartesian_z(self, target_z: float,
                     vel=0.1, accel=0.05,
                     min_fraction=0.9) -> bool:
        current = self._current_eef_pose()
        if current is None:
            return False
        target = Pose()
        target.position.x  = current.position.x
        target.position.y  = current.position.y
        target.position.z  = target_z
        # 固定用 grasp_q，不繼承 TF 當前姿態偏差
        target.orientation = self.grasp_q
        return self._run_cartesian([target], vel, accel, min_fraction)

    def _cartesian_xyz(self, tx: float, ty: float, tz: float,
                       vel=0.1, accel=0.05,
                       min_fraction=0.9) -> bool:
        target = Pose()
        target.position.x  = tx
        target.position.y  = ty
        target.position.z  = tz
        # 固定用 grasp_q，不繼承 TF 當前姿態偏差
        target.orientation = self.grasp_q
        return self._run_cartesian([target], vel, accel, min_fraction)

    # ─── Joint7 reset ────────────────────────────────────────────────────────

    def _reset_joint7(self):
        # 使用 MoveGroup action 控制單一關節歸零
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 3

        # 只限制 joint7；其他關節不設約束（OMPL 自由選擇）
        c = Constraints()
        jc = JointConstraint()
        jc.joint_name    = "openarm_left_joint7"
        jc.position      = 0.0
        jc.tolerance_above = 0.05
        jc.tolerance_below = 0.05
        jc.weight = 1.0
        c.joint_constraints.append(jc)
        req.goal_constraints.append(c)

        self.log.info("Resetting joint7 to 0...")
        self._send_move(req, vel=0.2, accel=0.1)

    # ─── 規劃場景碰撞物件 ────────────────────────────────────────────────────

    def _apply_scene(self, scene: PlanningScene):
        req = ApplyPlanningScene.Request()
        req.scene = scene
        resp = _wait(self._scene_sc.call_async(req), timeout=2.0)
        if resp is None or not resp.success:
            self.log.warning("ApplyPlanningScene failed or timed out")

    def _add_box(self, obj_id: str, x, y, z, size: float):
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = obj_id
        co.operation = CollisionObject.ADD
        box = SolidPrimitive(type=SolidPrimitive.BOX,
                             dimensions=[size, size, size])
        co.primitives.append(box)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self._apply_scene(scene)
        self.log.info(f"Added collision object '{obj_id}'.")

    def _add_platform(self):
        p = self.plat
        # 碰撞物件向上延伸 5 cm 作為安全裕度：
        # 讓 OMPL 規劃時手臂必須在高於實際平台頂面 5cm 的位置才能越過邊緣，
        # 避免路徑在 XY 越過邊緣時高度不足而碰撞。底部位置不變。
        EDGE_PADDING_Z = 0.02
        sz_padded = p['sz'] + EDGE_PADDING_Z
        z_padded  = p['z'] + EDGE_PADDING_Z / 2   # 中心上移，底部不變

        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = "platform"
        co.operation = CollisionObject.ADD
        box = SolidPrimitive(type=SolidPrimitive.BOX,
                             dimensions=[p['sx'], p['sy'], sz_padded])
        co.primitives.append(box)
        pose = Pose()
        pose.position.x = p['x']
        pose.position.y = p['y']
        pose.position.z = z_padded
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self._apply_scene(scene)
        self.log.info(
            f"Platform: center=({p['x']:.4f}, {p['y']:.4f}, {p['z']:.4f})  "
            f"size=({p['sx']:.4f}, {p['sy']:.4f}, {p['sz']:.4f})  "
            f"top_z={p['z'] + p['sz']/2:.4f}  (collision top_z={z_padded + sz_padded/2:.4f})"
        )
        time.sleep(0.3)

    def _add_basket(self):
        b = _BASKET
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = "basket"
        co.operation = CollisionObject.ADD
        box = SolidPrimitive(type=SolidPrimitive.BOX,
                             dimensions=[b['sx'], b['sy'], b['sz']])
        co.primitives.append(box)
        pose = Pose()
        pose.position.x = b['x']
        pose.position.y = b['y']
        pose.position.z = b['z']
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)
        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self._apply_scene(scene)
        self.log.info("Added collision object 'basket'.")
        time.sleep(0.2)

    def _cleanup_and_home(self):
        """失敗時清除碰撞物件並嘗試回 home。
        若手臂卡在奇異配置（如籃子上方），先 Cartesian 逃脫再 OMPL home。
        """
        for obj in ("basket", "cube", "platform"):
            self._remove_box(obj)
        eef = self._current_eef_pose()
        if eef is None or eef.position.z < 0.12:
            return
        # 若手臂在高位（z > 0.20），先 Cartesian 逃脫到安全位置
        if eef.position.z > 0.20:
            _esc_wp = Pose()
            _esc_wp.position.x  = eef.position.x
            _esc_wp.position.y  = eef.position.y
            _esc_wp.position.z  = 0.45
            _esc_wp.orientation = eef.orientation
            self._run_cartesian([_esc_wp], vel=0.10, accel=0.05, min_fraction=0.3)
            eef2 = self._current_eef_pose()
            if eef2 is not None and abs(eef2.position.y) < 0.20:
                _lat_wp = Pose()
                _lat_wp.position.x  = eef2.position.x
                _lat_wp.position.y  = 0.30
                _lat_wp.position.z  = eef2.position.z
                _lat_wp.orientation = eef2.orientation
                self._run_cartesian([_lat_wp], vel=0.10, accel=0.05, min_fraction=0.3)
        self._add_platform()
        for _ in range(3):
            self._move_home(vel=0.4, accel=0.2)
            time.sleep(1.0)
            eef = self._current_eef_pose()
            if eef is not None and eef.position.z < 0.12:
                break
        self._remove_box("platform")

    def _remove_box(self, obj_id: str):
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = obj_id
        co.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self._apply_scene(scene)
        self.log.info(f"Removed collision object '{obj_id}'.")
        time.sleep(0.2)

    # ─── 診斷 ────────────────────────────────────────────────────────────────

    def _grasp_diagnosis(self, cx, cy, cz, cs, top_cube, target_z):
        tcp = self._current_eef_pose()
        if tcp is None:
            self.log.warning("[GRASP DIAGNOSIS] Could not get TCP pose")
            return
        tx, ty, tz = tcp.position.x, tcp.position.y, tcp.position.z
        q = tcp.orientation
        # 從 quaternion 反推 RPY（供調試 grasp_roll/pitch/yaw 用）
        sinr = 2*(q.w*q.x + q.y*q.z)
        cosr = 1 - 2*(q.x*q.x + q.y*q.y)
        roll = math.atan2(sinr, cosr)
        sinp = 2*(q.w*q.y - q.z*q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny, cosy)
        gq = self.grasp_q
        self.log.warning("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.log.warning("[GRASP DIAGNOSIS]")
        self.log.warning(f"  Cube center    : ({cx:.5f}, {cy:.5f}, {cz:.5f})")
        self.log.warning(f"  cube_size      : {cs:.5f}  → top_cube={top_cube:.5f}")
        self.log.warning(f"  grasp target Z : {target_z:.5f}")
        self.log.warning(f"  TCP actual pos : ({tx:.5f}, {ty:.5f}, {tz:.5f})")
        self.log.warning(f"  TCP - cube XYZ : ({tx-cx:.5f}, {ty-cy:.5f}, {tz-cz:.5f})")
        self.log.warning(f"  TCP Z - top    : {tz - top_cube:.5f}  (ideal: ~0.0)")
        self.log.warning(f"  TCP orient quat: x={q.x:.4f} y={q.y:.4f} z={q.z:.4f} w={q.w:.4f}")
        self.log.warning(f"  TCP orient RPY : r={math.degrees(roll):.2f}° p={math.degrees(pitch):.2f}° y={math.degrees(yaw):.2f}°")
        self.log.warning(f"  Target grasp q : x={gq.x:.4f} y={gq.y:.4f} z={gq.z:.4f} w={gq.w:.4f}")
        self.log.warning("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _log_step(self, step, total, msg):
        self.log.info(f"── {step}/{total}  {msg} ──")


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MoveIt2 platform pick-and-place")
    # 夾取 / 放置
    p.add_argument("--cube_x",           type=float, default=0.22321)
    p.add_argument("--cube_y",           type=float, default=0.16423)
    p.add_argument("--cube_z",           type=float, default=0.21344)
    p.add_argument("--cube_size",        type=float, default=0.04)
    p.add_argument("--place_x",          type=float, default=0.25)
    p.add_argument("--place_y",          type=float, default=0.1)
    p.add_argument("--place_z",          type=float, default=0.25)
    p.add_argument("--pre_grasp_offset", type=float, default=0.15)
    p.add_argument("--grasp_z_offset",   type=float, default=-0.020)
    p.add_argument("--grasp_x_offset",   type=float, default=0.015)
    p.add_argument("--grasp_y_offset",   type=float, default=0.006)
    p.add_argument("--grasp_roll",       type=float, default=math.pi)
    p.add_argument("--grasp_pitch",      type=float, default=0.0)
    p.add_argument("--grasp_yaw",        type=float, default=0.0)
    # 平台
    p.add_argument("--platform_x",      type=float, default=_PLAT['x'])
    p.add_argument("--platform_y",      type=float, default=_PLAT['y'])
    p.add_argument("--platform_z",      type=float, default=_PLAT['z'])
    p.add_argument("--platform_size_x", type=float, default=_PLAT['sx'])
    p.add_argument("--platform_size_y", type=float, default=_PLAT['sy'])
    p.add_argument("--platform_size_z", type=float, default=_PLAT['sz'])
    return p.parse_args()


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rclpy.init()
    node = rclpy.create_node("platform_grasp")

    # 背景 executor 讓 spin 能在等待 action/service 時繼續處理 callbacks
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        pg = PlatformGrasp(node, args)
        pg.run()
    except AssertionError as e:
        node.get_logger().error(str(e))
    except KeyboardInterrupt:
        pass

    # 不呼叫 executor.shutdown() / rclpy.shutdown()：
    # 這兩個呼叫會在 spin_thread 活躍時觸發 rclcpp C++ 解構子，
    # 造成 "terminate called without an active exception" (SIGABRT, exit -6)，
    # 進而污染 move_group 的狀態機，讓下一個 episode 的所有規劃都回傳 -26 (CRASH)。
    # os._exit(0) 直接結束 process，讓 move_group 通過正常的連線逾時機制清理狀態。
    time.sleep(0.3)  # 等待最後一批 log 輸出
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
