#!/usr/bin/env python3
"""
bimanual_handover.py ─ OpenArm 雙臂空中手遞手 pick→handoff→place（Phase 2）

本檔是 openarm_manipulation/src/bimanual_handoff.cpp 的 Python port（沿用
platform_grasp.cpp→platform_grasp.py 的模式），交接幾何/17 步序列忠實對應 C++，
另加 Isaac Sim 必要修正：
  - finger_joint2（mimic）需手動鏡射發指令（Isaac 不自動跑 URDF mimic）
  - MoveGroup -26 CRASH 重試
  - 交接瞬間 force-close 夾爪（繞過 move_group 碰撞檢查；兩臂貼近時 plan 會誤判碰撞）
  - 結尾 os._exit 避免 rclcpp 解構子污染 move_group 狀態

交接幾何（兩夾爪正交，握物體不同對立面，互不干涉）：
  左手交接：roll=π/2      → EEF z→-y，從 +y 側握 ±x 面
  右手交接：RPY(0,-π/2,-π/2) → EEF z→+y 從 -y 側插入，夾爪沿 ±z（垂直左臂 ±x）

前置（3 終端，見 run_scene_bridge.py）：
  T1: isaaclab.sh -p scripts/isaac/run_scene_bridge.py          # Isaac 場景 + ROS2 bridge
  T2: ros2 launch openarm_bimanual_moveit_config demo.launch.py use_isaac_sim:=true launch_rviz:=false
  T3: python3 scripts/ros2/bimanual_handover.py [--basket red|yellow] [--stop_after present|all]
"""

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
    PositionConstraint, OrientationConstraint, JointConstraint,
    CollisionObject, PlanningScene, RobotState,
    RobotTrajectory, MoveItErrorCodes, PlanningOptions,
)
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
from tf2_ros import TransformException

PLANNING_FRAME = "world"
GRIPPER_OPEN   = 0.044
GRIPPER_CLOSED = 0.005

# 手臂關節位置限位（從 URDF xacro 展開取得）。交接姿態會把 joint6(±π/4)/joint7(±π/2) 等頂到限位，
# 物理過衝 ~0.0003 rad → move_group 的 CheckStartStateBounds 直接 abort 規劃（就是一直看到的「-26」根因）。
# 送規劃前用此表把 start_state 夾回限內（補 C++ 漏掉的 enforceBounds）。
ARM_JOINT_LIMITS = {
    "openarm_left_joint1": (-1.39626, 3.49066), "openarm_right_joint1": (-1.39626, 3.49066),
    "openarm_left_joint2": (-1.74533, 1.74533), "openarm_right_joint2": (-1.74533, 1.74533),
    "openarm_left_joint3": (-1.57080, 1.57080), "openarm_right_joint3": (-1.57080, 1.57080),
    "openarm_left_joint4": (0.00000, 2.44346),  "openarm_right_joint4": (0.00000, 2.44346),
    "openarm_left_joint5": (-1.57080, 1.57080), "openarm_right_joint5": (-1.57080, 1.57080),
    "openarm_left_joint6": (-0.78540, 0.78540), "openarm_right_joint6": (-0.78540, 0.78540),
    "openarm_left_joint7": (-1.57080, 1.57080), "openarm_right_joint7": (-1.57080, 1.57080),
}

# 場景物件（apple 2026-06-18；strawberry 2026-06-22 USD bbox 量測）
# sx/sy/sz = 各軸 bbox 尺寸；grasp_zoff = 下探深度（descend z = top + 0.01 + grasp_zoff）
# gx_comp/gy_comp = 夾取目標的 xy 補償：手臂遠伸時控制器系統性短 ~x（apple 短 1.1cm、strawberry 短 2cm），
# 往前補讓夾爪落在水果真實中心。box(step2) 仍用真實 xy，只補償夾取目標(step4/7)。
# grasp_tilt = 夾取姿態前傾 pitch（負值=approach 往 +x 前傾，讓左手前伸；只用於 step4-7，step8 轉回 top-down）
APPLE      = dict(x=0.2309, y=0.1445, z=0.2233, sx=0.050, sy=0.050, sz=0.050, grasp_zoff=-0.030, gx_comp=0.000, gy_comp=0.000, grasp_tilt=0.000, grasp_yaw=0.000)  # 5cm 球
STRAWBERRY = dict(x=0.3146, y=0.1468, z=0.2108, sx=0.039, sy=0.040, sz=0.025, grasp_zoff=-0.018, gx_comp=0.013, gy_comp=0.000, grasp_tilt=-0.650, grasp_yaw=0.000)  # 矮+遠：前傾擴可達區 + gx_comp 把 TCP 對到捕捉窗口中心~0.310
# grasp_yaw = 繞垂直軸轉夾爪讓兩指闔在物件短軸（長條形用）；預設 top-down 指尖已沿世界 Y 闔
BANANA     = dict(x=0.2789, y=0.2500, z=0.2109, sx=0.138, sy=0.055, sz=0.033, grasp_zoff=-0.018, gx_comp=0.000, gy_comp=0.000, grasp_tilt=0.000, grasp_yaw=0.000, ho_dx=0.000, place_dx=0.000)  # ho_dx=0：右手夾質心(同apple)；偏心0.045會讓左手放手後香蕉繞夾持點翻轉~130°脫出(HDF5實證)
FRUITS     = {"apple": APPLE, "strawberry": STRAWBERRY, "banana": BANANA}
PLATFORM = dict(x=0.38591, y=0.0, z=0.10435, sx=0.41433, sy=0.8, sz=0.17818)
BASKETS  = {  # 動態剛體，bbox 頂面 Z≈0.242
    "red":    dict(x=0.2735, y=-0.0729, top_z=0.242),
    "yellow": dict(x=0.2735, y=-0.1785, top_z=0.242),
}

SIDE_CFG = {
    "left": dict(
        arm_group="left_arm", grip_group="left_gripper", eef="openarm_left_hand_tcp",
        arm_joints=[f"openarm_left_joint{i}" for i in range(1, 8)],
        grip_joint="openarm_left_finger_joint1", finger2="openarm_left_finger_joint2",
        joint7="openarm_left_joint7", isaac_topic="/Isaac/left_joint_command",
    ),
    "right": dict(
        arm_group="right_arm", grip_group="right_gripper", eef="openarm_right_hand_tcp",
        arm_joints=[f"openarm_right_joint{i}" for i in range(1, 8)],
        grip_joint="openarm_right_finger_joint1", finger2="openarm_right_finger_joint2",
        joint7="openarm_right_joint7", isaac_topic="/Isaac/right_joint_command",
    ),
}


# ─── 工具函式（沿用 platform_grasp.py）───────────────────────────────────────

def _rpy_to_quaternion(roll, pitch, yaw) -> Quaternion:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def _wait(future, timeout=60.0):
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


def _time_parameterize(traj_msg, vel_scale):
    pts = traj_msg.joint_trajectory.points
    if len(pts) < 2:
        return traj_msg
    max_joint_vel = 3.14 * max(vel_scale, 0.05)
    t = 0.0
    for i, pt in enumerate(pts):
        if i > 0:
            max_disp = max(abs(a - b) for a, b in zip(pts[i - 1].positions, pt.positions))
            t += max_disp / max_joint_vel if max_disp > 1e-6 else 0.01
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        pt.velocities = [0.0] * len(pt.positions)
        pt.accelerations = [0.0] * len(pt.positions)
    return traj_msg


# ─── 單臂控制器（platform_grasp helper + C++ handoff 用到的擴充，以 side 參數化）──

class Arm:
    def __init__(self, node: Node, side: str, tf_buffer, joint_cache: dict):
        cfg = SIDE_CFG[side]
        self.node = node
        self.log = node.get_logger()
        self.side = side
        self.arm_group, self.grip_group, self.eef = cfg["arm_group"], cfg["grip_group"], cfg["eef"]
        self.arm_joints, self.grip_joint = cfg["arm_joints"], cfg["grip_joint"]
        self.finger2, self.joint7 = cfg["finger2"], cfg["joint7"]
        self._tf = tf_buffer
        self._jcache = joint_cache          # 共用 /joint_states 快取（name→pos）
        self._last_error_code = 0
        self._rec = False                   # 去程 dense 取樣開關（回程反向 replay 用）
        self._path = []                     # 去程實際走過的密集關節構型序列

        self._finger2_pub = node.create_publisher(JointState, cfg["isaac_topic"], 1)
        self._move_ac  = ActionClient(node, MoveGroup,         "/move_action")
        self._exec_ac  = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
        self._cart_sc  = node.create_client(GetCartesianPath,  "/compute_cartesian_path")
        self._scene_sc = node.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def wait_servers(self):
        assert self._move_ac.wait_for_server(timeout_sec=15.0),  "/move_action unavailable"
        assert self._exec_ac.wait_for_server(timeout_sec=10.0),  "/execute_trajectory unavailable"
        assert self._cart_sc.wait_for_service(timeout_sec=10.0), "/compute_cartesian_path unavailable"
        assert self._scene_sc.wait_for_service(timeout_sec=10.0), "/apply_planning_scene unavailable"

    def _clamped_start_state(self) -> RobotState:
        """補 enforceBounds：用當前 joint 值建 start_state，超界的夾回限內（留 margin），
        避免交接姿態關節在限位過衝就被 CheckStartStateBounds 拒絕規劃（diff 在 allowed_start_tolerance 內）。"""
        margin = 0.01
        js = JointState()
        for name, (lo, up) in ARM_JOINT_LIMITS.items():
            if name in self._jcache:
                v = max(lo + margin, min(up - margin, float(self._jcache[name])))
                js.name.append(name)
                js.position.append(v)
        rs = RobotState()
        rs.joint_state = js
        rs.is_diff = True   # 只覆寫提供的關節，其餘（夾爪等）沿用 move_group 監看狀態
        return rs

    # ── MoveGroup ──
    def _send_move(self, req, vel=0.3, accel=0.1, timeout=90.0) -> bool:
        req.max_velocity_scaling_factor = vel
        req.max_acceleration_scaling_factor = accel
        req.start_state = self._clamped_start_state()   # 夾回限內，避免 start-state-out-of-bounds 拒絕
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        gh = _wait(self._move_ac.send_goal_async(goal), timeout=3.0)
        if gh is None or not gh.accepted:
            self.log.error(f"[{self.side}] MoveGroup goal rejected")
            return False
        r = _wait(gh.get_result_async(), timeout=timeout)
        if r is None:
            self.log.error(f"[{self.side}] MoveGroup result timed out")
            self._last_error_code = 0
            return False
        self._last_error_code = r.result.error_code.val
        ok = r.result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.log.error(f"[{self.side}] MoveGroup failed: error_code={r.result.error_code.val}")
        return ok

    def move_home(self, vel=0.3, accel=0.1) -> bool:
        req = MotionPlanRequest()
        req.group_name = self.arm_group
        req.allowed_planning_time = 15.0
        req.num_planning_attempts = 10
        req.goal_constraints.append(_joint_constraints(self.arm_joints, [0.0] * 7, tol=0.01))
        return self._send_move(req, vel, accel)

    def move_pose(self, x, y, z, q: Quaternion | None = None,
                  vel=0.3, accel=0.1, pos_tol=0.01, ori_tol=0.3) -> bool:
        """q=None → position-only（方向交給 OMPL，對應 C++ move_position_only）。"""
        req = MotionPlanRequest()
        req.group_name = self.arm_group
        req.allowed_planning_time = 15.0
        req.num_planning_attempts = 10
        c = Constraints()
        pos = PositionConstraint()
        pos.header.frame_id = PLANNING_FRAME
        pos.link_name = self.eef
        pos.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[pos_tol]))
        pos.constraint_region.primitive_poses.append(
            Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(w=1.0)))
        pos.weight = 1.0
        c.position_constraints.append(pos)
        if q is not None:
            ori = OrientationConstraint()
            ori.header.frame_id = PLANNING_FRAME
            ori.link_name = self.eef
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
            self.log.warning(f"[{self.side}] attempt {attempt}/5 failed for "
                             f"({x:.3f},{y:.3f},{z:.3f}), retrying...")
            if self._last_error_code == -26:
                self.log.warning(f"[{self.side}] MoveGroup CRASH (-26), waiting 5s...")
                time.sleep(5.0)
        self.log.error(f"[{self.side}] planning failed after 5 attempts ({x:.3f},{y:.3f},{z:.3f})")
        return False

    # ── 夾爪 ──
    def control_gripper(self, position, vel=0.5, accel=0.3, force=False):
        """force=True：繞過 move_group 規劃，直接 ExecuteTrajectory（兩臂貼近交接時用）。"""
        label = "open" if position > 0.02 else "close"
        if force:
            # jcache 缺值時依開/合方向給合理起點，避免 start==target 變 no-op
            default_start = GRIPPER_CLOSED if position > 0.02 else GRIPPER_OPEN
            start = self._jcache.get(self.grip_joint, default_start)
            jt = JointTrajectory()
            jt.joint_names = [self.grip_joint]
            p1 = JointTrajectoryPoint(positions=[start], velocities=[0.0])
            p1.time_from_start.sec = 0
            p2 = JointTrajectoryPoint(positions=[position], velocities=[0.0])
            p2.time_from_start.sec = 1
            jt.points = [p1, p2]
            rt = RobotTrajectory()
            rt.joint_trajectory = jt
            self.log.info(f"[{self.side}] gripper {label} ({position:.4f} m) [force]...")
            self._execute(rt)
        else:
            req = MotionPlanRequest()
            req.group_name = self.grip_group
            req.allowed_planning_time = 5.0
            req.num_planning_attempts = 3
            req.goal_constraints.append(_joint_constraints([self.grip_joint], [position], tol=0.005))
            self.log.info(f"[{self.side}] gripper {label} ({position:.4f} m, vel={vel})...")
            self._send_move(req, vel=vel, accel=accel)
        self._mirror_finger2(position)

    def _mirror_finger2(self, position):
        """Isaac 不自動跑 mimic：finger_joint2 需另外發指令到該側合併 topic。"""
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = [self.finger2]
        msg.position = [position]
        msg.velocity = [0.0]
        msg.effort = [0.0]
        for _ in range(5):
            self._finger2_pub.publish(msg)
            time.sleep(0.02)

    # ── Cartesian / TF ──
    def current_eef_pose(self) -> Pose | None:
        try:
            t = self._tf.lookup_transform(PLANNING_FRAME, self.eef, rclpy.time.Time(),
                                          timeout=rclpy.duration.Duration(seconds=2.0))
            p = Pose()
            p.position.x, p.position.y, p.position.z = (
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
            p.orientation = Quaternion(
                x=t.transform.rotation.x, y=t.transform.rotation.y,
                z=t.transform.rotation.z, w=t.transform.rotation.w)
            return p
        except TransformException as e:
            self.log.error(f"[{self.side}] TF lookup failed: {e}")
            return None

    def _execute(self, robot_traj: RobotTrajectory) -> bool:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_traj
        gh = _wait(self._exec_ac.send_goal_async(goal), timeout=10.0)
        if gh is None or not gh.accepted:
            self.log.error(f"[{self.side}] ExecuteTrajectory goal rejected")
            return False
        r = _wait(gh.get_result_async(), timeout=60.0)
        if r is None:
            self.log.error(f"[{self.side}] ExecuteTrajectory timed out")
            return False
        ok = r.result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.log.error(f"[{self.side}] ExecuteTrajectory failed: {r.result.error_code.val}")
        return ok

    def run_cartesian(self, waypoints, vel=0.1, accel=0.05,
                      min_fraction=0.9, avoid_collisions=True) -> bool:
        if not waypoints:
            return True
        req = GetCartesianPath.Request()
        req.header.frame_id = PLANNING_FRAME
        req.group_name = self.arm_group
        req.link_name = self.eef
        req.waypoints = waypoints
        req.max_step = 0.005
        req.jump_threshold = 1.5
        req.avoid_collisions = avoid_collisions
        resp = _wait(self._cart_sc.call_async(req), timeout=15.0)
        if resp is None:
            self.log.error(f"[{self.side}] GetCartesianPath timed out")
            return False
        if resp.fraction < min_fraction:
            self.log.error(f"[{self.side}] Cartesian {resp.fraction*100:.0f}% "
                           f"(min={min_fraction*100:.0f}%)")
            return False
        if resp.fraction < 1.0:
            self.log.warning(f"[{self.side}] partial Cartesian: {resp.fraction*100:.0f}%")
        traj = resp.solution
        _time_parameterize(traj, vel)
        return self._execute(traj)

    def cartesian_z(self, target_z, q: Quaternion, vel=0.1, accel=0.05, min_fraction=0.9) -> bool:
        cur = self.current_eef_pose()
        if cur is None:
            return False
        target = Pose()
        target.position.x, target.position.y, target.position.z = cur.position.x, cur.position.y, target_z
        target.orientation = q
        return self.run_cartesian([target], vel, accel, min_fraction)

    def cartesian_xyz(self, tx, ty, tz, q: Quaternion, vel=0.1, accel=0.05,
                      min_fraction=0.9, avoid_collisions=True) -> bool:
        target = Pose()
        target.position.x, target.position.y, target.position.z = tx, ty, tz
        target.orientation = q
        return self.run_cartesian([target], vel, accel, min_fraction, avoid_collisions)

    def cartesian_rotate(self, target_q: Quaternion, min_fraction=0.5) -> bool:
        """原地旋轉到目標方向（停用碰撞，對應 C++ 右手 Phase 2）。"""
        cur = self.current_eef_pose()
        if cur is None:
            return False
        tgt = Pose()
        tgt.position = cur.position
        tgt.orientation = target_q
        return self.run_cartesian([tgt], vel=0.15, accel=0.08,
                                  min_fraction=min_fraction, avoid_collisions=False)

    def _orientation_error(self, q: Quaternion):
        """當前 EEF 方向與目標 q 的夾角（rad）；TF 取不到回 None。"""
        cur = self.current_eef_pose()
        if cur is None:
            return None
        c = cur.orientation
        dot = abs(c.x * q.x + c.y * q.y + c.z * q.z + c.w * q.w)
        return 2.0 * math.acos(min(1.0, dot))

    def reorient_to(self, x, y, z, q: Quaternion, ori_tol=0.2) -> bool:
        """把 EEF 原地轉到目標方向並『驗證真的轉到位』。
        取代原本只靠 cartesian 純旋轉(min_fraction=0.4)：那個常只轉 <40°(36°)就當成功→水果沒轉平。
        先用 OMPL 帶緊方向約束原地 reorient（-26 已由放寬限位根治，可帶緊）；沒到位再 cartesian 補一刀。"""
        self.move_pose(x, y, z, q, ori_tol=ori_tol, pos_tol=0.02, vel=0.2, accel=0.1)
        err = self._orientation_error(q)
        if err is not None and err > 0.25:   # >~14°：還沒轉平
            self.log.warning(f"[{self.side}] reorient 差 {math.degrees(err):.0f}°，cartesian 補轉")
            self.cartesian_rotate(q, min_fraction=0.4)
            err = self._orientation_error(q)
        self.log.info(f"[{self.side}] reorient err="
                      + (f"{math.degrees(err):.0f}°" if err is not None else "NA"))
        return err is not None and err < 0.25

    def reset_joint7(self):
        """回 home 前把 joint7 歸零（避免手腕纏繞），對應 C++ reset_joint7。"""
        cur = self._jcache.get(self.joint7)
        if cur is not None and abs(cur) < 0.05:
            return
        req = MotionPlanRequest()
        req.group_name = self.arm_group
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 3
        c = Constraints()
        jc = JointConstraint()
        jc.joint_name = self.joint7
        jc.position = 0.0
        jc.tolerance_above = jc.tolerance_below = 0.05
        jc.weight = 1.0
        c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        self.log.info(f"[{self.side}] reset {self.joint7} → 0")
        self._send_move(req, vel=0.2, accel=0.1)

    def force_home(self):
        """直接送關節軌跡回 home(zeros)，繞過 move_group 規劃（planner -26 也能執行）。
        每集結束呼叫，確保 bridge 最後指令=home，避免下一集手臂卡在上一集失敗位置。"""
        cur = [float(self._jcache.get(j, 0.0)) for j in self.arm_joints]
        jt = JointTrajectory()
        jt.joint_names = list(self.arm_joints)
        p1 = JointTrajectoryPoint(positions=cur, velocities=[0.0] * 7)
        p1.time_from_start.sec = 0
        p2 = JointTrajectoryPoint(positions=[0.0] * 7, velocities=[0.0] * 7)
        p2.time_from_start.sec = 3
        jt.points = [p1, p2]
        rt = RobotTrajectory()
        rt.joint_trajectory = jt
        self.log.info(f"[{self.side}] force-home (direct joint traj)")
        self._execute(rt)

    # ── 回程反向 replay（去程 dense 取樣關節路徑，回退時反向密集播放 → 乖乖沿原路回，不甩大圈）──
    #   稀疏關鍵構型直線內插會在關節空間掃出巨大笛卡爾弧（實測甩到 z=1.08m）；dense 取樣後每兩點只差
    #   零點幾度，等同 run_cartesian 的密集軌跡，能忠實 retrace 去程實際走過的路徑。
    def rec_on(self):
        self._rec = True

    def rec_off(self):
        self._rec = False

    def sample_if_recording(self):
        """由 Task 的 30Hz timer 呼叫：錄影中就把當前實測關節構型收進去程路徑（去重靜止點）。"""
        if not self._rec:
            return
        if not all(j in self._jcache for j in self.arm_joints):
            return
        cfg = [float(self._jcache[j]) for j in self.arm_joints]
        if self._path and max(abs(a - b) for a, b in zip(self._path[-1], cfg)) < 0.005:
            return   # <0.3° 幾乎沒動，跳過（避免靜止時堆大量重複點）
        self._path.append(cfg)

    def replay_return(self, vel=0.3) -> bool:
        """反向 dense replay 去程記錄的關節路徑回 home。去程從 home 開始錄，故反向序列尾點=home。"""
        if len(self._path) < 2:
            self.log.warning(f"[{self.side}] replay_return: 去程路徑太短（{len(self._path)} 點）")
            return False
        pts = list(reversed(self._path))
        pts[0] = [float(self._jcache.get(j, pts[0][k])) for k, j in enumerate(self.arm_joints)]
        jt = JointTrajectory()
        jt.joint_names = list(self.arm_joints)
        max_v = 3.14 * max(vel, 0.05)
        t = 0.0
        for i, cfg in enumerate(pts):
            if i > 0:
                disp = max(abs(a - b) for a, b in zip(pts[i - 1], cfg))
                t += max(disp / max_v, 0.01)
            p = JointTrajectoryPoint(positions=list(cfg), velocities=[0.0] * 7)
            p.time_from_start.sec = int(t)
            p.time_from_start.nanosec = int((t - int(t)) * 1e9)
            jt.points.append(p)
        rt = RobotTrajectory()
        rt.joint_trajectory = jt
        self.log.info(f"[{self.side}] reverse-replay 回程（dense {len(pts)} 點, {t:.1f}s）")
        return self._execute(rt)

    # ── 規劃場景 ──
    def _apply_scene(self, scene):
        req = ApplyPlanningScene.Request()
        req.scene = scene
        resp = _wait(self._scene_sc.call_async(req), timeout=2.0)
        if resp is None or not resp.success:
            self.log.warning(f"[{self.side}] ApplyPlanningScene failed/timed out")

    def add_box(self, obj_id, x, y, z, dims):
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = obj_id
        co.operation = CollisionObject.ADD
        co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(dims)))
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)
        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self._apply_scene(scene)
        self.log.info(f"[{self.side}] added '{obj_id}'")
        time.sleep(0.2)

    def remove_box(self, obj_id):
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = obj_id
        co.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self._apply_scene(scene)
        time.sleep(0.2)

    def add_platform(self):
        p = PLATFORM
        pad = 0.02
        self.add_box("platform", p["x"], p["y"], p["z"] + pad / 2,
                     [p["sx"], p["sy"], p["sz"] + pad])


# ─── 任務編排（對應 C++ BimanualHandoffNode::run 的 17 步）─────────────────────

class HandoverTask:
    def __init__(self, node: Node, args):
        self.node = node
        self.log = node.get_logger()
        self.args = args
        self._tf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf, node)
        self._jcache = {}
        node.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.left = Arm(node, "left", self._tf, self._jcache)
        self.right = Arm(node, "right", self._tf, self._jcache)
        # 30Hz 背景取樣：去程 rec_on 期間把兩臂實際走過的關節路徑收起來（回程反向 dense replay 用）
        node.create_timer(1.0 / 30.0, self._sample_paths)

        # 抓取/交接/放置姿態（對應 C++）
        self.left_grasp_q   = _rpy_to_quaternion(math.pi, 0.0, 0.0)       # top-down
        self.left_handoff_q = _rpy_to_quaternion(math.pi / 2, 0.0, 0.0)   # 水平，握 ±x
        self.right_handoff_q = _rpy_to_quaternion(0.0, -math.pi / 2, -math.pi / 2)  # 從 -y 插入，握 ±z
        # 長條物（banana）交接：右手需把兩指轉成跨「香蕉短軸」閉合，否則沿長軸(world+X)閉合會戳本體夾不住。
        # right_handoff_q_long = rpy(-π/2,0,0)=繞接近軸轉90°讓指沿world-Z閉合，但實測會把右腕joint7頂到極限→
        #   所有 Cartesian(插入/拉開)失敗，棄用。
        self.right_handoff_q_long = _rpy_to_quaternion(-math.pi / 2, 0.0, 0.0)
        # banana 用：與 _long 同樣指沿 world-Z 上下閉合(approach +y 從-y滑入把香蕉含在上下兩指間)，但繞接近軸
        #   反向180°→joint7 落到中段可動，避開 _long 的極限牆。
        self.right_handoff_q_banana = _rpy_to_quaternion(math.pi / 2, 0.0, math.pi)
        self.right_place_q  = _rpy_to_quaternion(math.pi, 0.0, 0.0)       # top-down 放下

        self.log.info("Waiting for move_group servers...")
        self.left.wait_servers()
        self.log.info("Servers ready.")

    def _on_js(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self._jcache[n] = p

    def _sample_paths(self):
        self.left.sample_if_recording()
        self.right.sample_if_recording()

    def _step(self, i, msg):
        self.log.info(f"── {i}/17  {msg} ──")

    def _home_arm(self, arm: Arm, retries=5) -> bool:
        """回 home，以 EEF z<0.12 判定（home 時 hand_tcp z≈0.082）。
        容忍 move_group 在『手臂已在 home』時規劃 home→home 退化回 -2 的情況
        （對應 platform_grasp.py 的 home 判定）。"""
        for attempt in range(retries):
            arm.move_home()
            time.sleep(1.5)
            eef = arm.current_eef_pose()
            if eef is not None and eef.position.z < 0.12:
                return True
            z = f"{eef.position.z:.3f}" if eef else "NA"
            self.log.warning(f"[{arm.side}] home not reached (eef z={z}), retry {attempt+1}/{retries}")
            time.sleep(1.0)
        return False

    def run(self):
        try:
            self._run_phases()
        finally:
            # 每集結束強制把雙臂直接送回 home（繞過可能 crash 的 planner），讓 bridge 最後指令=home，
            # 下一集 reset 才不會被卡在上一集失敗位置（MoveIt/bridge 不自動重置場景的問題）
            self.left.force_home()
            self.right.force_home()

    def _run_phases(self):
        a = FRUITS[self.args.fruit]
        b = BASKETS[self.args.basket]
        top_apple = a["z"] + a["sz"] / 2.0
        hx = self.args.handoff_x
        hy = a.get("handoff_y", self.args.handoff_y)   # 長條物把交接點往 -y 移，右手不用伸那麼遠 +y
        hz = self.args.handoff_z
        offset = self.args.pre_grasp_offset
        # 下探深度：未給 --grasp_z_offset 時用各水果自帶值（草莓比 apple 淺，避免頂桌面）
        grasp_zoff = a["grasp_zoff"] if self.args.grasp_z_offset is None else self.args.grasp_z_offset
        grasp_x = a["x"] + a.get("gx_comp", 0.0)   # 夾取目標補償遠伸系統性誤差（box 仍用真實 xy）
        grasp_y = a["y"] + a.get("gy_comp", 0.0)
        # 夾取姿態：前傾 grasp_tilt 補 reach（apple=0 維持 top-down）；step8 之後仍用 self.left_grasp_q 轉正
        grasp_q = _rpy_to_quaternion(math.pi, a.get("grasp_tilt", 0.0), a.get("grasp_yaw", 0.0))
        L, R = self.left, self.right

        # 1 雙臂回 home（乾淨場景；home 姿態手在 z≈0.082，collision box 移到 home 之後再加，
        #   避免規劃時與障礙框互動造成 INVALID_MOTION_PLAN）
        self._step(1, "Both arms to home")
        if not self._home_arm(L):
            self.log.error("left home failed, aborting")
            return
        if not self._home_arm(R):
            self.log.error("right home failed, aborting")
            return

        # 1b 在 home 就把雙夾爪打開（= eval reset 的雙爪開狀態）。解 train/eval home 不一致：
        #    舊收集起步雙爪閉(0.0)、eval 雙爪開(0.04，改不動)→ policy 起步觀測 OOD 崩回 home。
        #    右爪本來就該開著等接，左爪 step3 再開為冗餘無害。
        self._step(1, "Open both grippers at home (train/eval home 對齊)")
        L.control_gripper(GRIPPER_OPEN)
        R.control_gripper(GRIPPER_OPEN)

        # 2 加入 apple + 平台至規劃場景（籃子碰撞物移到交接後才加，避免擋到右手交接接近）
        self._step(2, "Add apple and platform to planning scene")
        L.add_box("apple", a["x"], a["y"], a["z"], [a["sx"], a["sy"], a["sz"]])
        L.add_platform()
        # 另兩顆沒被夾的水果也當靜態障礙加入（全程保留）：讓去程 OMPL 避開它們，
        # 回程反向 replay 才 retrace 到乾淨路徑、不掃到桌面其他水果。
        for _fn, _fd in FRUITS.items():
            if _fn == self.args.fruit:
                continue
            L.add_box(f"obs_{_fn}", _fd["x"], _fd["y"], _fd["z"],
                      [_fd["sx"], _fd["sy"], _fd["sz"]])

        # 3 開左夾爪
        self._step(3, "Open left gripper")
        L.control_gripper(GRIPPER_OPEN)

        # 4 左手預夾取（OMPL 直接到 apple 上方 pre-grasp 高度，沿用 platform_grasp 的 proven 做法；
        #   不走「先到 0.45 再 Cartesian 下降」——那段會強制轉正姿態導致手腕卡住 61% 早停）
        self._step(4, "Left pre-grasp")
        L.rec_on()   # 從 home 開始錄去程路徑（home→pre-grasp 這段供回程反向 replay）
        if not L.move_pose(grasp_x, grasp_y, top_apple + offset, grasp_q, vel=0.3, accel=0.1):
            return

        # 5 左手下降夾取（移除碰撞物件後 Cartesian 下降）
        self._step(5, "Left descend to grasp")
        L.rec_off()  # 暫停錄影：下探+夾取不進回程路徑（回程不重新下探到桌面）
        L.remove_box("apple")
        L.remove_box("platform")
        # 移除另兩顆水果障礙框：obs_ 框只為 step4 去程避障用；夾取/抬起階段留著會擋路——
        # strawberry 遠伸(x=0.31)夾取後手臂壓到 obs_apple(x=0.23)，step7 lift start state 被判 -10 collision。
        for _fn in FRUITS:
            if _fn != self.args.fruit:
                L.remove_box(f"obs_{_fn}")
        if not L.cartesian_z(top_apple + 0.01 + grasp_zoff, grasp_q,
                             vel=0.1, accel=0.05, min_fraction=0.9):
            return
        eef = L.current_eef_pose()
        if eef:
            self.log.warning(f"[GRASP] left TCP=({eef.position.x:.4f},{eef.position.y:.4f},"
                             f"{eef.position.z:.4f})")

        # 6 關左夾爪
        # force：閉合夾爪去夾物體時，normal MoveGroup 會把「手指闔進物體」判為碰撞→規劃 -2/99999 夾爪根本沒閉
        # （strawberry 實測整套空夾）。同 step11 交接 close 的理由，抓取 close 也 force 直接執行。
        self._step(6, "Left close gripper [force]")
        L.control_gripper(0.018, force=True)
        L.control_gripper(GRIPPER_CLOSED, force=True)
        time.sleep(1.5)

        # 7 左手抬起（空場景），完成後加回平台
        # Cartesian 直上 + avoid_collisions=False：夾住 strawberry(前傾姿態)後，closed gripper 在此姿態被
        # MoveIt 判 START_STATE_IN_COLLISION(-10)、OMPL 一遇起始碰撞就 abort。往上抬空間本淨空，跳過檢查安全。
        self._step(7, "Left lift")
        L.rec_on()   # 續錄：lift 回 pre-grasp 高度，與 step4 尾點接續（回程從交接點順回 pre-grasp→home）
        if not L.cartesian_xyz(grasp_x, grasp_y, top_apple + offset, grasp_q,
                               vel=0.1, accel=0.05, min_fraction=0.7, avoid_collisions=False):
            self.log.warning("[left] Cartesian lift 未達，fallback OMPL move_pose")
            if not L.move_pose(grasp_x, grasp_y, top_apple + offset, grasp_q,
                               vel=0.1, accel=0.05, ori_tol=0.5):
                return
        L.add_platform()

        # 8 左手移到交接點再轉水平。把「移動+90°大轉向」一次 OMPL（會機率性把 move_group 弄 -26 crash，
        #    是目前成功率頭號殺手）拆成：先維持朝下(roll=π)平移到交接點（單純移動不觸發大轉向 crash）
        #    → 再 Cartesian 原地旋轉到水平(roll=π/2)（用 computeCartesianPath，不走會 crash 的 OMPL 大轉向）。
        # step8 到中線交接點：約束方向到目前的舉起姿態 left_grasp_q（手臂本來就在此姿態）→ 直線平移、
        # 不繞大風車 → 再 Cartesian 旋轉到水平。position-only(q=None) 會讓 RRTConnect 甩一大圈肘+腕。
        # 原本改 position-only 是為避 -26，但 -26 已由放寬 URDF joint 限位根治，可放心約束方向。
        self._step(8, "Left to handoff (move top-down -> reorient horizontal)")
        if not L.move_pose(hx, hy, hz, q=self.left_grasp_q, ori_tol=0.6, vel=0.2, accel=0.1):
            self.log.warning("[left] 方向約束到交接點失敗，fallback 回 position-only")
            if not L.move_pose(hx, hy, hz, q=None, vel=0.2, accel=0.1):
                return
        if not L.reorient_to(hx, hy, hz, self.left_handoff_q, ori_tol=0.2):
            self.log.warning("left handoff 未完全轉平（見上面 err 角度）— 水果可能沒轉水平")
        L.rec_off()  # 去程錄到交接點(水平)為止；回程即從這裡反向 replay 回 home
        eef = L.current_eef_pose()
        if eef:
            self.log.warning(f"[HANDOFF] left TCP=({eef.position.x:.4f},{eef.position.y:.4f},"
                             f"{eef.position.z:.4f})")

        if self.args.stop_after == "present":
            self.log.info("Stopped after left present at handoff (步驟 1+2a 驗證點)。"
                          "左手應在交接點水平握住 apple。")
            return

        # 9 開右夾爪
        self._step(9, "Open right gripper")
        R.control_gripper(GRIPPER_OPEN)

        # 10 右手接近交接點（3-phase，重試 3 次）。關鍵：對準「左手實際握 apple 的位置」（讀左 TCP），
        #    不用固定交接點——左手 present 落點每集有 ~1cm 變異，右手插到固定 y=0 常差 0.6-1.1cm 夾空。
        #    插入再 overshoot +1.5cm 過 apple 中心，讓右爪把 apple 含進指間（不只到邊緣）。
        self._step(10, "Right approach handoff (3-phase, track apple)")
        ap = L.current_eef_pose()       # 左 TCP ≈ apple 實際位置
        if ap is None:
            return
        ax, ay, az = ap.position.x, ap.position.y, ap.position.z + 0.006  # +6mm 補償右臂插入下垂，避免上指擦到 apple
        # 長條物（banana）：右手用鏡像姿態 + X 錯開夾另一端（不撞本體）；其餘水果走原正交交接。
        if self.args.fruit == "banana":
            rq = self.right_handoff_q_banana  # 兩指轉成跨香蕉短軸(world-Z)上下閉合，不戳長條本體
            rx = ax + a.get("ho_dx", 0.0)     # ho_dx=0：夾質心截面(同apple)，最穩；偏心會讓香蕉翻轉脫出
            insert_y = ay + 0.015             # 從 -y 滑入 overshoot 過中心，把香蕉含進上下兩指間
        else:
            rq = self.right_handoff_q
            rx = ax
            insert_y = ay + 0.015       # 正交插入 overshoot 過中心
        approached = False
        R.rec_on()   # 錄 chunk1: home→ -y 側接近點。回程 chunk1 反向 = approach→home 的 dense 退回，
        #              取代會甩肘的 OMPL move_home（實測 move_home 從 pull-back 甩到 z=0.9）。
        for attempt in range(3):
            if not R.move_pose(rx, ay - 0.15, az, rq, ori_tol=0.6, vel=0.2, accel=0.1):
                R.move_pose(rx, ay - 0.15, az, q=None, vel=0.2, accel=0.1)
            R.rec_off()  # chunk1 錄到 approach（attempt1）為止。approach 順時分支與 pull-back 一致→接縫安全。
            R.cartesian_rotate(rq, min_fraction=0.3)
            if R.cartesian_xyz(rx, insert_y, az, rq, vel=0.15, accel=0.08,
                               min_fraction=0.8, avoid_collisions=False):  # 0.6→0.8：70% 插入差 3cm 夾空
                approached = True
                break
            self.log.warning(f"right approach attempt {attempt + 1}/3 failed, retrying...")
        if not approached:
            self.log.error("right approach failed after 3 attempts, aborting")
            return
        time.sleep(0.8)   # 等右臂物理到位再夾，避免「太早關」夾空

        # 11 關右夾爪抓住 apple（force：兩臂貼近，跳過碰撞檢查）
        self._step(11, "Right close gripper [force]")
        R.control_gripper(GRIPPER_CLOSED, force=True)
        time.sleep(0.5)

        # 12 開左夾爪放手（force：交接時左右爪與 apple 互鎖，normal MoveGroup 規劃會被碰撞檢查
        #    擋下而沒真的開；force 直接 ExecuteTrajectory 並等執行完，確保退開前左爪已張開）
        self._step(12, "Left release")
        L.control_gripper(GRIPPER_OPEN, force=True)
        time.sleep(1.0)

        # 落下偵測：apple 是否仍跟著右 TCP（右手附近）
        rp = R.current_eef_pose()
        if rp:
            self.log.warning(f"[TRANSFER] right TCP=({rp.position.x:.4f},{rp.position.y:.4f},"
                             f"{rp.position.z:.4f}) — 目視確認 apple 是否在右爪")

        # 順序：先小幅拉開右手打破左右臂互鎖 → 左手先復位回 home（你要的：左手不卡在交接點）→ 右手再放籃。
        # 交接瞬間左右臂互鎖，左手原地任何規劃都動不了；先把右手帶開 ~13cm，互鎖破除、apple 也離開交接點，
        # 左手才能正常規劃退開回 home、且不會勾到 apple。

        # 13 右手小幅 -y 拉開（打破互鎖，把 apple 帶離左手）
        self._step(13, "Right pull back to break interlock")
        R.cartesian_xyz(rx, ay - 0.13, az, rq, vel=0.15, accel=0.08,
                        min_fraction=0.4, avoid_collisions=False)
        # pull-back 後才開始錄右臂回程路徑（只錄 pull-back→放籃 這段）。回程反向即 place→籃上→pull-back，
        # 全在 -y 側，不 retrace 回 approach/insert(中線)，避免「橫移回交接處」的怪動作；pull-back→home 交給 move_home。
        R.rec_on()

        # 14 左手先復位：往 +y 退到左臂工作區（互鎖已破，這次能動）再回 home
        # z 由 0.40 降到 0.30：左手退到 +y=0.22 與右臂(−y側)橫向已離很遠，不需高抬避碰，避免「舉很高」
        # 反向 replay 去程「舉起→交接點」的乾淨軌跡：交接點(水平)→轉回朝下→回蘋果上方→home。
        # retrace 已驗證且避開桌面其他水果的路徑，不再用每次不同的 OMPL 重規劃（消除繞路/掃到水果）。
        # 左手回程：dense reverse-replay 去程「舉起→交接點」的乾淨軌跡回 home（平順、不甩肘、確實到 home）。
        # 註：曾試「直退朝 home」想避開經過夾取點上方，但交接姿態近奇異，OMPL 直退反而把手肘甩到 +y=0.5，
        #     比 retrace 更糟；故用回 reverse-replay。經夾取點上方是「平順逃離近奇異交接點」的代價。
        self._step(14, "Left retreat home (reverse-replay lift path)")
        if not L.replay_return():
            self.log.warning("[left] reverse-replay 失敗，fallback OMPL 中繼點回 home")
            if not L.move_pose(0.10, 0.14, 0.22, self.left_grasp_q, ori_tol=0.7, vel=0.2, accel=0.1):
                L.move_pose(0.10, 0.14, 0.22, q=None, vel=0.2, accel=0.1)
            L.reset_joint7()
            for _ in range(3):
                if L.move_home():
                    break

        # 交接已結束，此時才加兩籃子碰撞物（不會擋到前面的右手交接接近）：右手放籃/收回會避開籃子。
        # 此刻右手在 z≈0.33 之上、不被框困住；放籃夾爪在籃緣上方 0.28 不受擋；物理真籃中空，香蕉照樣落入。
        for _bn, _bb in BASKETS.items():
            L.add_box(f"basket_{_bn}", _bb["x"], _bb["y"], 0.2204, [0.1503, 0.0848, 0.0439])

        # 15 右手移到籃子上方（垂直放置姿態）。place_dx 補償長條物夾偏一端→讓質心對準籃中心。
        self._step(15, "Right move to pre-place above basket")
        place_x = b["x"] + a.get("place_dx", 0.0)
        if not R.move_pose(place_x, b["y"], b["top_z"] + offset, self.right_place_q,
                           vel=0.2, accel=0.1, ori_tol=0.5):
            return

        # 16 右手下降至放置高度 + 放開（降低高度讓物體輕放、置中）。籃子碰撞框頂在 top_z，
        #    故降到 top_z+0.01 仍在框上方不受擋；要再低需先移除目標籃框。
        self._step(16, "Right lower + release into basket")
        R.cartesian_z(b["top_z"] + 0.01, self.right_place_q, vel=0.1, accel=0.05, min_fraction=0.5)
        R.rec_off()  # 去程錄到放籃位置為止；回程從這裡反向 replay 回 home
        R.control_gripper(GRIPPER_OPEN, force=True)   # force：normal MoveGroup 偶發 -2 沒真的開→水果沒放掉
        time.sleep(0.5)

        # 17 右手退開並回 home
        # 回程：反向 dense replay 放籃→籃上→pull-back→approach→home（全程 -y 側 dense retrace，直接到 home，
        # 不用會甩肘的 OMPL move_home）。buffer = chunk1(home→approach)+chunk2(pull-back→放籃)，反向尾點=home。
        self._step(17, "Right retreat and home (reverse-replay place → -y → home)")
        if not R.replay_return():
            self.log.warning("[right] reverse-replay 失敗，fallback cartesian 抬高 + move_home")
            R.cartesian_z(b["top_z"] + offset + 0.1, self.right_place_q, vel=0.2, accel=0.1, min_fraction=0.3)
            R.reset_joint7()
            for _ in range(3):
                if R.move_home():
                    break

        self.log.info(f"Bimanual handover complete! apple → basket_{self.args.basket}")


def parse_args():
    p = argparse.ArgumentParser(description="OpenArm bimanual handover (Phase 2, port of bimanual_handoff.cpp)")
    p.add_argument("--fruit", choices=["apple", "strawberry", "banana"], default="apple")
    p.add_argument("--basket", choices=["red", "yellow"], default="red")
    p.add_argument("--stop_after", choices=["present", "all"], default="all",
                   help="present=跑到左手交接點停（早期驗證）；all=完整 pick→handoff→place")
    p.add_argument("--pre_grasp_offset", type=float, default=0.15)
    p.add_argument("--grasp_z_offset",   type=float, default=None,
                   help="下探深度；未給則用各水果自帶 grasp_zoff（apple −0.030、strawberry −0.018）")
    # 交接點 (0.15,0,0.35)：實測往前+上移到 (0.20,0,0.38) 雖改善左手 step8，卻讓右手放籃 step15 crash 暴增
    # (5/10)，淨虧損 → 回退。step8 crash 改用 position-only OMPL 緩解（不靠搬幾何）。
    p.add_argument("--handoff_x", type=float, default=0.15)
    p.add_argument("--handoff_y", type=float, default=0.0)
    p.add_argument("--handoff_z", type=float, default=0.35)
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("bimanual_handover")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        HandoverTask(node, args).run()
    except AssertionError as e:
        node.get_logger().error(str(e))
    except KeyboardInterrupt:
        pass

    # 與 platform_grasp 相同：os._exit 避免 rclcpp 解構子污染 move_group（否則下次規劃 -26）
    time.sleep(0.3)
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
