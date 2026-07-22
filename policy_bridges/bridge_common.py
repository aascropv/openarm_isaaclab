#!/usr/bin/env python3
"""
OpenArm bridge 共用常數（policy 無關）。

各 policy bridge（act / smolvla ...）共用的關節名單與相機 topic 對應。
原本寫在 pi05_bridge_node.py，pi0.5 棄用後抽出成獨立模組。
"""

MODEL_JOINT_NAMES_16D = [
    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4",
    "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
    "openarm_right_finger_joint1",   # 右 gripper
    "openarm_left_joint1",  "openarm_left_joint2",  "openarm_left_joint3",  "openarm_left_joint4",
    "openarm_left_joint5",  "openarm_left_joint6",  "openarm_left_joint7",
    "openarm_left_finger_joint1",    # 左 gripper
]

# ROS publish 順序（articulation 18D，含 mimic finger_joint2）
# OmniGraph SubscribeJointState 按 name 配對 → 順序不重要，但要含全部 18 個
ROS_JOINT_NAMES_18D = [
    "openarm_left_joint1",         "openarm_right_joint1",
    "openarm_left_joint2",         "openarm_right_joint2",
    "openarm_left_joint3",         "openarm_right_joint3",
    "openarm_left_joint4",         "openarm_right_joint4",
    "openarm_left_joint5",         "openarm_right_joint5",
    "openarm_left_joint6",         "openarm_right_joint6",
    "openarm_left_joint7",         "openarm_right_joint7",
    "openarm_left_finger_joint1",  "openarm_left_finger_joint2",
    "openarm_right_finger_joint1", "openarm_right_finger_joint2",
]

CAM_TOPIC_TO_KEY = {
    "/camera/overhead/image_raw": "main",
    "/camera/left/image_raw":     "wrist_left",
    "/camera/right/image_raw":    "wrist_right",
}
