"""
在 Isaac Sim Script Editor 內執行此 script，
為場景中的三個攝影機建立 ROS2 image publisher Action Graph。

執行方式：
  Isaac Sim → Window → Script Editor → 貼上 → 執行
  執行完後 File → Save 存入 USD 場景

建立的 topics：
  /camera/overhead/image_raw   ← /CameraMain
  /camera/left/image_raw       ← CameraWristLeft
  /camera/right/image_raw      ← CameraWristRight
"""

import omni.graph.core as og
import omni.usd
from pxr import Sdf

stage = omni.usd.get_context().get_stage()

CAMERAS = [
    {
        "graph_path":  "/Graph/CameraOverhead",
        "camera_prim": "/CameraMain",
        "namespace":   "camera/overhead",
        "topic":       "image_raw",
        "frame_id":    "Camera_Main",
    },
    {
        "graph_path":  "/Graph/CameraLeft",
        "camera_prim": "/World/openarm/openarm_left_link7/openarm_left_hand/CameraWristLeft",
        "namespace":   "camera/left",
        "topic":       "image_raw",
        "frame_id":    "CameraWristLeft",
    },
    {
        "graph_path":  "/Graph/CameraRight",
        "camera_prim": "/World/openarm/openarm_right_link7/openarm_right_hand/CameraWristRight",
        "namespace":   "camera/right",
        "topic":       "image_raw",
        "frame_id":    "CameraWristRight",
    },
]

keys = og.Controller.Keys

for cam in CAMERAS:
    graph_path  = cam["graph_path"]
    camera_prim = cam["camera_prim"]
    namespace   = cam["namespace"]
    topic       = cam["topic"]
    frame_id    = cam["frame_id"]

    # 如果已存在就跳過
    if stage.GetPrimAtPath(graph_path).IsValid():
        print(f"[SKIP] {graph_path} already exists")
        continue

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick",      "omni.graph.action.OnPlaybackTick"),
                ("Context",             "isaacsim.ros2.bridge.ROS2Context"),
                ("RunOneFrame",         "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CameraHelper",        "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("CameraHelper.inputs:nodeNamespace", namespace),
                ("CameraHelper.inputs:topicName",     topic),
                ("CameraHelper.inputs:type",          "rgb"),
                ("CameraHelper.inputs:frameId",       frame_id),
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

    # cameraPrim 是 relationship，需要用 USD API 另外設定
    render_product_prim = stage.GetPrimAtPath(f"{graph_path}/CreateRenderProduct")
    rel = render_product_prim.GetRelationship("inputs:cameraPrim")
    if rel:
        rel.SetTargets([Sdf.Path(camera_prim)])

    print(f"[OK] Created {graph_path} → topic: /{namespace}/{topic}")

print("\nDone. 請執行 File → Save 儲存場景。")
