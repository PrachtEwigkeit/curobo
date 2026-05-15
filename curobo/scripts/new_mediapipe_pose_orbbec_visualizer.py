#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MediaPipe Right Arm / Right Hand Orbbec Teleop Publisher

功能：
1. 调用 Orbbec Python SDK 获取 RGB / aligned depth；
2. 用 MediaPipe PoseLandmarker 获取右臂和双肩关键点；
3. 用双肩中点定义 shoulder-centered local origin；
4. 用 RGB-D 反投影得到 RIGHT_WRIST 相对 shoulder center 的位置和增量；
5. 用 MediaPipe HandLandmarker 构造视觉 palm frame 和 delta_R_camera_palm；
6. 可视化右臂、右手、palm frame；
7. 通过 UDP 发布 right wrist shoulder-centered target 和 palm orientation。

按键：
q / ESC：退出
p：暂停/继续
r：重置 right wrist shoulder-centered position origin
o：重置 palm orientation origin

不再支持：
- screenshot；
- CSV；
- full-body skeleton；
- face/legs/left arm tracking display；
- MediaPipe pose_world_landmarks display。

注意：
- shoulder-centered camera coordinate 只改变原点，坐标轴仍与 Orbbec camera frame 平行；
- 本脚本不做 camera->world 旋转，不做 palm->robot EE 对齐，不调用 cuRobo，不做 IK；
- palm orientation 是视觉估计的 palm frame，不是严格解剖意义 wrist joint rotation。
"""

import argparse
import json
import math
import os
import socket
import time
import urllib.request
from contextlib import ExitStack
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pyorbbecsdk import (
    Config,
    OBAlignMode,
    OBError,
    OBFormat,
    OBSensorType,
    Pipeline,
    VideoFrame,
    VideoStreamProfile,
)


# ============================================================
# MediaPipe indices
# ============================================================

LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}

POSE_USED_NAMES = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "RIGHT_INDEX",
    "RIGHT_THUMB",
    "RIGHT_PINKY",
]

POSE_USED_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("RIGHT_WRIST", "RIGHT_INDEX"),
    ("RIGHT_WRIST", "RIGHT_THUMB"),
    ("RIGHT_WRIST", "RIGHT_PINKY"),
    ("RIGHT_INDEX", "RIGHT_PINKY"),
]

POSE_RGBD_NAMES = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "RIGHT_INDEX",
    "RIGHT_THUMB",
    "RIGHT_PINKY",
]

HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

HAND_IDX = {name: i for i, name in enumerate(HAND_LANDMARK_NAMES)}

HAND_CONNECTIONS = [
    (HAND_IDX["WRIST"], HAND_IDX["THUMB_CMC"]),
    (HAND_IDX["THUMB_CMC"], HAND_IDX["THUMB_MCP"]),
    (HAND_IDX["THUMB_MCP"], HAND_IDX["THUMB_IP"]),
    (HAND_IDX["THUMB_IP"], HAND_IDX["THUMB_TIP"]),
    (HAND_IDX["WRIST"], HAND_IDX["INDEX_FINGER_MCP"]),
    (HAND_IDX["INDEX_FINGER_MCP"], HAND_IDX["INDEX_FINGER_PIP"]),
    (HAND_IDX["INDEX_FINGER_PIP"], HAND_IDX["INDEX_FINGER_DIP"]),
    (HAND_IDX["INDEX_FINGER_DIP"], HAND_IDX["INDEX_FINGER_TIP"]),
    (HAND_IDX["WRIST"], HAND_IDX["MIDDLE_FINGER_MCP"]),
    (HAND_IDX["MIDDLE_FINGER_MCP"], HAND_IDX["MIDDLE_FINGER_PIP"]),
    (HAND_IDX["MIDDLE_FINGER_PIP"], HAND_IDX["MIDDLE_FINGER_DIP"]),
    (HAND_IDX["MIDDLE_FINGER_DIP"], HAND_IDX["MIDDLE_FINGER_TIP"]),
    (HAND_IDX["WRIST"], HAND_IDX["RING_FINGER_MCP"]),
    (HAND_IDX["RING_FINGER_MCP"], HAND_IDX["RING_FINGER_PIP"]),
    (HAND_IDX["RING_FINGER_PIP"], HAND_IDX["RING_FINGER_DIP"]),
    (HAND_IDX["RING_FINGER_DIP"], HAND_IDX["RING_FINGER_TIP"]),
    (HAND_IDX["WRIST"], HAND_IDX["PINKY_MCP"]),
    (HAND_IDX["PINKY_MCP"], HAND_IDX["PINKY_PIP"]),
    (HAND_IDX["PINKY_PIP"], HAND_IDX["PINKY_DIP"]),
    (HAND_IDX["PINKY_DIP"], HAND_IDX["PINKY_TIP"]),
    (HAND_IDX["INDEX_FINGER_MCP"], HAND_IDX["MIDDLE_FINGER_MCP"]),
    (HAND_IDX["MIDDLE_FINGER_MCP"], HAND_IDX["RING_FINGER_MCP"]),
    (HAND_IDX["RING_FINGER_MCP"], HAND_IDX["PINKY_MCP"]),
]


# ============================================================
# Model files
# ============================================================

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

MODEL_FILES = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_FILE = "hand_landmarker.task"


def parse_args():
    parser = argparse.ArgumentParser(description="Right-arm/right-hand Orbbec teleop publisher")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-depth", action="store_true", default=False)
    parser.add_argument("--orbbec-align", type=str, default="HW", choices=["HW", "SW", "NONE"])
    parser.add_argument("--disable-frame-sync", action="store_true", default=False)

    parser.add_argument("--model", type=str, default="full", choices=["lite", "full", "heavy"])
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--num-poses", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.45)

    parser.add_argument("--enable-hand", action="store_true", default=True)
    parser.add_argument("--disable-hand", action="store_true", default=False)
    parser.add_argument("--hand-model-path", type=str, default="")
    parser.add_argument("--hand-models-dir", type=str, default="models")
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--handedness",
        type=str,
        default="Right",
        choices=["Right", "Left", "Any"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--palm-depth-radius", type=int, default=5)
    parser.add_argument("--draw-hand", action="store_true", default=True)
    parser.add_argument("--no-draw-hand", action="store_true", default=False)
    parser.add_argument("--draw-palm-frame", action="store_true", default=True)
    parser.add_argument("--palm-frame-axis-length", type=float, default=0.08)
    parser.add_argument("--hand-every-n-frames", type=int, default=1)
    parser.add_argument("--palm-auto-init", action="store_true", default=True)
    parser.add_argument("--palm-reset-key", type=str, default="o")
    parser.add_argument("--palm-filter-alpha", type=float, default=0.25)
    parser.add_argument("--palm-max-angle-step-deg", type=float, default=60.0)

    parser.add_argument("--udp-publish", action="store_true", default=False)
    parser.add_argument("--udp-host", type=str, default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5557)
    parser.add_argument("--udp-frame", type=str, default="orbbec_color_optical_frame")
    parser.add_argument("--teleop-scale", type=float, default=1.0)
    parser.add_argument(
        "--publish-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish teleop_scale * shoulder-centered wrist delta as target_position_m",
    )
    parser.add_argument("--reset-origin-key", type=str, default="r")

    parser.add_argument("--no-mirror", action="store_true", default=False)
    parser.add_argument("--no-draw-names", action="store_true", default=False)
    parser.add_argument("--panel-width", type=int, default=460)
    parser.add_argument("--display-scale", type=float, default=0.8)
    parser.add_argument(
        "--panel-skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show normalized right arm + right hand skeleton in the info panel",
    )
    parser.add_argument("--panel-skeleton-height", type=int, default=260)
    return parser.parse_args()


def ensure_model(model_name: str, model_path: str, models_dir: str) -> str:
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model file does not exist: {model_path}")
        return model_path

    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, MODEL_FILES[model_name])
    if os.path.exists(local_path):
        return local_path

    url = MODEL_URLS[model_name]
    print(f"[INFO] Downloading pose model: {url}")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def ensure_hand_model(model_path: str, models_dir: str) -> str:
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hand model file does not exist: {model_path}")
        return model_path

    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, HAND_MODEL_FILE)
    if os.path.exists(local_path):
        return local_path

    print(f"[INFO] Downloading hand model: {HAND_MODEL_URL}")
    urllib.request.urlretrieve(HAND_MODEL_URL, local_path)
    return local_path


# ============================================================
# Orbbec camera
# ============================================================

def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.cvtColor(frame.reshape((height, width, 2)), cv2.COLOR_YUV2BGR_YUY2)


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.cvtColor(frame.reshape((height, width, 2)), cv2.COLOR_YUV2BGR_UYVY)


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.cvtColor(frame.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_I420)


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.cvtColor(frame.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV12)


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.cvtColor(frame.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV21)


def orbbec_frame_to_bgr_image(frame: VideoFrame) -> Optional[np.ndarray]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        return cv2.cvtColor(np.resize(data, (height, width, 3)), cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.YUYV:
        return yuyv_to_bgr(data, width, height)
    if color_format == OBFormat.UYVY:
        return uyvy_to_bgr(data, width, height)
    if color_format == OBFormat.I420:
        return i420_to_bgr(data, width, height)
    if color_format == OBFormat.NV12:
        return nv12_to_bgr(data, width, height)
    if color_format == OBFormat.NV21:
        return nv21_to_bgr(data, width, height)

    print(f"[WARN] Unsupported Orbbec color format: {color_format}")
    return None


def orbbec_depth_frame_to_mm(depth_frame) -> np.ndarray:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    return depth_data.astype(np.float32) * scale


class OrbbecCamera:
    def __init__(self, width: int, height: int, fps: int, enable_depth: bool, align_mode: str, enable_sync: bool):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.align_mode = align_mode
        self.enable_sync = enable_sync
        self.pipeline = Pipeline()
        self.config = Config()
        self.color_intrinsics = None

    def _select_color_profile(self) -> VideoStreamProfile:
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            return profile_list.get_video_stream_profile(self.width, self.height, OBFormat.RGB, self.fps)
        except OBError as exc:
            print(f"[WARN] Requested Orbbec color profile unavailable: {exc}")
            color_profile = profile_list.get_default_video_stream_profile()
            print("[INFO] Using default color profile:", color_profile)
            return color_profile

    def _select_depth_profile(self) -> Optional[VideoStreamProfile]:
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            print("[INFO] Using default depth profile:", depth_profile)
            return depth_profile
        except Exception as exc:
            print(f"[WARN] Depth stream unavailable, continuing with color only: {exc}")
            self.enable_depth = False
            return None

    def start(self):
        color_profile = self._select_color_profile()
        print("[INFO] Using Orbbec color profile:", color_profile)
        self.config.enable_stream(color_profile)
        self.color_intrinsics = self._extract_color_intrinsics(color_profile)

        depth_profile = self._select_depth_profile() if self.enable_depth else None
        if depth_profile is not None:
            self.config.enable_stream(depth_profile)
            self._configure_align_mode()
            if self.enable_sync:
                try:
                    self.pipeline.enable_frame_sync()
                except Exception as exc:
                    print(f"[WARN] Failed to enable Orbbec frame sync: {exc}")

        self.pipeline.start(self.config)
        if self.color_intrinsics is None:
            self.color_intrinsics = self._extract_color_intrinsics(color_profile)
        if self.color_intrinsics is None:
            print("[WARN] Orbbec color intrinsics unavailable; RGB-D deprojection disabled.")
        else:
            print("[INFO] Orbbec color intrinsics:", self.color_intrinsics)
        return self

    def _read_value(self, obj, names):
        for name in names:
            if obj is None or not hasattr(obj, name):
                continue
            value = getattr(obj, name)
            try:
                return value() if callable(value) else value
            except Exception:
                continue
        return None

    def _intrinsics_to_dict(self, intrinsics_obj, color_profile) -> Optional[Dict[str, float]]:
        if intrinsics_obj is None:
            return None

        if isinstance(intrinsics_obj, dict):
            fx = intrinsics_obj.get("fx", intrinsics_obj.get("focal_length_x"))
            fy = intrinsics_obj.get("fy", intrinsics_obj.get("focal_length_y"))
            cx = intrinsics_obj.get("cx", intrinsics_obj.get("principal_point_x"))
            cy = intrinsics_obj.get("cy", intrinsics_obj.get("principal_point_y"))
            width = intrinsics_obj.get("width")
            height = intrinsics_obj.get("height")
        else:
            fx = self._read_value(intrinsics_obj, ["fx", "focal_length_x"])
            fy = self._read_value(intrinsics_obj, ["fy", "focal_length_y"])
            cx = self._read_value(intrinsics_obj, ["cx", "principal_point_x"])
            cy = self._read_value(intrinsics_obj, ["cy", "principal_point_y"])
            width = self._read_value(intrinsics_obj, ["width", "w"])
            height = self._read_value(intrinsics_obj, ["height", "h"])

        width = width if width is not None else self._read_value(color_profile, ["get_width", "width"])
        height = height if height is not None else self._read_value(color_profile, ["get_height", "height"])
        width = width if width is not None else self.width
        height = height if height is not None else self.height

        try:
            intrinsics = {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "width": float(width),
                "height": float(height),
            }
        except (TypeError, ValueError):
            return None

        if intrinsics["fx"] <= 0.0 or intrinsics["fy"] <= 0.0:
            return None
        return intrinsics

    def _extract_color_intrinsics(self, color_profile) -> Optional[Dict[str, float]]:
        for method_name in ["get_intrinsic", "get_intrinsics", "get_camera_intrinsic", "get_video_stream_intrinsic"]:
            if not hasattr(color_profile, method_name):
                continue
            try:
                intrinsics = self._intrinsics_to_dict(getattr(color_profile, method_name)(), color_profile)
                if intrinsics is not None:
                    return intrinsics
            except Exception as exc:
                print(f"[WARN] Failed reading intrinsics via {method_name}: {exc}")

        try:
            camera_param = self.pipeline.get_camera_param()
            for attr_name in ["rgb_intrinsic", "color_intrinsic", "rgbIntrinsic", "colorIntrinsic", "left_intrinsic"]:
                intrinsics_obj = self._read_value(camera_param, [attr_name])
                intrinsics = self._intrinsics_to_dict(intrinsics_obj, color_profile)
                if intrinsics is not None:
                    return intrinsics
        except Exception as exc:
            print(f"[WARN] Failed reading intrinsics from pipeline camera param: {exc}")
        return None

    def get_color_intrinsics(self) -> Optional[Dict[str, float]]:
        return self.color_intrinsics

    def _configure_align_mode(self):
        if self.align_mode == "NONE":
            self.config.set_align_mode(OBAlignMode.DISABLE)
            return
        if self.align_mode == "SW":
            self.config.set_align_mode(OBAlignMode.SW_MODE)
            return
        try:
            device = self.pipeline.get_device()
            device_pid = device.get_device_info().get_pid()
            self.config.set_align_mode(OBAlignMode.SW_MODE if device_pid == 0x066B else OBAlignMode.HW_MODE)
        except Exception as exc:
            print(f"[WARN] Failed to configure HW align, falling back to SW align: {exc}")
            self.config.set_align_mode(OBAlignMode.SW_MODE)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames(500)
        if frames is None:
            return False, None, None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return False, None, None

        color_image = orbbec_frame_to_bgr_image(color_frame)
        if color_image is None:
            return False, None, None

        depth_mm = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                try:
                    depth_mm = orbbec_depth_frame_to_mm(depth_frame)
                except Exception as exc:
                    print(f"[WARN] Failed to convert depth frame: {exc}")
                    depth_mm = None

        return True, color_image, depth_mm

    def stop(self):
        self.pipeline.stop()


# ============================================================
# Geometry and drawing helpers
# ============================================================

def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg: bool = True,
):
    x, y = org
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def get_visibility(lm) -> float:
    return float(getattr(lm, "visibility", 1.0))


def lm_to_pixel(lm, width: int, height: int, mirror: bool = False) -> Tuple[int, int]:
    x = float(lm.x)
    y = float(lm.y)
    if mirror:
        x = 1.0 - x
    px = int(np.clip(x * width, 0, width - 1))
    py = int(np.clip(y * height, 0, height - 1))
    return px, py


def deproject_pixel_to_camera(u, v, depth_m, intrinsics) -> Optional[np.ndarray]:
    if intrinsics is None:
        return None
    try:
        z = float(depth_m)
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError):
        return None
    if z <= 0.0 or fx <= 0.0 or fy <= 0.0 or not np.isfinite(z):
        return None
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def landmark_to_camera_point(
    lm,
    frame_w: int,
    frame_h: int,
    depth_mm,
    intrinsics,
    search_radius: int = 3,
) -> Optional[np.ndarray]:
    if depth_mm is None or intrinsics is None or depth_mm.ndim != 2:
        return None

    u, v = lm_to_pixel(lm, frame_w, frame_h, mirror=False)
    depth_h, depth_w = depth_mm.shape[:2]
    u = int(np.clip(u, 0, depth_w - 1))
    v = int(np.clip(v, 0, depth_h - 1))
    depth_value = float(depth_mm[v, u])

    if not np.isfinite(depth_value) or depth_value <= 0.0:
        x0 = max(0, u - search_radius)
        x1 = min(depth_w, u + search_radius + 1)
        y0 = max(0, v - search_radius)
        y1 = min(depth_h, v + search_radius + 1)
        patch = depth_mm[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        depth_value = float(np.median(valid))

    return deproject_pixel_to_camera(u, v, depth_value / 1000.0, intrinsics)


def compute_pose_rgbd_points(
    lms,
    depth_mm,
    intrinsics,
    frame_w: int,
    frame_h: int,
    visibility_threshold: float,
) -> Dict[str, Optional[np.ndarray]]:
    points = {name: None for name in POSE_RGBD_NAMES}
    if lms is None or depth_mm is None or intrinsics is None:
        return points

    for name in POSE_RGBD_NAMES:
        idx = IDX[name]
        if idx >= len(lms) or get_visibility(lms[idx]) < visibility_threshold:
            continue
        try:
            points[name] = landmark_to_camera_point(lms[idx], frame_w, frame_h, depth_mm, intrinsics)
        except Exception as exc:
            print(f"[WARN] RGB-D deprojection failed for {name}: {exc}")
            points[name] = None
    return points


def compute_shoulder_center_camera(rgbd_camera_points: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
    left = rgbd_camera_points.get("LEFT_SHOULDER")
    right = rgbd_camera_points.get("RIGHT_SHOULDER")
    if left is None or right is None:
        return None
    return 0.5 * (left + right)


def compute_shoulder_centered_points(
    rgbd_camera_points: Dict[str, Optional[np.ndarray]],
    shoulder_center_camera: Optional[np.ndarray],
) -> Dict[str, Optional[np.ndarray]]:
    centered = {name: None for name in POSE_RGBD_NAMES}
    if shoulder_center_camera is None:
        return centered
    for name in POSE_RGBD_NAMES:
        point = rgbd_camera_points.get(name)
        centered[name] = None if point is None else point - shoulder_center_camera
    return centered


def project_camera_point_to_pixel(P_camera, intrinsics) -> Optional[Tuple[int, int]]:
    if P_camera is None or intrinsics is None:
        return None
    try:
        X, Y, Z = [float(v) for v in P_camera]
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (TypeError, KeyError, ValueError):
        return None
    if Z <= 1e-6 or fx <= 0.0 or fy <= 0.0:
        return None
    return int(round(fx * X / Z + cx)), int(round(fy * Y / Z + cy))


def _handedness_label_and_score(handedness_item) -> Tuple[str, float]:
    if not handedness_item:
        return "Unknown", 0.0
    category = handedness_item[0]
    label = (
        getattr(category, "category_name", None)
        or getattr(category, "display_name", None)
        or getattr(category, "label", None)
        or "Unknown"
    )
    try:
        score = float(getattr(category, "score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return str(label), score


# This script is a right-hand teleoperation publisher. For safety and consistency,
# it never falls back to the left hand.
def select_right_hand_result(hand_result):
    if hand_result is None or not getattr(hand_result, "hand_landmarks", None):
        return None

    hand_lms_list = hand_result.hand_landmarks
    handedness_list = getattr(hand_result, "handedness", None)
    for i, hand_lms in enumerate(hand_lms_list):
        if handedness_list is not None and i < len(handedness_list) and handedness_list[i]:
            label, score = _handedness_label_and_score(handedness_list[i])
        else:
            label, score = "Unknown", 0.0

        if str(label).strip().lower() == "right":
            return {
                "hand_lms": hand_lms,
                "handedness_label": "Right",
                "handedness_score": score,
            }
    return None


def compute_hand_camera_points(hand_lms, depth_mm, intrinsics, frame_w: int, frame_h: int, search_radius: int):
    points = {name: None for name in HAND_LANDMARK_NAMES}
    if hand_lms is None or depth_mm is None or intrinsics is None:
        return points
    for name, idx in HAND_IDX.items():
        if idx >= len(hand_lms):
            continue
        try:
            points[name] = landmark_to_camera_point(
                hand_lms[idx],
                frame_w=frame_w,
                frame_h=frame_h,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                search_radius=search_radius,
            )
        except Exception as exc:
            print(f"[WARN] Hand RGB-D deprojection failed for {name}: {exc}")
    return points


def normalize_vec(v, eps=1e-8) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return None
    return v / n


def project_to_so3(R: np.ndarray) -> Optional[np.ndarray]:
    try:
        U, _S, Vt = np.linalg.svd(np.asarray(R, dtype=np.float32))
        R_so3 = U @ Vt
        if np.linalg.det(R_so3) < 0.0:
            U[:, -1] *= -1.0
            R_so3 = U @ Vt
    except Exception:
        return None
    if not np.all(np.isfinite(R_so3)):
        return None
    return R_so3.astype(np.float32)


def compute_palm_frame_from_points(hand_points_camera: Dict[str, Optional[np.ndarray]]) -> dict:
    required = ["WRIST", "INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP", "PINKY_MCP"]
    for name in required:
        if hand_points_camera.get(name) is None:
            return {"valid": False, "reason": f"missing {name}"}

    p_wrist = hand_points_camera["WRIST"]
    p_index = hand_points_camera["INDEX_FINGER_MCP"]
    p_middle = hand_points_camera["MIDDLE_FINGER_MCP"]
    p_pinky = hand_points_camera["PINKY_MCP"]

    x_palm = normalize_vec(p_index - p_pinky)
    y_raw = normalize_vec(p_middle - p_wrist)
    if x_palm is None or y_raw is None:
        return {"valid": False, "reason": "degenerate palm vectors"}

    z_palm = normalize_vec(np.cross(x_palm, y_raw))
    if z_palm is None:
        return {"valid": False, "reason": "degenerate palm normal"}

    y_palm = normalize_vec(np.cross(z_palm, x_palm))
    if y_palm is None:
        return {"valid": False, "reason": "degenerate palm y axis"}

    R_camera_palm = project_to_so3(np.column_stack([x_palm, y_palm, z_palm]))
    if R_camera_palm is None:
        return {"valid": False, "reason": "SO(3) projection failed"}

    origin_points = [p_wrist, p_index, p_middle, p_pinky]
    p_ring = hand_points_camera.get("RING_FINGER_MCP")
    if p_ring is not None:
        origin_points.append(p_ring)
    palm_origin = np.mean(np.stack(origin_points, axis=0), axis=0).astype(np.float32)

    return {
        "valid": True,
        "R_camera_palm": R_camera_palm,
        "palm_origin_camera": palm_origin,
        "x_axis_camera": R_camera_palm[:, 0],
        "y_axis_camera": R_camera_palm[:, 1],
        "z_axis_camera": R_camera_palm[:, 2],
        "reason": "ok",
    }


def matrix_to_quat_wxyz(R: np.ndarray) -> Optional[np.ndarray]:
    R = project_to_so3(R)
    if R is None:
        return None
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8 or not np.isfinite(n):
        return None
    return q / n


def quat_to_matrix_wxyz(q: np.ndarray) -> Optional[np.ndarray]:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8 or not np.isfinite(n):
        return None
    w, x, y, z = q / n
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return project_to_so3(R)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-8)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-8)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    q = s0 * q0 + s1 * q1
    return q / max(float(np.linalg.norm(q)), 1e-8)


def rotation_angle_deg(R_delta: np.ndarray) -> Optional[float]:
    R_delta = project_to_so3(R_delta)
    if R_delta is None:
        return None
    cos_angle = (float(np.trace(R_delta)) - 1.0) * 0.5
    return float(math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0))))


def empty_palm_state(reason: str = "hand unavailable") -> dict:
    return {
        "valid": False,
        "reason": reason,
        "handedness": "none",
        "handedness_score": None,
        "R_camera_palm": None,
        "R_camera_palm_0": None,
        "delta_R_camera_palm": None,
        "q_camera_palm_wxyz": None,
        "q_delta_palm_wxyz": None,
        "palm_origin_camera_m": None,
        "palm_normal_camera": None,
        "angle_from_origin_deg": None,
    }


def update_palm_orientation_state(
    palm_frame_camera: dict,
    selected_hand,
    runtime_state: dict,
    auto_init: bool,
    filter_alpha: float,
    max_angle_step_deg: float,
) -> dict:
    label = "none"
    score = None
    if selected_hand is not None:
        label = selected_hand.get("handedness_label", "Unknown")
        score = selected_hand.get("handedness_score", None)

    if palm_frame_camera is None or not palm_frame_camera.get("valid", False):
        state = empty_palm_state(palm_frame_camera.get("reason", "palm invalid") if palm_frame_camera else "palm invalid")
        state["handedness"] = label
        state["handedness_score"] = score
        return state

    R_now = palm_frame_camera["R_camera_palm"]
    R_filtered_prev = runtime_state.get("R_filtered")
    if R_filtered_prev is not None:
        step_angle = rotation_angle_deg(R_now @ R_filtered_prev.T)
        if step_angle is not None and step_angle > max_angle_step_deg:
            state = empty_palm_state(f"palm rotation jump {step_angle:.1f} deg")
            state["handedness"] = label
            state["handedness_score"] = score
            return state
        q_prev = matrix_to_quat_wxyz(R_filtered_prev)
        q_now = matrix_to_quat_wxyz(R_now)
        q_filtered = quat_slerp(q_prev, q_now, filter_alpha) if q_prev is not None and q_now is not None else None
        R_use = quat_to_matrix_wxyz(q_filtered) if q_filtered is not None else R_now
    else:
        R_use = R_now

    runtime_state["R_filtered"] = R_use.copy()
    runtime_state["R_prev"] = R_now.copy()
    if runtime_state.get("R0") is None and auto_init:
        runtime_state["R0"] = R_use.copy()
        print("[INFO] Auto set palm orientation origin.")

    R0 = runtime_state.get("R0")
    if R0 is None:
        state = empty_palm_state("palm origin not initialized")
        state["handedness"] = label
        state["handedness_score"] = score
        return state

    delta_R = project_to_so3(R_use @ R0.T)
    q_camera = matrix_to_quat_wxyz(R_use)
    q_delta = matrix_to_quat_wxyz(delta_R) if delta_R is not None else None
    angle = rotation_angle_deg(delta_R) if delta_R is not None else None

    return {
        "valid": delta_R is not None and q_camera is not None and q_delta is not None,
        "reason": "ok",
        "handedness": label,
        "handedness_score": score,
        "R_camera_palm": R_use,
        "R_camera_palm_0": R0,
        "delta_R_camera_palm": delta_R,
        "q_camera_palm_wxyz": q_camera,
        "q_delta_palm_wxyz": q_delta,
        "palm_origin_camera_m": palm_frame_camera["palm_origin_camera"],
        "palm_normal_camera": R_use[:, 2],
        "angle_from_origin_deg": angle,
    }


def arr_to_list(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).tolist()


def palm_state_udp_fields(palm_state: Optional[dict], frame_name: str) -> dict:
    valid = bool(palm_state and palm_state.get("valid", False))
    if not valid:
        return {
            "palm_orientation_valid": False,
            "palm_frame": frame_name,
            "palm_handedness": palm_state.get("handedness") if palm_state else "none",
            "palm_handedness_score": palm_state.get("handedness_score") if palm_state else None,
            "palm_reason": palm_state.get("reason") if palm_state else "palm invalid",
            "R_camera_palm": None,
            "delta_R_camera_palm": None,
            "q_camera_palm_wxyz": None,
            "q_delta_palm_wxyz": None,
            "palm_origin_camera_m": None,
            "palm_normal_camera": None,
            "palm_angle_from_origin_deg": None,
        }

    return {
        "palm_orientation_valid": True,
        "palm_frame": frame_name,
        "palm_handedness": palm_state.get("handedness"),
        "palm_handedness_score": palm_state.get("handedness_score"),
        "palm_reason": palm_state.get("reason"),
        "R_camera_palm": arr_to_list(palm_state.get("R_camera_palm")),
        "delta_R_camera_palm": arr_to_list(palm_state.get("delta_R_camera_palm")),
        "q_camera_palm_wxyz": arr_to_list(palm_state.get("q_camera_palm_wxyz")),
        "q_delta_palm_wxyz": arr_to_list(palm_state.get("q_delta_palm_wxyz")),
        "palm_origin_camera_m": arr_to_list(palm_state.get("palm_origin_camera_m")),
        "palm_normal_camera": arr_to_list(palm_state.get("palm_normal_camera")),
        "palm_angle_from_origin_deg": palm_state.get("angle_from_origin_deg"),
    }


def draw_pose_used_2d(img, lms, mirror: bool, visibility_threshold: float, draw_names: bool):
    if lms is None:
        return
    h, w = img.shape[:2]
    for a_name, b_name in POSE_USED_CONNECTIONS:
        a = IDX[a_name]
        b = IDX[b_name]
        if a >= len(lms) or b >= len(lms):
            continue
        if get_visibility(lms[a]) < visibility_threshold or get_visibility(lms[b]) < visibility_threshold:
            continue
        pa = lm_to_pixel(lms[a], w, h, mirror=mirror)
        pb = lm_to_pixel(lms[b], w, h, mirror=mirror)
        cv2.line(img, pa, pb, (0, 220, 255), 2, cv2.LINE_AA)

    for name in POSE_USED_NAMES:
        idx = IDX[name]
        if idx >= len(lms) or get_visibility(lms[idx]) < visibility_threshold:
            continue
        p = lm_to_pixel(lms[idx], w, h, mirror=mirror)
        cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 6, (0, 180, 255), 1, cv2.LINE_AA)

    if draw_names:
        labels = {
            "LEFT_SHOULDER": "L_SHO",
            "RIGHT_SHOULDER": "R_SHO",
            "RIGHT_ELBOW": "R_ELB",
            "RIGHT_WRIST": "R_WRI",
            "RIGHT_INDEX": "R_IDX",
            "RIGHT_THUMB": "R_THU",
            "RIGHT_PINKY": "R_PIN",
        }
        for name, label in labels.items():
            idx = IDX[name]
            if idx >= len(lms) or get_visibility(lms[idx]) < visibility_threshold:
                continue
            x, y = lm_to_pixel(lms[idx], w, h, mirror=mirror)
            put_text(img, label, (x + 6, y - 6), scale=0.40, color=(255, 255, 0), bg=True)


def draw_rgbd_valid_points(img, lms, rgbd_camera_points, mirror: bool, visibility_threshold: float):
    if lms is None or not rgbd_camera_points:
        return
    h, w = img.shape[:2]
    for name in POSE_RGBD_NAMES:
        point = rgbd_camera_points.get(name)
        if point is None:
            continue
        idx = IDX[name]
        if idx >= len(lms) or get_visibility(lms[idx]) < visibility_threshold:
            continue
        x, y = lm_to_pixel(lms[idx], w, h, mirror=mirror)
        cv2.circle(img, (x, y), 8, (0, 255, 0), 2, cv2.LINE_AA)


def draw_shoulder_center_marker(img, lms, mirror: bool, visibility_threshold: float):
    if lms is None:
        return
    li = IDX["LEFT_SHOULDER"]
    ri = IDX["RIGHT_SHOULDER"]
    if get_visibility(lms[li]) < visibility_threshold or get_visibility(lms[ri]) < visibility_threshold:
        return
    h, w = img.shape[:2]
    lp = np.asarray(lm_to_pixel(lms[li], w, h, mirror=mirror), dtype=np.float32)
    rp = np.asarray(lm_to_pixel(lms[ri], w, h, mirror=mirror), dtype=np.float32)
    p = tuple(np.round(0.5 * (lp + rp)).astype(np.int32).tolist())
    cv2.circle(img, p, 7, (255, 255, 0), 2, cv2.LINE_AA)
    put_text(img, "SHO_C", (p[0] + 8, p[1] + 4), scale=0.38, color=(255, 255, 0), bg=True)


def draw_hand_landmarks_2d(img, hand_lms, mirror: bool, draw_names: bool = False):
    if hand_lms is None:
        return
    h, w = img.shape[:2]
    for a, b in HAND_CONNECTIONS:
        if a >= len(hand_lms) or b >= len(hand_lms):
            continue
        pa = lm_to_pixel(hand_lms[a], w, h, mirror=mirror)
        pb = lm_to_pixel(hand_lms[b], w, h, mirror=mirror)
        cv2.line(img, pa, pb, (255, 80, 220), 2, cv2.LINE_AA)
    for lm in hand_lms:
        p = lm_to_pixel(lm, w, h, mirror=mirror)
        cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 5, (255, 80, 220), 1, cv2.LINE_AA)
    if draw_names:
        for name in ["WRIST", "INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP", "PINKY_MCP"]:
            idx = HAND_IDX[name]
            if idx >= len(hand_lms):
                continue
            x, y = lm_to_pixel(hand_lms[idx], w, h, mirror=mirror)
            short = name.replace("_FINGER", "").replace("_MCP", "")
            put_text(img, short, (x + 5, y - 5), scale=0.34, color=(255, 160, 255), bg=True)


def draw_palm_frame_axes(img, palm_frame_camera, intrinsics, mirror: bool, axis_length_m: float):
    if palm_frame_camera is None or not palm_frame_camera.get("valid", False) or intrinsics is None:
        return
    origin = palm_frame_camera.get("palm_origin_camera")
    R = palm_frame_camera.get("R_camera_palm")
    if origin is None or R is None:
        return
    h, w = img.shape[:2]

    def project_for_display(point):
        px = project_camera_point_to_pixel(point, intrinsics)
        if px is None:
            return None
        x, y = px
        if mirror:
            x = w - 1 - x
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        return int(x), int(y)

    p0 = project_for_display(origin)
    if p0 is None:
        return
    axes = [(R[:, 0], (0, 0, 255), "x"), (R[:, 1], (0, 255, 0), "y"), (R[:, 2], (255, 0, 0), "z")]
    for axis, color, label in axes:
        p1 = project_for_display(origin + axis_length_m * axis)
        if p1 is None:
            continue
        cv2.line(img, p0, p1, color, 3, cv2.LINE_AA)
        put_text(img, label, (p1[0] + 4, p1[1] + 4), scale=0.42, color=color, bg=True)


def _normalized_xy(lm, mirror: bool) -> Tuple[float, float]:
    x = float(lm.x)
    y = float(lm.y)
    if mirror:
        x = 1.0 - x
    return x, y


def collect_normalized_points_for_panel(pose_lms, hand_lms, visibility_threshold: float, mirror: bool):
    points = []
    if pose_lms is not None:
        for name in POSE_USED_NAMES:
            idx = IDX[name]
            if idx < len(pose_lms) and get_visibility(pose_lms[idx]) >= visibility_threshold:
                points.append(_normalized_xy(pose_lms[idx], mirror))
    if hand_lms is not None:
        for lm in hand_lms:
            points.append(_normalized_xy(lm, mirror))
    return points


def make_panel_bbox(points, padding: float = 0.08):
    if len(points) < 2:
        return 0.0, 1.0, 0.0, 1.0
    arr = np.asarray(points, dtype=np.float32)
    xmin = float(np.clip(np.min(arr[:, 0]) - padding, 0.0, 1.0))
    xmax = float(np.clip(np.max(arr[:, 0]) + padding, 0.0, 1.0))
    ymin = float(np.clip(np.min(arr[:, 1]) - padding, 0.0, 1.0))
    ymax = float(np.clip(np.max(arr[:, 1]) + padding, 0.0, 1.0))
    if xmax - xmin < 1e-3 or ymax - ymin < 1e-3:
        return 0.0, 1.0, 0.0, 1.0
    return xmin, xmax, ymin, ymax


def normalized_to_canvas_xy(x, y, bbox, canvas_w: int, canvas_h: int, margin: int = 12, title_h: int = 30):
    xmin, xmax, ymin, ymax = bbox
    draw_w = max(1, canvas_w - 2 * margin)
    draw_h = max(1, canvas_h - title_h - margin)
    span_x = max(xmax - xmin, 1e-3)
    span_y = max(ymax - ymin, 1e-3)
    scale = min(draw_w / span_x, draw_h / span_y)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    px = canvas_w * 0.5 + (x - cx) * scale
    py = title_h + draw_h * 0.5 + (y - cy) * scale
    return int(np.clip(round(px), 0, canvas_w - 1)), int(np.clip(round(py), 0, canvas_h - 1))


def draw_normalized_right_arm_hand_canvas(
    canvas: np.ndarray,
    pose_lms,
    hand_lms,
    mirror: bool,
    visibility_threshold: float,
):
    canvas[:] = (10, 10, 10)
    h, w = canvas.shape[:2]
    put_text(canvas, "Normalized right arm + right hand", (10, 22), scale=0.42, color=(255, 255, 255), bg=False)
    points = collect_normalized_points_for_panel(pose_lms, hand_lms, visibility_threshold, mirror)
    bbox = make_panel_bbox(points)

    def to_xy(lm):
        x, y = _normalized_xy(lm, mirror)
        return normalized_to_canvas_xy(x, y, bbox, w, h)

    if pose_lms is not None:
        for a_name, b_name in POSE_USED_CONNECTIONS:
            a = IDX[a_name]
            b = IDX[b_name]
            if a >= len(pose_lms) or b >= len(pose_lms):
                continue
            if get_visibility(pose_lms[a]) < visibility_threshold or get_visibility(pose_lms[b]) < visibility_threshold:
                continue
            cv2.line(canvas, to_xy(pose_lms[a]), to_xy(pose_lms[b]), (0, 220, 255), 2, cv2.LINE_AA)

        for name in POSE_USED_NAMES:
            idx = IDX[name]
            if idx >= len(pose_lms) or get_visibility(pose_lms[idx]) < visibility_threshold:
                continue
            cv2.circle(canvas, to_xy(pose_lms[idx]), 4, (255, 255, 255), -1, cv2.LINE_AA)

        li = IDX["LEFT_SHOULDER"]
        ri = IDX["RIGHT_SHOULDER"]
        if get_visibility(pose_lms[li]) >= visibility_threshold and get_visibility(pose_lms[ri]) >= visibility_threshold:
            lx, ly = _normalized_xy(pose_lms[li], mirror)
            rx, ry = _normalized_xy(pose_lms[ri], mirror)
            sho_c = normalized_to_canvas_xy(0.5 * (lx + rx), 0.5 * (ly + ry), bbox, w, h)
            cv2.circle(canvas, sho_c, 6, (255, 255, 0), 2, cv2.LINE_AA)
            put_text(canvas, "SHO_C", (sho_c[0] + 6, sho_c[1] + 4), scale=0.32, color=(255, 255, 0), bg=True)
    else:
        put_text(canvas, "no pose", (12, h - 36), scale=0.42, color=(100, 100, 255), bg=False)

    if hand_lms is None:
        put_text(canvas, "no selected right hand", (12, h - 14), scale=0.42, color=(100, 100, 255), bg=False)
        return

    for a, b in HAND_CONNECTIONS:
        if a >= len(hand_lms) or b >= len(hand_lms):
            continue
        cv2.line(canvas, to_xy(hand_lms[a]), to_xy(hand_lms[b]), (255, 80, 220), 2, cv2.LINE_AA)
    for lm in hand_lms:
        cv2.circle(canvas, to_xy(lm), 3, (255, 255, 255), -1, cv2.LINE_AA)

    labels = {
        "WRIST": "H_WRI",
        "INDEX_FINGER_MCP": "I_MCP",
        "MIDDLE_FINGER_MCP": "M_MCP",
        "PINKY_MCP": "P_MCP",
    }
    for name, label in labels.items():
        idx = HAND_IDX[name]
        if idx >= len(hand_lms):
            continue
        x, y = to_xy(hand_lms[idx])
        put_text(canvas, label, (x + 5, y - 5), scale=0.30, color=(255, 160, 255), bg=True)


# ============================================================
# UDP
# ============================================================

class UdpJsonPublisher:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self.addr = (self.host, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.last_warn_t = 0.0

    def publish(self, packet: dict):
        try:
            payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
            self.sock.sendto(payload, self.addr)
        except Exception as exc:
            now_t = time.time()
            if now_t - self.last_warn_t > 1.0:
                print(f"[WARN] UDP publish failed: {exc}")
                self.last_warn_t = now_t

    def close(self):
        self.sock.close()


def make_wrist_udp_packet(
    frame_id: int,
    timestamp_ms: int,
    rgbd_camera_points: Dict[str, Optional[np.ndarray]],
    rgbd_shoulder_points: Dict[str, Optional[np.ndarray]],
    shoulder_center_camera: Optional[np.ndarray],
    wrist_origin_shoulder: Optional[np.ndarray],
    frame_name: str,
    publish_delta: bool,
    teleop_scale: float,
    palm_state: Optional[dict],
) -> dict:
    right_wrist_camera = rgbd_camera_points.get("RIGHT_WRIST")
    right_wrist_shoulder = rgbd_shoulder_points.get("RIGHT_WRIST")

    position_valid = right_wrist_camera is not None and shoulder_center_camera is not None and right_wrist_shoulder is not None
    delta_shoulder = None
    target_position_shoulder = None
    if position_valid and wrist_origin_shoulder is not None:
        delta_shoulder = right_wrist_shoulder - wrist_origin_shoulder

    if position_valid:
        if publish_delta:
            if delta_shoulder is not None:
                target_position_shoulder = float(teleop_scale) * delta_shoulder
        else:
            target_position_shoulder = right_wrist_shoulder

    # target_position_m is intentionally shoulder-centered in this refactored script.
    # It is kept only for bridge / downstream compatibility, and no longer means
    # absolute camera optical-frame position.
    packet = {
        "stamp_ms": int(timestamp_ms),
        "frame_id": int(frame_id),
        "valid": bool(position_valid and target_position_shoulder is not None),
        "valid_position": bool(position_valid),
        "source_frame": str(frame_name),
        "origin_frame": "shoulder_center_camera_aligned",
        "landmark_name": "RIGHT_WRIST",
        "position_camera_m": arr_to_list(right_wrist_camera),
        "shoulder_center_camera_m": arr_to_list(shoulder_center_camera),
        "position_shoulder_m": arr_to_list(right_wrist_shoulder),
        "delta_shoulder_m": arr_to_list(delta_shoulder),
        "target_position_shoulder_m": arr_to_list(target_position_shoulder),
        "delta_camera_m": arr_to_list(delta_shoulder),
        "target_position_m": arr_to_list(target_position_shoulder),
        "teleop_scale": float(teleop_scale),
        "publish_delta": bool(publish_delta),
    }
    packet.update(palm_state_udp_fields(palm_state, str(frame_name)))
    return packet


# ============================================================
# Panel
# ============================================================

def fmt_vec(v, digits=3):
    if v is None:
        return "--"
    a = np.asarray(v, dtype=np.float32)
    return f"{a[0]:+.{digits}f} {a[1]:+.{digits}f} {a[2]:+.{digits}f}"


def build_info_panel(
    panel_h: int,
    panel_w: int,
    fps: float,
    pose_detected: bool,
    hand_detected: bool,
    intrinsics_available: bool,
    udp_enabled: bool,
    udp_port: int,
    shoulder_center_camera,
    right_wrist_camera,
    right_wrist_shoulder,
    delta_shoulder,
    target_position,
    palm_state: dict,
    paused: bool,
    pose_lms=None,
    hand_lms=None,
    mirror: bool = False,
    visibility_threshold: float = 0.45,
    show_skeleton_panel: bool = True,
    skeleton_panel_height: int = 260,
) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    y = 28

    def line(text, color=(230, 230, 230), scale=0.43, gap=20):
        nonlocal y
        put_text(panel, text, (12, y), scale=scale, color=color, bg=False)
        y += gap

    put_text(panel, "Right-arm / right-hand teleop", (12, y), scale=0.58, color=(0, 255, 255), bg=False)
    y += 30
    line(f"{'PAUSED' if paused else 'RUNNING'} | FPS {fps:.1f}", gap=18)
    line(f"Pose: {'yes' if pose_detected else 'no'} | Right hand: {'yes' if hand_detected else 'no'}", gap=18)
    line(f"RGB-D: {'yes' if intrinsics_available else 'no'} | UDP: {'on' if udp_enabled else 'off'} {udp_port if udp_enabled else '--'}", gap=18)
    line("Origin: shoulder_center camera-aligned", color=(180, 220, 220), scale=0.36, gap=20)

    if show_skeleton_panel:
        canvas_w = max(120, panel_w - 24)
        reserved_text_h = 180
        available_h = panel_h - y - reserved_text_h
        canvas_h = min(int(skeleton_panel_height), max(120, available_h))
        if y + canvas_h + 8 < panel_h:
            skel_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            draw_normalized_right_arm_hand_canvas(
                skel_canvas,
                pose_lms=pose_lms,
                hand_lms=hand_lms,
                mirror=mirror,
                visibility_threshold=visibility_threshold,
            )
            panel[y:y + canvas_h, 12:12 + canvas_w] = skel_canvas
            y += canvas_h + 14

    put_text(panel, "Position", (12, y), scale=0.54, color=(0, 255, 0), bg=False)
    y += 25
    line(f"Shoulder center cam: {fmt_vec(shoulder_center_camera)}", scale=0.38)
    line(f"R wrist camera     : {fmt_vec(right_wrist_camera)}", scale=0.38)
    line(f"R wrist shoulder   : {fmt_vec(right_wrist_shoulder)}", scale=0.38)
    line(f"Delta shoulder     : {fmt_vec(delta_shoulder)}", scale=0.38)
    line(f"target_position_m  : {fmt_vec(target_position)}", scale=0.38, gap=24)

    put_text(panel, "Palm", (12, y), scale=0.54, color=(255, 120, 255), bg=False)
    y += 25
    palm_valid = bool(palm_state and palm_state.get("valid", False))
    line(f"Palm valid: {'yes' if palm_valid else 'no'} ({palm_state.get('reason', '--')})", color=(0, 255, 0) if palm_valid else (80, 80, 255), scale=0.38)
    line("Palm source: selected right hand only", color=(180, 180, 180), scale=0.34, gap=17)
    score = palm_state.get("handedness_score")
    line(f"Hand: {palm_state.get('handedness', 'none')} score={'--' if score is None else f'{score:.3f}'}", scale=0.38)
    angle = palm_state.get("angle_from_origin_deg")
    line(f"Angle from origin: {'--' if angle is None else f'{angle:.1f} deg'}", scale=0.38)
    q_delta = palm_state.get("q_delta_palm_wxyz")
    if q_delta is None:
        line("q_delta wxyz: --", scale=0.36)
    else:
        line(f"q_delta wxyz: {q_delta[0]:+.3f} {q_delta[1]:+.3f}", scale=0.34, gap=16)
        line(f"              {q_delta[2]:+.3f} {q_delta[3]:+.3f}", scale=0.34)
    line(f"Palm normal cam: {fmt_vec(palm_state.get('palm_normal_camera'))}", scale=0.36, gap=26)

    put_text(panel, "Keys", (12, y), scale=0.52, color=(0, 255, 255), bg=False)
    y += 24
    line("p pause/resume", scale=0.38)
    line("r reset wrist origin", scale=0.38)
    line("o reset palm origin", scale=0.38)
    line("q / ESC quit", scale=0.38)
    return panel


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    mirror = not args.no_mirror
    draw_names = not args.no_draw_names
    hand_enabled = args.enable_hand and not args.disable_hand
    draw_hand = args.draw_hand and not args.no_draw_hand
    if args.handedness != "Right":
        print(f"[WARN] This script is right-hand locked; ignoring --handedness={args.handedness}")

    if args.model_complexity is not None:
        args.model = {0: "lite", 1: "full", 2: "heavy"}[args.model_complexity]

    model_path = ensure_model(args.model, args.model_path, args.models_dir)
    print("[INFO] Using pose model:", model_path)

    hand_model_path = None
    if hand_enabled:
        try:
            hand_model_path = ensure_hand_model(args.hand_model_path, args.hand_models_dir)
            print("[INFO] Using hand model:", hand_model_path)
        except Exception as exc:
            print(f"[WARN] Hand model unavailable, disabling hand detection: {exc}")
            hand_enabled = False

    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    HandLandmarker = getattr(vision, "HandLandmarker", None)
    HandLandmarkerOptions = getattr(vision, "HandLandmarkerOptions", None)
    VisionRunningMode = vision.RunningMode
    if hand_enabled and (HandLandmarker is None or HandLandmarkerOptions is None):
        print("[WARN] MediaPipe HandLandmarker API unavailable, disabling hand detection.")
        hand_enabled = False

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    hand_options = None
    if hand_enabled:
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.min_hand_detection_confidence,
            min_hand_presence_confidence=args.min_hand_presence_confidence,
            min_tracking_confidence=args.min_hand_tracking_confidence,
        )

    camera = OrbbecCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_depth=not args.no_depth,
        align_mode=args.orbbec_align,
        enable_sync=not args.disable_frame_sync,
    ).start()

    udp_pub = UdpJsonPublisher(args.udp_host, args.udp_port) if args.udp_publish else None
    if udp_pub is not None:
        print(f"[INFO] UDP publishing enabled: {args.udp_host}:{args.udp_port}")

    paused = False
    frame_id = 0
    fps_smooth = 0.0
    last_t = time.time()
    start_monotonic = time.monotonic()
    last_frame_bgr = None
    last_depth_mm = None
    last_pose_result = None
    last_hand_result = None
    last_non_right_hand_warn_t = 0.0
    wrist_origin_shoulder = None
    palm_runtime = {"R0": None, "R_prev": None, "R_filtered": None}
    latest_valid_palm_R = None

    window_name = "Right Arm / Right Hand Orbbec Teleop Publisher"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("[INFO] Started.")
    print(f"[INFO] Keys: q/ESC quit, p pause/resume, {args.reset_origin_key} reset wrist origin, {args.palm_reset_key} reset palm origin")
    print(f"[INFO] mirror={mirror}, hand={hand_enabled}, origin=shoulder_center")

    try:
        with ExitStack() as stack:
            pose_landmarker = stack.enter_context(PoseLandmarker.create_from_options(pose_options))
            hand_landmarker = None
            if hand_enabled:
                try:
                    hand_landmarker = stack.enter_context(HandLandmarker.create_from_options(hand_options))
                except Exception as exc:
                    print(f"[WARN] HandLandmarker initialization failed, disabling hand detection: {exc}")
                    hand_enabled = False

            while True:
                if not paused:
                    ok, frame_bgr, depth_mm = camera.read()
                    if not ok:
                        if last_frame_bgr is None:
                            frame_bgr = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                            put_text(frame_bgr, "Waiting for Orbbec frame...", (16, 60), scale=0.7, color=(0, 255, 255), bg=True)
                            depth_mm = None
                            pose_result = None
                            hand_result = None
                            timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)
                        else:
                            frame_bgr = last_frame_bgr.copy()
                            depth_mm = last_depth_mm
                            pose_result = last_pose_result
                            hand_result = last_hand_result
                            timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)
                    else:
                        frame_id += 1
                        last_frame_bgr = frame_bgr.copy()
                        last_depth_mm = depth_mm

                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frame_rgb = np.ascontiguousarray(frame_rgb)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                        timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)

                        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                        last_pose_result = pose_result
                        hand_result = last_hand_result
                        if hand_enabled and hand_landmarker is not None:
                            hand_every = max(1, int(args.hand_every_n_frames))
                            if frame_id % hand_every == 0:
                                try:
                                    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                                    last_hand_result = hand_result
                                except Exception as exc:
                                    print(f"[WARN] HandLandmarker detect failed: {exc}")

                        now_t = time.time()
                        dt = now_t - last_t
                        last_t = now_t
                        if dt > 1e-6:
                            fps = 1.0 / dt
                            fps_smooth = fps if fps_smooth <= 0.0 else 0.9 * fps_smooth + 0.1 * fps
                else:
                    if last_frame_bgr is None:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                        continue
                    frame_bgr = last_frame_bgr.copy()
                    depth_mm = last_depth_mm
                    pose_result = last_pose_result
                    hand_result = last_hand_result
                    timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)

                intrinsics = camera.get_color_intrinsics()
                intrinsics_available = intrinsics is not None

                lms = None
                pose_detected = bool(pose_result is not None and getattr(pose_result, "pose_landmarks", None))
                if pose_detected:
                    lms = pose_result.pose_landmarks[0]

                selected_hand = select_right_hand_result(hand_result) if hand_enabled else None
                if (
                    hand_enabled
                    and selected_hand is None
                    and hand_result is not None
                    and getattr(hand_result, "hand_landmarks", None)
                ):
                    warn_t = time.time()
                    if warn_t - last_non_right_hand_warn_t > 1.0:
                        print("[INFO] HandLandmarker saw non-right hand, ignored.")
                        last_non_right_hand_warn_t = warn_t
                hand_lms = selected_hand["hand_lms"] if selected_hand is not None else None
                hand_detected = hand_lms is not None

                rgbd_camera_points = compute_pose_rgbd_points(
                    lms=lms,
                    depth_mm=depth_mm,
                    intrinsics=intrinsics,
                    frame_w=frame_bgr.shape[1],
                    frame_h=frame_bgr.shape[0],
                    visibility_threshold=args.visibility_threshold,
                )
                shoulder_center_camera = compute_shoulder_center_camera(rgbd_camera_points)
                rgbd_shoulder_points = compute_shoulder_centered_points(rgbd_camera_points, shoulder_center_camera)
                right_wrist_camera = rgbd_camera_points.get("RIGHT_WRIST")
                right_wrist_shoulder = rgbd_shoulder_points.get("RIGHT_WRIST")

                hand_camera_points = compute_hand_camera_points(
                    hand_lms=hand_lms,
                    depth_mm=depth_mm,
                    intrinsics=intrinsics,
                    frame_w=frame_bgr.shape[1],
                    frame_h=frame_bgr.shape[0],
                    search_radius=args.palm_depth_radius,
                ) if hand_lms is not None else {name: None for name in HAND_LANDMARK_NAMES}
                palm_frame_camera = compute_palm_frame_from_points(hand_camera_points) if hand_lms is not None else {"valid": False, "reason": "no selected right hand"}
                palm_state = update_palm_orientation_state(
                    palm_frame_camera=palm_frame_camera,
                    selected_hand=selected_hand,
                    runtime_state=palm_runtime,
                    auto_init=args.palm_auto_init,
                    filter_alpha=args.palm_filter_alpha,
                    max_angle_step_deg=args.palm_max_angle_step_deg,
                )
                if palm_state.get("valid", False):
                    latest_valid_palm_R = palm_state["R_camera_palm"].copy()

                if right_wrist_shoulder is not None and wrist_origin_shoulder is None:
                    wrist_origin_shoulder = right_wrist_shoulder.copy()
                    print("[INFO] Auto set wrist shoulder-centered origin:", wrist_origin_shoulder.tolist())

                delta_shoulder = None if right_wrist_shoulder is None or wrist_origin_shoulder is None else right_wrist_shoulder - wrist_origin_shoulder
                target_position = None
                if right_wrist_shoulder is not None:
                    if args.publish_delta:
                        if delta_shoulder is not None:
                            target_position = float(args.teleop_scale) * delta_shoulder
                    else:
                        target_position = right_wrist_shoulder

                if udp_pub is not None:
                    packet = make_wrist_udp_packet(
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        rgbd_camera_points=rgbd_camera_points,
                        rgbd_shoulder_points=rgbd_shoulder_points,
                        shoulder_center_camera=shoulder_center_camera,
                        wrist_origin_shoulder=wrist_origin_shoulder,
                        frame_name=args.udp_frame,
                        publish_delta=args.publish_delta,
                        teleop_scale=args.teleop_scale,
                        palm_state=palm_state,
                    )
                    udp_pub.publish(packet)

                vis_frame = frame_bgr.copy()
                if mirror:
                    vis_frame = cv2.flip(vis_frame, 1)

                draw_pose_used_2d(vis_frame, lms, mirror=mirror, visibility_threshold=args.visibility_threshold, draw_names=draw_names)
                draw_shoulder_center_marker(vis_frame, lms, mirror=mirror, visibility_threshold=args.visibility_threshold)
                draw_rgbd_valid_points(vis_frame, lms, rgbd_camera_points, mirror=mirror, visibility_threshold=args.visibility_threshold)
                if draw_hand and hand_lms is not None:
                    draw_hand_landmarks_2d(vis_frame, hand_lms, mirror=mirror, draw_names=draw_names)
                if args.draw_palm_frame:
                    draw_palm_frame_axes(vis_frame, palm_frame_camera, intrinsics, mirror=mirror, axis_length_m=args.palm_frame_axis_length)

                status = "PAUSED" if paused else "RUNNING"
                put_text(
                    vis_frame,
                    f"{status} | FPS {fps_smooth:.1f} | pose={pose_detected} | right_hand={hand_detected} | UDP={udp_pub is not None} | origin=shoulder_center",
                    (16, 32),
                    scale=0.55,
                    color=(0, 255, 255),
                    bg=True,
                )

                h, _w = vis_frame.shape[:2]
                panel_w = max(360, args.panel_width)
                panel = build_info_panel(
                    panel_h=h,
                    panel_w=panel_w,
                    fps=fps_smooth,
                    pose_detected=pose_detected,
                    hand_detected=hand_detected,
                    intrinsics_available=intrinsics_available,
                    udp_enabled=udp_pub is not None,
                    udp_port=args.udp_port,
                    shoulder_center_camera=shoulder_center_camera,
                    right_wrist_camera=right_wrist_camera,
                    right_wrist_shoulder=right_wrist_shoulder,
                    delta_shoulder=delta_shoulder,
                    target_position=target_position,
                    palm_state=palm_state,
                    paused=paused,
                    pose_lms=lms,
                    hand_lms=hand_lms,
                    mirror=mirror,
                    visibility_threshold=args.visibility_threshold,
                    show_skeleton_panel=args.panel_skeleton,
                    skeleton_panel_height=args.panel_skeleton_height,
                )

                final_vis = np.hstack([vis_frame, panel])
                if args.display_scale != 1.0:
                    scale = float(np.clip(args.display_scale, 0.25, 1.5))
                    final_vis = cv2.resize(
                        final_vis,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                    )

                cv2.imshow(window_name, final_vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break
                if key == ord("p"):
                    paused = not paused
                    print("[INFO] paused =", paused)

                reset_key = (args.reset_origin_key or "r").lower()[:1]
                if reset_key and key == ord(reset_key):
                    if right_wrist_shoulder is not None:
                        wrist_origin_shoulder = right_wrist_shoulder.copy()
                        print("[INFO] Reset wrist shoulder-centered origin:", wrist_origin_shoulder.tolist())
                    else:
                        print("[WARN] Cannot reset wrist origin: right wrist shoulder-centered point invalid")

                palm_reset_key = (args.palm_reset_key or "o").lower()[:1]
                if palm_reset_key and key == ord(palm_reset_key):
                    if latest_valid_palm_R is not None:
                        palm_runtime["R0"] = latest_valid_palm_R.copy()
                        print("[INFO] Reset palm orientation origin.")
                    else:
                        print("[WARN] Cannot reset palm origin: no valid palm frame")

    finally:
        camera.stop()
        if udp_pub is not None:
            udp_pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
