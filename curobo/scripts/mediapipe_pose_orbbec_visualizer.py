#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MediaPipe Pose Landmarker V2 Orbbec Visualizer

适配新版 MediaPipe Tasks API：
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

功能：
1. 调用 Orbbec Python SDK 获取 RGB / depth 帧
2. 绘制 2D 骨架
3. 绘制关键点名称
4. 绘制 segmentation mask
5. 显示 world landmarks 的 front / side 简化视图
6. 显示左右肩、肘、髋、膝角度
7. 显示 FPS、关键点 visibility
8. 可保存截图
9. 可保存 landmarks 到 CSV
10. 可选通过 UDP 发布 RGB-D 反投影得到的右腕/手部目标点
11. 可选运行 Hand Landmarker，估计视觉侧 palm frame 和相对旋转

按键：
    q / ESC : 退出
    p       : 暂停 / 继续
    w / s   : 上下滚动右侧信息栏
    c       : 保存截图
    r       : 重置 UDP 发布用的人体初始目标点
    o       : 重置 palm orientation origin

示例：
    python mediapipe_pose_orbbec_visualizer.py --width 1280 --height 720 --model full
"""

import argparse
import csv
import json
import math
import os
import socket
import time
import urllib.request
from contextlib import ExitStack
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Orbbec Python SDK 的核心对象：
# - Pipeline/Config 负责打开设备、配置流、启动/停止相机；
# - OBSensorType 用来选择 COLOR / DEPTH 传感器；
# - OBFormat 用来判断彩色帧编码格式，并转换为 OpenCV 可用的 BGR 图像；
# - OBAlignMode 用来配置深度帧与彩色帧的对齐方式。
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
# MediaPipe Pose Landmark index
# ============================================================

# MediaPipe PoseLandmarker 固定输出 33 个人体关键点。
# 这里显式维护名字列表，后面可以通过 IDX["RIGHT_WRIST"] 这类写法
# 读出对应 landmark 下标，避免在计算角度、连线、可视化时直接写魔法数字。
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

# 自定义骨架连线拓扑。MediaPipe 本身也提供连接关系，但这里手写一份，
# 方便同时复用于主画面、normalized skeleton、world front/side 三种视图。
POSE_CONNECTIONS = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),

    # torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),

    # right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),

    # left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),

    # right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]

RIGHT_ARM_RGBD_NAMES = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "RIGHT_INDEX",
    "RIGHT_THUMB",
    "RIGHT_PINKY",
]

RGBD_LANDMARK_NAMES = RIGHT_ARM_RGBD_NAMES + [
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
]

# MediaPipe HandLandmarker 固定输出 21 个手部关键点。
# palm frame 是视觉侧估计的手掌姿态，不是严格解剖意义 wrist joint rotation。
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


MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# 本地缓存的 MediaPipe Tasks 模型文件名。
# ensure_model() 会优先复用本地文件，不存在时才下载。
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


# ============================================================
# 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MediaPipe Tasks API PoseLandmarker webcam visualizer"
    )

    # --camera 保留只是为了兼容旧 webcam 版本命令行；
    # Orbbec SDK 当前通过 Pipeline() 自动打开默认 Orbbec 设备。
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="kept for CLI compatibility; Orbbec SDK opens the default Orbbec device",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=10)

    # Orbbec 深度流默认打开：
    # - 彩色帧用于 MediaPipe 姿态检测；
    # - 深度帧会被同步读取并缓存，后续可以用于 2D landmark + depth 反投影。
    parser.add_argument(
        "--no-depth",
        action="store_true",
        default=False,
        help="disable Orbbec depth stream and use color only",
    )
    # 对齐模式控制 depth 和 color 是否在同一视角/分辨率坐标系下输出。
    # HW 通常性能最好；SW 作为兼容后备；NONE 则完全不做对齐。
    parser.add_argument(
        "--orbbec-align",
        type=str,
        default="HW",
        choices=["HW", "SW", "NONE"],
        help="depth-to-color align mode when depth is enabled",
    )
    # 帧同步用于让 color/depth 时间上尽量对应。
    # 如果某些设备或驱动不支持，可以用 --disable-frame-sync 关闭。
    parser.add_argument(
        "--disable-frame-sync",
        action="store_true",
        default=False,
        help="disable Orbbec color/depth frame sync",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="full",
        choices=["lite", "full", "heavy"],
        help="pose landmarker model type",
    )
    # 兼容旧 MediaPipe Solutions 风格的 --model-complexity 参数：
    # 0/1/2 会在 main() 里映射到 lite/full/heavy 三种 Tasks 模型。
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="MediaPipe-style alias: 0=lite, 1=full, 2=heavy",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="absolute or relative path to .task model file. If empty, auto-download to ./models/",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="directory to store downloaded .task model",
    )

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
    parser.add_argument("--handedness", type=str, default="Right", choices=["Right", "Left", "Any"])
    parser.add_argument("--palm-depth-radius", type=int, default=5)
    parser.add_argument("--draw-hand", action="store_true", default=True)
    parser.add_argument("--no-draw-hand", action="store_true", default=False)
    parser.add_argument("--draw-palm-frame", action="store_true", default=True)
    parser.add_argument("--palm-frame-axis-length", type=float, default=0.08)
    parser.add_argument("--hand-every-n-frames", type=int, default=1)
    parser.add_argument("--palm-auto-init", action="store_true", default=True)
    parser.add_argument("--palm-reset-key", type=str, default="o")
    parser.add_argument("--palm-filter-alpha", type=float, default=0.25)
    parser.add_argument("--palm-max-angle-step-deg", type=float, default=15.0)

    # 旧版参数保留但隐藏在 help 中，避免历史命令直接报错。
    # 实际开关采用下面的 --no-draw-names / --no-segmentation 语义。
    parser.add_argument("--draw-names", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--segmentation", action="store_true", default=False, help=argparse.SUPPRESS)
    # 默认显示关键点名称、默认开启 segmentation；
    # 如需关闭，使用对应的 no-* 参数。
    parser.add_argument("--no-draw-names", action="store_true", default=False)
    parser.add_argument("--no-segmentation", action="store_true", default=False)
    # 默认镜像显示更接近普通自拍预览；加 --no-mirror 才关闭镜像。
    parser.add_argument("--no-mirror", action="store_true", default=False)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--show-index", action="store_true")

    # UDP 发布只负责把感知侧目标点发出去：
    # - 不做 cuRobo IK；
    # - 不做机器人控制；
    # - 不引入 ROS/ZeroMQ，保持脚本和下游控制程序低耦合。
    parser.add_argument("--udp-publish", action="store_true", default=True)
    parser.add_argument("--udp-host", type=str, default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5557)
    parser.add_argument("--udp-frame", type=str, default="orbbec_color_optical_frame")
    parser.add_argument(
        "--publish-target",
        type=str,
        default="right_wrist",
        choices=["right_wrist", "right_index", "right_thumb"],
    )
    parser.add_argument("--teleop-scale", type=float, default=1.0)
    parser.add_argument("--publish-delta", action="store_true", default=False)
    parser.add_argument("--reset-origin-key", type=str, default="r")

    # 右侧信息栏参数：panel-width 控制横向宽度；
    # panel-scroll-height 控制右侧虚拟画布高度，内容多时用滚轮/w/s 滚动查看。
    parser.add_argument(
        "--panel-width",
        type=int,
        default=460,
        help="right information panel width in pixels",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.8,
        help="scale the final composed window, e.g. 0.9 for small screens",
    )
    
    parser.add_argument(
        "--panel-scroll-height",
        type=int,
        default=1100,
        help="virtual right panel height; use mouse wheel to scroll it",
    )

    return parser.parse_args()


# ============================================================
# 模型下载
# ============================================================

def ensure_model(model_name: str, model_path: str, models_dir: str) -> str:
    # 用户显式给了模型路径时，完全信任该路径，不做自动下载。
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"指定的模型文件不存在: {model_path}")
        return model_path

    # 没有指定模型路径时，把模型缓存到 models_dir。
    # 这样第一次运行下载，后续运行直接复用本地 .task 文件。
    os.makedirs(models_dir, exist_ok=True)

    local_path = os.path.join(models_dir, MODEL_FILES[model_name])
    if os.path.exists(local_path):
        return local_path

    # MediaPipe Tasks 模型是 .task 文件，不是旧版 solutions 的 pbtxt/tflite 组合。
    url = MODEL_URLS[model_name]
    print(f"[INFO] 模型不存在，开始下载: {url}")
    print(f"[INFO] 保存到: {local_path}")

    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        raise RuntimeError(
            "模型自动下载失败。\n"
            "你可以手动执行：\n"
            f"mkdir -p {models_dir}\n"
            f"wget -O {local_path} '{url}'\n"
            "然后重新运行脚本。\n"
            f"原始错误: {e}"
        ) from e

    return local_path


def ensure_hand_model(model_path: str, models_dir: str) -> str:
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"指定的手部模型文件不存在: {model_path}")
        return model_path

    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, HAND_MODEL_FILE)
    if os.path.exists(local_path):
        return local_path

    print(f"[INFO] 手部模型不存在，开始下载: {HAND_MODEL_URL}")
    print(f"[INFO] 保存到: {local_path}")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, local_path)
    except Exception as e:
        raise RuntimeError(
            "手部模型自动下载失败。\n"
            "你可以手动执行：\n"
            f"mkdir -p {models_dir}\n"
            f"wget -O {local_path} '{HAND_MODEL_URL}'\n"
            "然后重新运行脚本，或使用 --disable-hand 关闭手部检测。\n"
            f"原始错误: {e}"
        ) from e

    return local_path


# ============================================================
# Orbbec camera utilities
# ============================================================

# 以下几个函数只负责“像素格式转换”：
# Orbbec 彩色流可能以 RGB、MJPG、YUYV、UYVY、I420、NV12、NV21 等格式输出。
# MediaPipe 和本脚本后续可视化统一使用 OpenCV BGR，所以这里先全部归一化。
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
    # Orbbec VideoFrame 给出原始字节、宽高和像素格式；
    # 这里不做姿态检测，只把相机帧转换成 frame_bgr。
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    # RGB 是 Orbbec examples 中最常见的彩色格式；
    # OpenCV 绘图/显示默认 BGR，所以需要 RGB -> BGR。
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 如果设备已经输出 BGR，则只需要整理成 HxWx3。
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    # MJPG 是压缩帧，需要先用 OpenCV 解码。
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    # 其余 YUV 格式按对应 OpenCV code 转 BGR。
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
    # Orbbec 深度帧通常是 uint16 原始深度值；
    # get_depth_scale() 给出该设备/模式下的尺度因子，乘完后得到实际深度单位。
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    return depth_data.astype(np.float32) * scale


class OrbbecCamera:
    # 这个类是本文件与 webcam 版本唯一真正不同的核心：
    # 它把 Orbbec Pipeline/Config/read/stop 包装成类似 cv2.VideoCapture 的接口。
    # main() 后面仍然只关心 ok, frame_bgr, depth_mm，不关心底层 SDK 细节。
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        enable_depth: bool,
        align_mode: str,
        enable_sync: bool,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.align_mode = align_mode
        self.enable_sync = enable_sync
        # Pipeline 是 Orbbec SDK 的运行入口；Config 保存将要启用的 stream/profile。
        self.pipeline = Pipeline()
        self.config = Config()
        self.color_intrinsics = None

    def _select_color_profile(self) -> VideoStreamProfile:
        # 优先请求用户命令行指定的 width/height/fps + RGB 格式。
        # 如果设备不支持该 profile，就回退到 SDK 默认 color profile。
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            return profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.RGB, self.fps
            )
        except OBError as exc:
            print(f"[WARN] Requested Orbbec color profile unavailable: {exc}")
            color_profile = profile_list.get_default_video_stream_profile()
            print("[INFO] Using default color profile:", color_profile)
            return color_profile

    def _select_depth_profile(self) -> Optional[VideoStreamProfile]:
        # 深度流用于获取 depth map。这里采用默认 depth profile，
        # 因为不同 Orbbec 型号支持的深度分辨率/格式差异比较大。
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
        # 先启用 color stream：这是 MediaPipe 姿态检测的输入来源。
        color_profile = self._select_color_profile()
        print("[INFO] Using Orbbec color profile:", color_profile)
        self.config.enable_stream(color_profile)
        self.color_intrinsics = self._extract_color_intrinsics(color_profile)

        # depth stream 可选启用。启用后会配置 align 和 frame sync，
        # 但 MediaPipe 本身仍然只使用 color image。
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
            print("[WARN] Orbbec color camera intrinsics unavailable; RGB-D deprojection disabled.")
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
        # RGB-D camera coords 使用的是 Orbbec aligned depth + color image 像素坐标；
        # 因此反投影必须使用 color camera intrinsics。
        profile_methods = [
            "get_intrinsic",
            "get_intrinsics",
            "get_camera_intrinsic",
            "get_video_stream_intrinsic",
        ]
        for method_name in profile_methods:
            if not hasattr(color_profile, method_name):
                continue
            try:
                intrinsics = self._intrinsics_to_dict(getattr(color_profile, method_name)(), color_profile)
                if intrinsics is not None:
                    return intrinsics
            except Exception as exc:
                print(f"[WARN] Failed reading intrinsics from color profile via {method_name}: {exc}")

        try:
            camera_param = self.pipeline.get_camera_param()
            for attr_name in [
                "rgb_intrinsic",
                "color_intrinsic",
                "rgbIntrinsic",
                "colorIntrinsic",
                "left_intrinsic",
            ]:
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
        # NONE: 完全禁用对齐，depth 保持原始深度视角。
        if self.align_mode == "NONE":
            self.config.set_align_mode(OBAlignMode.DISABLE)
            return
        # SW: 使用软件对齐，兼容性好但可能更耗 CPU。
        if self.align_mode == "SW":
            self.config.set_align_mode(OBAlignMode.SW_MODE)
            return

        # HW: 默认尝试硬件对齐；部分设备 PID 已知需要软件对齐。
        # 如果查询设备信息失败，也回退到 SW，优先保证程序能跑起来。
        try:
            device = self.pipeline.get_device()
            device_pid = device.get_device_info().get_pid()
            if device_pid == 0x066B:
                self.config.set_align_mode(OBAlignMode.SW_MODE)
            else:
                self.config.set_align_mode(OBAlignMode.HW_MODE)
        except Exception as exc:
            print(f"[WARN] Failed to configure HW align, falling back to SW align: {exc}")
            self.config.set_align_mode(OBAlignMode.SW_MODE)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        # wait_for_frames() 是 Orbbec examples 的典型取帧方式。
        # 这里 timeout=500ms；超时返回 False，由主循环继续等待下一帧。
        frames = self.pipeline.wait_for_frames(500)
        if frames is None:
            return False, None, None

        # 彩色帧是必需的：没有 color frame，MediaPipe 没有输入图像。
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return False, None, None

        # 把 Orbbec color frame 统一转换为 OpenCV BGR。
        color_image = orbbec_frame_to_bgr_image(color_frame)
        if color_image is None:
            return False, None, None

        # 深度帧是可选的：当前可视化主流程不直接使用 depth，
        # 但保留 depth_mm 便于后续做 2D landmark + depth 的三维反投影。
        depth_mm = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                depth_mm = orbbec_depth_frame_to_mm(depth_frame)

        return True, color_image, depth_mm

    def stop(self):
        # 退出时必须停止 pipeline，释放 Orbbec 设备句柄。
        self.pipeline.stop()


# ============================================================
# 基础工具
# ============================================================

def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        cv2.rectangle(
            img,
            (x - 3, y - th - 5),
            (x + tw + 3, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def get_visibility(lm) -> float:
    return float(getattr(lm, "visibility", 1.0))


def get_presence(lm) -> float:
    return float(getattr(lm, "presence", 1.0))


def lm_to_pixel(
    lm,
    width: int,
    height: int,
    mirror: bool = False,
) -> Tuple[int, int]:
    x = float(lm.x)
    y = float(lm.y)

    if mirror:
        x = 1.0 - x

    px = int(np.clip(x * width, 0, width - 1))
    py = int(np.clip(y * height, 0, height - 1))
    return px, py


def deproject_pixel_to_camera(
    u,
    v,
    depth_m,
    intrinsics,
) -> Optional[np.ndarray]:
    # RGB-D camera coords:
    # - 使用 MediaPipe 2D landmark 的原始相机像素坐标；
    # - 结合 Orbbec aligned depth；
    # - 再通过 pinhole camera model 反投影；
    # - 输出坐标系是 Orbbec color/depth aligned 后的相机坐标系；
    # - 单位是 meter。
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
    mirror: bool = False,
    search_radius: int = 3,
) -> Optional[np.ndarray]:
    # 反投影必须使用原始未镜像图像坐标。
    # mirror 参数保留在签名里用于明确调用意图，但本函数强制用 mirror=False。
    if depth_mm is None or intrinsics is None:
        return None
    if depth_mm.ndim != 2:
        return None

    u, v = lm_to_pixel(lm, frame_w, frame_h, mirror=False)
    depth_h, depth_w = depth_mm.shape[:2]
    if depth_w <= 0 or depth_h <= 0:
        return None

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

    depth_m = depth_value / 1000.0
    return deproject_pixel_to_camera(u, v, depth_m, intrinsics)


def compute_rgbd_camera_landmarks(
    lms,
    depth_mm,
    intrinsics,
    frame_w: int,
    frame_h: int,
    visibility_threshold: float,
) -> Dict[str, Optional[np.ndarray]]:
    points = {name: None for name in RGBD_LANDMARK_NAMES}
    if lms is None or depth_mm is None or intrinsics is None:
        return points

    for name in RGBD_LANDMARK_NAMES:
        idx = IDX[name]
        if idx >= len(lms) or get_visibility(lms[idx]) < visibility_threshold:
            continue
        try:
            points[name] = landmark_to_camera_point(
                lms[idx],
                frame_w=frame_w,
                frame_h=frame_h,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                mirror=False,
            )
        except Exception as exc:
            print(f"[WARN] RGB-D deprojection failed for {name}: {exc}")
            points[name] = None

    return points


# ============================================================
# Hand Landmarker palm frame utilities
# ============================================================

def select_hand_result(hand_result, handedness: str):
    if hand_result is None or not getattr(hand_result, "hand_landmarks", None):
        return None

    hand_lms_list = hand_result.hand_landmarks
    hand_world_list = getattr(hand_result, "hand_world_landmarks", None)
    handedness_list = getattr(hand_result, "handedness", None)

    candidates = []
    for i, hand_lms in enumerate(hand_lms_list):
        label = "Unknown"
        score = 0.0
        if handedness_list is not None and i < len(handedness_list) and handedness_list[i]:
            category = handedness_list[i][0]
            label = (
                getattr(category, "category_name", None)
                or getattr(category, "display_name", None)
                or getattr(category, "label", None)
                or "Unknown"
            )
            score = float(getattr(category, "score", 0.0))

        hand_world_lms = None
        if hand_world_list is not None and i < len(hand_world_list):
            hand_world_lms = hand_world_list[i]

        candidates.append(
            {
                "hand_lms": hand_lms,
                "hand_world_lms": hand_world_lms,
                "handedness_label": str(label),
                "handedness_score": score,
            }
        )

    if not candidates:
        return None
    if handedness == "Any":
        return candidates[0]

    for item in candidates:
        if item["handedness_label"].lower() == handedness.lower():
            return item
    return candidates[0]


def hand_landmark_to_camera_point(
    hand_lm,
    frame_w: int,
    frame_h: int,
    depth_mm,
    intrinsics,
    search_radius: int = 5,
) -> Optional[np.ndarray]:
    # HandLandmarker 的 hand_lm.x/y 也是 normalized image coordinate。
    # 反投影必须使用原始未镜像图像坐标，与 aligned depth 一致。
    return landmark_to_camera_point(
        hand_lm,
        frame_w=frame_w,
        frame_h=frame_h,
        depth_mm=depth_mm,
        intrinsics=intrinsics,
        mirror=False,
        search_radius=search_radius,
    )


def compute_hand_camera_points(
    hand_lms,
    depth_mm,
    intrinsics,
    frame_w: int,
    frame_h: int,
    search_radius: int,
) -> Dict[str, Optional[np.ndarray]]:
    points = {name: None for name in HAND_LANDMARK_NAMES}
    if hand_lms is None or depth_mm is None or intrinsics is None:
        return points

    for name, idx in HAND_IDX.items():
        if idx >= len(hand_lms):
            continue
        try:
            points[name] = hand_landmark_to_camera_point(
                hand_lms[idx],
                frame_w=frame_w,
                frame_h=frame_h,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                search_radius=search_radius,
            )
        except Exception as exc:
            print(f"[WARN] Hand RGB-D deprojection failed for {name}: {exc}")
            points[name] = None
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

    origin_points = [p for p in [p_wrist, p_index, p_middle, p_pinky] if p is not None]
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

    # q_delta_palm_wxyz 是视觉估计的手掌坐标系相对初始手掌坐标系的相对旋转，
    # 不是 wrist joint angle，也不是 robot end-effector target rotation。
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
    u = int(round(fx * X / Z + cx))
    v = int(round(fy * Y / Z + cy))
    return u, v


def draw_hand_landmarks_2d(
    img,
    hand_lms,
    mirror: bool,
    draw_names: bool = False,
):
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
            short = name.replace("_FINGER", "").replace("_MCP", "").replace("WRIST", "PALM_WRI")
            put_text(img, short, (x + 5, y - 5), scale=0.35, color=(255, 160, 255), bg=True)


def draw_palm_frame_axes(
    img,
    palm_frame_camera,
    intrinsics,
    mirror: bool,
    axis_length_m: float,
):
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

    axes = [
        (R[:, 0], (0, 0, 255), "x"),
        (R[:, 1], (0, 255, 0), "y"),
        (R[:, 2], (255, 0, 0), "z"),
    ]
    for axis, color, label in axes:
        p1 = project_for_display(origin + axis_length_m * axis)
        if p1 is None:
            continue
        cv2.line(img, p0, p1, color, 3, cv2.LINE_AA)
        put_text(img, label, (p1[0] + 4, p1[1] + 4), scale=0.42, color=color, bg=True)


def _array_to_list(value):
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
        "R_camera_palm": _array_to_list(palm_state.get("R_camera_palm")),
        "delta_R_camera_palm": _array_to_list(palm_state.get("delta_R_camera_palm")),
        "q_camera_palm_wxyz": _array_to_list(palm_state.get("q_camera_palm_wxyz")),
        "q_delta_palm_wxyz": _array_to_list(palm_state.get("q_delta_palm_wxyz")),
        "palm_origin_camera_m": _array_to_list(palm_state.get("palm_origin_camera_m")),
        "palm_normal_camera": _array_to_list(palm_state.get("palm_normal_camera")),
        "palm_angle_from_origin_deg": palm_state.get("angle_from_origin_deg"),
    }


# ============================================================
# UDP target publisher
# ============================================================

PUBLISH_TARGET_TO_LANDMARK = {
    "right_wrist": "RIGHT_WRIST",
    "right_index": "RIGHT_INDEX",
    "right_thumb": "RIGHT_THUMB",
}


def publish_target_to_landmark_name(publish_target: str) -> str:
    return PUBLISH_TARGET_TO_LANDMARK.get(publish_target, "RIGHT_WRIST")


class UdpJsonPublisher:
    # 这个 publisher 只负责发布感知目标：
    # - 数据来源是 RGB-D camera coords，也就是 MediaPipe 2D landmark + Orbbec aligned depth；
    # - 不使用 MediaPipe pose_world_landmarks 当作相机坐标；
    # - 不做 cuRobo IK，不做机器人控制，不做外参变换。
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


def _point_to_list(point: np.ndarray) -> List[float]:
    return [float(point[0]), float(point[1]), float(point[2])]


def make_wrist_udp_packet(
    frame_id: int,
    timestamp_ms: int,
    rgbd_camera_points: dict,
    lms,
    frame_name: str,
    publish_target: str,
    visibility_threshold: float,
    origin_point: Optional[np.ndarray],
    publish_delta: bool,
    teleop_scale: float,
    palm_state: Optional[dict] = None,
) -> dict:
    landmark_name = publish_target_to_landmark_name(publish_target)
    point = None
    if rgbd_camera_points is not None:
        point = rgbd_camera_points.get(landmark_name)

    visibility = None
    idx = IDX.get(landmark_name)
    if lms is not None and idx is not None and idx < len(lms):
        visibility = float(get_visibility(lms[idx]))

    valid = (
        point is not None
        and visibility is not None
        and visibility >= visibility_threshold
    )

    position_camera_m = None
    delta_camera_m = None
    target_position_m = None

    if valid:
        point = np.asarray(point, dtype=np.float32)
        position_camera_m = _point_to_list(point)

        if publish_delta:
            if origin_point is not None:
                delta = point - np.asarray(origin_point, dtype=np.float32)
                target = float(teleop_scale) * delta
                delta_camera_m = _point_to_list(delta)
                target_position_m = _point_to_list(target)
        else:
            target_position_m = position_camera_m

    packet = {
        "stamp_ms": int(timestamp_ms),
        "frame_id": int(frame_id),
        "valid": bool(valid),
        "source_frame": str(frame_name),
        "landmark_name": landmark_name,
        "position_camera_m": position_camera_m,
        "delta_camera_m": delta_camera_m,
        "target_position_m": target_position_m,
        "visibility": visibility,
        "teleop_scale": float(teleop_scale),
        "publish_delta": bool(publish_delta),
    }
    packet.update(palm_state_udp_fields(palm_state, str(frame_name)))
    return packet


def lm_to_vec3(lms: List, index: int) -> np.ndarray:
    lm = lms[index]
    return np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=np.float32)


def angle_abc_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    ba = a - b
    bc = c - b

    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)

    if nba < 1e-8 or nbc < 1e-8:
        return None

    cos_value = np.dot(ba, bc) / (nba * nbc)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_value)))


def valid_landmark(lms: List, index: int, threshold: float) -> bool:
    if lms is None or index >= len(lms):
        return False
    return get_visibility(lms[index]) >= threshold


def joint_angle(
    lms: List,
    a_idx: int,
    b_idx: int,
    c_idx: int,
    threshold: float,
) -> Optional[float]:
    if not (
        valid_landmark(lms, a_idx, threshold)
        and valid_landmark(lms, b_idx, threshold)
        and valid_landmark(lms, c_idx, threshold)
    ):
        return None

    a = lm_to_vec3(lms, a_idx)
    b = lm_to_vec3(lms, b_idx)
    c = lm_to_vec3(lms, c_idx)
    return angle_abc_deg(a, b, c)


def calculate_angles(lms: List, threshold: float) -> Dict[str, Optional[float]]:
    return {
        "left_elbow": joint_angle(lms, IDX["LEFT_SHOULDER"], IDX["LEFT_ELBOW"], IDX["LEFT_WRIST"], threshold),
        "right_elbow": joint_angle(lms, IDX["RIGHT_SHOULDER"], IDX["RIGHT_ELBOW"], IDX["RIGHT_WRIST"], threshold),

        "left_shoulder": joint_angle(lms, IDX["LEFT_ELBOW"], IDX["LEFT_SHOULDER"], IDX["LEFT_HIP"], threshold),
        "right_shoulder": joint_angle(lms, IDX["RIGHT_ELBOW"], IDX["RIGHT_SHOULDER"], IDX["RIGHT_HIP"], threshold),

        "left_hip": joint_angle(lms, IDX["LEFT_SHOULDER"], IDX["LEFT_HIP"], IDX["LEFT_KNEE"], threshold),
        "right_hip": joint_angle(lms, IDX["RIGHT_SHOULDER"], IDX["RIGHT_HIP"], IDX["RIGHT_KNEE"], threshold),

        "left_knee": joint_angle(lms, IDX["LEFT_HIP"], IDX["LEFT_KNEE"], IDX["LEFT_ANKLE"], threshold),
        "right_knee": joint_angle(lms, IDX["RIGHT_HIP"], IDX["RIGHT_KNEE"], IDX["RIGHT_ANKLE"], threshold),
    }


# ============================================================
# 绘图
# ============================================================

def draw_segmentation_mask(frame_bgr: np.ndarray, result, alpha: float = 0.35) -> np.ndarray:
    if not hasattr(result, "segmentation_masks"):
        return frame_bgr

    if result.segmentation_masks is None or len(result.segmentation_masks) == 0:
        return frame_bgr

    mask_mp = result.segmentation_masks[0]

    try:
        mask = mask_mp.numpy_view()
    except Exception:
        return frame_bgr

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
    person = mask > 0.5

    overlay = frame_bgr.copy()
    overlay[person] = (60, 180, 75)

    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)


def draw_pose_2d(
    img: np.ndarray,
    lms: List,
    mirror: bool,
    visibility_threshold: float,
    draw_names: bool,
    show_index: bool,
):
    h, w = img.shape[:2]

    # 画骨架连线
    for a, b in POSE_CONNECTIONS:
        if not valid_landmark(lms, a, visibility_threshold):
            continue
        if not valid_landmark(lms, b, visibility_threshold):
            continue

        pa = lm_to_pixel(lms[a], w, h, mirror=mirror)
        pb = lm_to_pixel(lms[b], w, h, mirror=mirror)
        cv2.line(img, pa, pb, (0, 220, 255), 2, cv2.LINE_AA)

    # 画关键点
    for i, lm in enumerate(lms):
        if get_visibility(lm) < visibility_threshold:
            continue

        p = lm_to_pixel(lm, w, h, mirror=mirror)
        cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 6, (0, 180, 255), 1, cv2.LINE_AA)

    if draw_names:
        key_names = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ]

        for name in key_names:
            i = IDX[name]
            if not valid_landmark(lms, i, visibility_threshold):
                continue

            x, y = lm_to_pixel(lms[i], w, h, mirror=mirror)
            short_name = (
                name.replace("LEFT_", "L_")
                    .replace("RIGHT_", "R_")
                    .replace("_", "")
            )
            put_text(img, short_name, (x + 6, y - 6), scale=0.42, color=(255, 255, 0), bg=True)

    if show_index:
        for i, lm in enumerate(lms):
            if get_visibility(lm) < visibility_threshold:
                continue
            x, y = lm_to_pixel(lm, w, h, mirror=mirror)
            put_text(img, str(i), (x + 4, y + 15), scale=0.35, color=(0, 255, 255), bg=True)


def draw_rgbd_valid_points(
    img: np.ndarray,
    lms: Optional[List],
    rgbd_camera_points: Optional[Dict[str, Optional[np.ndarray]]],
    mirror: bool,
    visibility_threshold: float,
):
    if lms is None or not rgbd_camera_points:
        return

    h, w = img.shape[:2]
    for name in RIGHT_ARM_RGBD_NAMES:
        point = rgbd_camera_points.get(name)
        if point is None:
            continue
        idx = IDX[name]
        if idx >= len(lms) or get_visibility(lms[idx]) < visibility_threshold:
            continue
        x, y = lm_to_pixel(lms[idx], w, h, mirror=mirror)
        cv2.circle(img, (x, y), 8, (0, 255, 0), 2, cv2.LINE_AA)
        put_text(img, "3D", (x + 8, y + 18), scale=0.38, color=(0, 255, 0), bg=True)


def draw_normalized_skeleton_canvas(
    canvas: np.ndarray,
    lms: Optional[List],
    mirror: bool,
    threshold: float,
):
    h, w = canvas.shape[:2]
    canvas[:] = (8, 8, 8)

    put_text(canvas, "2D normalized skeleton", (10, 24), scale=0.55, color=(255, 255, 255), bg=False)

    if lms is None:
        put_text(canvas, "no pose", (10, h // 2), scale=0.7, color=(80, 80, 255), bg=False)
        return

    for a, b in POSE_CONNECTIONS:
        if not valid_landmark(lms, a, threshold) or not valid_landmark(lms, b, threshold):
            continue
        pa = lm_to_pixel(lms[a], w, h, mirror=mirror)
        pb = lm_to_pixel(lms[b], w, h, mirror=mirror)
        cv2.line(canvas, pa, pb, (0, 220, 255), 2, cv2.LINE_AA)

    for lm in lms:
        if get_visibility(lm) < threshold:
            continue
        p = lm_to_pixel(lm, w, h, mirror=mirror)
        cv2.circle(canvas, p, 4, (255, 255, 255), -1, cv2.LINE_AA)


def world_landmarks_to_array(world_lms: List) -> np.ndarray:
    points = []
    for lm in world_lms:
        points.append([float(lm.x), float(lm.y), float(lm.z)])
    return np.asarray(points, dtype=np.float32)


def project_world_points(
    world_lms: List,
    view: str,
    width: int,
    height: int,
) -> np.ndarray:
    pts = world_landmarks_to_array(world_lms)

    if len(pts) >= 25:
        hip_center = 0.5 * (pts[IDX["LEFT_HIP"]] + pts[IDX["RIGHT_HIP"]])
        pts = pts - hip_center

    if view == "front":
        # x-y 视图
        xy = np.stack([pts[:, 0], -pts[:, 1]], axis=1)
    elif view == "side":
        # z-y 视图
        xy = np.stack([pts[:, 2], -pts[:, 1]], axis=1)
    else:
        raise ValueError("view must be front or side")

    max_abs = float(np.nanmax(np.abs(xy)))
    if not np.isfinite(max_abs) or max_abs < 1e-6:
        max_abs = 1.0

    scale = 0.42 * min(width, height) / max_abs
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    pixels = center + xy * scale
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return pixels.astype(np.int32)


def draw_world_canvas(
    canvas: np.ndarray,
    world_lms: Optional[List],
    title: str,
    view: str,
):
    h, w = canvas.shape[:2]
    canvas[:] = (8, 8, 8)

    put_text(canvas, title, (10, 24), scale=0.55, color=(255, 255, 255), bg=False)

    cv2.line(canvas, (w // 2, 0), (w // 2, h), (50, 50, 50), 1)
    cv2.line(canvas, (0, h // 2), (w, h // 2), (50, 50, 50), 1)

    if world_lms is None:
        put_text(canvas, "no world landmarks", (10, h // 2), scale=0.55, color=(80, 80, 255), bg=False)
        return

    pts2d = project_world_points(world_lms, view=view, width=w, height=h)

    for a, b in POSE_CONNECTIONS:
        if a >= len(pts2d) or b >= len(pts2d):
            continue
        cv2.line(canvas, tuple(pts2d[a]), tuple(pts2d[b]), (0, 220, 255), 2, cv2.LINE_AA)

    for p in pts2d:
        cv2.circle(canvas, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)

def build_info_panel(
    panel_h: int,
    panel_w: int,
    fps: float,
    lms: Optional[List],
    world_lms: Optional[List],
    rgbd_camera_points: Optional[Dict[str, Optional[np.ndarray]]],
    palm_state: Optional[dict],
    intrinsics_available: bool,
    angles: Dict[str, Optional[float]],
    threshold: float,
) -> Tuple[np.ndarray, int]:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    y = 28
    line_gap = 24

    put_text(panel, "MediaPipe Tasks PoseLandmarker", (12, y), scale=0.62, color=(0, 255, 255), bg=False)
    y += line_gap + 4

    visible_count = 0
    if lms is not None:
        visible_count = sum(1 for lm in lms if get_visibility(lm) >= threshold)

    lines = [
        f"FPS: {fps:.1f}",
        f"visible landmarks: {visible_count}/33",
        f"world landmarks: {'yes' if world_lms is not None else 'no'}",
        f"RGB-D intrinsics: {'yes' if intrinsics_available else 'unavailable'}",
    ]

    for line in lines:
        put_text(panel, line, (12, y), scale=0.52, color=(230, 230, 230), bg=False)
        y += line_gap

    y += 8
    put_text(panel, "Joint angles", (12, y), scale=0.58, color=(0, 255, 255), bg=False)
    y += line_gap

    for name, value in angles.items():
        if value is None:
            text = f"{name}: --"
            color = (120, 120, 120)
        else:
            text = f"{name}: {value:6.1f} deg"
            color = (255, 255, 255)

        put_text(panel, text, (12, y), scale=0.48, color=color, bg=False)
        y += 20

    y += 10
    put_text(panel, "Key visibility", (12, y), scale=0.58, color=(0, 255, 255), bg=False)
    y += line_gap

    key_items = [
        ("L_SHO", "LEFT_SHOULDER"),
        ("R_SHO", "RIGHT_SHOULDER"),
        ("L_ELB", "LEFT_ELBOW"),
        ("R_ELB", "RIGHT_ELBOW"),
        ("L_WRI", "LEFT_WRIST"),
        ("R_WRI", "RIGHT_WRIST"),
        ("L_HIP", "LEFT_HIP"),
        ("R_HIP", "RIGHT_HIP"),
    ]

    if lms is not None:
        for short, name in key_items:
            idx = IDX[name]
            vis = get_visibility(lms[idx])
            bar_w = int(np.clip(vis, 0.0, 1.0) * 120)
            put_text(panel, f"{short}: {vis:.2f}", (12, y), scale=0.43, color=(230, 230, 230), bg=False)
            cv2.rectangle(panel, (110, y - 11), (230, y - 2), (70, 70, 70), -1)
            cv2.rectangle(panel, (110, y - 11), (110 + bar_w, y - 2), (0, 200, 0), -1)
            y += 18

    y += 14
    put_text(panel, "Right arm world landmarks", (12, y), scale=0.55, color=(0, 255, 255), bg=False)
    y += line_gap
    put_text(
        panel,
        "MediaPipe estimated body-relative 3D, not Orbbec camera/depth",
        (12, y),
        scale=0.34,
        color=(170, 220, 220),
        bg=False,
    )
    y += 18

    if world_lms is None:
        put_text(panel, "no world landmarks", (12, y), scale=0.45, color=(120, 120, 120), bg=False)
        y += 20
    else:
        right_arm_names = [
            "RIGHT_SHOULDER",
            "RIGHT_ELBOW",
            "RIGHT_WRIST",
            "RIGHT_INDEX",
            "RIGHT_THUMB",
            "RIGHT_PINKY",
        ]
        col_name = 12
        col_x = min(125, panel_w - 250)
        col_y = min(195, panel_w - 175)
        col_z = min(265, panel_w - 95)

        put_text(panel, "point", (col_name, y), scale=0.39, color=(180, 180, 180), bg=False)
        put_text(panel, "x", (col_x, y), scale=0.39, color=(180, 180, 180), bg=False)
        put_text(panel, "y", (col_y, y), scale=0.39, color=(180, 180, 180), bg=False)
        put_text(panel, "z", (col_z, y), scale=0.39, color=(180, 180, 180), bg=False)
        y += 18

        for name in right_arm_names:
            idx = IDX[name]
            if idx >= len(world_lms):
                continue
            lm = world_lms[idx]
            short_name = (
                name.replace("RIGHT_", "R_")
                    .replace("SHOULDER", "SHO")
                    .replace("ELBOW", "ELB")
                    .replace("WRIST", "WRI")
                    .replace("INDEX", "IDX")
                    .replace("THUMB", "THU")
                    .replace("PINKY", "PIN")
            )
            put_text(panel, short_name, (col_name, y), scale=0.38, color=(255, 255, 255), bg=False)
            put_text(panel, f"{lm.x:+.3f}", (col_x - 8, y), scale=0.38, color=(255, 255, 255), bg=False)
            put_text(panel, f"{lm.y:+.3f}", (col_y - 8, y), scale=0.38, color=(255, 255, 255), bg=False)
            put_text(panel, f"{lm.z:+.3f}", (col_z - 8, y), scale=0.38, color=(255, 255, 255), bg=False)
            y += 18

        def vec(name: str) -> np.ndarray:
            lm = world_lms[IDX[name]]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        try:
            r_shoulder = vec("RIGHT_SHOULDER")
            r_elbow = vec("RIGHT_ELBOW")
            r_wrist = vec("RIGHT_WRIST")
            upper_arm_len = np.linalg.norm(r_shoulder - r_elbow)
            forearm_len = np.linalg.norm(r_elbow - r_wrist)
            shoulder_wrist_len = np.linalg.norm(r_shoulder - r_wrist)

            y += 8
            put_text(panel, "segment length", (12, y), scale=0.45, color=(0, 255, 255), bg=False)
            y += 20
            put_text(panel, f"upper arm     : {upper_arm_len:.3f} m", (12, y), scale=0.40, color=(230, 230, 230), bg=False)
            y += 18
            put_text(panel, f"forearm       : {forearm_len:.3f} m", (12, y), scale=0.40, color=(230, 230, 230), bg=False)
            y += 18
            put_text(panel, f"shoulder-wrist: {shoulder_wrist_len:.3f} m", (12, y), scale=0.40, color=(230, 230, 230), bg=False)
            y += 18
        except Exception as e:
            put_text(panel, f"length calc failed: {e}", (12, y), scale=0.36, color=(80, 80, 255), bg=False)

    y += 14
    put_text(panel, "Right arm RGB-D camera coords", (12, y), scale=0.55, color=(0, 255, 0), bg=False)
    y += line_gap
    put_text(
        panel,
        "Orbbec aligned depth + pinhole camera, unit=m",
        (12, y),
        scale=0.36,
        color=(170, 220, 170),
        bg=False,
    )
    y += 18

    if not intrinsics_available:
        put_text(panel, "camera intrinsics unavailable", (12, y), scale=0.45, color=(80, 80, 255), bg=False)
        y += 22
    else:
        rgbd_camera_points = rgbd_camera_points or {}
        valid_count = sum(1 for name in RIGHT_ARM_RGBD_NAMES if rgbd_camera_points.get(name) is not None)
        put_text(panel, f"RGB-D valid points: {valid_count}/6", (12, y), scale=0.45, color=(230, 230, 230), bg=False)
        y += 22

        col_name = 12
        col_x = min(92, panel_w - 320)
        col_y = min(190, panel_w - 220)
        col_z = min(288, panel_w - 120)
        put_text(panel, "point", (col_name, y), scale=0.38, color=(180, 180, 180), bg=False)
        put_text(panel, "X", (col_x, y), scale=0.38, color=(180, 180, 180), bg=False)
        put_text(panel, "Y", (col_y, y), scale=0.38, color=(180, 180, 180), bg=False)
        put_text(panel, "Z", (col_z, y), scale=0.38, color=(180, 180, 180), bg=False)
        y += 18

        for name in RIGHT_ARM_RGBD_NAMES:
            short_name = (
                name.replace("RIGHT_", "R_")
                    .replace("SHOULDER", "SHO")
                    .replace("ELBOW", "ELB")
                    .replace("WRIST", "WRI")
                    .replace("INDEX", "IDX")
                    .replace("THUMB", "THU")
                    .replace("PINKY", "PIN")
            )
            point = rgbd_camera_points.get(name)
            if point is None:
                put_text(panel, f"{short_name}: --", (col_name, y), scale=0.38, color=(120, 120, 120), bg=False)
            else:
                put_text(panel, short_name, (col_name, y), scale=0.38, color=(255, 255, 255), bg=False)
                put_text(panel, f"{point[0]:+.3f}", (col_x - 8, y), scale=0.38, color=(255, 255, 255), bg=False)
                put_text(panel, f"{point[1]:+.3f}", (col_y - 8, y), scale=0.38, color=(255, 255, 255), bg=False)
                put_text(panel, f"{point[2]:+.3f}", (col_z - 8, y), scale=0.38, color=(255, 255, 255), bg=False)
            y += 18

        r_shoulder = rgbd_camera_points.get("RIGHT_SHOULDER")
        r_elbow = rgbd_camera_points.get("RIGHT_ELBOW")
        r_wrist = rgbd_camera_points.get("RIGHT_WRIST")
        y += 8
        put_text(panel, "RGB-D segment length", (12, y), scale=0.45, color=(0, 255, 0), bg=False)
        y += 20
        if r_shoulder is not None and r_elbow is not None:
            put_text(panel, f"upper arm RGB-D     : {np.linalg.norm(r_shoulder - r_elbow):.3f} m", (12, y), scale=0.38, color=(230, 230, 230), bg=False)
        else:
            put_text(panel, "upper arm RGB-D     : --", (12, y), scale=0.38, color=(120, 120, 120), bg=False)
        y += 18
        if r_elbow is not None and r_wrist is not None:
            put_text(panel, f"forearm RGB-D       : {np.linalg.norm(r_elbow - r_wrist):.3f} m", (12, y), scale=0.38, color=(230, 230, 230), bg=False)
        else:
            put_text(panel, "forearm RGB-D       : --", (12, y), scale=0.38, color=(120, 120, 120), bg=False)
        y += 18
        if r_shoulder is not None and r_wrist is not None:
            put_text(panel, f"shoulder-wrist RGB-D: {np.linalg.norm(r_shoulder - r_wrist):.3f} m", (12, y), scale=0.38, color=(230, 230, 230), bg=False)
        else:
            put_text(panel, "shoulder-wrist RGB-D: --", (12, y), scale=0.38, color=(120, 120, 120), bg=False)
        y += 18

    y += 16
    put_text(panel, "Palm orientation / Hand Landmarker", (12, y), scale=0.53, color=(255, 120, 255), bg=False)
    y += line_gap
    palm_state = palm_state or empty_palm_state("hand disabled")
    palm_valid = bool(palm_state.get("valid", False))
    put_text(panel, f"Hand: {palm_state.get('handedness', 'none')}", (12, y), scale=0.43, color=(230, 230, 230), bg=False)
    y += 18
    score = palm_state.get("handedness_score")
    put_text(panel, f"Hand score: {'--' if score is None else f'{score:.3f}'}", (12, y), scale=0.43, color=(230, 230, 230), bg=False)
    y += 18
    put_text(
        panel,
        f"Palm frame: {'valid' if palm_valid else 'invalid'} ({palm_state.get('reason', '--')})",
        (12, y),
        scale=0.40,
        color=(0, 255, 0) if palm_valid else (80, 80, 255),
        bg=False,
    )
    y += 20
    normal = palm_state.get("palm_normal_camera")
    if normal is not None:
        put_text(panel, f"normal cam: {normal[0]:+.3f} {normal[1]:+.3f} {normal[2]:+.3f}", (12, y), scale=0.39, color=(230, 230, 230), bg=False)
    else:
        put_text(panel, "normal cam: --", (12, y), scale=0.39, color=(120, 120, 120), bg=False)
    y += 18
    angle = palm_state.get("angle_from_origin_deg")
    put_text(panel, f"angle from origin: {'--' if angle is None else f'{angle:.1f} deg'}", (12, y), scale=0.39, color=(230, 230, 230), bg=False)
    y += 18
    q_delta = palm_state.get("q_delta_palm_wxyz")
    if q_delta is not None:
        put_text(panel, f"q_delta wxyz: {q_delta[0]:+.3f} {q_delta[1]:+.3f}", (12, y), scale=0.36, color=(230, 230, 230), bg=False)
        y += 16
        put_text(panel, f"              {q_delta[2]:+.3f} {q_delta[3]:+.3f}", (12, y), scale=0.36, color=(230, 230, 230), bg=False)
    else:
        put_text(panel, "q_delta wxyz: --", (12, y), scale=0.36, color=(120, 120, 120), bg=False)
    y += 20
    delta_R = palm_state.get("delta_R_camera_palm")
    put_text(panel, "Delta R camera palm:", (12, y), scale=0.39, color=(255, 120, 255), bg=False)
    y += 18
    if delta_R is not None:
        for row in np.asarray(delta_R):
            put_text(panel, f"[{row[0]:+.2f} {row[1]:+.2f} {row[2]:+.2f}]", (18, y), scale=0.36, color=(230, 230, 230), bg=False)
            y += 16
    else:
        put_text(panel, "--", (18, y), scale=0.36, color=(120, 120, 120), bg=False)
        y += 16
    y += 4
    put_text(panel, "Palm orientation in camera frame,", (12, y), scale=0.34, color=(180, 180, 180), bg=False)
    y += 15
    put_text(panel, "not anatomical wrist joint rotation", (12, y), scale=0.34, color=(180, 180, 180), bg=False)
    y += 15
    put_text(panel, "camera/world/EE alignment is in UDP bridge", (12, y), scale=0.34, color=(180, 180, 180), bg=False)
    y += 18

    return panel, y + 24


# ============================================================
# CSV
# ============================================================

def init_csv(csv_path: str):
    if not csv_path:
        return None, None

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    writer.writerow([
        "frame_id",
        "timestamp_ms",
        "landmark_id",
        "landmark_name",
        "x_norm",
        "y_norm",
        "z_norm",
        "visibility",
        "presence",
        "world_x_m",
        "world_y_m",
        "world_z_m",
        "world_visibility",
        "world_presence",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "camera_depth_valid",
    ])

    return csv_file, writer


def write_csv(
    writer,
    frame_id: int,
    timestamp_ms: int,
    lms: Optional[List],
    world_lms: Optional[List],
    rgbd_camera_points: Optional[Dict[str, Optional[np.ndarray]]] = None,
):
    if writer is None or lms is None:
        return

    for i, lm in enumerate(lms):
        if world_lms is not None and i < len(world_lms):
            wlm = world_lms[i]
            wx, wy, wz = wlm.x, wlm.y, wlm.z
            wvis = get_visibility(wlm)
            wp = get_presence(wlm)
        else:
            wx, wy, wz, wvis, wp = "", "", "", "", ""

        camera_point = None
        camera_depth_valid = ""
        if rgbd_camera_points is not None and LANDMARK_NAMES[i] in rgbd_camera_points:
            camera_point = rgbd_camera_points.get(LANDMARK_NAMES[i])
            camera_depth_valid = 1 if camera_point is not None else 0

        if camera_point is not None:
            camera_x, camera_y, camera_z = camera_point.tolist()
        else:
            camera_x, camera_y, camera_z = "", "", ""

        writer.writerow([
            frame_id,
            timestamp_ms,
            i,
            LANDMARK_NAMES[i],
            lm.x,
            lm.y,
            lm.z,
            get_visibility(lm),
            get_presence(lm),
            wx,
            wy,
            wz,
            wvis,
            wp,
            camera_x,
            camera_y,
            camera_z,
            camera_depth_valid,
        ])


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_args()
    # 把命令行参数整理成明确布尔变量，后面不要反复读 args。
    mirror = not args.no_mirror
    draw_names = not args.no_draw_names
    segmentation = not args.no_segmentation
    hand_enabled = args.enable_hand and not args.disable_hand
    draw_hand = args.draw_hand and not args.no_draw_hand
    # 兼容 --model-complexity：保持旧命令习惯，同时内部仍使用 Tasks 的模型名。
    if args.model_complexity is not None:
        args.model = {0: "lite", 1: "full", 2: "heavy"}[args.model_complexity]

    # MediaPipe Tasks 模型加载：这里和 webcam 版本一致。
    model_path = ensure_model(
        model_name=args.model,
        model_path=args.model_path,
        models_dir=args.models_dir,
    )

    print("[INFO] Using model:", model_path)
    hand_model_path = None
    if hand_enabled:
        try:
            hand_model_path = ensure_hand_model(args.hand_model_path, args.hand_models_dir)
            print("[INFO] Using hand model:", hand_model_path)
        except Exception as exc:
            print(f"[WARN] HandLandmarker model unavailable, disabling hand detection: {exc}")
            hand_enabled = False

    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    HandLandmarker = getattr(vision, "HandLandmarker", None)
    HandLandmarkerOptions = getattr(vision, "HandLandmarkerOptions", None)
    if hand_enabled and (HandLandmarker is None or HandLandmarkerOptions is None):
        print("[WARN] MediaPipe HandLandmarker API unavailable, disabling hand detection.")
        hand_enabled = False
    VisionRunningMode = vision.RunningMode

    # PoseLandmarker 仍使用 VIDEO 模式：
    # 主循环每帧传入单调递增 timestamp_ms，MediaPipe 内部做视频追踪。
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        output_segmentation_masks=segmentation,
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

    # 这里是 Orbbec 版本替代 webcam 版本的关键位置：
    # 不再创建 cv2.VideoCapture，而是创建 OrbbecCamera 包装器。
    camera = OrbbecCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_depth=not args.no_depth,
        align_mode=args.orbbec_align,
        enable_sync=not args.disable_frame_sync,
    ).start()

    csv_file, csv_writer = init_csv(args.csv)

    if args.udp_publish:
        udp_pub = UdpJsonPublisher(args.udp_host, args.udp_port)
        print(f"[INFO] UDP publishing enabled: {args.udp_host}:{args.udp_port}")
    else:
        udp_pub = None

    # wrist_origin 是 publish_delta 模式的零点。
    # 第一帧目标点有效时自动设置；也可以按 reset-origin-key 手动重置。
    wrist_origin = None
    last_udp_warn_t = 0.0
    udp_last_valid = False

    # paused=True 时，不继续从相机取新帧，而是复用 last_frame_bgr / last_result。
    paused = False
    frame_id = 0
    fps_smooth = 0.0
    last_t = time.time()
    start_monotonic = time.monotonic()

    last_frame_bgr = None
    last_depth_mm = None
    last_result = None
    last_hand_result = None
    palm_runtime = {"R0": None, "R_prev": None, "R_filtered": None}
    latest_valid_palm_R = None

    print("[INFO] Started.")
    print(
        "[INFO] Keys: q/ESC quit, p pause/resume, w/s scroll panel, "
        f"c screenshot, {args.reset_origin_key} reset UDP origin, "
        f"{args.palm_reset_key} reset palm orientation origin"
    )
    print("[INFO] Mouse wheel or w/s: scroll right information panel")
    print(f"[INFO] mirror={mirror}, segmentation={segmentation}, draw_names={draw_names}, hand={hand_enabled}")

    window_name = "MediaPipe PoseLandmarker V2 Webcam Visualizer"
    panel_scroll = {"offset": 0, "max": 0}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEWHEEL:
            return
        wheel_delta = cv2.getMouseWheelDelta(flags)
        if wheel_delta == 0:
            return
        step = 70
        direction = -1 if wheel_delta > 0 else 1
        panel_scroll["offset"] = int(
            np.clip(
                panel_scroll["offset"] + direction * step,
                0,
                panel_scroll["max"],
            )
        )

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    # MediaPipe landmarker 生命周期由 ExitStack 管理；
    # HandLandmarker 初始化失败时不影响 PoseLandmarker / Orbbec / UDP 主流程。
    with ExitStack() as stack:
        landmarker = stack.enter_context(PoseLandmarker.create_from_options(options))
        hand_landmarker = None
        if hand_enabled:
            try:
                hand_landmarker = stack.enter_context(HandLandmarker.create_from_options(hand_options))
            except Exception as exc:
                print(f"[WARN] HandLandmarker initialization failed, disabling hand detection: {exc}")
                hand_enabled = False

        while True:
            if not paused:
                # Orbbec read() 返回：
                # - ok: 本次是否拿到可用彩色帧；
                # - frame_bgr: OpenCV BGR 彩色图；
                # - depth_mm: 深度图，当前缓存但不改变既有 MediaPipe 可视化逻辑。
                ok, frame_bgr, depth_mm = camera.read()
                if not ok:
                    print("[WARN] failed to read frame from Orbbec camera")
                    continue

                frame_id += 1
                last_frame_bgr = frame_bgr.copy()
                last_depth_mm = depth_mm

                # OpenCV 是 BGR，MediaPipe Image 需要 SRGB/RGB
                # 这一步和 webcam 版本完全一致：只要上游给 frame_bgr，
                # 后续 MediaPipe 调用不用关心图像来自 USB webcam 还是 Orbbec。
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = np.ascontiguousarray(frame_rgb)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # VIDEO 模式必须提供单调递增 timestamp_ms
                timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                last_result = result
                hand_result = last_hand_result
                if hand_enabled and hand_landmarker is not None:
                    hand_every = max(1, int(args.hand_every_n_frames))
                    if frame_id % hand_every == 0:
                        try:
                            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                            last_hand_result = hand_result
                        except Exception as exc:
                            print(f"[WARN] HandLandmarker detect failed: {exc}")
                            hand_result = last_hand_result

                # 平滑 FPS 显示，避免每帧瞬时 FPS 抖动太明显。
                now_t = time.time()
                dt = now_t - last_t
                last_t = now_t

                if dt > 1e-6:
                    fps = 1.0 / dt
                    if fps_smooth <= 0.0:
                        fps_smooth = fps
                    else:
                        fps_smooth = 0.9 * fps_smooth + 0.1 * fps
            else:
                if last_frame_bgr is None:
                    continue
                # 暂停状态下沿用上一帧图像和上一帧 MediaPipe result，
                # 这样可以冻结画面，同时仍能滚动右侧信息栏/保存截图。
                frame_bgr = last_frame_bgr.copy()
                depth_mm = last_depth_mm
                result = last_result
                hand_result = last_hand_result
                timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)

            vis_frame = frame_bgr.copy()

            # 取第一人
            lms = None
            world_lms = None
            rgbd_camera_points = {}
            intrinsics = camera.get_color_intrinsics()
            intrinsics_available = intrinsics is not None
            angles = {}
            selected_hand = select_hand_result(hand_result, args.handedness) if hand_enabled else None
            hand_lms = selected_hand["hand_lms"] if selected_hand is not None else None
            hand_camera_points = compute_hand_camera_points(
                hand_lms=hand_lms,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                frame_w=frame_bgr.shape[1],
                frame_h=frame_bgr.shape[0],
                search_radius=args.palm_depth_radius,
            ) if hand_lms is not None else {name: None for name in HAND_LANDMARK_NAMES}
            palm_frame_camera = compute_palm_frame_from_points(hand_camera_points) if hand_lms is not None else {"valid": False, "reason": "no selected hand"}
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

            if result is not None and result.pose_landmarks:
                # 当前脚本只取第一个人体姿态。
                # 如果 num_poses > 1，这里仍然只展示第 0 个结果，保持原可视化结构不变。
                lms = result.pose_landmarks[0]

                if result.pose_world_landmarks:
                    world_lms = result.pose_world_landmarks[0]

                rgbd_camera_points = compute_rgbd_camera_landmarks(
                    lms=lms,
                    depth_mm=depth_mm,
                    intrinsics=intrinsics,
                    frame_w=frame_bgr.shape[1],
                    frame_h=frame_bgr.shape[0],
                    visibility_threshold=args.visibility_threshold,
                )

                if segmentation:
                    vis_frame = draw_segmentation_mask(vis_frame, result)

                # 先镜像图像，再按镜像坐标画点
                # draw_pose_2d 里也会根据 mirror 翻转 landmark 的 x 坐标。
                if mirror:
                    vis_frame = cv2.flip(vis_frame, 1)

                draw_pose_2d(
                    img=vis_frame,
                    lms=lms,
                    mirror=mirror,
                    visibility_threshold=args.visibility_threshold,
                    draw_names=draw_names,
                    show_index=args.show_index,
                )
                draw_rgbd_valid_points(
                    img=vis_frame,
                    lms=lms,
                    rgbd_camera_points=rgbd_camera_points,
                    mirror=mirror,
                    visibility_threshold=args.visibility_threshold,
                )
                if draw_hand and hand_lms is not None:
                    draw_hand_landmarks_2d(
                        img=vis_frame,
                        hand_lms=hand_lms,
                        mirror=mirror,
                        draw_names=draw_names,
                    )
                if args.draw_palm_frame:
                    draw_palm_frame_axes(
                        img=vis_frame,
                        palm_frame_camera=palm_frame_camera,
                        intrinsics=intrinsics,
                        mirror=mirror,
                        axis_length_m=args.palm_frame_axis_length,
                    )

                angles = calculate_angles(lms, args.visibility_threshold)

                # CSV 保留原有 MediaPipe 字段，并在末尾追加 RGB-D camera coords 字段。
                # 只有成功反投影的关键点会写 camera_x/y/z，其他 landmark 留空。
                if not paused:
                    write_csv(csv_writer, frame_id, timestamp_ms, lms, world_lms, rgbd_camera_points)

            else:
                if mirror:
                    vis_frame = cv2.flip(vis_frame, 1)

                put_text(
                    vis_frame,
                    "No pose detected",
                    (16, 92),
                    scale=0.75,
                    color=(0, 0, 255),
                    bg=True,
                )
                if draw_hand and hand_lms is not None:
                    draw_hand_landmarks_2d(
                        img=vis_frame,
                        hand_lms=hand_lms,
                        mirror=mirror,
                        draw_names=draw_names,
                    )
                if args.draw_palm_frame:
                    draw_palm_frame_axes(
                        img=vis_frame,
                        palm_frame_camera=palm_frame_camera,
                        intrinsics=intrinsics,
                        mirror=mirror,
                        axis_length_m=args.palm_frame_axis_length,
                    )

            if udp_pub is not None:
                # UDP 发布的是 RGB-D camera coords：Orbbec aligned depth + color 像素反投影。
                # 这里故意不使用 MediaPipe pose_world_landmarks，也不在本脚本里做 IK/控制。
                target_landmark_name = publish_target_to_landmark_name(args.publish_target)
                current_point = None
                if rgbd_camera_points is not None:
                    current_point = rgbd_camera_points.get(target_landmark_name)

                if current_point is not None and wrist_origin is None:
                    wrist_origin = np.asarray(current_point, dtype=np.float32).copy()
                    print(f"[INFO] Auto set UDP origin ({target_landmark_name}): {wrist_origin.tolist()}")

                udp_packet = make_wrist_udp_packet(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    rgbd_camera_points=rgbd_camera_points,
                    lms=lms,
                    frame_name=args.udp_frame,
                    publish_target=args.publish_target,
                    visibility_threshold=args.visibility_threshold,
                    origin_point=wrist_origin,
                    publish_delta=args.publish_delta,
                    teleop_scale=args.teleop_scale,
                    palm_state=palm_state,
                )
                udp_pub.publish(udp_packet)
                udp_last_valid = bool(udp_packet.get("valid", False))
            else:
                udp_last_valid = False

            status = "PAUSED" if paused else "RUNNING"
            # 左上角状态条只显示全局运行状态，不承载复杂调试信息；
            # 详细角度、visibility、world landmark 等都放在右侧滚动面板。
            put_text(
                vis_frame,
                f"{status} | FPS {fps_smooth:.1f} | model={args.model} | mirror={mirror} | seg={segmentation}",
                (16, 32),
                scale=0.62,
                color=(0, 255, 255),
                bg=True,
            )
            udp_status_text = "UDP: off"
            if udp_pub is not None:
                udp_status_text = f"UDP: {args.udp_host}:{args.udp_port} valid={udp_last_valid}"
            put_text(
                vis_frame,
                udp_status_text,
                (16, 58),
                scale=0.50,
                color=(0, 255, 0) if udp_pub is not None else (160, 160, 160),
                bg=True,
            )

            # 右侧信息面板
            # build_info_panel 负责文字信息，下面再把 skeleton/world 小视图接在文字之后。
            # panel_full_h 是虚拟高度，最终只裁剪出一段 panel_view 拼到画面右侧。
            h, w = vis_frame.shape[:2]
            panel_w = max(420, args.panel_width)
            panel_full_h = max(h, args.panel_scroll_height)
            panel, info_bottom_y = build_info_panel(
                panel_h=panel_full_h,
                panel_w=panel_w,
                fps=fps_smooth,
                lms=lms,
                world_lms=world_lms,
                rgbd_camera_points=rgbd_camera_points,
                palm_state=palm_state,
                intrinsics_available=intrinsics_available,
                angles=angles,
                threshold=args.visibility_threshold,
            )
            # 插入 2D skeleton canvas
            # info_bottom_y 来自 build_info_panel，可避免 skeleton 盖住状态文字。
            margin = 10
            view_w = panel_w - 2 * margin
            skeleton_h = max(160, int(h * 0.28))
            world_h = max(150, int(h * 0.25))

            skeleton_y0 = info_bottom_y + 24
            world_y0 = skeleton_y0 + skeleton_h + 12
            required_panel_h = world_y0 + world_h + 20
            # 如果文字 + skeleton + world panels 超出初始虚拟面板高度，
            # 这里动态扩展右侧面板，滚动条范围随后会自动变大。
            if required_panel_h > panel.shape[0]:
                extra_h = required_panel_h - panel.shape[0]
                extra_panel = np.zeros((extra_h, panel_w, 3), dtype=np.uint8)
                extra_panel[:] = (20, 20, 20)
                panel = np.vstack([panel, extra_panel])
                panel_full_h = panel.shape[0]

            # normalized skeleton 使用 MediaPipe normalized landmarks；
            # 这里显示的是 2D 归一化人体骨架，不是相机深度坐标。
            if skeleton_y0 + skeleton_h < panel_full_h:
                skeleton_canvas = np.zeros((skeleton_h, view_w, 3), dtype=np.uint8)
                draw_normalized_skeleton_canvas(
                    canvas=skeleton_canvas,
                    lms=lms,
                    mirror=mirror,
                    threshold=args.visibility_threshold,
                )
                panel[
                    skeleton_y0:skeleton_y0 + skeleton_h,
                    margin:margin + view_w,
                ] = skeleton_canvas

            # world front/side 使用 MediaPipe pose_world_landmarks，
            # 它是 MediaPipe 估计的人体相对三维坐标，不是 Orbbec depth map。
            if world_y0 + world_h < panel_full_h:
                half_w = (view_w - 8) // 2
                front_canvas = np.zeros((world_h, half_w, 3), dtype=np.uint8)
                side_canvas = np.zeros((world_h, half_w, 3), dtype=np.uint8)

                draw_world_canvas(
                    canvas=front_canvas,
                    world_lms=world_lms,
                    title="World front x-y",
                    view="front",
                )
                draw_world_canvas(
                    canvas=side_canvas,
                    world_lms=world_lms,
                    title="World side z-y",
                    view="side",
                )

                panel[
                    world_y0:world_y0 + world_h,
                    margin:margin + half_w,
                ] = front_canvas

                panel[
                    world_y0:world_y0 + world_h,
                    margin + half_w + 8:margin + 2 * half_w + 8,
                ] = side_canvas

            # 根据当前滚动偏移，从右侧虚拟长面板中裁剪出与主画面同高的一段。
            panel_scroll["max"] = max(0, panel_full_h - h)
            panel_scroll["offset"] = int(
                np.clip(panel_scroll["offset"], 0, panel_scroll["max"])
            )
            scroll_y = panel_scroll["offset"]
            panel_view = panel[scroll_y:scroll_y + h, :].copy()

            # 画一个简易滚动条，提示当前右侧状态栏的位置。
            if panel_scroll["max"] > 0:
                track_x = panel_w - 8
                cv2.rectangle(panel_view, (track_x, 4), (track_x + 4, h - 4), (70, 70, 70), -1)
                thumb_h = max(36, int(h * h / panel_full_h))
                thumb_y = 4 + int((h - 8 - thumb_h) * scroll_y / panel_scroll["max"])
                cv2.rectangle(
                    panel_view,
                    (track_x - 1, thumb_y),
                    (track_x + 5, thumb_y + thumb_h),
                    (0, 220, 255),
                    -1,
                )
                put_text(
                    panel_view,
                    f"panel scroll {scroll_y}/{panel_scroll['max']}",
                    (12, h - 14),
                    scale=0.48,
                    color=(0, 255, 255),
                    bg=True,
                )

            final_vis = np.hstack([vis_frame, panel_view])

            # display_scale 是最终合成图的显示缩放，不影响 MediaPipe 输入分辨率。
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

            # w/s 或方向键用于滚动右侧虚拟面板。
            if key in (ord("w"), 82):
                panel_scroll["offset"] = int(
                    np.clip(panel_scroll["offset"] - 70, 0, panel_scroll["max"])
                )

            if key in (ord("s"), 84):
                panel_scroll["offset"] = int(
                    np.clip(panel_scroll["offset"] + 70, 0, panel_scroll["max"])
                )

            if key == ord("p"):
                paused = not paused
                print("[INFO] paused =", paused)

            reset_key = (args.reset_origin_key or "r").lower()[:1]
            if reset_key and key == ord(reset_key):
                target_landmark_name = publish_target_to_landmark_name(args.publish_target)
                current_point = None
                if rgbd_camera_points is not None:
                    current_point = rgbd_camera_points.get(target_landmark_name)

                if current_point is not None:
                    wrist_origin = np.asarray(current_point, dtype=np.float32).copy()
                    print(f"[INFO] Reset wrist origin ({target_landmark_name}): {wrist_origin.tolist()}")
                else:
                    now_t = time.time()
                    if now_t - last_udp_warn_t > 0.5:
                        print(f"[WARN] Cannot reset wrist origin: {target_landmark_name} is invalid")
                        last_udp_warn_t = now_t

            palm_reset_key = (args.palm_reset_key or "o").lower()[:1]
            if palm_reset_key and key == ord(palm_reset_key):
                if latest_valid_palm_R is not None:
                    palm_runtime["R0"] = latest_valid_palm_R.copy()
                    print("[INFO] Reset palm orientation origin.")
                else:
                    print("[WARN] Cannot reset palm orientation origin: no valid palm frame")

            # 截图保存的是最终合成窗口：左侧彩色姿态画面 + 右侧当前滚动位置的状态栏。
            if key == ord("c"):
                os.makedirs("screenshots", exist_ok=True)
                save_path = os.path.join("screenshots", f"pose_v2_{now_string()}.jpg")
                cv2.imwrite(save_path, final_vis)
                print("[INFO] saved screenshot:", save_path)

    camera.stop()
    if udp_pub is not None:
        udp_pub.close()
    cv2.destroyAllWindows()

    if csv_file is not None:
        csv_file.close()
        print("[INFO] saved csv:", args.csv)


if __name__ == "__main__":
    main()
