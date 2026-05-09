#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MediaPipe Pose Landmarker V2 Webcam Visualizer

适配新版 MediaPipe Tasks API：
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

功能：
1. 调用 webcam 实时提取人体 33 个 Pose landmarks
2. 绘制 2D 骨架
3. 绘制关键点名称
4. 绘制 segmentation mask
5. 显示 world landmarks 的 front / side 简化视图
6. 显示左右肩、肘、髋、膝角度
7. 显示 FPS、关键点 visibility
8. 可保存截图
9. 可保存 landmarks 到 CSV

按键：
    q / ESC : 退出
    p       : 暂停 / 继续
    w / s   : 上下滚动右侧信息栏
    c       : 保存截图

示例：
    python mediapipe_pose_webcam_visualizer_v2.py --camera 0 --width 1280 --height 720 --model full
"""

import argparse
import csv
import math
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MediaPipe Pose Landmark index
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


# ============================================================
# 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MediaPipe Tasks API PoseLandmarker webcam visualizer"
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--model",
        type=str,
        default="full",
        choices=["lite", "full", "heavy"],
        help="pose landmarker model type",
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

    parser.add_argument("--draw-names", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--segmentation", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--no-draw-names", action="store_true", default=False)
    parser.add_argument("--no-segmentation", action="store_true", default=False)
    parser.add_argument("--no-mirror", action="store_true", default=False)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--show-index", action="store_true")
    parser.add_argument(
        "--panel-width",
        type=int,
        default=460,
        help="right information panel width in pixels",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
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
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"指定的模型文件不存在: {model_path}")
        return model_path

    os.makedirs(models_dir, exist_ok=True)

    local_path = os.path.join(models_dir, MODEL_FILES[model_name])
    if os.path.exists(local_path):
        return local_path

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
    ])

    return csv_file, writer


def write_csv(
    writer,
    frame_id: int,
    timestamp_ms: int,
    lms: Optional[List],
    world_lms: Optional[List],
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
        ])


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_args()
    mirror = not args.no_mirror
    draw_names = not args.no_draw_names
    segmentation = not args.no_segmentation

    model_path = ensure_model(
        model_name=args.model,
        model_path=args.model_path,
        models_dir=args.models_dir,
    )

    print("[INFO] Using model:", model_path)

    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        output_segmentation_masks=segmentation,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开 camera index={args.camera}。"
            "请先执行 ls /dev/video*，然后尝试 --camera 1 或 --camera 2。"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # 有些 USB 摄像头用 MJPG 更容易跑到 720p/30fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    csv_file, csv_writer = init_csv(args.csv)

    paused = False
    frame_id = 0
    fps_smooth = 0.0
    last_t = time.time()
    start_monotonic = time.monotonic()

    last_frame_bgr = None
    last_result = None

    print("[INFO] Started.")
    print("[INFO] Keys: q/ESC quit, p pause/resume, w/s scroll panel, c screenshot")
    print("[INFO] Mouse wheel or w/s: scroll right information panel")
    print(f"[INFO] mirror={mirror}, segmentation={segmentation}, draw_names={draw_names}")

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

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            if not paused:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("[WARN] failed to read frame from camera")
                    break

                frame_id += 1
                last_frame_bgr = frame_bgr.copy()

                # OpenCV 是 BGR，MediaPipe Image 需要 SRGB/RGB
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
                frame_bgr = last_frame_bgr.copy()
                result = last_result
                timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)

            vis_frame = frame_bgr.copy()

            # 取第一人
            lms = None
            world_lms = None
            angles = {}

            if result is not None and result.pose_landmarks:
                lms = result.pose_landmarks[0]

                if result.pose_world_landmarks:
                    world_lms = result.pose_world_landmarks[0]

                if segmentation:
                    vis_frame = draw_segmentation_mask(vis_frame, result)

                # 先镜像图像，再按镜像坐标画点
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

                angles = calculate_angles(lms, args.visibility_threshold)

                if not paused:
                    write_csv(csv_writer, frame_id, timestamp_ms, lms, world_lms)

            else:
                if mirror:
                    vis_frame = cv2.flip(vis_frame, 1)

                put_text(
                    vis_frame,
                    "No pose detected",
                    (16, 70),
                    scale=0.75,
                    color=(0, 0, 255),
                    bg=True,
                )

            status = "PAUSED" if paused else "RUNNING"
            put_text(
                vis_frame,
                f"{status} | FPS {fps_smooth:.1f} | model={args.model} | mirror={mirror} | seg={segmentation}",
                (16, 32),
                scale=0.62,
                color=(0, 255, 255),
                bg=True,
            )

            # 右侧信息面板
            h, w = vis_frame.shape[:2]
            panel_w = max(420, args.panel_width)
            panel_full_h = max(h, args.panel_scroll_height)
            panel, info_bottom_y = build_info_panel(
                panel_h=panel_full_h,
                panel_w=panel_w,
                fps=fps_smooth,
                lms=lms,
                world_lms=world_lms,
                angles=angles,
                threshold=args.visibility_threshold,
            )
            # 插入 2D skeleton canvas
            margin = 10
            view_w = panel_w - 2 * margin
            skeleton_h = max(160, int(h * 0.28))
            world_h = max(150, int(h * 0.25))

            skeleton_y0 = info_bottom_y + 24
            world_y0 = skeleton_y0 + skeleton_h + 12
            required_panel_h = world_y0 + world_h + 20
            if required_panel_h > panel.shape[0]:
                extra_h = required_panel_h - panel.shape[0]
                extra_panel = np.zeros((extra_h, panel_w, 3), dtype=np.uint8)
                extra_panel[:] = (20, 20, 20)
                panel = np.vstack([panel, extra_panel])
                panel_full_h = panel.shape[0]

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

            panel_scroll["max"] = max(0, panel_full_h - h)
            panel_scroll["offset"] = int(
                np.clip(panel_scroll["offset"], 0, panel_scroll["max"])
            )
            scroll_y = panel_scroll["offset"]
            panel_view = panel[scroll_y:scroll_y + h, :].copy()

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

            if key == ord("c"):
                os.makedirs("screenshots", exist_ok=True)
                save_path = os.path.join("screenshots", f"pose_v2_{now_string()}.jpg")
                cv2.imwrite(save_path, final_vis)
                print("[INFO] saved screenshot:", save_path)

    cap.release()
    cv2.destroyAllWindows()

    if csv_file is not None:
        csv_file.close()
        print("[INFO] saved csv:", args.csv)


if __name__ == "__main__":
    main()
