#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Teleop UDP protocol adapter from camera-frame targets to world-frame targets.

This bridge sits between:

    MediaPipe Right Arm / Right Hand Orbbec Teleop Publisher
        -> camera-frame UDP packet on 127.0.0.1:5557
        -> this bridge
        -> world-frame UDP packet on 127.0.0.1:5558
        -> cuRobo interactive IK viewer

This script is not an IK solver. It does not open an Orbbec camera, does not run
MediaPipe, does not import OpenCV, and does not import cuRobo. It only adapts the
teleoperation UDP protocol.

Coordinate notes:
- camera coords originate at the Orbbec optical center;
- absolute camera points use p_world = R_world_camera @ p_camera + t_world_camera;
- vectors such as delta_camera_m only use rotation;
- palm orientation is converted from camera frame to world frame;
- palm frame is a visual hand-palm frame, not a robot end-effector frame.

Default R_world_camera:

    X_world =  Z_camera
    Y_world = -X_camera
    Z_world = -Y_camera

Recommended no-double-scale setup:

    python camera_to_world_udp_bridge.py --output-mode curobo_delta --no-apply-bridge-scale
    python curobo_udp_viewer.py --target-source udp --udp-port 5558 --mapping-mode delta --teleop-scale 0.5

In that setup, this bridge outputs raw human delta in world axes, while the cuRobo
viewer owns human_origin, robot_ee_origin, viewer teleop_scale, filtering, clamp,
and IK. If a future calibration provides R_palm_ee, robot EE orientation can be
computed downstream as R_world_ee = R_world_palm @ R_palm_ee.
"""

import argparse
import json
import socket
import time
from typing import Optional, Tuple

import numpy as np


BRIDGE_VERSION = "teleop_protocol_adapter_v2"
DEFAULT_ROTATION_PRESET = "camera_z_forward_y_down_to_world_x_forward_z_up"

DEFAULT_R_WORLD_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

# R_palm_ee expresses the robot EE frame axes in the visual palm frame.
# The bridge computes R_world_ee_target = R_world_palm @ R_palm_ee.
# If your convention is the opposite, edit this matrix to its transpose.
DEFAULT_R_PALM_EE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Teleop UDP camera/world protocol adapter")
    parser.add_argument("--input-host", type=str, default="127.0.0.1")
    parser.add_argument("--input-port", type=int, default=5557)
    parser.add_argument("--output-host", type=str, default="127.0.0.1")
    parser.add_argument("--output-port", type=int, default=5558)

    parser.add_argument(
        "--rotation-preset",
        type=str,
        default=DEFAULT_ROTATION_PRESET,
        choices=[DEFAULT_ROTATION_PRESET],
    )
    parser.add_argument(
        "--rotation-matrix",
        type=str,
        default="",
        help='nine row-major comma-separated numbers, e.g. "0,0,1,-1,0,0,0,-1,0"',
    )
    parser.add_argument(
        "--translation",
        type=str,
        default="0,0,0",
        help='three comma-separated numbers for t_world_camera, e.g. "0.2,0,0.5"',
    )
    parser.add_argument(
        "--output-mode",
        choices=["curobo_delta", "absolute_passthrough"],
        default="curobo_delta",
        help="curobo_delta publishes world human delta; absolute_passthrough is for debugging",
    )
    parser.add_argument(
        "--apply-bridge-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply --bridge-scale inside this bridge; off by default to avoid double scaling",
    )
    parser.add_argument("--bridge-scale", type=float, default=1.0)
    parser.add_argument(
        "--elbow-swing-delta-lower-deg",
        type=float,
        default=0.0,
        help="lower band-pass threshold for signed elbow swing angle changes; smaller changes are treated as jitter",
    )
    parser.add_argument(
        "--elbow-swing-delta-upper-deg",
        type=float,
        default=180.0,
        help="upper band-pass threshold for signed elbow swing angle changes; larger changes are treated as spikes",
    )
    parser.add_argument(
        "--elbow-swing-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="open a 0-180 degree slider and publish its value as the human elbow swing angle",
    )
    parser.add_argument(
        "--elbow-swing-test-initial-deg",
        type=float,
        default=0.0,
        help="initial slider value for --elbow-swing-test",
    )
    parser.add_argument(
        "--validate-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate R_world_camera at startup",
    )
    # Deprecated compatibility options from the first bridge version. They are
    # accepted so old shell commands do not fail, but the v2 protocol is driven
    # by --output-mode and --bridge-scale.
    parser.add_argument("--target-mode", choices=["delta", "absolute"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--teleop-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def normalize_legacy_args(args):
    if args.target_mode is not None:
        args.output_mode = "curobo_delta" if args.target_mode == "delta" else "absolute_passthrough"
        print(f"[WARN] --target-mode is deprecated; using --output-mode {args.output_mode}")

    if args.teleop_scale is not None:
        args.bridge_scale = float(args.teleop_scale)
        args.apply_bridge_scale = True
        print("[WARN] --teleop-scale is deprecated; using it as --bridge-scale with --apply-bridge-scale")


def parse_float_list(text: str, expected_len: int, name: str) -> np.ndarray:
    try:
        values = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric comma-separated values") from exc

    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(values)}")

    arr = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def load_rotation(args) -> np.ndarray:
    if args.rotation_matrix:
        return parse_float_list(args.rotation_matrix, 9, "--rotation-matrix").reshape(3, 3)
    return DEFAULT_R_WORLD_CAMERA.copy()


def load_palm_to_ee_rotation() -> np.ndarray:
    R_palm_ee = DEFAULT_R_PALM_EE.copy()
    ok, msg = validate_rotation_matrix(R_palm_ee)
    if not ok:
        raise SystemExit(f"[ERROR] Invalid R_palm_ee: {msg}")
    return R_palm_ee.astype(np.float32)


class ElbowSwingTestSlider:
    """Small stdlib Tk slider used only for manual elbow-swing protocol tests."""

    def __init__(self, initial_deg: float):
        import tkinter as tk

        self._tk = tk
        self._closed = False
        self._last_value = float(np.clip(initial_deg, 0.0, 180.0))

        self._root = tk.Tk()
        self._root.title("Elbow swing test")
        self._value = tk.DoubleVar(value=self._last_value)
        tk.Label(
            self._root,
            text="Human elbow swing test angle (deg)",
            padx=12,
            pady=8,
        ).pack()
        self._scale = tk.Scale(
            self._root,
            from_=0.0,
            to=180.0,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            length=420,
            variable=self._value,
        )
        self._scale.pack(padx=12, pady=8)
        self._label = tk.Label(self._root, text=f"{self._last_value:.1f} deg", padx=12, pady=8)
        self._label.pack()
        self._root.protocol("WM_DELETE_WINDOW", self.close)

    def update(self):
        if self._closed:
            return
        try:
            value = float(self._value.get())
            self._last_value = float(np.clip(value, 0.0, 180.0))
            self._label.configure(text=f"{self._last_value:.1f} deg")
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            self._closed = True

    def get_deg(self) -> float:
        return self._last_value

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except Exception:
            pass


# ============================================================
# SO(3) / Quaternion
# ============================================================

def project_to_so3(R: np.ndarray) -> Optional[np.ndarray]:
    if R is None:
        return None
    R = np.asarray(R, dtype=np.float32)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return None

    try:
        U, _S, Vt = np.linalg.svd(R)
    except np.linalg.LinAlgError:
        return None

    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0.0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    return R_proj.astype(np.float32)


def validate_rotation_matrix(R: np.ndarray, atol: float = 1e-3) -> Tuple[bool, str]:
    if R is None:
        return False, "rotation is None"
    R = np.asarray(R, dtype=np.float32)
    if R.shape != (3, 3):
        return False, f"rotation shape is {R.shape}, expected (3, 3)"
    if not np.all(np.isfinite(R)):
        return False, "rotation contains non-finite values"

    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3, dtype=np.float32)))
    det = float(np.linalg.det(R))
    if orth_err > atol:
        return False, f"R.T @ R is not close to I, error={orth_err:.6f}"
    if abs(det - 1.0) > atol:
        return False, f"det(R) is not close to +1, det={det:.6f}"
    return True, "ok"


def matrix_to_quat_wxyz(R: np.ndarray) -> Optional[np.ndarray]:
    R = project_to_so3(R)
    if R is None:
        return None

    trace = float(np.trace(R))
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0)) * 2.0
        w = 0.25 * s
        x = (float(R[2, 1]) - float(R[1, 2])) / s
        y = (float(R[0, 2]) - float(R[2, 0])) / s
        z = (float(R[1, 0]) - float(R[0, 1])) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = float(np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        w = (float(R[2, 1]) - float(R[1, 2])) / s
        x = 0.25 * s
        y = (float(R[0, 1]) + float(R[1, 0])) / s
        z = (float(R[0, 2]) + float(R[2, 0])) / s
    elif R[1, 1] > R[2, 2]:
        s = float(np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        w = (float(R[0, 2]) - float(R[2, 0])) / s
        x = (float(R[0, 1]) + float(R[1, 0])) / s
        y = 0.25 * s
        z = (float(R[1, 2]) + float(R[2, 1])) / s
    else:
        s = float(np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        w = (float(R[1, 0]) - float(R[0, 1])) / s
        x = (float(R[0, 2]) + float(R[2, 0])) / s
        y = (float(R[1, 2]) + float(R[2, 1])) / s
        z = 0.25 * s

    q = np.asarray([w, x, y, z], dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8 or not np.isfinite(norm):
        return None
    return q / norm


def quat_to_matrix_wxyz(q) -> Optional[np.ndarray]:
    if q is None:
        return None
    try:
        q = np.asarray(q, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None

    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return None
    w, x, y, z = q / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def parse_rotation_field(value) -> Optional[np.ndarray]:
    R = parse_mat3(value)
    if R is None:
        return None
    return project_to_so3(R)


def parse_vec3(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def parse_mat3(value) -> Optional[np.ndarray]:
    if value is None:
        return None

    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if arr.shape == (9,):
        arr = arr.reshape(3, 3)
    elif arr.shape != (3, 3):
        return None

    if not np.all(np.isfinite(arr)):
        return None
    return arr


def vec_to_list(value: Optional[np.ndarray]):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(x) for x in arr]


def mat_to_list(value: Optional[np.ndarray]):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).astype(float).tolist()


def normalize_vec(v: Optional[np.ndarray], eps: float = 1e-6) -> Optional[np.ndarray]:
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    if n < eps or not np.isfinite(n):
        return None
    return (v / n).astype(np.float32)


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def visibility_confidence(packet: dict, valid: bool) -> float:
    visibility = packet.get("arm_visibility")
    if isinstance(visibility, dict):
        vals = []
        for name in ("right_shoulder", "right_elbow", "right_wrist"):
            try:
                value = visibility.get(name)
                if value is not None and np.isfinite(float(value)):
                    vals.append(float(value))
            except (TypeError, ValueError):
                pass
        if len(vals) == 3:
            return float(min(vals))
    return 1.0 if valid else 0.0


def filter_elbow_swing_angle(raw_signed: Optional[float], state: dict, args) -> Tuple[Optional[float], dict]:
    """Band-pass filter signed elbow swing angle changes.

    The filter is intentionally simple and stateful:
    - first valid angle initializes the output;
    - changes below the lower threshold are considered jitter and held;
    - changes above the upper threshold are considered spikes and held;
    - changes inside the band update the filtered output.
    """
    lower = max(0.0, float(np.deg2rad(args.elbow_swing_delta_lower_deg)))
    upper = max(lower, float(np.deg2rad(args.elbow_swing_delta_upper_deg)))
    info = {
        "enabled": True,
        "delta_lower_rad": lower,
        "delta_upper_rad": upper,
        "delta_lower_deg": float(np.rad2deg(lower)),
        "delta_upper_deg": float(np.rad2deg(upper)),
        "delta_rad": None,
        "delta_deg": None,
        "accepted": False,
        "status": "no_raw_angle",
    }

    if raw_signed is None or not np.isfinite(float(raw_signed)):
        return None, info

    raw_signed = wrap_to_pi(float(raw_signed))
    previous = state.get("elbow_swing_angle_signed_filtered_rad")
    if previous is None:
        state["elbow_swing_angle_signed_filtered_rad"] = raw_signed
        info.update({"accepted": True, "status": "initialized", "delta_rad": 0.0, "delta_deg": 0.0})
        return raw_signed, info

    delta = wrap_to_pi(raw_signed - float(previous))
    abs_delta = abs(delta)
    info["delta_rad"] = float(delta)
    info["delta_deg"] = float(np.rad2deg(delta))

    if abs_delta < lower:
        info["status"] = "held_below_lower"
        return float(previous), info

    if abs_delta > upper:
        info["status"] = "held_above_upper"
        return float(previous), info

    filtered = wrap_to_pi(float(previous) + delta)
    state["elbow_swing_angle_signed_filtered_rad"] = filtered
    info.update({"accepted": True, "status": "accepted"})
    return filtered, info


def transform_arm_points(packet: dict, R: np.ndarray, t: np.ndarray) -> dict:
    arm_points_camera = packet.get("arm_points_camera_m")
    if not isinstance(arm_points_camera, dict):
        arm_points_camera = {}

    right_shoulder_camera = parse_vec3(arm_points_camera.get("right_shoulder"))
    right_elbow_camera = parse_vec3(arm_points_camera.get("right_elbow"))
    right_wrist_camera = parse_vec3(arm_points_camera.get("right_wrist"))

    right_shoulder_world = None if right_shoulder_camera is None else R @ right_shoulder_camera + t
    right_elbow_world = None if right_elbow_camera is None else R @ right_elbow_camera + t
    right_wrist_world = None if right_wrist_camera is None else R @ right_wrist_camera + t

    return {
        "right_shoulder": right_shoulder_world,
        "right_elbow": right_elbow_world,
        "right_wrist": right_wrist_world,
    }


def compute_right_arm_config_world(arm_points_world: dict, packet: dict, state: dict, args) -> dict:
    eps = 1e-6
    p_s = arm_points_world.get("right_shoulder")
    p_e = arm_points_world.get("right_elbow")
    p_w = arm_points_world.get("right_wrist")

    upper_arm = None
    forearm = None
    shoulder_to_wrist = None
    elbow_internal_angle = None
    elbow_flexion_angle = None
    arm_plane_normal = None
    arm_plane_valid = False
    elbow_swing_angle = None
    elbow_swing_angle_signed = None
    elbow_swing_angle_raw = None
    elbow_swing_angle_signed_raw = None
    swing_delta_lower = max(0.0, float(np.deg2rad(args.elbow_swing_delta_lower_deg)))
    swing_delta_upper = max(swing_delta_lower, float(np.deg2rad(args.elbow_swing_delta_upper_deg)))
    elbow_swing_filter_info = {
        "enabled": True,
        "delta_lower_rad": swing_delta_lower,
        "delta_upper_rad": swing_delta_upper,
        "delta_lower_deg": float(np.rad2deg(swing_delta_lower)),
        "delta_upper_deg": float(np.rad2deg(swing_delta_upper)),
        "delta_rad": None,
        "delta_deg": None,
        "accepted": False,
        "status": "not_computed",
    }

    arm_config_valid = bool(p_s is not None and p_e is not None and p_w is not None)
    if arm_config_valid:
        upper_arm = p_e - p_s
        forearm = p_w - p_e
        shoulder_to_wrist = p_w - p_s

        to_shoulder = p_s - p_e
        to_wrist = p_w - p_e
        len_to_shoulder = float(np.linalg.norm(to_shoulder))
        len_to_wrist = float(np.linalg.norm(to_wrist))
        len_upper = float(np.linalg.norm(upper_arm))
        len_forearm = float(np.linalg.norm(forearm))
        arm_config_valid = bool(len_upper > eps and len_forearm > eps)

        if arm_config_valid and len_to_shoulder > eps and len_to_wrist > eps:
            cos_internal = float(np.dot(to_shoulder, to_wrist) / (len_to_shoulder * len_to_wrist))
            elbow_internal_angle = float(np.arccos(np.clip(cos_internal, -1.0, 1.0)))
            elbow_flexion_angle = float(np.pi - elbow_internal_angle)

            normal_raw = np.cross(upper_arm, forearm)
            arm_plane_normal = normalize_vec(normal_raw, eps)
            if arm_plane_normal is not None:
                last_normal = state.get("last_valid_arm_plane_normal_world")
                if last_normal is not None and float(np.dot(arm_plane_normal, last_normal)) < 0.0:
                    arm_plane_normal = -arm_plane_normal
                state["last_valid_arm_plane_normal_world"] = arm_plane_normal.copy()
                arm_plane_valid = True

                normal_ref = state.get("arm_plane_normal_ref_world")
                if normal_ref is None:
                    normal_ref = arm_plane_normal.copy()
                    state["arm_plane_normal_ref_world"] = normal_ref

                cross_ref_cur = np.cross(normal_ref, arm_plane_normal)
                dot_ref_cur = float(np.clip(np.dot(normal_ref, arm_plane_normal), -1.0, 1.0))
                elbow_swing_angle = float(np.arctan2(np.linalg.norm(cross_ref_cur), dot_ref_cur))

                swing_axis = normalize_vec(shoulder_to_wrist, eps)
                if swing_axis is not None:
                    elbow_swing_angle_signed_raw = float(
                        np.arctan2(float(np.dot(swing_axis, cross_ref_cur)), dot_ref_cur)
                    )
                    elbow_swing_angle_raw = float(abs(elbow_swing_angle_signed_raw))
                    if bool(getattr(args, "elbow_swing_test", False)):
                        test_deg = float(np.clip(getattr(args, "elbow_swing_test_current_deg", 0.0), 0.0, 180.0))
                        test_rad = float(np.deg2rad(test_deg))
                        elbow_swing_angle_raw = test_rad
                        elbow_swing_angle_signed_raw = test_rad
                        elbow_swing_angle = test_rad
                        elbow_swing_angle_signed = test_rad
                        state["elbow_swing_angle_signed_filtered_rad"] = test_rad
                        elbow_swing_filter_info = {
                            "enabled": False,
                            "delta_lower_rad": swing_delta_lower,
                            "delta_upper_rad": swing_delta_upper,
                            "delta_lower_deg": float(np.rad2deg(swing_delta_lower)),
                            "delta_upper_deg": float(np.rad2deg(swing_delta_upper)),
                            "delta_rad": None,
                            "delta_deg": None,
                            "accepted": True,
                            "status": "test_slider_override",
                        }
                    else:
                        elbow_swing_angle_signed, elbow_swing_filter_info = filter_elbow_swing_angle(
                            elbow_swing_angle_signed_raw, state, args
                        )
                        elbow_swing_angle = (
                            None if elbow_swing_angle_signed is None else float(abs(elbow_swing_angle_signed))
                        )
                else:
                    elbow_swing_angle_raw = elbow_swing_angle

        if bool(getattr(args, "elbow_swing_test", False)) and arm_config_valid:
            test_deg = float(np.clip(getattr(args, "elbow_swing_test_current_deg", 0.0), 0.0, 180.0))
            test_rad = float(np.deg2rad(test_deg))
            elbow_swing_angle_raw = test_rad
            elbow_swing_angle_signed_raw = test_rad
            elbow_swing_angle = test_rad
            elbow_swing_angle_signed = test_rad
            state["elbow_swing_angle_signed_filtered_rad"] = test_rad
            if not arm_plane_valid:
                arm_plane_valid = True
            elbow_swing_filter_info = {
                "enabled": False,
                "delta_lower_rad": swing_delta_lower,
                "delta_upper_rad": swing_delta_upper,
                "delta_lower_deg": float(np.rad2deg(swing_delta_lower)),
                "delta_upper_deg": float(np.rad2deg(swing_delta_upper)),
                "delta_rad": None,
                "delta_deg": None,
                "accepted": True,
                "status": "test_slider_override",
            }

    confidence = visibility_confidence(packet, arm_config_valid)
    normal_ref = state.get("arm_plane_normal_ref_world")

    return {
        "arm_points_world_m": {
            "right_shoulder": vec_to_list(p_s),
            "right_elbow": vec_to_list(p_e),
            "right_wrist": vec_to_list(p_w),
        },
        "arm_vectors_world": {
            "upper_arm": vec_to_list(upper_arm),
            "forearm": vec_to_list(forearm),
            "shoulder_to_wrist": vec_to_list(shoulder_to_wrist),
        },
        "elbow_internal_angle_rad": elbow_internal_angle,
        "elbow_internal_angle_deg": None if elbow_internal_angle is None else float(np.degrees(elbow_internal_angle)),
        "elbow_flexion_angle_rad": elbow_flexion_angle,
        "elbow_flexion_angle_deg": None if elbow_flexion_angle is None else float(np.degrees(elbow_flexion_angle)),
        "arm_plane_valid": bool(arm_plane_valid),
        "arm_plane_normal_world": vec_to_list(arm_plane_normal) if arm_plane_valid else None,
        "arm_plane_normal_ref_world": vec_to_list(normal_ref),
        "elbow_swing_angle_rad": elbow_swing_angle,
        "elbow_swing_angle_deg": None if elbow_swing_angle is None else float(np.degrees(elbow_swing_angle)),
        "elbow_swing_angle_signed_rad": elbow_swing_angle_signed,
        "elbow_swing_angle_signed_deg": None if elbow_swing_angle_signed is None else float(np.degrees(elbow_swing_angle_signed)),
        "elbow_swing_angle_raw_rad": elbow_swing_angle_raw,
        "elbow_swing_angle_raw_deg": None if elbow_swing_angle_raw is None else float(np.degrees(elbow_swing_angle_raw)),
        "elbow_swing_angle_signed_raw_rad": elbow_swing_angle_signed_raw,
        "elbow_swing_angle_signed_raw_deg": (
            None if elbow_swing_angle_signed_raw is None else float(np.degrees(elbow_swing_angle_signed_raw))
        ),
        "elbow_swing_filter": elbow_swing_filter_info,
        "arm_config_valid": bool(arm_config_valid),
        "arm_config_frame": "world",
        "arm_config_semantics": "right_arm_shoulder_elbow_wrist_geometry",
        "arm_config_confidence": float(confidence),
    }


def transform_position_fields(packet: dict, R: np.ndarray, t: np.ndarray):
    position_camera = parse_vec3(packet.get("position_camera_m"))
    delta_camera = parse_vec3(packet.get("delta_camera_m"))
    target_camera = parse_vec3(packet.get("target_position_m"))

    position_world = None if position_camera is None else R @ position_camera + t
    delta_world = None if delta_camera is None else R @ delta_camera
    target_world_vector = None if target_camera is None else R @ target_camera

    return {
        "position_camera": position_camera,
        "delta_camera": delta_camera,
        "target_camera": target_camera,
        "position_world": position_world,
        "delta_world": delta_world,
        "target_world_vector": target_world_vector,
    }


def make_target_position(packet: dict, fields: dict, R: np.ndarray, t: np.ndarray, args):
    if args.output_mode == "curobo_delta":
        target = fields["delta_world"]
        if target is None:
            target = fields["target_world_vector"]
        if target is not None and args.apply_bridge_scale:
            target = float(args.bridge_scale) * target
        semantics = (
            "bridge_scaled_human_delta_world_m"
            if args.apply_bridge_scale
            else "raw_human_delta_world_m_unscaled"
        )
        return target, semantics

    target_camera = fields["target_camera"]
    if bool(packet.get("publish_delta", False)):
        target = None if target_camera is None else R @ target_camera
        return target, "camera_delta_vector_transformed_to_world"

    if target_camera is None:
        target_camera = fields["position_camera"]
    target = None if target_camera is None else R @ target_camera + t
    return target, "absolute_camera_point_transformed_to_world"


def transform_palm_orientation(
    packet: dict,
    R_world_camera: np.ndarray,
    t_world_camera: np.ndarray,
    R_palm_ee: np.ndarray,
):
    upstream_valid = bool(packet.get("palm_orientation_valid", False))

    R_camera_palm = parse_rotation_field(packet.get("R_camera_palm"))
    R_camera_palm_0 = parse_rotation_field(packet.get("R_camera_palm_0"))
    delta_R_camera_palm = parse_rotation_field(packet.get("delta_R_camera_palm"))
    palm_origin_camera = parse_vec3(packet.get("palm_origin_camera_m"))
    palm_normal_camera = parse_vec3(packet.get("palm_normal_camera"))

    R_world_palm = None if R_camera_palm is None else project_to_so3(R_world_camera @ R_camera_palm)
    R_world_palm_0 = None if R_camera_palm_0 is None else project_to_so3(R_world_camera @ R_camera_palm_0)

    delta_R_world_palm = None
    if R_world_palm is not None and R_world_palm_0 is not None:
        delta_R_world_palm = project_to_so3(R_world_palm @ R_world_palm_0.T)
    elif delta_R_camera_palm is not None:
        delta_R_world_palm = project_to_so3(R_world_camera @ delta_R_camera_palm @ R_world_camera.T)

    q_world_palm = matrix_to_quat_wxyz(R_world_palm) if R_world_palm is not None else None
    q_delta_world_palm = matrix_to_quat_wxyz(delta_R_world_palm) if delta_R_world_palm is not None else None
    palm_origin_world = None if palm_origin_camera is None else R_world_camera @ palm_origin_camera + t_world_camera
    palm_normal_world = None if palm_normal_camera is None else R_world_camera @ palm_normal_camera

    valid_orientation = bool(
        upstream_valid
        and R_world_palm is not None
        and delta_R_world_palm is not None
        and q_world_palm is not None
        and q_delta_world_palm is not None
    )

    if not upstream_valid:
        reason = packet.get("palm_reason", "palm orientation invalid upstream")
    elif not valid_orientation:
        reason = "palm orientation matrix parse/transform failed"
    else:
        reason = packet.get("palm_reason", "ok")

    R_world_ee_target = None
    R_world_ee_target_0 = None
    delta_R_world_ee_target = None
    q_world_ee_target = None
    q_delta_world_ee_target = None

    if R_world_palm is not None:
        R_world_ee_target = project_to_so3(R_world_palm @ R_palm_ee)

    if R_world_palm_0 is not None:
        R_world_ee_target_0 = project_to_so3(R_world_palm_0 @ R_palm_ee)

    if R_world_ee_target is not None and R_world_ee_target_0 is not None:
        delta_R_world_ee_target = project_to_so3(R_world_ee_target @ R_world_ee_target_0.T)
    elif delta_R_world_palm is not None:
        # With a fixed right-multiplied R_palm_ee, relative rotation is
        # equivalent to palm delta. Keep this fallback so downstream can still
        # receive a delta orientation when only delta_R_camera_palm is available.
        delta_R_world_ee_target = delta_R_world_palm

    if R_world_ee_target is not None:
        q_world_ee_target = matrix_to_quat_wxyz(R_world_ee_target)

    if delta_R_world_ee_target is not None:
        q_delta_world_ee_target = matrix_to_quat_wxyz(delta_R_world_ee_target)

    ee_orientation_valid = bool(
        valid_orientation
        and R_world_ee_target is not None
        and q_world_ee_target is not None
    )

    return {
        "valid_orientation": valid_orientation,
        "palm_orientation_valid": valid_orientation,
        "palm_reason": reason,
        "R_world_palm": mat_to_list(R_world_palm) if valid_orientation else None,
        "R_world_palm_0": mat_to_list(R_world_palm_0) if valid_orientation else None,
        "delta_R_world_palm": mat_to_list(delta_R_world_palm) if valid_orientation else None,
        "q_world_palm_wxyz": vec_to_list(q_world_palm) if valid_orientation else None,
        "q_delta_world_palm_wxyz": vec_to_list(q_delta_world_palm) if valid_orientation else None,
        "palm_origin_world_m": vec_to_list(palm_origin_world) if valid_orientation else None,
        "palm_normal_world": vec_to_list(palm_normal_world) if valid_orientation else None,
        "palm_orientation_frame": "world",
        "palm_orientation_semantics": "visual_palm_frame_not_robot_ee_frame",
        "ee_orientation_valid": ee_orientation_valid,
        "R_palm_ee": mat_to_list(R_palm_ee),
        "R_world_ee_target": mat_to_list(R_world_ee_target) if ee_orientation_valid else None,
        "R_world_ee_target_0": mat_to_list(R_world_ee_target_0) if ee_orientation_valid else None,
        "delta_R_world_ee_target": mat_to_list(delta_R_world_ee_target) if ee_orientation_valid else None,
        "q_world_ee_target_wxyz": vec_to_list(q_world_ee_target) if ee_orientation_valid else None,
        "q_delta_world_ee_target_wxyz": vec_to_list(q_delta_world_ee_target) if ee_orientation_valid else None,
        "ee_orientation_frame": "world",
        "ee_orientation_semantics": "robot_end_effector_target_from_visual_palm_frame",
    }


def make_world_packet(packet: dict, R: np.ndarray, t: np.ndarray, args, arm_state: dict) -> dict:
    fields = transform_position_fields(packet, R, t)
    target_position_world, target_semantics = make_target_position(packet, fields, R, t, args)

    upstream_position_valid = bool(packet.get("valid_position", packet.get("valid", False)))
    if args.output_mode == "curobo_delta":
        valid_position = bool(
            upstream_position_valid
            and (fields["delta_world"] is not None or fields["target_world_vector"] is not None)
        )
    else:
        valid_position = bool(upstream_position_valid and target_position_world is not None)

    palm_fields = transform_palm_orientation(packet, R, t, args.R_palm_ee)
    arm_points_world = transform_arm_points(packet, R, t)
    arm_fields = compute_right_arm_config_world(arm_points_world, packet, arm_state, args)

    output = dict(packet)
    output.update(
        {
            "valid": valid_position,
            "valid_position": valid_position,
            "valid_orientation": palm_fields["valid_orientation"],
            "target_frame": "world",
            "position_camera_m": vec_to_list(fields["position_camera"]),
            "delta_camera_m": vec_to_list(fields["delta_camera"]),
            "position_world_m": vec_to_list(fields["position_world"]),
            "delta_world_m": vec_to_list(fields["delta_world"]),
            "target_position_world_m": vec_to_list(target_position_world),
            # Compatibility for older cuRobo receiver code. In bridge v2 output
            # this field is already in world frame and follows target semantics.
            "target_position_m": vec_to_list(target_position_world),
            "target_position_semantics": target_semantics,
            "bridge_output_mode": args.output_mode,
            "scale_applied_by_bridge": bool(args.apply_bridge_scale),
            "bridge_scale": float(args.bridge_scale),
            "rotation_world_camera": mat_to_list(R),
            "translation_world_camera_m": vec_to_list(t),
            "bridge_time_ms": int(time.time() * 1000),
            "bridge_version": BRIDGE_VERSION,
        }
    )
    output.update(palm_fields)
    output.update(arm_fields)
    return output


def main():
    args = parse_args()
    normalize_legacy_args(args)
    args.elbow_swing_test_current_deg = float(np.clip(args.elbow_swing_test_initial_deg, 0.0, 180.0))

    R = load_rotation(args)
    t = parse_float_list(args.translation, 3, "--translation")
    R_palm_ee = load_palm_to_ee_rotation()
    args.R_palm_ee = R_palm_ee

    if args.validate_rotation:
        ok, msg = validate_rotation_matrix(R)
        if not ok:
            raise SystemExit(f"[ERROR] Invalid R_world_camera: {msg}")

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((args.input_host, args.input_port))
    recv_sock.settimeout(0.1)

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    output_addr = (args.output_host, args.output_port)

    elbow_swing_slider = None
    if args.elbow_swing_test:
        try:
            elbow_swing_slider = ElbowSwingTestSlider(args.elbow_swing_test_current_deg)
            print("[INFO] Elbow swing test slider enabled; publishing slider value instead of computed swing.")
        except Exception as exc:
            args.elbow_swing_test = False
            print(f"[WARN] Could not open elbow swing test slider; disabling test mode: {exc}")

    received_count = 0
    published_count = 0
    latest_valid_position = False
    latest_valid_orientation = False
    latest_target_position_world = None
    arm_state = {
        "last_valid_arm_plane_normal_world": None,
        "arm_plane_normal_ref_world": None,
        "elbow_swing_angle_signed_filtered_rad": None,
    }
    next_status_t = time.time() + 1.0

    print(f"[INFO] camera_to_world_udp_bridge v2 input  {args.input_host}:{args.input_port}")
    print(f"[INFO] camera_to_world_udp_bridge v2 output {args.output_host}:{args.output_port}")
    print(
        f"[INFO] output_mode={args.output_mode}, "
        f"apply_bridge_scale={args.apply_bridge_scale}, bridge_scale={args.bridge_scale}"
    )
    print("[INFO] R_world_camera =")
    print(R)
    print(f"[INFO] t_world_camera = {t.tolist()}")
    print("[INFO] R_palm_ee =")
    print(R_palm_ee)
    if args.elbow_swing_test:
        print(f"[INFO] elbow swing test initial value = {args.elbow_swing_test_current_deg:.1f} deg")
    print(
        "[INFO] elbow swing delta band-pass: "
        f"lower={args.elbow_swing_delta_lower_deg} deg, "
        f"upper={args.elbow_swing_delta_upper_deg} deg"
    )
    print("[INFO] Press Ctrl+C to exit.")

    try:
        while True:
            if elbow_swing_slider is not None:
                elbow_swing_slider.update()
                args.elbow_swing_test_current_deg = elbow_swing_slider.get_deg()

            now_t = time.time()
            if now_t >= next_status_t:
                swing_test_status = (
                    f" elbow_swing_test_deg={args.elbow_swing_test_current_deg:.1f}"
                    if args.elbow_swing_test
                    else ""
                )
                print(
                    "[STATUS] "
                    f"received={received_count} published={published_count} "
                    f"latest_valid_position={latest_valid_position} "
                    f"latest_target_position_world_m={latest_target_position_world} "
                    f"latest_valid_orientation={latest_valid_orientation}"
                    f"{swing_test_status}"
                )
                next_status_t = now_t + 1.0

            try:
                data, _addr = recv_sock.recvfrom(65535)
            except socket.timeout:
                continue

            try:
                packet = json.loads(data.decode("utf-8"))
            except Exception as exc:
                print(f"[WARN] Invalid input JSON: {exc}")
                continue

            if not isinstance(packet, dict):
                print("[WARN] Input JSON is not an object; skipping")
                continue

            received_count += 1
            try:
                output_packet = make_world_packet(packet, R, t, args, arm_state)
            except Exception as exc:
                print(f"[WARN] Failed to adapt packet: {exc}")
                continue

            payload = json.dumps(output_packet, separators=(",", ":")).encode("utf-8")
            try:
                send_sock.sendto(payload, output_addr)
            except OSError as exc:
                print(f"[WARN] Failed to publish output UDP packet: {exc}")
                continue

            published_count += 1
            latest_valid_position = bool(output_packet.get("valid_position", False))
            latest_valid_orientation = bool(output_packet.get("valid_orientation", False))
            latest_target_position_world = output_packet.get("target_position_world_m")
            if args.verbose:
                print(json.dumps(output_packet, ensure_ascii=False, indent=2))

    except KeyboardInterrupt:
        print("\n[INFO] Bridge stopped.")
    finally:
        if elbow_swing_slider is not None:
            elbow_swing_slider.close()
        recv_sock.close()
        send_sock.close()


if __name__ == "__main__":
    main()
