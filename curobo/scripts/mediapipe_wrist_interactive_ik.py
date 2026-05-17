#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cuRobo interactive IK viewer driven by either a mouse gizmo or UDP wrist targets.

This script deliberately does not import Orbbec, MediaPipe, or OpenCV.  It only:
1. opens a cuRobo/Viser robot viewer;
2. receives wrist target JSON packets over UDP;
3. maps the perceived target into a robot EE target;
4. solves IK for visualization;
5. updates the robot mesh and target marker.

It never sends commands to a real robot.
"""

import argparse
import copy
import json
import math
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo.types import ContentPath, GoalToolPose, Pose
from curobo.viewer import ViserVisualizer


class UdpWristTargetReceiver:
    """Background UDP JSON receiver for MediaPipe wrist target packets."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_packet: Optional[dict] = None
        self._latest_recv_t: Optional[float] = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.05)
        self._thread = threading.Thread(target=self._run, name="udp-wrist-receiver", daemon=True)
        self._thread.start()
        print(f"[INFO] UDP receiver listening on {self.host}:{self.port}")
        return self

    def _run(self):
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                packet = json.loads(data.decode("utf-8"))
            except Exception as exc:
                print(f"[WARN] Dropping invalid UDP JSON packet: {exc}")
                continue

            with self._lock:
                self._latest_packet = packet
                self._latest_recv_t = time.time()

    def get_latest(self) -> Tuple[Optional[dict], Optional[float]]:
        with self._lock:
            if self._latest_packet is None:
                return None, None
            return copy.deepcopy(self._latest_packet), self._latest_recv_t

    def stop(self):
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="cuRobo UDP wrist interactive IK viewer")
    parser.add_argument("--robot", type=str, default="franka.yml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)

    parser.add_argument("--target-source", choices=["mouse", "udp"], default="udp")
    parser.add_argument("--udp-host", type=str, default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5558)
    parser.add_argument("--target-timeout-s", type=float, default=0.3)

    parser.add_argument("--mapping-mode", choices=["identity", "delta", "manual"], default="delta")
    parser.add_argument("--teleop-scale", type=float, default=1)
    parser.add_argument(
        "--fixed-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the initial robot EE orientation as a fixed wxyz quaternion",
    )
    parser.add_argument(
        "--use-udp-ee-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use q_world_ee_target_wxyz from UDP packet and map it as relative EE orientation",
    )

    parser.add_argument("--manual-axis-map", nargs=3, default=["x", "y", "z"])
    parser.add_argument("--manual-translation", nargs=3, type=float, default=[0.0, 0.0, 0.0])

    parser.add_argument("--workspace-x", nargs=2, type=float, default=[0.2, 0.8])
    parser.add_argument("--workspace-y", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--workspace-z", nargs=2, type=float, default=[0.1, 0.8])
    parser.add_argument("--filter-alpha", type=float, default=0.2)
    parser.add_argument("--max-step-m", type=float, default=0.03)
    parser.add_argument("--loop-hz", type=float, default=30.0)

    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--seed-solver-num-seeds", type=int, default=1)
    parser.add_argument("--return-seeds", type=int, default=32)
    parser.add_argument("--score-continuity-weight", type=float, default=1.0)
    parser.add_argument("--score-nominal-weight", type=float, default=0.1)
    parser.add_argument("--score-limit-weight", type=float, default=0.05)
    # Recommended first human-arm tuning:
    #   --score-elbow-swing-weight 0.2 --score-elbow-flexion-weight 0.05
    parser.add_argument("--score-elbow-flexion-weight", type=float, default=0.05)
    parser.add_argument("--score-elbow-swing-weight", type=float, default=0.2)
    parser.add_argument("--score-arm-plane-weight", type=float, default=0.0)
    parser.add_argument("--score-debug", action="store_true", default=False)
    parser.add_argument("--robot-shoulder-link", type=str, default="panda_link0")
    parser.add_argument("--robot-elbow-link", type=str, default="panda_link4")
    parser.add_argument(
        "--robot-wrist-link",
        type=str,
        default="",
        help="robot wrist/EE-equivalent link for arm scoring; empty means target_link",
    )
    parser.add_argument("--score-elbow-flexion-scale", type=float, default=1.0)
    parser.add_argument("--score-elbow-swing-scale", type=float, default=1.0)
    parser.add_argument("--score-arm-confidence-gate", type=float, default=0.3)
    parser.add_argument(
        "--auto-init-elbow-swing-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="auto initialize human/robot elbow swing offset when both are valid",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        default=True,
        help="use the heavier interactive IK CUDA graph config from cuRobo examples",
    )
    return parser.parse_args()


def packet_position(packet: dict, mapping_mode: str) -> Optional[np.ndarray]:
    if not packet or not bool(packet.get("valid", False)):
        return None

    if packet.get("target_position_world_m") is not None:
        values = packet.get("target_position_world_m")
    elif packet.get("target_frame") == "world" and packet.get("target_position_m") is not None:
        values = packet.get("target_position_m")
    elif mapping_mode in ("delta", "manual"):
        values = packet.get("position_camera_m")
    else:
        values = packet.get("target_position_m") or packet.get("position_camera_m")

    if values is None or len(values) != 3:
        return None

    try:
        pos = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if not np.all(np.isfinite(pos)):
        return None
    return pos


def apply_manual_axis_map(point: np.ndarray, axis_map, translation) -> np.ndarray:
    source = {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
    mapped = []
    for token in axis_map:
        token = str(token).strip().lower()
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token[-1] if token else ""
        if axis not in source:
            raise ValueError(f"Invalid axis token '{token}', expected x/y/z or -x/-y/-z")
        mapped.append(sign * source[axis])
    return np.asarray(mapped, dtype=np.float32) + np.asarray(translation, dtype=np.float32)


def clamp_position(position: np.ndarray, args) -> np.ndarray:
    lo = np.asarray([args.workspace_x[0], args.workspace_y[0], args.workspace_z[0]], dtype=np.float32)
    hi = np.asarray([args.workspace_x[1], args.workspace_y[1], args.workspace_z[1]], dtype=np.float32)
    return np.clip(position, lo, hi)


def filter_target(position: np.ndarray, previous: Optional[np.ndarray], args) -> np.ndarray:
    if previous is None:
        return position.astype(np.float32)

    alpha = float(np.clip(args.filter_alpha, 0.0, 1.0))
    position = alpha * position + (1.0 - alpha) * previous

    max_step = max(1e-6, float(args.max_step_m))
    delta = position - previous
    dist = float(np.linalg.norm(delta))
    if dist > max_step:
        position = previous + delta / dist * max_step

    return position.astype(np.float32)


def parse_quat_wxyz(value):
    if value is None:
        return None
    try:
        q = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return None
    return q / n


def quat_to_matrix_wxyz(q):
    q = parse_quat_wxyz(q)
    if q is None:
        return None
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return R


def matrix_to_quat_wxyz(R):
    if R is None:
        return None
    R = np.asarray(R, dtype=np.float32)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return None

    try:
        U, _S, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
    except Exception:
        return None

    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return parse_quat_wxyz([w, x, y, z])


def packet_ee_quat(packet):
    if not packet:
        return None
    if not bool(packet.get("ee_orientation_valid", packet.get("valid_orientation", False))):
        return None
    return parse_quat_wxyz(packet.get("q_world_ee_target_wxyz"))


def parse_packet_vec3(value):
    if value is None:
        return None
    try:
        vec = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        return None
    return vec


def parse_packet_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def packet_arm_config(packet: dict) -> dict:
    points = packet.get("arm_points_world_m") if packet else None
    vectors = packet.get("arm_vectors_world") if packet else None
    if not isinstance(points, dict):
        points = {}
    if not isinstance(vectors, dict):
        vectors = {}

    valid = bool(packet.get("arm_config_valid", False)) if packet else False
    plane_valid = bool(packet.get("arm_plane_valid", False)) if packet else False
    confidence = parse_packet_float(packet.get("arm_config_confidence")) if packet else None
    if confidence is None:
        confidence = 0.0

    return {
        "valid": valid,
        "plane_valid": plane_valid,
        "confidence": confidence,
        "points_world": {
            "right_shoulder": parse_packet_vec3(points.get("right_shoulder")),
            "right_elbow": parse_packet_vec3(points.get("right_elbow")),
            "right_wrist": parse_packet_vec3(points.get("right_wrist")),
        },
        "vectors_world": {
            "upper_arm": parse_packet_vec3(vectors.get("upper_arm")),
            "forearm": parse_packet_vec3(vectors.get("forearm")),
            "shoulder_to_wrist": parse_packet_vec3(vectors.get("shoulder_to_wrist")),
        },
        "elbow_internal_angle_rad": parse_packet_float(packet.get("elbow_internal_angle_rad")) if packet else None,
        "elbow_internal_angle_deg": parse_packet_float(packet.get("elbow_internal_angle_deg")) if packet else None,
        "elbow_flexion_angle_rad": parse_packet_float(packet.get("elbow_flexion_angle_rad")) if packet else None,
        "elbow_flexion_angle_deg": parse_packet_float(packet.get("elbow_flexion_angle_deg")) if packet else None,
        "arm_plane_normal_world": parse_packet_vec3(packet.get("arm_plane_normal_world")) if packet else None,
        "arm_plane_normal_ref_world": parse_packet_vec3(packet.get("arm_plane_normal_ref_world")) if packet else None,
        "elbow_swing_angle_rad": parse_packet_float(packet.get("elbow_swing_angle_rad")) if packet else None,
        "elbow_swing_angle_deg": parse_packet_float(packet.get("elbow_swing_angle_deg")) if packet else None,
        "elbow_swing_angle_signed_rad": parse_packet_float(packet.get("elbow_swing_angle_signed_rad")) if packet else None,
        "elbow_swing_angle_signed_deg": parse_packet_float(packet.get("elbow_swing_angle_signed_deg")) if packet else None,
        "frame": packet.get("arm_config_frame") if packet and isinstance(packet.get("arm_config_frame"), str) else None,
        "semantics": packet.get("arm_config_semantics") if packet and isinstance(packet.get("arm_config_semantics"), str) else None,
    }


def make_pose(position: np.ndarray, quaternion_wxyz: np.ndarray) -> Pose:
    # cuRobo Pose expects quaternion in wxyz order.
    pos = torch.tensor(position, device="cuda", dtype=torch.float32).view(1, 3)
    quat_np = np.asarray(quaternion_wxyz, dtype=np.float32)
    quat_norm = float(np.linalg.norm(quat_np))
    if quat_norm > 1e-8:
        quat_np = quat_np / quat_norm
    quat = torch.tensor(quat_np, device="cuda", dtype=torch.float32).view(1, 4)
    return Pose(position=pos, quaternion=quat)


def ik_success(result) -> bool:
    try:
        return bool(result.success.any().item())
    except Exception:
        return bool(result.success)


def update_marker(marker, position: np.ndarray, color):
    marker.position = np.asarray(position, dtype=np.float32)
    try:
        marker.color = color
    except Exception:
        pass


def gui_set(handle, value):
    handle.value = str(value)


def gui_float(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def gui_vec(value, digits=4):
    return "--" if value is None else np.round(value, digits).tolist()


def joint_state_position_1d(value) -> torch.Tensor:
    position = value.position if hasattr(value, "position") else value
    return position.squeeze().reshape(-1)


def joint_state_from_q(q: torch.Tensor, joint_names=None) -> JointState:
    q = q.reshape(1, 1, -1).clone()
    return JointState.from_position(q, joint_names=joint_names)


def extract_successful_ik_candidates(result) -> List[torch.Tensor]:
    """Extract successful IK candidate joint vectors from a cuRobo result."""
    if result is None or getattr(result, "js_solution", None) is None:
        return []

    solutions = result.js_solution.position if hasattr(result.js_solution, "position") else result.js_solution
    if solutions is None or solutions.numel() == 0:
        return []

    dof = int(solutions.shape[-1])
    flat_q = solutions.reshape(-1, dof)
    success = getattr(result, "success", None)

    if success is None:
        return [q.reshape(-1) for q in flat_q] if ik_success(result) else []

    if not isinstance(success, torch.Tensor):
        return [q.reshape(-1) for q in flat_q] if bool(success) else []

    success = success.detach()
    if success.numel() == 1:
        return [q.reshape(-1) for q in flat_q] if bool(success.reshape(-1)[0].item()) else []

    flat_success = success.reshape(-1).to(device=flat_q.device)
    if flat_success.numel() == flat_q.shape[0]:
        return [flat_q[i].reshape(-1) for i in range(flat_q.shape[0]) if bool(flat_success[i].item())]

    if not getattr(extract_successful_ik_candidates, "_warned_mismatch", False):
        print(
            "[WARN] IK success shape does not match candidate count; "
            "falling back to the first solution when overall success is true."
        )
        extract_successful_ik_candidates._warned_mismatch = True
    return [flat_q[0].reshape(-1)] if ik_success(result) else []


def _tensor_like_1d(value, q: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = joint_state_position_1d(value).to(device=q.device, dtype=q.dtype)
    if tensor.numel() != q.numel():
        return None
    return tensor.reshape(-1)


def get_active_joint_limits_or_none(ik, dof: int, device, dtype):
    """Best-effort joint-limit lookup across cuRobo versions/config layouts."""
    candidates = [
        lambda: ik.kinematics.get_joint_limits(),
        lambda: ik.core.kinematics.get_joint_limits(),
        lambda: ik.kinematics.config.get_joint_limits(),
        lambda: ik.core.kinematics.config.get_joint_limits(),
        lambda: ik.config.robot_cfg.kinematics.kinematics_config.joint_limits,
        lambda: ik.config.robot_cfg.kinematics.joint_limits,
        lambda: ik.config.kinematics_config.joint_limits,
    ]

    for getter in candidates:
        try:
            limits = getter()
            position = getattr(limits, "position", None)
            if position is None:
                continue
            position = position.to(device=device, dtype=dtype)
            if position.ndim != 2 or position.shape[0] != 2:
                continue
            if position.shape[1] == dof:
                return position[0].reshape(-1), position[1].reshape(-1)
            names = getattr(limits, "joint_names", None)
            active_names = getattr(ik, "joint_names", None)
            if names is not None and active_names is not None and len(active_names) == dof:
                indices = [names.index(name) for name in active_names if name in names]
                if len(indices) == dof:
                    active_position = position[:, indices]
                    return active_position[0].reshape(-1), active_position[1].reshape(-1)
        except Exception:
            continue

    if not getattr(get_active_joint_limits_or_none, "_warned", False):
        print("[WARN] Could not read joint limits; disabling limit-margin score.")
        get_active_joint_limits_or_none._warned = True
    return None, None


def wrap_to_pi_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def normalize_torch_vec(v: Optional[torch.Tensor], eps: float = 1.0e-8) -> Optional[torch.Tensor]:
    if v is None:
        return None
    v = v.reshape(-1)
    if v.numel() != 3 or not torch.isfinite(v).all():
        return None
    n = torch.linalg.norm(v)
    if not torch.isfinite(n) or float(n.detach().cpu()) < eps:
        return None
    return v / n.clamp_min(eps)


def as_torch_vec3(value, q: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, device=q.device, dtype=q.dtype).reshape(-1)
    except Exception:
        return None
    if tensor.numel() != 3 or not torch.isfinite(tensor).all():
        return None
    return tensor


def get_pose_position_from_state(kin_state, link_name: str) -> Optional[torch.Tensor]:
    tool_poses = getattr(kin_state, "tool_poses", None)
    if tool_poses is None:
        return None
    frames = getattr(tool_poses, "tool_frames", None)
    if frames is None or link_name not in frames:
        return None
    try:
        pose = tool_poses.get_link_pose(link_name)
        return pose.position.reshape(-1, 3)[0]
    except Exception:
        return None


def _joint_names_from_state(value) -> Optional[List[str]]:
    names = getattr(value, "joint_names", None)
    return list(names) if names is not None else None


def _filter_q_to_fk_joint_names(ik, arm_fk, q: torch.Tensor, model_dof: int):
    """Filter a full q vector, e.g. arm+fingers, down to the FK model's active joints."""
    target_names = None
    for names in (
        _joint_names_from_state(getattr(arm_fk, "default_joint_state", None)),
        _joint_names_from_state(getattr(ik, "default_joint_state", None)),
        list(getattr(ik, "joint_names", []) or []),
    ):
        if names is not None and len(names) == model_dof:
            target_names = names
            break
    if target_names is None:
        return q, None

    source_name_candidates = []
    ik_joint_names = list(getattr(ik, "joint_names", []) or [])
    if len(ik_joint_names) == q.numel():
        source_name_candidates.append(ik_joint_names)

    for state in (getattr(ik, "default_joint_state", None), getattr(arm_fk, "default_joint_state", None)):
        names = _joint_names_from_state(state)
        if names is not None and len(names) == q.numel():
            source_name_candidates.append(names)

    for getter in (
        lambda: ik.get_full_js(getattr(ik, "default_joint_state")),
        lambda: arm_fk.get_full_js(getattr(arm_fk, "default_joint_state")),
    ):
        try:
            names = _joint_names_from_state(getter())
            if names is not None and len(names) == q.numel():
                source_name_candidates.append(names)
        except Exception:
            continue

    for source_names in source_name_candidates:
        if all(name in source_names for name in target_names):
            indices = [source_names.index(name) for name in target_names]
            return q[indices].reshape(-1), target_names

    return q, None


def compute_robot_arm_points_world(
    ik,
    q,
    target_link,
    shoulder_link,
    elbow_link,
    wrist_link,
) -> dict:
    wrist_link = wrist_link or target_link
    if q is None:
        return {"valid": False, "shoulder": None, "elbow": None, "wrist": None, "reason": "q is None"}
    q = q.reshape(-1)
    arm_fk = getattr(ik, "_arm_scoring_kinematics", None) or getattr(ik, "kinematics", None)
    if arm_fk is None:
        return {"valid": False, "shoulder": None, "elbow": None, "wrist": None, "reason": "no FK model"}

    q_fk = q
    q_fk_names = None
    model_dof = getattr(arm_fk, "dof", None)
    if model_dof is not None and q_fk.numel() != int(model_dof):
        q_filtered, q_filtered_names = _filter_q_to_fk_joint_names(ik, arm_fk, q_fk, int(model_dof))
        if q_filtered.numel() == int(model_dof):
            q_fk = q_filtered
            q_fk_names = q_filtered_names

    if model_dof is not None and q_fk.numel() != int(model_dof):
        try:
            active_js = JointState.from_position(
                q.reshape(1, 1, -1),
                joint_names=ik.joint_names if q.numel() == len(ik.joint_names) else None,
            )
            full_js = ik.get_full_js(active_js)
            q_full = full_js.position.squeeze().reshape(-1)
            if q_full.numel() == int(model_dof):
                q_fk = q_full
        except Exception:
            pass

    if model_dof is not None and q_fk.numel() != int(model_dof):
        return {
            "valid": False,
            "shoulder": None,
            "elbow": None,
            "wrist": None,
            "reason": f"q dof {q_fk.numel()} != FK dof {model_dof}",
        }

    try:
        joint_state = JointState.from_position(q_fk.reshape(1, 1, -1), joint_names=q_fk_names)
        kin_state = arm_fk.compute_kinematics(joint_state)
    except Exception as exc:
        return {"valid": False, "shoulder": None, "elbow": None, "wrist": None, "reason": f"FK failed: {exc}"}

    shoulder = get_pose_position_from_state(kin_state, shoulder_link)
    elbow = get_pose_position_from_state(kin_state, elbow_link)
    wrist = get_pose_position_from_state(kin_state, wrist_link)
    if shoulder is None or elbow is None or wrist is None:
        frames = getattr(getattr(kin_state, "tool_poses", None), "tool_frames", [])
        reason = (
            f"missing link pose(s): shoulder={shoulder_link in frames}, "
            f"elbow={elbow_link in frames}, wrist={wrist_link in frames}; "
            f"available={frames}"
        )
        if not getattr(compute_robot_arm_points_world, "_warned_missing_link", False):
            print("[WARN] Could not find robot arm scoring links; human arm scoring terms disabled until links are fixed.")
            print(f"[WARN] {reason}")
            compute_robot_arm_points_world._warned_missing_link = True
        return {"valid": False, "shoulder": shoulder, "elbow": elbow, "wrist": wrist, "reason": reason}

    return {"valid": True, "shoulder": shoulder, "elbow": elbow, "wrist": wrist, "reason": "ok"}


def compute_flexion_angle_from_points(p_s, p_e, p_w, eps: float = 1.0e-8):
    v1 = p_s - p_e
    v2 = p_w - p_e
    n1 = torch.linalg.norm(v1)
    n2 = torch.linalg.norm(v2)
    if float(n1.detach().cpu()) < eps or float(n2.detach().cpu()) < eps:
        return None
    cos_theta = torch.dot(v1, v2) / (n1.clamp_min(eps) * n2.clamp_min(eps))
    theta_internal = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    return torch.as_tensor(math.pi, device=p_s.device, dtype=p_s.dtype) - theta_internal


def compute_arm_plane_normal_from_points(p_s, p_e, p_w, eps: float = 1.0e-8):
    upper = p_e - p_s
    forearm = p_w - p_e
    normal = torch.cross(upper, forearm, dim=0)
    return normalize_torch_vec(normal, eps=eps)


def compute_signed_swing_angle(n_ref, n_now, axis):
    n_ref = normalize_torch_vec(n_ref)
    n_now = normalize_torch_vec(n_now)
    axis = normalize_torch_vec(axis)
    if n_ref is None or n_now is None or axis is None:
        return None
    n_ref = normalize_torch_vec(n_ref - axis * torch.dot(n_ref, axis))
    n_now = normalize_torch_vec(n_now - axis * torch.dot(n_now, axis))
    if n_ref is None or n_now is None:
        return None
    numerator = torch.dot(axis, torch.cross(n_ref, n_now, dim=0))
    denominator = torch.dot(n_ref, n_now)
    return torch.atan2(numerator, denominator)


def human_arm_terms_requested(args) -> bool:
    return (
        float(args.score_elbow_flexion_weight) != 0.0
        or float(args.score_elbow_swing_weight) != 0.0
        or float(args.score_arm_plane_weight) != 0.0
    )


def safe_tensor_term(value: Optional[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
    if value is None or not torch.isfinite(value).all():
        return torch.zeros((), device=q.device, dtype=q.dtype)
    return value.reshape(())


def tensor_to_float_or_none(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    try:
        if not torch.isfinite(value).all():
            return None
        return float(value.detach().cpu().item())
    except Exception:
        return None


def compute_robot_arm_metric_info(
    ik,
    q: torch.Tensor,
    target_link: str,
    args,
    elbow_swing_state: Optional[dict] = None,
    initialize_ref: bool = False,
) -> dict:
    """Compute robot-side elbow flexion and swing for display/scoring diagnostics."""
    info = {
        "robot_arm_valid": False,
        "robot_arm_reason": "not computed",
        "robot_elbow_flexion_rad": None,
        "robot_elbow_flexion_deg": None,
        "robot_elbow_swing_rad": None,
        "robot_elbow_swing_deg": None,
        "robot_arm_plane_normal": None,
    }

    q = q.reshape(-1)
    robot_points = compute_robot_arm_points_world(
        ik=ik,
        q=q,
        target_link=target_link,
        shoulder_link=args.robot_shoulder_link,
        elbow_link=args.robot_elbow_link,
        wrist_link=args.robot_wrist_link or target_link,
    )
    info["robot_arm_valid"] = bool(robot_points.get("valid", False))
    info["robot_arm_reason"] = robot_points.get("reason", "unknown")
    if not info["robot_arm_valid"]:
        return info

    p_s = robot_points["shoulder"]
    p_e = robot_points["elbow"]
    p_w = robot_points["wrist"]
    robot_flex = compute_flexion_angle_from_points(p_s, p_e, p_w)
    robot_normal = compute_arm_plane_normal_from_points(p_s, p_e, p_w)
    robot_swing = None

    if robot_normal is not None and elbow_swing_state is not None:
        if initialize_ref and elbow_swing_state.get("robot_plane_ref") is None:
            elbow_swing_state["robot_plane_ref"] = robot_normal.detach().clone()
            print("[INFO] Auto set robot elbow plane reference for swing display/scoring.")
        robot_ref = elbow_swing_state.get("robot_plane_ref")
        axis = normalize_torch_vec(p_w - p_s)
        robot_swing = compute_signed_swing_angle(robot_ref, robot_normal, axis)

    flex_rad = tensor_to_float_or_none(robot_flex)
    swing_rad = tensor_to_float_or_none(robot_swing)
    info.update(
        {
            "robot_elbow_flexion_rad": flex_rad,
            "robot_elbow_flexion_deg": None if flex_rad is None else flex_rad * 180.0 / math.pi,
            "robot_elbow_swing_rad": swing_rad,
            "robot_elbow_swing_deg": None if swing_rad is None else swing_rad * 180.0 / math.pi,
            "robot_arm_plane_normal": robot_normal,
        }
    )
    return info


def score_candidate_q(
    q: torch.Tensor,
    q_prev: torch.Tensor,
    q_nominal: torch.Tensor,
    q_min: Optional[torch.Tensor],
    q_max: Optional[torch.Tensor],
    latest_arm_config: dict,
    ik: InverseKinematics,
    target_link: str,
    args,
    elbow_swing_state: dict,
) -> Tuple[torch.Tensor, dict]:
    """Score one IK candidate.

    External selection:
        Q_valid = {q_i | cuRobo IK succeeded}
        q* = argmin J_select(q_i)

    J_select =
        w_cont E_cont
      + w_nom E_nom
      + w_limit E_limit
      + c_arm (w_flex E_flex + w_swing E_swing + w_plane E_plane)

    Robot flexion:
        theta_flex_r = pi - angle(p_s - p_e, p_w - p_e)

    Robot arm plane:
        n_r = normalize((p_e - p_s) x (p_w - p_e))

    Robot swing:
        psi_r = atan2(axis dot (n_r0 x n_r), n_r0 dot n_r)
        axis = normalize(p_w - p_s)

    Swing matching uses an initial human/robot offset:
        offset0 = wrap_to_pi(psi_r0 - gamma_s * psi_h0)
        E_swing = wrap_to_pi(psi_r - gamma_s * psi_h - offset0)^2
    """
    q = q.reshape(-1)
    huge = torch.tensor(1.0e9, device=q.device, dtype=q.dtype)
    zero = torch.zeros((), device=q.device, dtype=q.dtype)
    if not torch.isfinite(q).all():
        return huge, {
            "total": 1.0e9,
            "continuity": 1.0e9,
            "nominal": 0.0,
            "limit": 0.0,
            "elbow_flexion": 0.0,
            "elbow_swing": 0.0,
            "arm_plane": 0.0,
            "robot_arm_valid": False,
            "robot_arm_reason": "candidate q invalid",
        }

    q_prev_for_fk = joint_state_position_1d(q_prev).detach().clone() if q_prev is not None else None
    q_prev = _tensor_like_1d(q_prev, q)
    q_nominal = _tensor_like_1d(q_nominal, q)

    q_range = torch.ones_like(q)
    limit_term = torch.zeros((), device=q.device, dtype=q.dtype)
    if q_min is not None and q_max is not None:
        q_min = _tensor_like_1d(q_min, q)
        q_max = _tensor_like_1d(q_max, q)
        if q_min is not None and q_max is not None:
            q_range = (q_max - q_min).abs().clamp_min(1.0e-6)
            q_mid = 0.5 * (q_min + q_max)
            radius = (0.5 * (q_max - q_min).abs()).clamp_min(1.0e-6)
            limit_term = torch.mean(((q - q_mid) / radius) ** 4)

    continuity = (
        torch.mean(((q - q_prev) / q_range) ** 2)
        if q_prev is not None
        else torch.zeros((), device=q.device, dtype=q.dtype)
    )
    nominal = (
        torch.mean(((q - q_nominal) / q_range) ** 2)
        if q_nominal is not None
        else torch.zeros((), device=q.device, dtype=q.dtype)
    )

    elbow_flexion = zero
    elbow_swing = zero
    arm_plane = zero
    robot_flex = None
    robot_swing = None
    human_flex = None
    human_swing = None
    arm_confidence = 0.0
    robot_arm_valid = False
    robot_arm_reason = "human arm terms disabled"

    confidence = float(latest_arm_config.get("confidence", 0.0) or 0.0) if latest_arm_config else 0.0
    arm_confidence = float(np.clip(confidence, 0.0, 1.0))
    confidence_ok = arm_confidence >= float(args.score_arm_confidence_gate)
    has_human_arm_config = bool(
        latest_arm_config
        and (latest_arm_config.get("valid", False) or latest_arm_config.get("plane_valid", False))
    )
    # Robot elbow metrics are cheap enough and useful as diagnostics, so compute them even when
    # human-arm scoring weights are zero or human confidence is currently gated out.
    needs_robot_arm = True
    can_score_human_arm = human_arm_terms_requested(args) and confidence_ok and has_human_arm_config

    if human_arm_terms_requested(args) and not confidence_ok:
        robot_arm_reason = f"arm confidence {arm_confidence:.3f} below gate"
    elif human_arm_terms_requested(args) and not has_human_arm_config:
        robot_arm_reason = "no valid human arm config"

    robot_points = None
    robot_normal = None
    if needs_robot_arm:
        robot_points = compute_robot_arm_points_world(
            ik=ik,
            q=q,
            target_link=target_link,
            shoulder_link=args.robot_shoulder_link,
            elbow_link=args.robot_elbow_link,
            wrist_link=args.robot_wrist_link or target_link,
        )
        robot_arm_valid = bool(robot_points.get("valid", False))
        robot_arm_reason = robot_points.get("reason", "unknown")
        if robot_arm_valid:
            p_s = robot_points["shoulder"]
            p_e = robot_points["elbow"]
            p_w = robot_points["wrist"]
            robot_flex = compute_flexion_angle_from_points(p_s, p_e, p_w)
            robot_normal = compute_arm_plane_normal_from_points(p_s, p_e, p_w)

            if robot_normal is not None and elbow_swing_state.get("robot_plane_ref") is None:
                prev_points = compute_robot_arm_points_world(
                    ik=ik,
                    q=q_prev_for_fk,
                    target_link=target_link,
                    shoulder_link=args.robot_shoulder_link,
                    elbow_link=args.robot_elbow_link,
                    wrist_link=args.robot_wrist_link or target_link,
                )
                if prev_points.get("valid", False):
                    prev_normal = compute_arm_plane_normal_from_points(
                        prev_points["shoulder"], prev_points["elbow"], prev_points["wrist"]
                    )
                    if prev_normal is not None:
                        elbow_swing_state["robot_plane_ref"] = prev_normal.detach().clone()
                        print("[INFO] Auto set robot elbow plane reference for swing scoring.")

            robot_ref = elbow_swing_state.get("robot_plane_ref")
            axis = normalize_torch_vec(p_w - p_s)
            robot_swing = compute_signed_swing_angle(robot_ref, robot_normal, axis)

    if can_score_human_arm and robot_arm_valid:
        if (
            float(args.score_elbow_flexion_weight) != 0.0
            and latest_arm_config.get("valid", False)
            and latest_arm_config.get("elbow_flexion_angle_rad") is not None
            and robot_flex is not None
        ):
            human_flex = torch.as_tensor(
                float(latest_arm_config["elbow_flexion_angle_rad"]),
                device=q.device,
                dtype=q.dtype,
            )
            elbow_flexion = (robot_flex - float(args.score_elbow_flexion_scale) * human_flex) ** 2

        if (
            float(args.score_arm_plane_weight) != 0.0
            and latest_arm_config.get("plane_valid", False)
            and robot_normal is not None
        ):
            human_normal = normalize_torch_vec(as_torch_vec3(latest_arm_config.get("arm_plane_normal_world"), q))
            if human_normal is not None:
                dot = torch.clamp(torch.dot(robot_normal, human_normal), -1.0, 1.0)
                arm_plane = 1.0 - torch.abs(dot)

        if (
            float(args.score_elbow_swing_weight) != 0.0
            and latest_arm_config.get("plane_valid", False)
            and latest_arm_config.get("elbow_swing_angle_signed_rad") is not None
            and robot_normal is not None
        ):
            human_swing = torch.as_tensor(
                float(latest_arm_config["elbow_swing_angle_signed_rad"]),
                device=q.device,
                dtype=q.dtype,
            )

            if (
                args.auto_init_elbow_swing_offset
                and elbow_swing_state.get("human_robot_swing_offset") is None
                and robot_ref is not None
            ):
                prev_points = compute_robot_arm_points_world(
                    ik=ik,
                    q=q_prev_for_fk,
                    target_link=target_link,
                    shoulder_link=args.robot_shoulder_link,
                    elbow_link=args.robot_elbow_link,
                    wrist_link=args.robot_wrist_link or target_link,
                )
                if prev_points.get("valid", False):
                    prev_normal = compute_arm_plane_normal_from_points(
                        prev_points["shoulder"], prev_points["elbow"], prev_points["wrist"]
                    )
                    prev_axis = normalize_torch_vec(prev_points["wrist"] - prev_points["shoulder"])
                    prev_swing = compute_signed_swing_angle(robot_ref, prev_normal, prev_axis)
                    if prev_swing is not None:
                        offset = wrap_to_pi_torch(prev_swing - float(args.score_elbow_swing_scale) * human_swing)
                        elbow_swing_state["human_robot_swing_offset"] = offset.detach().clone()
                        human_deg = float(human_swing.detach().cpu()) * 180.0 / math.pi
                        prev_deg = float(prev_swing.detach().cpu()) * 180.0 / math.pi
                        offset_deg = float(offset.detach().cpu()) * 180.0 / math.pi
                        elbow_swing_state["elbow_swing_offset_status"] = (
                            f"set human={human_deg:+.2f} prev={prev_deg:+.2f} offset={offset_deg:+.2f} deg"
                        )
                        print(
                            "[INFO] Auto set elbow swing offset: "
                            f"human_swing={human_deg:+.2f} deg, "
                            f"prev_robot_swing={prev_deg:+.2f} deg, "
                            f"offset={offset_deg:+.2f} deg"
                        )

            offset = elbow_swing_state.get("human_robot_swing_offset")
            if robot_swing is not None and offset is not None:
                swing_error = wrap_to_pi_torch(
                    robot_swing
                    - float(args.score_elbow_swing_scale) * human_swing
                    - offset.to(device=q.device, dtype=q.dtype)
                )
                elbow_swing = swing_error ** 2

    elbow_flexion = safe_tensor_term(elbow_flexion, q)
    elbow_swing = safe_tensor_term(elbow_swing, q)
    arm_plane = safe_tensor_term(arm_plane, q)

    total = (
        float(args.score_continuity_weight) * continuity
        + float(args.score_nominal_weight) * nominal
        + float(args.score_limit_weight) * limit_term
        + arm_confidence * float(args.score_elbow_flexion_weight) * elbow_flexion
        + arm_confidence * float(args.score_elbow_swing_weight) * elbow_swing
        + arm_confidence * float(args.score_arm_plane_weight) * arm_plane
    )
    if not torch.isfinite(total):
        total = huge

    def scalar(value: torch.Tensor) -> float:
        try:
            return float(value.detach().cpu().item())
        except Exception:
            return float("inf")

    return total, {
        "total": scalar(total),
        "continuity": scalar(continuity),
        "nominal": scalar(nominal),
        "limit": scalar(limit_term),
        "elbow_flexion": scalar(elbow_flexion),
        "elbow_swing": scalar(elbow_swing),
        "arm_plane": scalar(arm_plane),
        "robot_elbow_flexion_rad": tensor_to_float_or_none(robot_flex),
        "robot_elbow_flexion_deg": (
            None if tensor_to_float_or_none(robot_flex) is None else tensor_to_float_or_none(robot_flex) * 180.0 / math.pi
        ),
        "robot_elbow_swing_rad": tensor_to_float_or_none(robot_swing),
        "robot_elbow_swing_deg": (
            None if tensor_to_float_or_none(robot_swing) is None else tensor_to_float_or_none(robot_swing) * 180.0 / math.pi
        ),
        "human_elbow_flexion_rad": tensor_to_float_or_none(human_flex),
        "human_elbow_swing_rad": tensor_to_float_or_none(human_swing),
        "elbow_swing_offset_rad": tensor_to_float_or_none(elbow_swing_state.get("human_robot_swing_offset")),
        "elbow_swing_offset_status": elbow_swing_state.get("elbow_swing_offset_status"),
        "arm_confidence": arm_confidence,
        "robot_arm_valid": robot_arm_valid,
        "robot_arm_reason": robot_arm_reason,
    }


def select_best_ik_solution(
    candidates: List[torch.Tensor],
    q_prev: torch.Tensor,
    q_nominal: torch.Tensor,
    q_min,
    q_max,
    latest_arm_config: dict,
    ik,
    target_link,
    args,
    elbow_swing_state: dict,
) -> Tuple[Optional[torch.Tensor], dict]:
    if not candidates:
        return None, {"num_candidates": 0}

    best_q = None
    best_score = None
    best_info = {"num_candidates": len(candidates), "best_index": None}
    for idx, candidate in enumerate(candidates):
        score, info = score_candidate_q(
            candidate,
            q_prev=q_prev,
            q_nominal=q_nominal,
            q_min=q_min,
            q_max=q_max,
            latest_arm_config=latest_arm_config,
            ik=ik,
            target_link=target_link,
            args=args,
            elbow_swing_state=elbow_swing_state,
        )
        if args.score_debug:
            offset = info.get("elbow_swing_offset_rad")
            print(
                "[DEBUG] IK candidate "
                f"{idx}: total={info['total']:.6f}, cont={info['continuity']:.6f}, "
                f"nom={info['nominal']:.6f}, limit={info['limit']:.6f}, "
                f"flex={info['elbow_flexion']:.6f}, swing={info['elbow_swing']:.6f}, "
                f"plane={info['arm_plane']:.6f}, "
                f"robot_flex={gui_float(info.get('robot_elbow_flexion_deg'), 2)} deg, "
                f"robot_swing={gui_float(info.get('robot_elbow_swing_deg'), 2)} deg, "
                f"human_flex={gui_float(latest_arm_config.get('elbow_flexion_angle_deg'), 2)} deg, "
                f"human_swing={gui_float(latest_arm_config.get('elbow_swing_angle_signed_deg'), 2)} deg, "
                f"offset={gui_float(None if offset is None else offset * 180.0 / math.pi, 2)} deg"
            )

        if best_score is None or bool((score < best_score).item()):
            best_score = score
            best_q = candidate.reshape(-1)
            best_info = {"best_index": idx, "num_candidates": len(candidates), **info}

    if best_q is None or best_score is None or not torch.isfinite(best_score):
        return None, {"num_candidates": len(candidates), "best_index": None}

    return best_q, best_info


def main():
    args = parse_args()

    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        connect_ip=args.host,
        connect_port=args.port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )
    server = visualizer._server

    print("[INFO] Initializing cuRobo IK solver...")
    if args.use_cuda_graph:
        config = InverseKinematicsCfg.create(
            robot=args.robot,
            optimizer_configs=["ik/lbfgs_ik.yml"],
            metrics_rollout="metrics_base.yml",
            transition_model="ik/transition_ik.yml",
            use_cuda_graph=True,
            num_seeds=args.num_seeds,
            seed_solver_num_seeds=args.seed_solver_num_seeds,
        )
    else:
        config = InverseKinematicsCfg.create(
            robot=args.robot,
            num_seeds=args.num_seeds,
        )
    ik = InverseKinematics(config)
    if hasattr(visualizer, "_kinematics"):
        ik._arm_scoring_kinematics = visualizer._kinematics
    if hasattr(ik.config, "use_lm_seed"):
        ik.config.use_lm_seed = False
    if hasattr(ik.config, "exit_early"):
        ik.config.exit_early = False

    target_link = ik.tool_frames[0]
    initial_state = ik.default_joint_state.clone()
    kin_state = ik.compute_kinematics(initial_state).clone()
    initial_pose = kin_state.tool_poses[target_link]
    robot_ee_origin = initial_pose.position.squeeze().detach().cpu().numpy().astype(np.float32)
    fixed_quat_wxyz = initial_pose.quaternion.squeeze().detach().cpu().numpy().astype(np.float32)
    fixed_quat_wxyz = parse_quat_wxyz(fixed_quat_wxyz)
    if fixed_quat_wxyz is None:
        fixed_quat_wxyz = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    robot_ee_origin_R = quat_to_matrix_wxyz(fixed_quat_wxyz)
    if robot_ee_origin_R is None:
        robot_ee_origin_R = np.eye(3, dtype=np.float32)
    human_ee_orientation_origin_R = None
    robot_ee_orientation_origin_R = robot_ee_origin_R.copy()

    current_state = ik.get_active_js(initial_state).unsqueeze(0)
    latest_q = current_state.clone()
    q_nominal = joint_state_position_1d(ik.get_active_js(initial_state)).detach().clone()
    q_min, q_max = get_active_joint_limits_or_none(
        ik,
        dof=int(q_nominal.numel()),
        device=q_nominal.device,
        dtype=q_nominal.dtype,
    )

    target_control = None
    if hasattr(visualizer, "_control_frames"):
        target_control = visualizer._control_frames.get(target_link)

    target_marker = server.scene.add_icosphere(
        "/mediapipe_wrist_ik/target",
        radius=0.025,
        color=(140, 140, 140),
        subdivisions=2,
        position=robot_ee_origin,
    )

    with server.gui.add_folder("MediaPipe Wrist IK"):
        gui_source = server.gui.add_text("target_source", initial_value=args.target_source, disabled=True)
        gui_status = server.gui.add_text("ik_status", initial_value="lost", disabled=True)
        gui_udp_valid = server.gui.add_text("udp_valid", initial_value="false", disabled=True)
        gui_udp_age = server.gui.add_text("udp_age_ms", initial_value="--", disabled=True)
        gui_udp_visibility = server.gui.add_text("visibility", initial_value="--", disabled=True)
        gui_udp_position = server.gui.add_text("udp_position_m", initial_value="--", disabled=True)
        gui_target_position = server.gui.add_text("target_ee_pos_m", initial_value="--", disabled=True)
        gui_udp_ee_quat = server.gui.add_text("udp_ee_quat_wxyz", initial_value="--", disabled=True)
        gui_target_quat = server.gui.add_text("target_quat_wxyz", initial_value="--", disabled=True)
        gui_arm_config_valid = server.gui.add_text("arm_config_valid", initial_value="false", disabled=True)
        gui_arm_config_confidence = server.gui.add_text("arm_config_confidence", initial_value="--", disabled=True)
        gui_elbow_flexion_deg = server.gui.add_text("elbow_flexion_deg", initial_value="--", disabled=True)
        gui_elbow_swing_deg = server.gui.add_text("elbow_swing_deg", initial_value="--", disabled=True)
        gui_elbow_swing_signed_deg = server.gui.add_text("elbow_swing_signed_deg", initial_value="--", disabled=True)
        gui_arm_plane_valid = server.gui.add_text("arm_plane_valid", initial_value="false", disabled=True)
        gui_arm_plane_normal_world = server.gui.add_text("arm_plane_normal_world", initial_value="--", disabled=True)
        gui_ik_return_seeds = server.gui.add_text("ik_return_seeds", initial_value=str(args.return_seeds), disabled=True)
        gui_ik_candidate_count = server.gui.add_text("ik_candidate_count", initial_value="--", disabled=True)
        gui_selected_candidate_index = server.gui.add_text("selected_candidate_index", initial_value="--", disabled=True)
        gui_score_total = server.gui.add_text("score_total", initial_value="--", disabled=True)
        gui_score_continuity = server.gui.add_text("score_continuity", initial_value="--", disabled=True)
        gui_score_nominal = server.gui.add_text("score_nominal", initial_value="--", disabled=True)
        gui_score_limit = server.gui.add_text("score_limit", initial_value="--", disabled=True)
        gui_score_elbow_flexion = server.gui.add_text("score_elbow_flexion", initial_value="--", disabled=True)
        gui_score_elbow_swing = server.gui.add_text("score_elbow_swing", initial_value="--", disabled=True)
        gui_score_arm_plane = server.gui.add_text("score_arm_plane", initial_value="--", disabled=True)
        gui_robot_arm_valid = server.gui.add_text("robot_arm_valid", initial_value="false", disabled=True)
        gui_robot_arm_reason = server.gui.add_text("robot_arm_reason", initial_value="--", disabled=True)
        gui_robot_elbow_flexion_deg = server.gui.add_text("robot_elbow_flexion_deg", initial_value="--", disabled=True)
        gui_robot_elbow_swing_deg = server.gui.add_text("robot_elbow_swing_deg", initial_value="--", disabled=True)
        gui_elbow_swing_offset_deg = server.gui.add_text("elbow_swing_offset_deg", initial_value="--", disabled=True)
        gui_elbow_swing_offset_status = server.gui.add_text("elbow_swing_offset_status", initial_value="not initialized", disabled=True)
        gui_arm_score_confidence = server.gui.add_text("arm_score_confidence", initial_value="--", disabled=True)
        reset_origin_btn = server.gui.add_button("Reset delta origin")
        reset_swing_offset_btn = server.gui.add_button("Reset elbow swing offset")

    state = {
        "reset_delta_origin": False,
        "robot_plane_ref": None,
        "human_robot_swing_offset": None,
        "elbow_swing_offset_status": "not initialized",
    }

    @reset_origin_btn.on_click
    def _reset_delta_origin(_):
        state["reset_delta_origin"] = True
        state["robot_plane_ref"] = None
        state["human_robot_swing_offset"] = None
        state["elbow_swing_offset_status"] = "reset with delta origin"
        gui_set(gui_elbow_swing_offset_status, state["elbow_swing_offset_status"])
        update_current_robot_arm_gui()

    @reset_swing_offset_btn.on_click
    def _reset_swing_offset(_):
        state["robot_plane_ref"] = None
        state["human_robot_swing_offset"] = None
        state["elbow_swing_offset_status"] = "manual reset"
        gui_set(gui_elbow_swing_offset_status, state["elbow_swing_offset_status"])
        update_current_robot_arm_gui()

    def clear_score_gui():
        gui_set(gui_ik_candidate_count, "--")
        gui_set(gui_selected_candidate_index, "--")
        gui_set(gui_score_total, "--")
        gui_set(gui_score_continuity, "--")
        gui_set(gui_score_nominal, "--")
        gui_set(gui_score_limit, "--")
        gui_set(gui_score_elbow_flexion, "--")
        gui_set(gui_score_elbow_swing, "--")
        gui_set(gui_score_arm_plane, "--")
        gui_set(gui_elbow_swing_offset_deg, "--")
        gui_set(gui_elbow_swing_offset_status, state.get("elbow_swing_offset_status", "--"))
        gui_set(gui_arm_score_confidence, "--")

    def update_score_gui(info: dict):
        gui_set(gui_ik_candidate_count, info.get("num_candidates", "--"))
        gui_set(gui_selected_candidate_index, info.get("best_index", "--"))
        gui_set(gui_score_total, gui_float(info.get("total"), 6))
        gui_set(gui_score_continuity, gui_float(info.get("continuity"), 6))
        gui_set(gui_score_nominal, gui_float(info.get("nominal"), 6))
        gui_set(gui_score_limit, gui_float(info.get("limit"), 6))
        gui_set(gui_score_elbow_flexion, gui_float(info.get("elbow_flexion"), 6))
        gui_set(gui_score_elbow_swing, gui_float(info.get("elbow_swing"), 6))
        gui_set(gui_score_arm_plane, gui_float(info.get("arm_plane"), 6))
        if "robot_arm_valid" in info:
            gui_set(gui_robot_arm_valid, bool(info.get("robot_arm_valid", False)))
            gui_set(gui_robot_arm_reason, info.get("robot_arm_reason") or "--")
        if "robot_elbow_flexion_deg" in info:
            gui_set(gui_robot_elbow_flexion_deg, gui_float(info.get("robot_elbow_flexion_deg"), 2))
        if "robot_elbow_swing_deg" in info:
            gui_set(gui_robot_elbow_swing_deg, gui_float(info.get("robot_elbow_swing_deg"), 2))
        offset = info.get("elbow_swing_offset_rad")
        gui_set(gui_elbow_swing_offset_deg, "--" if offset is None else gui_float(offset * 180.0 / math.pi, 2))
        gui_set(gui_elbow_swing_offset_status, info.get("elbow_swing_offset_status") or state.get("elbow_swing_offset_status", "--"))
        gui_set(gui_arm_score_confidence, gui_float(info.get("arm_confidence"), 3))

    def update_current_robot_arm_gui():
        info = compute_robot_arm_metric_info(
            ik=ik,
            q=joint_state_position_1d(current_state).detach(),
            target_link=target_link,
            args=args,
            elbow_swing_state=state,
            initialize_ref=True,
        )
        gui_set(gui_robot_arm_valid, bool(info.get("robot_arm_valid", False)))
        gui_set(gui_robot_arm_reason, info.get("robot_arm_reason") or "--")
        gui_set(gui_robot_elbow_flexion_deg, gui_float(info.get("robot_elbow_flexion_deg"), 2))
        gui_set(gui_robot_elbow_swing_deg, gui_float(info.get("robot_elbow_swing_deg"), 2))

    update_current_robot_arm_gui()

    receiver = None
    if args.target_source == "udp":
        receiver = UdpWristTargetReceiver(args.udp_host, args.udp_port).start()

    print(f"\ncuRobo wrist IK viewer: http://localhost:{args.port}")
    print(f"Robot: {args.robot}, target link: {target_link}")
    print(f"Target source: {args.target_source}, mapping mode: {args.mapping_mode}")
    print(f"IK return_seeds: {args.return_seeds}")
    print(f"[INFO] robot shoulder link for scoring: {args.robot_shoulder_link}")
    print(f"[INFO] robot elbow link for scoring: {args.robot_elbow_link}")
    print(f"[INFO] robot wrist link for scoring: {args.robot_wrist_link or target_link}")
    print("This script only visualizes IK. It does not command a real robot.")
    print("Press Ctrl+C to exit.\n")

    human_origin = None
    filtered_target = None
    previous_mouse_pose = None
    last_packet_key = None
    # latest_arm_config will be used later for redundancy scoring:
    # E_elbow / E_plane / elbow_swing preference should be computed after multiple IK solutions are returned.
    latest_arm_config = packet_arm_config(None)
    sleep_dt = 1.0 / max(1.0, float(args.loop_hz))

    try:
        while True:
            now_t = time.time()
            target_valid = False
            target_position = None
            target_quat = fixed_quat_wxyz
            udp_packet = None
            udp_age_s = None
            solve_requested = False

            if args.target_source == "mouse":
                poses = visualizer.get_control_frame_pose()
                mouse_pose = poses[target_link]
                mouse_pos = mouse_pose.position.squeeze().detach().cpu().numpy().astype(np.float32)
                mouse_quat = mouse_pose.quaternion.squeeze().detach().cpu().numpy().astype(np.float32)
                target_position = mouse_pos
                if not args.fixed_orientation:
                    target_quat = mouse_quat
                target_valid = True

                pose_key = (
                    tuple(np.round(mouse_pos, 5).tolist()),
                    tuple(np.round(mouse_quat, 5).tolist()),
                )
                solve_requested = pose_key != previous_mouse_pose
                previous_mouse_pose = pose_key
                gui_set(gui_udp_valid, "n/a")
                gui_set(gui_udp_age, "n/a")
                gui_set(gui_udp_visibility, "n/a")
                gui_set(gui_udp_position, "n/a")
                gui_set(gui_udp_ee_quat, "n/a")
                gui_set(gui_arm_config_valid, "n/a")
                gui_set(gui_arm_config_confidence, "n/a")
                gui_set(gui_elbow_flexion_deg, "n/a")
                gui_set(gui_elbow_swing_deg, "n/a")
                gui_set(gui_elbow_swing_signed_deg, "n/a")
                gui_set(gui_arm_plane_valid, "n/a")
                gui_set(gui_arm_plane_normal_world, "n/a")
                clear_score_gui()

            else:
                udp_packet, recv_t = receiver.get_latest() if receiver is not None else (None, None)
                udp_age_s = None if recv_t is None else now_t - recv_t
                packet_is_fresh = udp_age_s is not None and udp_age_s <= args.target_timeout_s
                source_pos = packet_position(udp_packet, args.mapping_mode)
                source_quat = packet_ee_quat(udp_packet) if args.use_udp_ee_orientation else None
                arm_config = packet_arm_config(udp_packet)
                latest_arm_config = arm_config
                target_valid = bool(packet_is_fresh and source_pos is not None)

                if target_valid:
                    if state["reset_delta_origin"]:
                        human_origin = None
                        human_ee_orientation_origin_R = None
                        state["robot_plane_ref"] = None
                        state["human_robot_swing_offset"] = None
                        state["elbow_swing_offset_status"] = "reset with delta origin"
                        robot_ee_origin, current_ee_quat = latest_ee_pose(
                            ik, latest_q, target_link, robot_ee_origin, fixed_quat_wxyz
                        )
                        current_ee_R = quat_to_matrix_wxyz(current_ee_quat)
                        if current_ee_R is not None:
                            robot_ee_orientation_origin_R = current_ee_R
                        state["reset_delta_origin"] = False

                    if args.mapping_mode == "identity":
                        target_position = source_pos
                    elif args.mapping_mode == "manual":
                        target_position = apply_manual_axis_map(
                            source_pos, args.manual_axis_map, args.manual_translation
                        )
                    else:
                        if human_origin is None:
                            human_origin = source_pos.copy()
                            robot_ee_origin = latest_ee_position(ik, latest_q, target_link, robot_ee_origin)
                            print(
                                "[INFO] Set delta origins: "
                                f"human={human_origin.tolist()}, robot_ee={robot_ee_origin.tolist()}"
                            )
                        target_position = robot_ee_origin + float(args.teleop_scale) * (source_pos - human_origin)

                    if args.use_udp_ee_orientation and source_quat is not None:
                        R_human_now = quat_to_matrix_wxyz(source_quat)
                        if R_human_now is not None:
                            if human_ee_orientation_origin_R is None:
                                human_ee_orientation_origin_R = R_human_now.copy()
                                print("[INFO] Set orientation delta origin from UDP q_world_ee_target_wxyz")

                            delta_R = R_human_now @ human_ee_orientation_origin_R.T
                            R_robot_target = delta_R @ robot_ee_orientation_origin_R
                            mapped_quat = matrix_to_quat_wxyz(R_robot_target)
                            if mapped_quat is not None:
                                target_quat = mapped_quat

                    packet_key = (
                        udp_packet.get("frame_id"),
                        udp_packet.get("stamp_ms"),
                    )
                    solve_requested = packet_key != last_packet_key
                    last_packet_key = packet_key
                else:
                    if state["reset_delta_origin"]:
                        human_origin = None
                        human_ee_orientation_origin_R = None
                        state["robot_plane_ref"] = None
                        state["human_robot_swing_offset"] = None
                        state["elbow_swing_offset_status"] = "reset with delta origin"
                        state["reset_delta_origin"] = False

                gui_set(gui_udp_valid, bool(udp_packet.get("valid", False)) if udp_packet else False)
                gui_set(gui_udp_age, "--" if udp_age_s is None else f"{udp_age_s * 1000.0:.1f}")
                gui_set(gui_udp_visibility, "--" if not udp_packet else udp_packet.get("visibility"))
                source_display = packet_position(udp_packet, "delta") if udp_packet else None
                gui_set(gui_udp_position, "--" if source_display is None else np.round(source_display, 4).tolist())
                gui_set(gui_udp_ee_quat, "--" if source_quat is None else np.round(source_quat, 4).tolist())
                gui_set(gui_arm_config_valid, bool(arm_config["valid"]))
                gui_set(gui_arm_config_confidence, gui_float(arm_config["confidence"], 3))
                gui_set(gui_elbow_flexion_deg, gui_float(arm_config["elbow_flexion_angle_deg"], 2))
                gui_set(gui_elbow_swing_deg, gui_float(arm_config["elbow_swing_angle_deg"], 2))
                gui_set(gui_elbow_swing_signed_deg, gui_float(arm_config["elbow_swing_angle_signed_deg"], 2))
                gui_set(gui_arm_plane_valid, bool(arm_config["plane_valid"]))
                gui_set(gui_arm_plane_normal_world, gui_vec(arm_config["arm_plane_normal_world"], 4))

            if not target_valid or target_position is None:
                gui_set(gui_status, "lost")
                clear_score_gui()
                update_marker(target_marker, filtered_target if filtered_target is not None else robot_ee_origin, (140, 140, 140))
                time.sleep(sleep_dt)
                continue

            target_position = clamp_position(target_position.astype(np.float32), args)
            filtered_target = filter_target(target_position, filtered_target, args)

            if target_control is not None and args.target_source == "udp":
                target_control.position = filtered_target
                target_control.wxyz = target_quat

            gui_set(gui_target_position, np.round(filtered_target, 4).tolist())
            gui_set(gui_target_quat, np.round(target_quat, 4).tolist())

            if not solve_requested:
                time.sleep(sleep_dt)
                continue

            goal_pose = make_pose(filtered_target, target_quat)
            try:
                active_js = ik.get_active_js(current_state)
                result = ik.solve_pose(
                    goal_tool_poses=GoalToolPose.from_poses(
                        {target_link: goal_pose},
                        ordered_tool_frames=[target_link],
                        num_goalset=1,
                    ),
                    current_state=active_js.squeeze(1).clone(),
                    return_seeds=max(1, int(args.return_seeds)),
                )
            except Exception as exc:
                gui_set(gui_status, f"ik_error: {exc}")
                print(f"[WARN] IK solve error: {exc}")
                update_marker(target_marker, filtered_target, (255, 0, 0))
                clear_score_gui()
                time.sleep(sleep_dt)
                continue

            if ik_success(result):
                candidates = extract_successful_ik_candidates(result)
                q_prev = joint_state_position_1d(current_state).detach().clone()
                q_best, score_info = select_best_ik_solution(
                    candidates=candidates,
                    q_prev=q_prev,
                    q_nominal=q_nominal,
                    q_min=q_min,
                    q_max=q_max,
                    latest_arm_config=latest_arm_config,
                    ik=ik,
                    target_link=target_link,
                    args=args,
                    elbow_swing_state=state,
                )
                if q_best is not None:
                    joint_names = getattr(result.js_solution, "joint_names", None) or ik.joint_names
                    current_state = joint_state_from_q(q_best, joint_names=joint_names)
                    latest_q = current_state.clone()
                    visualizer.set_joint_state(current_state.squeeze(0).squeeze(0))
                    update_marker(target_marker, filtered_target, (0, 220, 0))
                    update_score_gui(score_info)
                    gui_set(gui_status, "success_selected")
                else:
                    update_marker(target_marker, filtered_target, (255, 0, 0))
                    update_score_gui(score_info)
                    gui_set(gui_status, "fail_no_candidate")
            else:
                update_marker(target_marker, filtered_target, (255, 0, 0))
                clear_score_gui()
                gui_set(gui_status, "fail")

            time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting wrist IK viewer.")
    finally:
        if receiver is not None:
            receiver.stop()


def latest_ee_position(
    ik: InverseKinematics,
    joint_state,
    target_link: str,
    fallback: np.ndarray,
) -> np.ndarray:
    try:
        state = ik.compute_kinematics(joint_state.squeeze(0).squeeze(0)).clone()
        pose = state.tool_poses[target_link]
        return pose.position.squeeze().detach().cpu().numpy().astype(np.float32)
    except Exception:
        return fallback.copy()


def latest_ee_pose(ik, joint_state, target_link, fallback_pos, fallback_quat):
    try:
        state = ik.compute_kinematics(joint_state.squeeze(0).squeeze(0)).clone()
        pose = state.tool_poses[target_link]
        pos = pose.position.squeeze().detach().cpu().numpy().astype(np.float32)
        quat = pose.quaternion.squeeze().detach().cpu().numpy().astype(np.float32)
        quat = parse_quat_wxyz(quat)
        if quat is None:
            quat = fallback_quat.copy()
        return pos, quat
    except Exception:
        return fallback_pos.copy(), fallback_quat.copy()


if __name__ == "__main__":
    main()
