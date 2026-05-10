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
import socket
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
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
    parser.add_argument("--teleop-scale", type=float, default=0.5)
    parser.add_argument(
        "--fixed-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the initial robot EE orientation as a fixed wxyz quaternion",
    )

    parser.add_argument("--manual-axis-map", nargs=3, default=["x", "y", "z"])
    parser.add_argument("--manual-translation", nargs=3, type=float, default=[0.0, 0.0, 0.0])

    parser.add_argument("--workspace-x", nargs=2, type=float, default=[0.2, 0.8])
    parser.add_argument("--workspace-y", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--workspace-z", nargs=2, type=float, default=[0.1, 0.8])
    parser.add_argument("--filter-alpha", type=float, default=0.2)
    parser.add_argument("--max-step-m", type=float, default=0.03)
    parser.add_argument("--loop-hz", type=float, default=30.0)

    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--seed-solver-num-seeds", type=int, default=1)
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        default=False,
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

    current_state = ik.get_active_js(initial_state).unsqueeze(0)
    latest_q = current_state.clone()

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
        reset_origin_btn = server.gui.add_button("Reset delta origin")

    state = {"reset_delta_origin": False}

    @reset_origin_btn.on_click
    def _reset_delta_origin(_):
        state["reset_delta_origin"] = True

    receiver = None
    if args.target_source == "udp":
        receiver = UdpWristTargetReceiver(args.udp_host, args.udp_port).start()

    print(f"\ncuRobo wrist IK viewer: http://localhost:{args.port}")
    print(f"Robot: {args.robot}, target link: {target_link}")
    print(f"Target source: {args.target_source}, mapping mode: {args.mapping_mode}")
    print("This script only visualizes IK. It does not command a real robot.")
    print("Press Ctrl+C to exit.\n")

    human_origin = None
    filtered_target = None
    previous_mouse_pose = None
    last_packet_key = None
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

            else:
                udp_packet, recv_t = receiver.get_latest() if receiver is not None else (None, None)
                udp_age_s = None if recv_t is None else now_t - recv_t
                packet_is_fresh = udp_age_s is not None and udp_age_s <= args.target_timeout_s
                source_pos = packet_position(udp_packet, args.mapping_mode)
                target_valid = bool(packet_is_fresh and source_pos is not None)

                if target_valid:
                    if state["reset_delta_origin"]:
                        human_origin = None
                        robot_ee_origin = latest_ee_position(ik, latest_q, target_link, robot_ee_origin)
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

                    packet_key = (
                        udp_packet.get("frame_id"),
                        udp_packet.get("stamp_ms"),
                    )
                    solve_requested = packet_key != last_packet_key
                    last_packet_key = packet_key
                else:
                    if state["reset_delta_origin"]:
                        human_origin = None
                        state["reset_delta_origin"] = False

                gui_set(gui_udp_valid, bool(udp_packet.get("valid", False)) if udp_packet else False)
                gui_set(gui_udp_age, "--" if udp_age_s is None else f"{udp_age_s * 1000.0:.1f}")
                gui_set(gui_udp_visibility, "--" if not udp_packet else udp_packet.get("visibility"))
                source_display = packet_position(udp_packet, "delta") if udp_packet else None
                gui_set(gui_udp_position, "--" if source_display is None else np.round(source_display, 4).tolist())

            if not target_valid or target_position is None:
                gui_set(gui_status, "lost")
                update_marker(target_marker, filtered_target if filtered_target is not None else robot_ee_origin, (140, 140, 140))
                time.sleep(sleep_dt)
                continue

            target_position = clamp_position(target_position.astype(np.float32), args)
            filtered_target = filter_target(target_position, filtered_target, args)

            if target_control is not None and args.target_source == "udp":
                target_control.position = filtered_target
                target_control.wxyz = fixed_quat_wxyz

            gui_set(gui_target_position, np.round(filtered_target, 4).tolist())

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
                    return_seeds=1,
                )
            except Exception as exc:
                gui_set(gui_status, f"ik_error: {exc}")
                print(f"[WARN] IK solve error: {exc}")
                update_marker(target_marker, filtered_target, (255, 0, 0))
                time.sleep(sleep_dt)
                continue

            if ik_success(result):
                current_state = result.js_solution.clone()
                latest_q = result.js_solution.clone()
                visualizer.set_joint_state(result.js_solution.squeeze(0).squeeze(0))
                update_marker(target_marker, filtered_target, (0, 220, 0))
                gui_set(gui_status, "success")
            else:
                update_marker(target_marker, filtered_target, (255, 0, 0))
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


if __name__ == "__main__":
    main()
