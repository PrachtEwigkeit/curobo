#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UDP bridge from camera-frame wrist targets to world-frame wrist targets.

This script is not an IK solver. It does not open a camera, run MediaPipe, or
import cuRobo. Its only job is:

    camera-frame UDP target
        -> fixed extrinsic transform
        -> world-frame UDP target

Coordinate notes:
- camera coords are assumed to originate at the Orbbec optical center;
- world coords originate at t_world_camera, provided by the user;
- delta vectors are affected only by rotation, not translation;
- absolute positions use rotation plus translation;
- the default rotation assumes camera Z is horizontal/forward, and maps the
  camera axes to world as X_world=Z_camera, Y_world=-X_camera, Z_world=-Y_camera.
"""

import argparse
import json
import socket
import time
from typing import Optional

import numpy as np


DEFAULT_ROTATION_PRESET = "camera_z_forward_y_down_to_world_x_forward_z_up"

DEFAULT_R_WORLD_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="UDP camera-frame to world-frame target bridge")
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
        help='nine comma-separated row-major numbers, e.g. "0,0,1,1,0,0,0,-1,0"',
    )
    parser.add_argument(
        "--translation",
        type=str,
        default="0,0,0",
        help='three comma-separated numbers for t_world_camera, e.g. "0.2,0,0.5"',
    )
    parser.add_argument("--target-mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--teleop-scale", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


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


def parse_vec3(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def vec_to_list(value: Optional[np.ndarray]):
    if value is None:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def read_scale(packet: dict, override_scale: Optional[float]) -> float:
    if override_scale is not None:
        return float(override_scale)
    try:
        return float(packet.get("teleop_scale", 1.0))
    except (TypeError, ValueError):
        return 1.0


def make_world_packet(packet: dict, R: np.ndarray, t: np.ndarray, args) -> dict:
    valid = bool(packet.get("valid", False))

    position_camera = parse_vec3(packet.get("position_camera_m"))
    delta_camera = parse_vec3(packet.get("delta_camera_m"))
    target_camera = parse_vec3(packet.get("target_position_m"))

    position_world = None if position_camera is None else R @ position_camera + t
    delta_world = None if delta_camera is None else R @ delta_camera

    scale = read_scale(packet, args.teleop_scale)
    target_position_world = None

    if valid:
        if args.target_mode == "delta":
            if delta_world is not None:
                target_position_world = scale * delta_world
            elif target_camera is not None:
                # Fallback: the upstream packet may only provide target_position_m.
                # In that case rotate it as the target vector without applying an
                # extra scale, because it may already be a scaled delta.
                target_position_world = R @ target_camera
        else:
            absolute_camera = target_camera if target_camera is not None else position_camera
            if absolute_camera is not None:
                target_position_world = R @ absolute_camera + t

    output = dict(packet)
    output.update(
        {
            "valid": valid,
            "target_frame": "world",
            "position_camera_m": vec_to_list(position_camera),
            "delta_camera_m": vec_to_list(delta_camera),
            "position_world_m": vec_to_list(position_world),
            "delta_world_m": vec_to_list(delta_world),
            "target_position_world_m": vec_to_list(target_position_world),
            # Compatibility field for older consumers. In bridge output this is
            # now in world frame, not camera frame.
            "target_position_m": vec_to_list(target_position_world),
            "rotation_world_camera": R.astype(float).tolist(),
            "translation_world_camera_m": vec_to_list(t),
            "target_mode": args.target_mode,
            "teleop_scale": scale,
            "bridge_time_ms": int(time.time() * 1000),
        }
    )
    return output


def main():
    args = parse_args()
    R = load_rotation(args)
    t = parse_float_list(args.translation, 3, "--translation")

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((args.input_host, args.input_port))
    recv_sock.settimeout(0.1)

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    output_addr = (args.output_host, args.output_port)

    received_count = 0
    published_count = 0
    latest_valid = False
    latest_target = None
    next_status_t = time.time() + 1.0

    print(f"[INFO] camera_to_world_udp_bridge input  {args.input_host}:{args.input_port}")
    print(f"[INFO] camera_to_world_udp_bridge output {args.output_host}:{args.output_port}")
    print(f"[INFO] target_mode={args.target_mode}, teleop_scale_override={args.teleop_scale}")
    print("[INFO] R_world_camera =")
    print(R)
    print(f"[INFO] t_world_camera = {t.tolist()}")
    print("[INFO] Press Ctrl+C to exit.")

    try:
        while True:
            now_t = time.time()
            if now_t >= next_status_t:
                print(
                    "[STATUS] "
                    f"received={received_count} published={published_count} "
                    f"latest_valid={latest_valid} latest_target_world={latest_target}"
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

            received_count += 1
            try:
                output_packet = make_world_packet(packet, R, t, args)
            except Exception as exc:
                print(f"[WARN] Failed to transform packet: {exc}")
                continue

            payload = json.dumps(output_packet, separators=(",", ":")).encode("utf-8")
            try:
                send_sock.sendto(payload, output_addr)
            except OSError as exc:
                print(f"[WARN] Failed to publish output UDP packet: {exc}")
                continue

            published_count += 1
            latest_valid = bool(output_packet.get("valid", False))
            latest_target = output_packet.get("target_position_world_m")
            if args.verbose:
                print(json.dumps(output_packet, ensure_ascii=False, indent=2))

    except KeyboardInterrupt:
        print("\n[INFO] Bridge stopped.")
    finally:
        recv_sock.close()
        send_sock.close()


if __name__ == "__main__":
    main()
