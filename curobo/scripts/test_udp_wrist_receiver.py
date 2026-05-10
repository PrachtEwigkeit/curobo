#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal UDP JSON receiver for mediapipe_pose_orbbec_visualizer.py.

It listens on 127.0.0.1:5557 and prints at most 10 packets per second so the
terminal stays readable while the visualizer publishes every frame.
"""

import json
import socket
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal UDP JSON receiver")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--max-print-hz", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)

    min_print_interval = 1.0 / max(1e-6, float(args.max_print_hz))
    next_print_t = 0.0

    print(f"[INFO] Listening UDP JSON on {args.host}:{args.port}")
    print("[INFO] Press Ctrl+C to exit.")

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue

            now_t = time.time()
            if now_t < next_print_t:
                continue
            next_print_t = now_t + min_print_interval

            try:
                packet = json.loads(data.decode("utf-8"))
            except Exception as exc:
                print(f"[WARN] Invalid JSON from {addr}: {exc}")
                continue

            print(f"\n[UDP] from {addr}")
            print(json.dumps(packet, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("\n[INFO] Receiver stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
