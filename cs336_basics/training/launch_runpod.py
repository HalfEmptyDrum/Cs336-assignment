"""Launch a 1x H100 RunPod instance for CS336 training.

Usage:
    python launch_runpod.py              # launch pod, print SSH command
    python launch_runpod.py --stop <id>  # terminate a pod
"""

import argparse
import sys
import time

import runpod

import os

runpod.api_key = os.environ["RUNPOD_API_KEY"]

POD_NAME = "cs336-training"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_TYPE = "NVIDIA H100 80GB HBM3"   # fallback: "NVIDIA H100 PCIe"


def launch() -> None:
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE,
        gpu_type_id=GPU_TYPE,
        gpu_count=1,
        volume_in_gb=50,
        container_disk_in_gb=20,
        volume_mount_path="/workspace",
        ports="22/tcp,8888/http",
        support_public_ip=True,
        start_ssh=True,
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    print("Waiting for SSH (30-90s)...")

    for _ in range(60):
        info = runpod.get_pod(pod_id) or {}
        runtime = info.get("runtime") or {}
        ports = runtime.get("ports") or []
        ssh = next((p for p in ports if p.get("privatePort") == 22), None)
        if ssh and ssh.get("ip") and ssh.get("publicPort"):
            print()
            print("Pod is live.")
            print(f"  ID:   {pod_id}")
            print(f"  SSH:  ssh root@{ssh['ip']} -p {ssh['publicPort']}")
            print()
            print(f"Stop when done:  python {sys.argv[0]} --stop {pod_id}")
            return
        time.sleep(5)

    print(f"Pod {pod_id} did not expose SSH in 5 minutes. Check https://runpod.io/console/pods")


def stop(pod_id: str) -> None:
    runpod.terminate_pod(pod_id)
    print(f"Terminated pod {pod_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", metavar="POD_ID", help="Terminate the given pod and exit")
    args = parser.parse_args()

    if args.stop:
        stop(args.stop)
    else:
        launch()


if __name__ == "__main__":
    main()
