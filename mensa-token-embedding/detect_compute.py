#!/usr/bin/env python3
import subprocess
import argparse
from collections import Counter

def get_gpus():
    """Try multiple methods to detect NVIDIA GPUs."""
    # 1) PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    # 2) NVIDIA-ML (pynvml)
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        names = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            names.append(pynvml.nvmlDeviceGetName(handle).decode())
        pynvml.nvmlShutdown()
        if names:
            return names
    except Exception:
        pass

    # 3) nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        )
        names = [l.strip() for l in out.splitlines() if l.strip()]
        if names:
            return names
    except Exception:
        pass

    # 4) lspci grep
    try:
        out = subprocess.check_output(
            ["lspci"], encoding="utf-8"
        )
        names = []
        for line in out.splitlines():
            if "NVIDIA" in line:
                # e.g. "01:00.0 VGA compatible controller: NVIDIA Corporation TU104 [GeForce RTX 2080] ..."
                part = line.split("NVIDIA Corporation")[-1].strip()
                names.append("NVIDIA" + part)
        if names:
            return names
    except Exception:
        pass

    return []

def get_cpus():
    """Try /proc/cpuinfo then lscpu to detect physical sockets and model names."""
    # 1) /proc/cpuinfo
    models = {}
    try:
        with open("/proc/cpuinfo") as f:
            phys = None
            for line in f:
                line = line.strip()
                if not line:
                    phys = None
                    continue
                if line.startswith("physical id"):
                    phys = line.split(":",1)[1].strip()
                elif line.startswith("model name") and phys is not None:
                    models[phys] = line.split(":",1)[1].strip()
        if models:
            return list(models.values())
    except Exception:
        pass

    # 2) lscpu
    try:
        out = subprocess.check_output(["lscpu"], encoding="utf-8")
        sockets = None
        model = None
        for line in out.splitlines():
            if line.startswith("Socket(s):"):
                sockets = int(line.split(":",1)[1].strip())
            elif line.startswith("Model name:"):
                model = line.split(":",1)[1].strip()
        if sockets and model:
            return [model] * sockets
    except Exception:
        pass

    return []

def plural(n, singular, plural=None):
    return singular if n == 1 else (plural or singular + "s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", help="Compute estimate (e.g. '1000 CPU-h')", default=None)
    args = parser.parse_args()

    gpus = get_gpus()
    if gpus:
        cnt = Counter(gpus)
        gpu_lines = [f"{n} {name} {plural(n,'GPU')}" for name,n in cnt.items()]
    else:
        gpu_lines = ["No NVIDIA GPU detected"]

    cpus = get_cpus()
    if cpus:
        cnt = Counter(cpus)
        cpu_lines = [f"{n} {model} {plural(n,'CPU')}" for model,n in cnt.items()]
    else:
        cpu_lines = ["Could not detect CPU model"]

    estimate = args.estimate or input("Enter total compute estimate (e.g. '1000 CPU-h'): ").strip()

    print("\nFor development:")
    for l in gpu_lines: print(l)
    for l in cpu_lines: print(l)
    print(f"Total compute estimate: {estimate}")

if __name__ == "__main__":
    main()
