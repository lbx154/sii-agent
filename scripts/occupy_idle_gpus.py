"""占卡脚本: transparently hold currently idle NVIDIA GPUs.

Run with:
  bash -c 'exec -a 占卡脚本 python scripts/occupy_idle_gpus.py'

The script only selects GPUs that look idle at startup. It sets the Linux
process name to "占卡脚本" and releases memory cleanly on SIGINT/SIGTERM.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


PROCESS_LABEL = "占卡脚本"
PR_SET_NAME = 15


@dataclass(frozen=True)
class GpuInfo:
    index: int
    total_mib: int
    used_mib: int
    util_pct: int


def _utf8_prefix(text: str, max_bytes: int) -> bytes:
    data = text.encode("utf-8")
    while len(data) > max_bytes:
        data = data[:-1]
        try:
            data.decode("utf-8")
            break
        except UnicodeDecodeError:
            continue
    return data


def set_process_label(label: str) -> None:
    name = _utf8_prefix(label, 15)
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(PR_SET_NAME, ctypes.c_char_p(name), 0, 0, 0)
    except Exception:
        pass
    try:
        with open("/proc/self/comm", "w", encoding="utf-8") as f:
            f.write(label[:15])
    except Exception:
        pass


def query_gpus() -> list[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        raise SystemExit("ERROR: nvidia-smi not found; this script requires NVIDIA GPUs.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ERROR: nvidia-smi failed:\n{exc.output}") from exc

    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            GpuInfo(
                index=int(parts[0]),
                total_mib=int(parts[1]),
                used_mib=int(parts[2]),
                util_pct=int(parts[3]),
            )
        )
    return gpus


def parse_gpu_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def select_idle_gpus(
    gpus: list[GpuInfo],
    requested: list[int] | None,
    max_used_mib: int,
    max_util_pct: int,
) -> list[GpuInfo]:
    by_index = {gpu.index: gpu for gpu in gpus}
    if requested is not None:
        missing = [idx for idx in requested if idx not in by_index]
        if missing:
            raise SystemExit(f"ERROR: requested GPU(s) not found: {missing}")
        return [by_index[idx] for idx in requested]
    return [
        gpu
        for gpu in gpus
        if gpu.used_mib <= max_used_mib and gpu.util_pct <= max_util_pct
    ]


def install_signal_handlers(allocations: list[object]) -> None:
    def _cleanup(signum: int, _frame: object) -> None:
        print(f"\n{PROCESS_LABEL}: received signal {signum}; releasing GPU memory.", flush=True)
        allocations.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)


def hold_gpu_memory(
    selected: list[GpuInfo],
    fraction: float,
    reserve_mib: int,
    chunk_mib: int,
    touch: bool,
) -> list[object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu.index) for gpu in selected)
    import torch

    allocations: list[object] = []
    install_signal_handlers(allocations)

    for logical_id, gpu in enumerate(selected):
        device = torch.device(f"cuda:{logical_id}")
        available_mib = max(0, gpu.total_mib - gpu.used_mib - reserve_mib)
        target_mib = min(int(gpu.total_mib * fraction), available_mib)
        if target_mib <= 0:
            print(f"{PROCESS_LABEL}: GPU {gpu.index} skipped; no allocatable memory after reserve.")
            continue

        remaining_mib = target_mib
        print(
            f"{PROCESS_LABEL}: holding GPU {gpu.index} "
            f"({target_mib} MiB target, reserve {reserve_mib} MiB).",
            flush=True,
        )
        while remaining_mib > 0:
            current_mib = min(chunk_mib, remaining_mib)
            try:
                tensor = torch.empty(current_mib * 1024 * 1024, dtype=torch.uint8, device=device)
                if touch:
                    tensor.fill_(1)
                allocations.append(tensor)
                remaining_mib -= current_mib
            except torch.cuda.OutOfMemoryError:
                if current_mib <= 64:
                    print(f"{PROCESS_LABEL}: GPU {gpu.index} hit OOM; keeping allocated chunks.")
                    break
                chunk_mib = max(64, current_mib // 2)
                torch.cuda.empty_cache()
    return allocations


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{PROCESS_LABEL}: hold idle GPUs with a clear process label.")
    parser.add_argument("--gpus", help="Comma-separated physical GPU indexes to hold. Default: auto-select idle GPUs.")
    parser.add_argument("--fraction", type=float, default=0.90, help="Fraction of each GPU's total memory to hold.")
    parser.add_argument("--reserve-mib", type=int, default=1024, help="Memory to leave free on each selected GPU.")
    parser.add_argument("--idle-max-used-mib", type=int, default=256, help="Auto-select GPUs with used memory at or below this.")
    parser.add_argument("--idle-max-util-pct", type=int, default=5, help="Auto-select GPUs with utilization at or below this.")
    parser.add_argument("--chunk-mib", type=int, default=1024, help="Allocation chunk size.")
    parser.add_argument("--sleep-seconds", type=int, default=60, help="Heartbeat interval while holding GPUs.")
    parser.add_argument("--touch", action="store_true", help="Write to allocated tensors after allocation.")
    args = parser.parse_args()

    set_process_label(PROCESS_LABEL)
    if not 0 < args.fraction <= 1:
        raise SystemExit("ERROR: --fraction must be in (0, 1].")

    all_gpus = query_gpus()
    selected = select_idle_gpus(
        all_gpus,
        parse_gpu_list(args.gpus),
        args.idle_max_used_mib,
        args.idle_max_util_pct,
    )
    if not selected:
        print(f"{PROCESS_LABEL}: no idle GPUs found; exiting.")
        return 0

    print(f"{PROCESS_LABEL}: selected physical GPU(s): {', '.join(str(g.index) for g in selected)}", flush=True)
    print(f"{PROCESS_LABEL}: kill this process to release the GPUs.", flush=True)
    allocations = hold_gpu_memory(selected, args.fraction, args.reserve_mib, args.chunk_mib, args.touch)
    if not allocations:
        print(f"{PROCESS_LABEL}: no GPU memory was allocated; exiting.")
        return 1

    while True:
        held_mib = sum(getattr(tensor, "numel", lambda: 0)() for tensor in allocations) // (1024 * 1024)
        print(f"{PROCESS_LABEL}: still holding about {held_mib} MiB across {len(selected)} GPU(s).", flush=True)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
