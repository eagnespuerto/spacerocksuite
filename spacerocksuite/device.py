"""Work out roughly how strong this device is and pick sensible limits.

A Raspberry Pi 3B+ (4 cores, 1 GB) gets small thread pools, fewer orbit
samples and smaller stamps. A desktop gets more. Users can override every
value in Settings.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict


def _total_ram_mb() -> int:
    try:  # Linux / Pi
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullTotalPhys // (1024 * 1024))
        except Exception:
            pass
    try:  # macOS / BSD
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 2048


def _pi_model() -> str | None:
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(p, "rb") as f:
                return f.read().decode(errors="ignore").strip("\x00 \n")
        except OSError:
            continue
    return None


@dataclass
class Profile:
    tier: str              # "light" | "medium" | "full"
    cores: int
    ram_mb: int
    machine: str
    pi_model: str | None
    net_workers: int       # parallel archive queries
    cpu_workers: int       # parallel image jobs
    mc_samples: int        # Monte Carlo orbit samples for sigma clouds
    track_step_deg: float  # spacing of archive query points along the track
    stamp_px: int          # default stamp size for stacks
    max_frames_in_memory: int

    def as_dict(self):
        return asdict(self)


def detect() -> Profile:
    cores = os.cpu_count() or 1
    ram = _total_ram_mb()
    machine = platform.machine().lower()
    pi = _pi_model()
    arm = machine.startswith(("arm", "aarch"))
    if ram < 1500 or (arm and cores <= 4 and ram < 3000):
        tier = "light"
    elif ram < 6000 or cores <= 4:
        tier = "medium"
    else:
        tier = "full"
    table = {
        #          net cpu  mc    step stamp frames
        "light":  (3, 1, 200, 0.5, 101, 40),
        "medium": (5, max(1, cores // 2), 500, 0.5, 151, 150),
        "full":   (8, max(1, cores - 1), 1000, 0.5, 201, 600),
    }[tier]
    return Profile(tier, cores, ram, machine, pi, *table)


PROFILE = detect()

if __name__ == "__main__":
    import json
    print(json.dumps(PROFILE.as_dict(), indent=2))
