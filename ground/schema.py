"""Binary layout of a Ground Truth phone session. The iOS recorder (ios/GroundTruth/Sources/SessionWriter.swift)
writes exactly these records, little endian and packed, one file per kind per 5 minute segment:

    imu_000.bin   100 Hz  t (unix s, f8), acc[3] (user acceleration, m/s^2), grav[3] (m/s^2), gyro[3] (rad/s),
                          quat[4] (x, y, z, w, attitude in a z-up frame with arbitrary corrected yaw), mag[3] (uT, NaN
                          until the magnetometer is calibrated)
    gps_000.bin   ~1 Hz   t, lat, lon (f8), alt (m above sea level), hacc, vacc (m), speed (m/s), course (deg)
    baro_000.bin  ~1 Hz   t, rel_alt (m since start), kpa, abs_alt (m), abs_acc (m). Relative and absolute altitude
                          arrive as separate updates, so each row fills one pair and leaves the other NaN.
    meta.json     written at the end: session id, placement, consent, device, gaps, pedometer, and every data file
                  with its byte size, so the server knows when the upload is complete.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

IMU = np.dtype([("t", "<f8"), ("acc", "<f4", 3), ("grav", "<f4", 3), ("gyro", "<f4", 3), ("quat", "<f4", 4),
                ("mag", "<f4", 3)])
GPS = np.dtype([("t", "<f8"), ("lat", "<f8"), ("lon", "<f8"), ("alt", "<f4"), ("hacc", "<f4"), ("vacc", "<f4"),
                ("speed", "<f4"), ("course", "<f4")])
BARO = np.dtype([("t", "<f8"), ("rel_alt", "<f4"), ("kpa", "<f4"), ("abs_alt", "<f4"), ("abs_acc", "<f4")])
KINDS = {"imu": IMU, "gps": GPS, "baro": BARO}
FILE_RE = re.compile(r"^(imu|gps|baro)_(\d{3})\.bin$")

assert (IMU.itemsize, GPS.itemsize, BARO.itemsize) == (72, 44, 24)  # must match SessionWriter.swift


def load_session(d: Path) -> tuple[dict, dict[str, np.ndarray], list[str]]:
    """meta, {kind: records sorted by time}, warnings. A trailing partial record (app killed mid write) is dropped
    and reported; anything else malformed raises."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    parts: dict[str, list[np.ndarray]] = {k: [] for k in KINDS}
    warnings = []
    for name in sorted(meta["files"]):
        m = FILE_RE.match(name)
        if not m:
            raise ValueError(f"unexpected file in meta: {name}")
        dt = KINDS[m.group(1)]
        raw = (d / name).read_bytes()
        extra = len(raw) % dt.itemsize
        if extra:
            warnings.append(f"{name}: dropped a partial trailing record ({extra} bytes)")
            raw = raw[: len(raw) - extra]
        parts[m.group(1)].append(np.frombuffer(raw, dtype=dt))
    data = {}
    for k, ps in parts.items():
        a = np.concatenate(ps) if ps else np.zeros(0, dtype=KINDS[k])
        data[k] = a[np.argsort(a["t"], kind="stable")]
    return meta, data, warnings
