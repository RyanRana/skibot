"""Body motion from a phone in a pocket: steps, and the per-second features a hike stores in SpacetimeDB.

Shared by ground/validate.py (accuracy checks) and ground/server.py (what goes into the hike_chunk table).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from course import LocalFrame

MAX_HACC_M = 30.0   # same GPS quality cut as register.py
TRIM_M = 200.0      # same privacy trim as register.py: nothing within 200 m of either end is stored
SLOPE_WINDOW_S = 10


def up_axis(imu: np.ndarray) -> np.ndarray:
    g = imu["grav"].astype(np.float64)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def vertical_acc(imu: np.ndarray) -> np.ndarray:
    """User acceleration along gravity, so it does not depend on how the phone sits in a pocket."""
    return (imu["acc"].astype(np.float64) * up_axis(imu)).sum(1)


def step_peaks(imu: np.ndarray, rate: float) -> np.ndarray:
    """Steps from vertical acceleration: low-passed at 3 Hz, one peak per step."""
    if len(imu) < rate * 2:
        return np.zeros(0, dtype=int)
    a = vertical_acc(imu)
    b, aa = butter(2, 3.0 / (rate / 2))
    v = filtfilt(b, aa, a)
    moving = np.abs(v) > 0.05
    if moving.sum() < rate:
        return np.zeros(0, dtype=int)
    peaks, _ = find_peaks(-v if np.abs(v.min()) > v.max() else v, distance=int(0.3 * rate),
                          prominence=0.6 * v[moving].std())
    return peaks


def per_second(meta: dict, data: dict[str, np.ndarray]) -> list[dict]:
    """One row per second of the hike where GPS is good, past the first and last TRIM_M metres. Keys are the
    SpacetimeDB HikeSample fields (snake_case, as its HTTP API expects for nested objects)."""
    imu, gps, baro = data["imu"], data["gps"], data["baro"]
    good = gps[(gps["hacc"] >= 0) & (gps["hacc"] <= MAX_HACC_M)] if len(gps) else gps
    if len(good) < 2:
        return []
    start = float(meta.get("started") or good["t"][0])
    t = np.arange(np.ceil(good["t"][0]), np.floor(good["t"][-1]) + 1, 1.0)
    if len(t) < 2:
        return []
    lat = np.interp(t, good["t"], good["lat"])
    lon = np.interp(t, good["t"], good["lon"])
    speed = np.interp(t, good["t"], np.maximum(good["speed"].astype(np.float64), 0))

    alt = np.interp(t, good["t"], good["alt"].astype(np.float64))
    rel = baro[np.isfinite(baro["rel_alt"])] if len(baro) else baro
    if len(rel) >= 2:  # barometer for shape, anchored to the GPS altitude
        r = np.interp(t, rel["t"], rel["rel_alt"].astype(np.float64))
        alt = r + np.median(alt - r)

    frame = LocalFrame(float(np.median(lat)), float(np.median(lon)))
    x, y = frame.to_xy(lat, lon)
    along = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    slope = np.zeros(len(t))
    h = SLOPE_WINDOW_S // 2
    for i in range(len(t)):
        a, b = max(0, i - h), min(len(t) - 1, i + h)
        run = along[b] - along[a]
        if run > 2.0:
            slope[i] = np.degrees(np.arctan2(alt[b] - alt[a], run))

    cadence = np.zeros(len(t))
    bounce = np.zeros(len(t))
    impact = np.zeros(len(t))
    if len(imu) > 200:
        rate = float(1 / np.median(np.diff(imu["t"])))
        va = vertical_acc(imu)
        steps = imu["t"][step_peaks(imu, rate)]
        sec = np.floor(imu["t"]).astype(np.int64)
        for i, s in enumerate(t.astype(np.int64)):
            m = sec == s
            if m.any():
                bounce[i] = va[m].std()
                impact[i] = np.abs(va[m]).max()
            cadence[i] = ((steps >= s - 5) & (steps < s + 5)).sum() * 6.0

    keep = (along >= TRIM_M) & (along <= along[-1] - TRIM_M)
    return [{"t_s": int(round(t[i] - start)), "lat": round(float(lat[i]), 6), "lon": round(float(lon[i]), 6),
             "alt_m": round(float(alt[i]), 2), "speed": round(float(speed[i]), 2), "slope_deg": round(float(slope[i]), 2),
             "cadence": round(float(cadence[i]), 1), "bounce": round(float(bounce[i]), 3), "impact": round(float(impact[i]), 3)}
            for i in np.nonzero(keep)[0]]
