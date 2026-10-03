"""Checks that a phone's motion data is accurate enough to train on. Record one short session per test with the app
(see ground/PHONE_TEST.md), then:

    python -m ground.validate list                              # sessions, newest first, to find the ids
    python -m ground.validate rest     <id|latest>              # phone flat and still for 60 s
    python -m ground.validate steps    <id|latest> --count 100  # walk exactly 100 counted steps
    python -m ground.validate rotate   <id|latest> --turns 3    # spin the phone flat on a table, same direction
    python -m ground.validate distance <id|latest> --meters 400 [--climb-m 12]   # a route of known length
    python -m ground.validate report                            # every result so far as one markdown table

Each check prints measured value, target and pass/fail, and saves validation_<test>.json in the session folder.
The targets are ours, set from typical phone IMU performance; they are not Apple specifications.

What this can and cannot show: the app records Core Motion's fused device motion. Gravity there comes out of
Apple's sensor fusion and is about 1 g by construction, so "does gravity read 9.81" proves nothing and is not
checked. These tests check what the data is used for: noise floor and bias at rest, drift, step timing, rotation
angle, distance and climb against known answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from ground.register import GAP_S, climb
from ground.schema import load_session

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "out" / "ground" / "sessions"
EDGE_S = 5.0  # ignore the first and last seconds of a rest/rotate test (hands on the phone)


# ------------------------------------------------------------------------------------------------ helpers

def resolve(root: Path, ref: str) -> Path:
    dirs = [d for d in root.iterdir() if (d / "meta.json").exists()] if root.exists() else []
    if not dirs:
        raise SystemExit(f"no uploaded sessions in {root}")
    if ref == "latest":
        return max(dirs, key=lambda d: json.loads((d / "meta.json").read_text()).get("started", 0))
    hits = [d for d in dirs if d.name.startswith(ref)]
    if len(hits) != 1:
        raise SystemExit(f"{ref!r} matches {len(hits)} sessions; use more of the id (python -m ground.validate list)")
    return hits[0]


def up_axis(imu: np.ndarray) -> np.ndarray:
    g = imu["grav"].astype(np.float64)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def yaw_deg(imu: np.ndarray) -> np.ndarray:
    x, y, z, w = (imu["quat"][:, i].astype(np.float64) for i in range(4))
    return np.degrees(np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def interior(imu: np.ndarray, edge: float = EDGE_S) -> np.ndarray:
    t = imu["t"]
    keep = (t >= t[0] + edge) & (t <= t[-1] - edge)
    return imu[keep] if keep.sum() > 100 else imu


def check(name: str, value: float, target: str, ok: bool, unit: str = "") -> dict:
    return {"check": name, "measured": f"{value:.4g}{unit}", "target": target, "pass": bool(ok)}


def timing(imu: np.ndarray) -> list[dict]:
    """Every test also checks the stream itself: rate, timing jitter, holes."""
    if len(imu) < 2:
        return [check("motion samples", len(imu), "> 0", False)]
    dt = np.diff(imu["t"])
    steady = dt[dt <= GAP_S]
    rate = 1 / np.median(steady)
    gaps = int((dt > GAP_S).sum())
    return [check("sample rate", rate, "95-105 Hz", 95 <= rate <= 105, " Hz"),
            check("timing jitter (p99 interval)", np.percentile(steady, 99) * 1000, "< 20 ms",
                  np.percentile(steady, 99) < 0.020, " ms"),
            check("gaps over 0.5 s", gaps, "0", gaps == 0)]


# ------------------------------------------------------------------------------------------------ tests

def test_rest(d: Path, args) -> tuple[list[dict], dict]:
    meta, data, _ = load_session(d)
    imu = interior(data["imu"])
    acc, gyro = imu["acc"].astype(np.float64), imu["gyro"].astype(np.float64)
    up = up_axis(imu)
    mean_up = up.mean(0) / np.linalg.norm(up.mean(0))
    tilt = np.degrees(np.arccos(np.clip(up @ mean_up, -1, 1)))
    yaw = yaw_deg(imu)
    minutes = (imu["t"][-1] - imu["t"][0]) / 60
    drift = abs(np.polyfit((imu["t"] - imu["t"][0]) / 60, yaw, 1)[0]) if len(imu) > 10 else np.nan
    m = {"duration_s": round(minutes * 60, 1), "acc_bias_mps2": np.abs(acc.mean(0)).round(4).tolist(),
         "acc_noise_mps2": acc.std(0).round(4).tolist(), "gyro_bias_rads": np.abs(gyro.mean(0)).round(5).tolist(),
         "gyro_noise_rads": gyro.std(0).round(5).tolist()}
    checks = [
        check("held still long enough", minutes * 60, ">= 45 s", minutes * 60 >= 45, " s"),
        check("acceleration bias (worst axis)", np.abs(acc.mean(0)).max(), "< 0.05 m/s²", np.abs(acc.mean(0)).max() < 0.05, " m/s²"),
        check("acceleration noise (worst axis)", acc.std(0).max(), "< 0.10 m/s²", acc.std(0).max() < 0.10, " m/s²"),
        check("gyro bias (worst axis)", np.abs(gyro.mean(0)).max(), "< 0.01 rad/s", np.abs(gyro.mean(0)).max() < 0.01, " rad/s"),
        check("gyro noise (worst axis)", gyro.std(0).max(), "< 0.02 rad/s", gyro.std(0).max() < 0.02, " rad/s"),
        check("tilt wobble", tilt.std(), "< 0.5°", tilt.std() < 0.5, "°"),
        check("heading drift", drift, "< 2°/min", drift < 2, "°/min"),
    ]
    if acc.std(0).max() > 0.5:
        m["note"] = "the phone was moving: put it flat on a table and don't touch it during the test"
    return checks, m


def step_peaks(imu: np.ndarray, rate: float) -> np.ndarray:
    """Steps from vertical acceleration (user acceleration on the gravity axis): low-passed at 3 Hz, one peak per step."""
    a = (imu["acc"].astype(np.float64) * up_axis(imu)).sum(1)
    b, aa = butter(2, 3.0 / (rate / 2))
    v = filtfilt(b, aa, a)
    moving = np.abs(v) > 0.05
    if moving.sum() < rate:
        return np.zeros(0, dtype=int)
    peaks, _ = find_peaks(-v if np.abs(v.min()) > v.max() else v, distance=int(0.3 * rate),
                          prominence=0.6 * v[moving].std())
    return peaks


def test_steps(d: Path, args) -> tuple[list[dict], dict]:
    meta, data, _ = load_session(d)
    imu = data["imu"]
    rate = 1 / np.median(np.diff(imu["t"]))
    peaks = step_peaks(imu, rate)
    n = args.count
    m = {"counted": n, "detected_from_motion": int(len(peaks))}
    checks = [check("steps found in motion data", len(peaks), f"{n} ± 5%", abs(len(peaks) - n) <= 0.05 * n)]
    ped = (meta.get("pedometer") or {}).get("steps")
    if ped:
        m["iphone_pedometer"] = int(ped)
        checks.append(check("iphone pedometer", ped, f"{n} ± 5%", abs(ped - n) <= 0.05 * n))
    if len(peaks) > 2:
        st = np.diff(imu["t"][peaks])
        m["cadence_spm"] = round(60 / np.median(st), 1)
        m["step_time_cv"] = round(float(st.std() / st.mean()), 3)
        checks.append(check("step timing regularity (CV)", st.std() / st.mean(), "< 0.15", st.std() / st.mean() < 0.15))
    return checks, m


def test_rotate(d: Path, args) -> tuple[list[dict], dict]:
    meta, data, _ = load_session(d)
    imu = data["imu"]
    w = (imu["gyro"].astype(np.float64) * up_axis(imu)).sum(1)  # rotation rate about the vertical axis
    gyro_deg = abs(np.degrees(np.trapezoid(w, imu["t"])))
    quat_deg = abs(yaw_deg(imu)[-1] - yaw_deg(imu)[0])
    target = 360.0 * args.turns
    m = {"turns": args.turns, "gyro_integrated_deg": round(gyro_deg, 1), "attitude_yaw_change_deg": round(quat_deg, 1)}
    checks = [check("gyro angle (integrated)", gyro_deg, f"{target:.0f}° ± 5%", abs(gyro_deg - target) <= 0.05 * target, "°"),
              check("attitude heading change", quat_deg, f"{target:.0f}° ± 3%", abs(quat_deg - target) <= 0.03 * target, "°")]
    return checks, m


def test_distance(d: Path, args) -> tuple[list[dict], dict]:
    if not (d / "summary.json").exists():
        raise SystemExit(f"{d.name} has no summary.json yet: wait for the server to register it, or run "
                         f"python -m ground.register {d}")
    s = json.loads((d / "summary.json").read_text())
    meta, data, _ = load_session(d)
    checks, m = [], {"known_m": args.meters, "gps_m": s.get("distance_m"), "gps_median_hacc_m": s["gps"].get("median_hacc_m")}
    if s.get("distance_m") is None:
        checks.append(check("gps distance", 0, "a GPS track", False))
    else:
        checks.append(check("gps distance", s["distance_m"], f"{args.meters:.0f} m ± 5%",
                            abs(s["distance_m"] - args.meters) <= 0.05 * args.meters, " m"))
    baro = data["baro"]
    rel = baro["rel_alt"][np.isfinite(baro["rel_alt"])].astype(np.float64)
    if len(rel) > 2:
        gain = climb(rel)[0]
        m["baro_gain_m"] = round(gain, 1)
        if args.climb_m is not None:
            tol = max(5.0, 0.25 * args.climb_m)
            checks.append(check("barometer climb vs known", gain, f"{args.climb_m:.0f} m ± {tol:.0f} m",
                                abs(gain - args.climb_m) <= tol, " m"))
        dem = (s.get("climb_dem_m") or {}).get("gain")
        if dem is not None:
            m["terrain_tile_gain_m"] = dem
            tol = max(5.0, 0.25 * dem)
            checks.append(check("barometer climb vs terrain tiles", gain, f"{dem:.0f} m ± {tol:.0f} m",
                                abs(gain - dem) <= tol, " m"))
    else:
        checks.append(check("barometer samples", len(rel), "> 0", False))
    return checks, m


TESTS = {"rest": test_rest, "steps": test_steps, "rotate": test_rotate, "distance": test_distance}


# ------------------------------------------------------------------------------------------------ cli

def run(kind: str, d: Path, args) -> dict:
    checks, metrics = TESTS[kind](d, args)
    _, data, warnings = load_session(d)
    checks = timing(data["imu"]) + checks
    meta = json.loads((d / "meta.json").read_text())
    res = {"test": kind, "session": d.name, "started": meta.get("started"), "placement": meta.get("placement"),
           "device": meta.get("device"), "checks": checks, "metrics": metrics, "warnings": warnings,
           "passed": all(c["pass"] for c in checks)}
    (d / f"validation_{kind}.json").write_text(json.dumps(res, indent=1))
    return res


def show(res: dict):
    print(f"\n{res['test']} test on {res['session'][:8]} ({(res.get('device') or {}).get('model', '?')}, "
          f"phone in {res.get('placement')})")
    w = max(len(c["check"]) for c in res["checks"])
    for c in res["checks"]:
        print(f"  {'pass' if c['pass'] else 'FAIL'}  {c['check']:<{w}}  {c['measured']:>14}   target {c['target']}")
    for k, v in res["metrics"].items():
        print(f"        {k}: {v}")
    for wmsg in res["warnings"]:
        print(f"  warning: {wmsg}")
    print(f"  => {'ALL PASS' if res['passed'] else 'SOME CHECKS FAILED'}")


def report(root: Path) -> str:
    rows = ["| test | session | device | check | measured | target | result |", "|---|---|---|---|---|---|---|"]
    for f in sorted(root.glob("*/validation_*.json")):
        r = json.loads(f.read_text())
        for c in r["checks"]:
            rows.append(f"| {r['test']} | {r['session'][:8]} | {(r.get('device') or {}).get('model', '?')} | {c['check']} "
                        f"| {c['measured']} | {c['target']} | {'pass' if c['pass'] else '**fail**'} |")
    return "\n".join(rows) if len(rows) > 2 else "no validation results yet"


def listing(root: Path):
    dirs = sorted((d for d in root.iterdir() if (d / "meta.json").exists()) if root.exists() else [],
                  key=lambda d: json.loads((d / "meta.json").read_text()).get("started", 0), reverse=True)
    import datetime as dt
    for d in dirs:
        m = json.loads((d / "meta.json").read_text())
        when = dt.datetime.fromtimestamp(m.get("started", 0)).strftime("%b %d %H:%M:%S")
        dur = (m.get("ended") or 0) - (m.get("started") or 0)
        done = ", ".join(sorted(f.stem.removeprefix("validation_") for f in d.glob("validation_*.json")))
        print(f"{d.name[:8]}  {when}  {dur:6.0f} s  {m.get('counts', {}).get('imu', 0):>7} motion  "
              f"phone in {m.get('placement', '?'):<9} {('validated: ' + done) if done else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=[*TESTS, "list", "report"])
    ap.add_argument("session", nargs="?", default="latest", help="session id or its first characters, or latest")
    ap.add_argument("--count", type=int, default=100, help="steps counted by hand (steps test)")
    ap.add_argument("--turns", type=float, default=3, help="full turns made (rotate test)")
    ap.add_argument("--meters", type=float, help="known route length (distance test)")
    ap.add_argument("--climb-m", type=float, help="known climb along the route (distance test, optional)")
    ap.add_argument("--root", type=Path, default=SESSIONS, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.test == "list":
        return listing(a.root)
    if a.test == "report":
        text = report(a.root)
        print(text)
        if a.root.exists():
            (a.root.parent / "validation_report.md").write_text(text + "\n")
        return
    if a.test == "distance" and a.meters is None:
        ap.error("distance needs --meters (the route's known length)")
    show(run(a.test, resolve(a.root, a.session), a))


if __name__ == "__main__":
    main()
