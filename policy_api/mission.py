"""Missions: send the robot to one chosen point on a route and prove it got there.

A mission holds a route polyline (local metres) and a target, given as metres along the route or as an x, y that
snaps to the nearest route point. Each control step the client sends its observation and pose; the server writes
the policy's command slice (sin and cos of heading error, target speed / 10) itself, steering by pure pursuit a few
metres ahead on the route, slowing on approach and commanding a stop inside the arrival radius. The client never
computes a heading, so the log is the server's own record of where the robot was told to go and where it went.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class Mission:
    policy: str
    xy: np.ndarray                 # (N, 2) route points
    s: np.ndarray                  # (N,) arc length at each point
    target_s: float
    speed: float = 3.0             # cruise speed, m/s
    arrive_radius: float = 3.0     # within this of the target counts as arrived
    lookahead: float = 6.0         # pure-pursuit distance ahead on the route, m
    slow_radius: float = 20.0      # speed ramps down to zero over this distance before the target
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    progress_s: float = 0.0
    arrived_at: float | None = None    # mission time (s) of first arrival
    steps: int = 0
    log: list = field(default_factory=list)

    @property
    def target_xy(self) -> np.ndarray:
        return self.point_at(self.target_s)

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0, self.s[-1]))
        return np.array([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])

    def project(self, p: np.ndarray, near: float, window: float = 30.0) -> float:
        """Arc length of the closest route point to p, searched near the last progress so loops don't jump."""
        k = np.flatnonzero((self.s >= near - window) & (self.s <= near + window))
        a, b = self.xy[k[:-1]], self.xy[k[1:]]
        d = b - a
        t = np.clip(np.sum((p - a) * d, 1) / np.maximum(np.sum(d * d, 1), 1e-9), 0, 1)
        dist = np.linalg.norm(a + t[:, None] * d - p, axis=1)
        j = int(np.argmin(dist))
        return float(self.s[k[j]] + t[j] * (self.s[k[j + 1]] - self.s[k[j]]))

    def command(self, x: float, y: float, yaw: float, t: float | None = None) -> dict:
        """Command for the robot at pose (x, y, yaw), plus progress. Heading error is target heading minus yaw,
        matching SkiEnv._observations."""
        p = np.array([x, y], dtype=float)
        self.progress_s = self.project(p, self.progress_s, window=30.0 if self.steps else np.inf)
        to_target = float(np.linalg.norm(self.target_xy - p))
        remaining = abs(self.target_s - self.progress_s)
        sign = 1.0 if self.target_s >= self.progress_s else -1.0     # target may be behind the start point
        aim = self.point_at(self.progress_s + sign * min(self.lookahead, remaining))
        if remaining < self.lookahead:                                 # last stretch: aim at the target itself
            aim = self.target_xy
        heading = float(np.arctan2(*(aim - p)[::-1]))
        err = wrap(heading - yaw)
        arrived = to_target <= self.arrive_radius
        speed = 0.0 if arrived else self.speed * float(np.clip(remaining / self.slow_radius, 0.15, 1.0))
        t = self.steps * 0.02 if t is None else t
        if arrived and self.arrived_at is None:
            self.arrived_at = t
        self.steps += 1
        rec = {"t": round(t, 3), "x": round(x, 3), "y": round(y, 3), "progress_s": round(self.progress_s, 2),
               "to_target_m": round(to_target, 2), "cmd_heading": round(heading, 4), "cmd_speed": round(speed, 3)}
        self.log.append(rec)
        return {"command": [np.sin(err), np.cos(err), speed / 10], "heading": heading, "speed": speed,
                "progress_s": self.progress_s, "to_target_m": to_target, "arrived": arrived}

    def status(self) -> dict:
        return {"id": self.id, "policy": self.policy, "route_length_m": float(self.s[-1]),
                "target": {"s": self.target_s, "x": float(self.target_xy[0]), "y": float(self.target_xy[1])},
                "arrive_radius_m": self.arrive_radius, "steps": self.steps, "progress_s": self.progress_s,
                "to_target_m": self.log[-1]["to_target_m"] if self.log else None,
                "arrived": self.arrived_at is not None, "arrived_at_s": self.arrived_at}


def make(policy: str, x: list[float], y: list[float], target_s: float | None = None,
         target_xy: list[float] | None = None, **kw) -> Mission:
    xy = np.stack([np.asarray(x, float), np.asarray(y, float)], 1)
    if len(xy) < 2:
        raise ValueError("route needs at least 2 points")
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if target_s is None and target_xy is None:
        raise ValueError("give target_s (metres along the route) or target_xy")
    if target_s is None:
        target_s = float(s[int(np.argmin(np.linalg.norm(xy - np.asarray(target_xy, float), axis=1)))])
    if not 0 <= target_s <= s[-1]:
        raise ValueError(f"target_s {target_s} outside route [0, {s[-1]:.1f}]")
    return Mission(policy=policy, xy=xy, s=s, target_s=float(target_s), **kw)
