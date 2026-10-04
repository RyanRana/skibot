"""End-to-end rough-terrain hiking: the policy drives the G1's 29 joints directly (no SONIC), sees a dense height map
and is rewarded for lifting its feet over roots and rocks. The mjlab G1 velocity-task recipe, on the bumpy Ann Arbor
trail tiles, the foot ground planes and the curriculum of train/hike_env.py.

Per policy step (50 Hz): joint targets = default pose + action * scale (the same per-joint scale SONIC's decoder uses),
PD at the actuators. Command: walk along the trail line at v_cmd (0.3 to 1.2 m/s, per episode).
Observations (242): base angular velocity, gravity, command (speed, heading error), joint positions and velocities,
last action, a 16 x 9 height map at 10 cm (0.5 m behind to 1 m ahead, 0.4 m either side), foot contact.
"""

from __future__ import annotations

import math

import torch

from train.hike_env import HikeEnv, wrap

SCAN_X = [round(-0.5 + 0.1 * i, 2) for i in range(16)]
SCAN_Y = [round(-0.4 + 0.1 * i, 2) for i in range(9)]
SOLE_Z = -0.035          # sole below the ankle roll link origin
UPPER = list(range(12, 29))  # waist and arms, MuJoCo order
HIP_ROLL_YAW = [1, 2, 7, 8]
CLEARANCE = 0.10         # swing foot target height over the terrain
AIR_TIME = 0.3           # seconds a step should spend in the air


class HikeRoughEnv(HikeEnv):
  def __init__(self, *args, speed_range=(0.3, 1.2), **kw):
    self.speed_range = speed_range
    super().__init__(*args, **kw)
    N, dev = self.num_envs, self.device
    self.num_actions = 29
    self.last_action = torch.zeros(N, 29, device=dev)
    self.prev_action = torch.zeros(N, 29, device=dev)
    self.v_cmd = torch.zeros(N, device=dev)
    self.air = torch.zeros(N, 2, device=dev)
    self.was_contact = torch.ones(N, 2, dtype=torch.bool, device=dev)
    self.foot_prev = torch.zeros(N, 2, 3, device=dev)
    self.clearance = torch.zeros(N, 2, device=dev)
    self.contact = torch.ones(N, 2, dtype=torch.bool, device=dev)
    gx, gy = torch.meshgrid(torch.tensor(SCAN_X, device=dev), torch.tensor(SCAN_Y, device=dev), indexing="ij")
    self.scan_x, self.scan_y = gx.flatten(), gy.flatten()
    self.sole3 = torch.cat([self.sole_xy, torch.full((5, 1), SOLE_Z, device=dev)], -1)  # (5, 3)
    ids = torch.arange(N, device=dev)
    self.reset_idx(ids)
    self._forward()
    self._feet_state(update=False)
    self.foot_prev[:] = self.xpos[:, self.feet]
    self._update_command(rate_limit=False)
    self.obs = self._observations()

  # ------------------------------------------------------------------------------------------------
  def reset_idx(self, ids):
    super().reset_idx(ids)
    if not hasattr(self, "v_cmd") or len(ids) == 0:
      return
    self.v_cmd[ids] = self._rand(len(ids), *self.speed_range)
    self.air[ids] = 0
    self.was_contact[ids] = True
    self.last_action[ids] = 0
    self.prev_action[ids] = 0
    yaw = self.ref_heading[ids]
    v0 = torch.where(self.walk_from[ids] != 50, self.v_cmd[ids] * 0.8, torch.zeros_like(yaw))  # mid-stride starts
    self.qvel[ids, 0], self.qvel[ids, 1] = v0 * torch.cos(yaw), v0 * torch.sin(yaw)

  def _feet_state(self, update=True):
    """Sole clearance over each foot's ground plane, contact, and foot velocity from the last control step."""
    p = self.xpos[:, self.feet]                                     # (N, 2, 3)
    R = self.xmat[:, self.feet]                                     # (N, 2, 3, 3)
    pts = p[..., None, :] + torch.einsum("nfij,kj->nfki", R, self.sole3)  # (N, 2, 5, 3)
    rel = pts - self.ground_top[..., None, :]
    self.clearance = (rel * self.ground_n[..., None, :]).sum(-1).amin(-1)
    self.contact = self.clearance < 0.015
    if update:
      self.foot_vel = (p - self.foot_prev) / (self.dt * self.decimation)
      self.foot_prev = p.clone()
    else:
      self.foot_vel = torch.zeros_like(p)

  # ------------------------------------------------------------------------------------------------
  def _privileged(self):
    if not hasattr(self, "v_cmd"):
      return super()._privileged()
    R, *_ = self._state()
    vb = (R.transpose(1, 2) @ self.qvel[:, 0:3, None])[..., 0]
    _, nrm = self._hn(self.qpos[:, 0], self.qpos[:, 1])
    t_frac = (self.episode_length_buf.float() / self.max_episode_length)[:, None]
    priv = torch.cat([vb * 0.5, nrm, self._trail_offset()[:, None], self.clearance * 5, self.air, t_frac], -1)
    return torch.nan_to_num(priv, 0.0, 2.0, -2.0)

  def _observations(self):
    if not hasattr(self, "v_cmd"):
      return super()._observations()
    R, ang, grav, q, dq = self._state()
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    px = self.qpos[:, 0:1] + c[:, None] * self.scan_x - s[:, None] * self.scan_y
    py = self.qpos[:, 1:2] + s[:, None] * self.scan_x + c[:, None] * self.scan_y
    h, _ = self._hn(px, py)
    heights = torch.clamp(h - self.qpos[:, 2:3] + self.stand_h, -1.0, 1.0)
    herr = wrap(self.cmd_heading - yaw)
    qm = self.qpos[:, 7:] - self.default
    obs = torch.cat([
      ang * 0.25, grav, torch.stack([self.v_cmd, torch.sin(herr), torch.cos(herr)], -1),
      qm, self.qvel[:, 6:] * 0.05, self.last_action, heights * 5, self.contact.float(),
    ], -1)
    return torch.nan_to_num(obs, 0.0, 10.0, -10.0)

  # ------------------------------------------------------------------------------------------------
  def _step(self, actions):
    actions = torch.clamp(actions, -4, 4)
    self.prev_action = self.last_action
    self.last_action = actions
    self.ctrl[:] = self.default + actions * self.scale
    for _ in range(self.decimation):
      self._physics_step()
    self.episode_length_buf += 1
    self._update_command()
    self._feet_state()
    dtc = self.dt * self.decimation

    R, ang, grav, q, dq = self._state()
    v = self.qvel[:, 0:3]
    ch, sh = torch.cos(self.cmd_heading), torch.sin(self.cmd_heading)
    fwd = v[:, 0] * ch + v[:, 1] * sh
    lat = -v[:, 0] * sh + v[:, 1] * ch
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    herr = wrap(yaw - self.cmd_heading)
    first = self.contact & ~self.was_contact
    air_rew = (torch.clamp(self.air - AIR_TIME, -AIR_TIME, 0.4) * first.float()).sum(-1) / dtc
    self.air = torch.where(self.contact, torch.zeros_like(self.air), self.air + dtc)
    self.was_contact = self.contact.clone()
    vxy = self.foot_vel[..., :2].norm(dim=-1)
    qm = self.qpos[:, 7:] - self.default
    r = {
      "track_speed": 2.0 * torch.exp(-(fwd - self.v_cmd) ** 2 / 0.1),
      "lateral_vel": -1.0 * lat ** 2,
      "heading": 0.5 * torch.exp(-herr ** 2 / 0.15),
      "on_trail": -0.3 * self._trail_offset().abs(),
      "upright": -2.0 * (grav[:, 0] ** 2 + grav[:, 1] ** 2),
      "ang_vel_xy": -0.05 * (ang[:, :2] ** 2).sum(-1),
      "feet_air_time": 1.0 * air_rew,
      "foot_clearance": -20.0 * (torch.relu(CLEARANCE - self.clearance) ** 2 * vxy * (~self.contact).float()).sum(-1),
      "feet_slip": -0.5 * (self.contact.float() * vxy ** 2).sum(-1),
      "upper_pose": -0.2 * (qm[:, UPPER] ** 2).sum(-1),
      "hip_roll_yaw": -0.5 * (qm[:, HIP_ROLL_YAW] ** 2).sum(-1),
      "action_rate": -0.01 * ((self.last_action - self.prev_action) ** 2).sum(-1),
      "joint_vel": -1e-4 * (self.qvel[:, 6:] ** 2).sum(-1),
      "alive": 0.5 * torch.ones_like(fwd),
    }

    hp, _ = self._hn(self.qpos[:, 0], self.qpos[:, 1])
    fb = self.xpos[:, self.fall_ids]
    hb, _ = self._hn(fb[..., 0], fb[..., 1])
    blowup = ~torch.isfinite(self.qpos).all(-1) | (self.qvel.abs().max(-1).values > 200)
    fell = (R[:, 2, 2] < 0.5) | (self.qpos[:, 2] - hp < 0.45) | (fb[..., 2] - hb < 0.05).any(-1) | blowup
    finished = self._finished()
    off_tile = self._off_course()
    timeout = (self.episode_length_buf >= self.max_episode_length) | finished | off_tile
    for k in r:
      r[k] = torch.where(blowup, torch.zeros_like(r[k]), torch.nan_to_num(r[k], 0.0, 0.0, 0.0))
    r["termination"] = -20.0 * fell.float()
    reward = torch.clamp(sum(r.values()), -60.0, 20.0) * dtc
    for k, val in r.items():
      m = val.mean().item()
      self.rew_ema[k] = m if k not in self.rew_ema else 0.99 * self.rew_ema[k] + 0.01 * m

    self.last_fell = fell
    if not self.auto_reset:
      self.obs = self._observations()
      return self.get_observations(), reward, torch.zeros_like(fell), {"time_outs": torch.zeros_like(fell), "log": {}}
    done = fell | timeout
    ids = done.nonzero().flatten()
    extras = {"time_outs": timeout & ~fell, "log": {}}
    if len(ids):
      dist = torch.nan_to_num(self.qpos[ids, 0] - self.start_x[ids], 0.0, 0.0, 0.0)
      t = self.episode_length_buf[ids].float() * dtc
      expect = self.v_cmd[ids] * torch.clamp(t - 1.0, min=0.0)
      good = (~fell[ids] & (dist > 0.6 * expect) & (t > 0.5 * self.cfg["episode_s"])) | finished[ids]
      self.level[ids] = torch.clamp(self.level[ids] + self.level_step * (good.long() - fell[ids].long()), 0, self.n_tiles - 1)
      tg = self.tile_grade[self.tile[ids]]
      extras["log"] = {
        "Episode/fell_frac": fell[ids].float().mean().item(),
        "Episode/blowup_frac": blowup[ids].float().mean().item(),
        "Episode/finished_frac": finished[ids].float().mean().item(),
        "Episode/distance_m": dist.mean().item(),
        "Episode/length_s": t.mean().item(),
        "Curriculum/mean_level": self.level.float().mean().item(),
        "Curriculum/max_level": self.level.float().max().item(),
        "Task/speed_mps": fwd.mean().item(),
        "Task/speed_cmd_mps": self.v_cmd.mean().item(),
        "Task/heading_err_deg": math.degrees(herr.abs().mean().item()),
        "Task/foot_clearance_m": self.clearance.mean().item(),
        **{f"Reward/{k}": v for k, v in self.rew_ema.items()},
      }
      climbs = ids[tg > 8]
      if len(climbs):
        extras["log"]["Episode/fell_frac_climbs_over_8deg"] = fell[climbs].float().mean().item()
      self.reset_idx(ids)
      self._forward()
      self.foot_prev[ids] = self.xpos[ids][:, self.feet]
      snap = self.ref_heading.clone()
      self._update_command(rate_limit=False)
      keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
      keep[ids] = False
      self.ref_heading = torch.where(keep, snap, self.ref_heading)
    self.obs = self._observations()
    return self.get_observations(), reward, done, extras
