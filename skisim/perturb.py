"""Terrain perturbation: many training slopes from a few real runs.

A tile is a window cut along a real course (x = downhill, y = across, z = 0 at the top), or a synthetic slope,
then perturbed: pitch scale, cross tilt, rolling bumps, moguls, rollers, mirroring. Tiles go side by side
into one mosaic HeightGrid so one MuJoCo scene (or one GPU batch) holds every variant.

Run: .venv/bin/python -m skisim.perturb --n 24 --out out/terrains
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from skisim.terrain import HeightGrid, from_course

ROOT = Path(__file__).resolve().parents[1]
COURSES = sorted(p.parent.name for p in (ROOT / "out" / "courses").glob("*/elevation.npy"))

# Snow conditions: multipliers on the groomed ski model (SkiParams defaults).
SNOW = {
  "groomed": {"mu_glide": 0.04, "mu_skid": 0.25, "mu_carve": 1.5, "k_normal": 40000.0},
  "ice": {"mu_glide": 0.025, "mu_skid": 0.12, "mu_carve": 0.7, "k_normal": 80000.0},
  "hardpack": {"mu_glide": 0.03, "mu_skid": 0.18, "mu_carve": 1.1, "k_normal": 60000.0},
  "soft": {"mu_glide": 0.06, "mu_skid": 0.35, "mu_carve": 1.8, "k_normal": 20000.0},
  "slush": {"mu_glide": 0.09, "mu_skid": 0.4, "mu_carve": 1.2, "k_normal": 25000.0},
  "powder": {"mu_glide": 0.08, "mu_skid": 0.5, "mu_carve": 2.0, "k_normal": 9000.0},
}


@dataclass
class TileSpec:
  source: str  # course slug or "synthetic"
  s0_m: float = 0.0  # distance along the run where the window starts
  pitch_scale: float = 1.0
  cross_tilt_deg: float = 0.0
  bump_amp_m: float = 0.0
  bump_wavelength_m: float = 20.0
  mogul_amp_m: float = 0.0
  mogul_spacing_m: float = 5.0
  rollers: int = 0
  mirror: bool = False
  base_pitch_deg: float = 0.0  # synthetic only
  jumps: int = 0  # kickers: a ramp with a sharp lip and a drop to the landing
  jump_height_m: float = 0.0
  compressions: int = 0  # dips across the fall line
  chute_extra_deg: float = 0.0  # a steeper section partway down
  flat_section: bool = False  # a near-flat cat-track section
  snow: str = "groomed"
  stats: dict = field(default_factory=dict)


def run_line(course_dir: Path) -> np.ndarray:
  d = json.loads((course_dir / "run.json").read_text())
  for key in ("course", "course_xy", "centerline_xy", "xy", "polyline"):
    if key in d:
      return np.asarray(d[key], float)[:, :2]
  for v in d.values():
    if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) >= 2:
      return np.asarray(v, float)[:, :2]
  raise KeyError(f"no polyline in {course_dir / 'run.json'}: keys {list(d)}")


def cut_window(grid: HeightGrid, line: np.ndarray, s0: float, length: float, width: float, cell: float):
  """Resample a straight window starting s0 m along the run, aligned with the run direction there."""
  seg = np.diff(line, axis=0)
  s = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
  s0 = float(np.clip(s0, 0, max(s[-1] - length, 0)))
  p0 = np.array([np.interp(s0, s, line[:, 0]), np.interp(s0, s, line[:, 1])])
  p1 = np.array([np.interp(s0 + length, s, line[:, 0]), np.interp(s0 + length, s, line[:, 1])])
  fwd = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-6)
  left = np.array([-fwd[1], fwd[0]])
  xs = np.arange(0, length + cell / 2, cell)
  ys = np.arange(-width / 2, width / 2 + cell / 2, cell)
  X, Y = np.meshgrid(xs, ys)
  wx = p0[0] + fwd[0] * X + left[0] * Y
  wy = p0[1] + fwd[1] * X + left[1] * Y
  h, _ = grid.height_normal(wx, wy)
  return h - h[len(ys) // 2, 0], s0


def synthetic(rng, length, width, cell, pitch_deg):
  xs = np.arange(0, length + cell / 2, cell)
  ys = np.arange(-width / 2, width / 2 + cell / 2, cell)
  X, Y = np.meshgrid(xs, ys)
  # Pitch that changes down the hill: a flat-ish start, a steeper middle, a runout.
  t = X / length
  local = np.radians(pitch_deg) * (0.75 + 0.5 * np.sin(np.pi * t) * rng.uniform(0.5, 1.2))
  z = -np.cumsum(np.tan(local) * cell, axis=1)
  return z - z[len(ys) // 2, 0]


def perturb(z: np.ndarray, cell: float, spec: TileSpec, rng) -> np.ndarray:
  ny, nx = z.shape
  ys = (np.arange(ny) - ny // 2) * cell
  xs = np.arange(nx) * cell
  X, Y = np.meshgrid(xs, ys)
  z = z * spec.pitch_scale
  z = z + np.tan(np.radians(spec.cross_tilt_deg)) * Y
  if spec.bump_amp_m > 0:
    noise = gaussian_filter(rng.standard_normal(z.shape), spec.bump_wavelength_m / (2.5 * cell))
    z = z + spec.bump_amp_m * noise / (np.abs(noise).max() + 1e-9)
  if spec.mogul_amp_m > 0:
    k = 2 * np.pi / spec.mogul_spacing_m
    # Offset rows of bumps, like a mogul field on a fall line.
    z = z + spec.mogul_amp_m * 0.5 * (np.sin(k * X) * np.sin(k * Y + 0.5 * k * X) + 1) - spec.mogul_amp_m * 0.5
  for _ in range(spec.rollers):
    x0 = rng.uniform(0.25, 0.85) * xs[-1]
    z = z + rng.uniform(0.3, 0.9) * np.exp(-((X - x0) / rng.uniform(3, 7)) ** 2)
  for _ in range(spec.jumps):
    x0 = rng.uniform(0.3, 0.8) * xs[-1]
    up = np.exp(-(((X - x0) / rng.uniform(2.5, 4.0)) ** 2))
    lip = np.where(X <= x0, up, np.exp(-(((X - x0) / 0.9) ** 2)))  # gentle ramp up, sharp drop after the lip
    z = z + spec.jump_height_m * lip
  for _ in range(spec.compressions):
    x0 = rng.uniform(0.25, 0.85) * xs[-1]
    z = z - rng.uniform(0.4, 1.2) * np.exp(-(((X - x0) / rng.uniform(4, 8)) ** 2))
  if spec.chute_extra_deg > 0:
    a, L = rng.uniform(0.3, 0.6) * xs[-1], rng.uniform(25, 50)
    z = z - np.tan(np.radians(spec.chute_extra_deg)) * np.clip(X - a, 0, L)
  if spec.flat_section:
    a, L = rng.uniform(0.3, 0.6) * xs[-1], rng.uniform(15, 30)
    gx = np.gradient(z, cell, axis=1)
    inside = (X >= a) & (X <= a + L)
    z = z - np.cumsum(np.where(inside, gx * 0.85, 0.0), axis=1) * cell  # take 85% of the pitch out of that stretch
  if spec.mirror:
    z = z[::-1]
  return z - z[ny // 2, 0]


def slope_stats(z: np.ndarray, cell: float) -> dict:
  gy, gx = np.gradient(z, cell)
  ang = np.degrees(np.arctan(np.hypot(gx, gy)))
  return {
    "drop_m": round(float(z[:, 0].mean() - z[:, -1].mean()), 1),
    "mean_slope_deg": round(float(ang.mean()), 1),
    "p95_slope_deg": round(float(np.percentile(ang, 95)), 1),
    "roughness_m": round(float(np.std(z - gaussian_filter(z, 8 / cell))), 3),
  }


def make_bank(n: int, seed: int = 0, length: float = 160.0, width: float = 50.0, cell: float = 1.0,
              synthetic_frac: float = 0.25, steep: float = 1.0):
  """n tiles in rising difficulty: index 0 is gentle and smooth, n-1 is steep and rough."""
  rng = np.random.default_rng(seed)
  courses = []
  for slug in COURSES:
    d = ROOT / "out" / "courses" / slug
    if (d / "elevation.npy").exists():
      courses.append((slug, from_course(d), run_line(d)))
  tiles = []
  for i in range(n):
    level = i / max(n - 1, 1)  # curriculum: difficulty rises with the tile index
    use_synth = rng.random() < synthetic_frac or not courses
    spec = TileSpec(
      source="synthetic" if use_synth else "",
      pitch_scale=float(rng.uniform(0.45, 0.7) + 0.55 * level * steep),
      cross_tilt_deg=float(rng.uniform(-1, 1) * (1 + 7 * level)),
      bump_amp_m=float(rng.uniform(0.1, 0.4) + 1.2 * level * rng.random()),
      bump_wavelength_m=float(rng.uniform(12, 35)),
      mogul_amp_m=float(0.45 * level * rng.random()) if rng.random() < 0.35 else 0.0,
      mogul_spacing_m=float(rng.uniform(3.5, 6.0)),
      rollers=int(rng.integers(0, 1 + int(3 * level))),
      mirror=bool(rng.random() < 0.5),
      jumps=int(rng.random() < 0.15 + 0.35 * level) * int(rng.integers(1, 3)),
      jump_height_m=float(rng.uniform(0.2, 0.4) + 0.6 * level * rng.random()),
      compressions=int(rng.integers(0, 1 + int(2 * level))),
      chute_extra_deg=float(rng.uniform(4, 12) * level * steep) if rng.random() < 0.3 + 0.1 * (steep - 1) else 0.0,
      flat_section=bool(rng.random() < 0.2),
      snow=str(rng.choice(list(SNOW), p=[0.4, 0.1, 0.15, 0.15, 0.1, 0.1])),
    )
    if use_synth:
      spec.base_pitch_deg = float(8 + 22 * steep * level * rng.uniform(0.6, 1.0))
      spec.pitch_scale = 1.0
      z = synthetic(rng, length, width, cell, spec.base_pitch_deg)
    else:
      slug, grid, line = courses[rng.integers(len(courses))]
      spec.source = slug
      s_max = max(float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) - length, 0.0)
      z, spec.s0_m = cut_window(grid, line, rng.uniform(0, s_max), length, width, cell)
    z = perturb(z, cell, spec, rng)
    spec.stats = slope_stats(z, cell)
    if spec.stats["mean_slope_deg"] > 42:  # keep it skiable: scale the whole tile back to a 42 degree mean
      z = z * np.tan(np.radians(42)) / np.tan(np.radians(spec.stats["mean_slope_deg"]))
      spec.stats = slope_stats(z, cell)
    tiles.append((z, spec))
  return tiles


def mosaic(tiles, cell: float = 1.0, gap: float = 8.0) -> tuple[HeightGrid, list[dict]]:
  """Tiles side by side across y with blended gaps. Downhill is +x for every tile; tile tops sit at x=0."""
  ny, nx = tiles[0][0].shape
  g = int(round(gap / cell))
  rows = []
  info = []
  y = 0
  for k, (z, spec) in enumerate(tiles):
    if k:
      prev = rows[-1][-1]
      w = np.linspace(0, 1, g + 2)[1:-1, None]
      rows.append(prev[None, :] * (1 - w) + z[0][None, :] * w)
      y += g
    rows.append(z)
    info.append({"index": k, "y_center_m": (y + ny // 2) * cell, "source": spec.source, "snow": spec.snow,
                 "snow_params": SNOW[spec.snow], "jumps": spec.jumps, "compressions": spec.compressions,
                 "chute_extra_deg": round(spec.chute_extra_deg, 1), "flat_section": spec.flat_section, **spec.stats})
    y += ny
  Z = np.concatenate(rows, axis=0)
  return HeightGrid(z=Z, x0=0.0, y0=0.0, cell=cell), info


def hillshade(z: np.ndarray, cell: float) -> np.ndarray:
  gy, gx = np.gradient(z, cell)
  n = np.stack([-gx, -gy, np.ones_like(z)], -1)
  n /= np.linalg.norm(n, axis=-1, keepdims=True)
  light = np.array([-0.4, 0.5, 0.75])
  light /= np.linalg.norm(light)
  return np.clip(n @ light, 0, 1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=24)
  ap.add_argument("--cols", type=int, default=6)
  ap.add_argument("--steep", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=str(ROOT / "out/terrains"))
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  tiles = make_bank(args.n, args.seed, steep=args.steep)
  grid, info = mosaic(tiles)
  np.save(out / "mosaic_elevation.npy", grid.z.astype(np.float32))
  (out / "mosaic.json").write_text(json.dumps({"x0": grid.x0, "y0": grid.y0, "cell": grid.cell,
                                               "nrow": grid.nrow, "ncol": grid.ncol, "tiles": info,
                                               "specs": [asdict(s) for _, s in tiles]}, indent=1))

  from PIL import Image, ImageDraw
  cols = args.cols
  tw, th = 320, 110
  sheet = Image.new("RGB", (cols * tw, int(np.ceil(len(tiles) / cols)) * (th + 22)), (245, 247, 250))
  draw = ImageDraw.Draw(sheet)
  for k, (z, spec) in enumerate(tiles):
    shade = hillshade(z, 1.0)
    rel = (z - z.min()) / max(np.ptp(z), 1e-6)
    rgb = np.stack([0.80 + 0.2 * shade, 0.84 + 0.16 * shade, 0.9 + 0.1 * shade], -1) * (0.75 + 0.25 * rel[..., None])
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((tw - 6, th))
    x, y = (k % cols) * tw, (k // cols) * (th + 22)
    sheet.paste(im, (x + 3, y + 20))
    feats = "".join(c for c, on in (("J", spec.jumps), ("C", spec.compressions), ("H", spec.chute_extra_deg > 0), ("F", spec.flat_section), ("M", spec.mogul_amp_m > 0)) if on)
    label = f"{k:02d} {spec.source.split('-')[0][:9]} {spec.stats['mean_slope_deg']}° {spec.snow} {feats}"
    draw.text((x + 4, y + 4), label, fill=(20, 30, 45))
  sheet.save(out / "tiles.png")
  for t in info:
    print(f"{t['index']:02d} {t['source'][:28]:28s} drop {t['drop_m']:6.1f} m  mean {t['mean_slope_deg']:5.1f}°  p95 {t['p95_slope_deg']:5.1f}°  rough {t['roughness_m']:.3f} m")
  print(f"mosaic {grid.ncol} x {grid.nrow} cells ({grid.ncol * grid.cell:.0f} m downhill x {grid.nrow * grid.cell:.0f} m across)")


if __name__ == "__main__":
  main()
