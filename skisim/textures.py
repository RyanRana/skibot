"""Procedural textures for the render-only look (skisim.scene with pretty=True): sky with mountains, snow, rock.

Files are cached under out/textures/ (gitignored). File names carry their parameters and a version number,
because MuJoCo caches texture files by path inside a process.

Conventions, checked by rendering:
- 2D textures on an hfield with texuniform="true": image columns run along +x, image row 0 is the +y edge of a
  tile, and one tile spans 2 / texrepeat meters.
- Skybox (gridlayout ".U..LFRB.D..", gridsize "3 4"): a face pixel whose OpenGL cube-map direction is
  (tx, ty, tz) shows the world direction (tx, -tz, ty), z up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEX_DIR = ROOT / "out" / "textures"
# Bump a version whenever its generator changes, so a fresh file name is written and loaded.
SNOW_VERSION, ROCK_VERSION, SKY_VERSION = 1, 2, 8

SNOW_TILE_M = 8.0  # one snow texture tile covers 8 m x 8 m
ROCK_TILE_M = 12.0
SKY_HORIZON = np.array([0.80, 0.86, 0.93])  # haze at the horizon; the fog color matches it


def _noise(shape, feature_px: float, rng: np.random.Generator) -> np.ndarray:
  """Smooth noise that tiles seamlessly (white noise low-passed in Fourier space), zero mean, unit std."""
  fy = np.fft.fftfreq(shape[0])[:, None]
  fx = np.fft.fftfreq(shape[1])[None, :]
  filt = np.exp(-2.0 * (np.hypot(fx, fy) * feature_px) ** 2)
  n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * filt))
  n -= n.mean()
  return n / (n.std() + 1e-9)


def _save(rgb: np.ndarray, path: Path) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(".tmp.png")
  Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)).save(tmp)
  tmp.replace(path)  # atomic, so a parallel render never reads half a file
  return path


def snow_texture(groom_deg: float = 0.0, groomed: bool = True, size: int = 1024, seed: int = 0) -> Path:
  """Snow tile (SNOW_TILE_M square). Groomed snow has corduroy ridges about 5 cm apart running along groom_deg
  (degrees from +x, the direction the groomer drove, i.e. the fall line) and a faint seam every groomer pass."""
  deg = int(round(groom_deg)) % 180
  path = TEX_DIR / (f"snow_groomed_{deg:03d}_v{SNOW_VERSION}.png" if groomed else f"snow_natural_v{SNOW_VERSION}.png")
  if path.exists():
    return path
  rng = np.random.default_rng(seed)
  px_per_m = size / SNOW_TILE_M
  v = 0.020 * _noise((size, size), 1.5 * px_per_m, rng)  # broad drifts
  v += 0.010 * _noise((size, size), 0.2 * px_per_m, rng)
  v += 0.012 * rng.standard_normal((size, size))  # grain
  v += 0.06 * (rng.random((size, size)) > 0.9985)  # a few sparkles
  i, j = np.mgrid[0:size, 0:size].astype(float)
  if groomed:
    th = np.radians(deg)
    perp = np.array([-np.sin(th), np.cos(th)])  # ridges vary across the groom direction
    a, b = (int(round(c)) for c in perp * (SNOW_TILE_M / 0.05))
    # World x = j / size * tile, world y = -i / size * tile; integer cycles keep the tile seamless.
    wob = 0.6 * _noise((size, size), 0.8 * px_per_m, rng)  # ridges meander a little
    phase = 2 * np.pi * (a * j - b * i) / size + wob
    ridge = np.sin(phase)
    ridge = np.sign(ridge) * np.abs(ridge) ** 0.5  # flat-topped ridges with narrow grooves
    fade = np.clip(0.75 + 0.25 * _noise((size, size), 1.0 * px_per_m, rng), 0.3, 1.0)  # worn in places
    v += 0.034 * ridge * fade
    sa, sb = (int(round(c)) for c in perp * (SNOW_TILE_M / 4.0))  # a groomer pass is about 4 m wide
    if sa or sb:
      seam = np.cos(2 * np.pi * (sa * j - sb * i) / size + 0.3 * wob)
      v -= 0.035 * np.clip((seam - 0.985) / 0.015, 0, 1)
  else:
    # Wind-packed natural snow: soft elongated ripples, no corduroy.
    v += 0.018 * np.sin(2 * np.pi * (3 * j + 7 * i) / size + 2.5 * _noise((size, size), 0.6 * px_per_m, rng))
  base = np.array([0.93, 0.95, 0.98])
  rgb = base[None, None, :] * (1 + v[..., None]) + np.array([0.0, 0.002, 0.008]) * v[..., None] * 10
  return _save(rgb, path)


def rock_texture(size: int = 512, seed: int = 1) -> Path:
  """Grey-brown rock tile (ROCK_TILE_M square) with darker cracks and a little snow caught in them."""
  path = TEX_DIR / f"rock_v{ROCK_VERSION}.png"
  if path.exists():
    return path
  rng = np.random.default_rng(seed)
  px_per_m = size / ROCK_TILE_M
  n1 = _noise((size, size), 1.5 * px_per_m, rng)
  n2 = _noise((size, size), 0.3 * px_per_m, rng)
  n3 = rng.standard_normal((size, size))
  i = np.arange(size)[:, None]
  strata = np.sin(2 * np.pi * 9 * i / size + 1.5 * n1)  # layered bands across the face
  shade = 0.40 + 0.07 * n1 + 0.05 * n2 + 0.03 * n3 + 0.035 * strata
  cracks = np.clip(1 - np.abs(_noise((size, size), 0.6 * px_per_m, rng)) / 0.06, 0, 1)
  shade -= 0.08 * cracks
  rgb = shade[..., None] * np.array([1.0, 0.96, 0.90])
  snow = (_noise((size, size), 0.25 * px_per_m, rng) + 0.5 * n1) > 1.9  # small pockets of snow on ledges
  rgb[snow] = np.array([0.84, 0.87, 0.92])
  return _save(rgb, path)


def _ridge(az: np.ndarray, rng: np.random.Generator, base: float, amp: float, n: int = 60) -> np.ndarray:
  """A periodic mountain skyline (degrees above the horizon) as a function of azimuth (radians): broad massifs from
  the low harmonics, and sharp summits from folding the higher ones (ridged noise)."""
  broad = np.zeros_like(az)
  for k in range(1, 41):
    broad += rng.normal() * k ** -1.15 * np.cos(k * az + rng.uniform(0, 2 * np.pi))
  sharp = np.zeros_like(az)
  for k in range(10, n + 1):
    sharp += rng.normal() * k ** -1.3 * np.cos(k * az + rng.uniform(0, 2 * np.pi))
  sharp = 1 - np.abs(sharp) / (np.abs(sharp).max() + 1e-9)  # crests where the noise crosses zero
  out = (broad - broad.min()) / (broad.max() - broad.min() + 1e-9) + 0.12 * sharp ** 2
  out = (out - out.min()) / (out.max() - out.min() + 1e-9)
  return base + amp * out ** 1.5


def sky_texture(face: int = 1024, seed: int = 3) -> Path:
  """Cube-map skybox: blue gradient sky, three hazy mountain ranges on the horizon, pale valley haze below."""
  path = TEX_DIR / f"sky_mountains_{face}_v{SKY_VERSION}.png"
  if path.exists():
    return path
  rng = np.random.default_rng(seed)
  i, j = np.meshgrid(np.arange(face), np.arange(face), indexing="ij")
  sc = 2 * (j + 0.5) / face - 1
  tc = 2 * (i + 0.5) / face - 1
  one = np.ones_like(sc)
  gl = {"R": (one, -tc, -sc), "L": (-one, -tc, sc), "U": (sc, one, tc), "D": (sc, -one, -tc), "F": (sc, -tc, one),
        "B": (-sc, -tc, -one)}
  cells = {"U": (0, 1), "L": (1, 0), "F": (1, 1), "R": (1, 2), "B": (1, 3), "D": (2, 1)}
  floor = -4.0  # every range stands on this line; below it is valley haze
  # Three ranges, far to near: higher and paler far away, lower and darker close by.
  ranges = [
    dict(base=1.5, amp=9.0, rock=np.array([0.46, 0.52, 0.64]), snow=np.array([0.92, 0.94, 0.98]), line=0.50, haze=0.45),
    dict(base=0.5, amp=6.0, rock=np.array([0.33, 0.39, 0.49]), snow=np.array([0.94, 0.96, 0.99]), line=0.45, haze=0.28),
    dict(base=-1.0, amp=3.5, rock=np.array([0.34, 0.40, 0.47]), snow=np.array([0.90, 0.93, 0.97]), line=0.35, haze=0.32,
         gully_amp=0.05),
  ]
  az_s = np.linspace(-np.pi, np.pi, 8192, endpoint=False)
  for r in ranges:
    r["ridge"] = _ridge(az_s, rng, r["base"], r["amp"])
    soft = np.convolve(np.r_[r["ridge"][-32:], r["ridge"], r["ridge"][:32]], np.hanning(65) / np.hanning(65).sum(), "same")[32:-32]
    r["slope"] = np.gradient(soft) / np.degrees(az_s[1] - az_s[0])  # degrees of rise per degree of azimuth, smoothed
    g = sum(rng.normal() * np.cos(m * az_s + rng.uniform(0, 2 * np.pi)) / np.sqrt(m) for m in range(60, 400, 7))
    r["gully"] = g / (np.abs(g).max() + 1e-9)  # rock ribs and gullies streaking down through the snow
    r["line_noise"] = sum(rng.normal() * np.cos(m * az_s + rng.uniform(0, 2 * np.pi)) / m for m in range(8, 90, 3))
  zenith = np.array([0.20, 0.41, 0.77])
  img = np.zeros((3 * face, 4 * face, 3))
  for name, (r0, c0) in cells.items():
    tx, ty, tz = gl[name]
    w = np.stack([tx, -tz, ty], -1)
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    el = np.degrees(np.arcsin(np.clip(w[..., 2], -1, 1)))
    az = np.arctan2(w[..., 1], w[..., 0])
    t = np.clip(el / 90.0, 0, 1) ** 0.45
    col = SKY_HORIZON * (1 - t[..., None]) + zenith * t[..., None]
    below = np.clip(-el / 40.0, 0, 1)[..., None]  # the valley floor lost in haze, a little darker straight down
    col = np.where(el[..., None] < 0, SKY_HORIZON * (1 - 0.15 * below) + np.array([0.0, 0.0, 0.01]) * below, col)
    band = (el > floor - 1) & (el < 13)
    if band.any():
      k = ((az[band] + np.pi) / (2 * np.pi) * len(az_s)).astype(int) % len(az_s)
      e = el[band]
      c = col[band]
      for rr in ranges:
        ridge = rr["ridge"][k]
        # Soft one-texel edges (coverage, not yes/no) so the skyline and snow line do not stair-step when magnified.
        texel = 90.0 / face
        inside = np.clip((ridge - e) / texel, 0, 1) * np.clip((e - floor) / texel, 0, 1)
        h = np.clip((e - floor) / (ridge - floor + 1e-6), 0, 1)  # 0 at the foot, 1 on the skyline
        snowline = 1 - rr["line"] + 0.12 * rr["line_noise"][k] + rr.get("gully_amp", 0.18) * np.clip(rr["gully"][k], 0, 1) * (1 - h)
        snowy = np.clip((h - snowline) * (ridge - floor) / (1.5 * texel), 0, 1)[:, None]
        lit = np.clip(0.5 + 0.45 * np.tanh(rr["slope"][k] * 1.5), 0.1, 0.95)[:, None]  # sun from one side
        rock = rr["rock"] * (0.75 + 0.45 * lit)
        snow = rr["snow"] * (0.80 + 0.22 * lit) + np.array([-0.05, -0.02, 0.03]) * (1 - lit)
        m = rock * (1 - snowy) + snow * snowy
        m = m * (1 - rr["haze"]) + SKY_HORIZON * rr["haze"]  # aerial perspective
        foot = np.clip(1 - h / 0.3, 0, 1)[:, None] ** 1.5  # haze pooling at the foot, fully hazy at the floor line
        m = m * (1 - foot) + SKY_HORIZON * foot
        c = c * (1 - inside[:, None]) + m * inside[:, None]
      col[band] = c
    img[r0 * face:(r0 + 1) * face, c0 * face:(c0 + 1) * face] = col
  return _save(img, path)


def ensure(groom_deg: float = 0.0) -> dict[str, Path]:
  """Paths of every texture the pretty scene needs, generating any that are missing."""
  return {
    "sky": sky_texture(),
    "snow_groomed": snow_texture(groom_deg, groomed=True),
    "snow_natural": snow_texture(groomed=False),
    "rock": rock_texture(),
  }
