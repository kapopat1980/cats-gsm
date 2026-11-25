"""Turn the Porto taxi (ECML/PKDD 2015) trips into 1 Hz routes for the CATS-GSM simulator.

Real: the paths and speeds of the trips (15 s GPS samples).  Interpolated: everything between samples.
Modelled (as in the synthetic study): IMU, GPS noise/jamming/spoofing, serving cell / timing advance, energy.
"""
import json, glob, os
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline

LAT0, LON0 = 41.1579, -8.6291      # Porto centre
R_EARTH_X = 111320.0 * np.cos(np.radians(LAT0))
R_EARTH_Y = 110574.0


def find_csv(root='/kaggle/input'):
    cands = [p for p in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
             if 'train' in os.path.basename(p).lower()] or glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
    if not cands: raise FileNotFoundError('No CSV found under ' + root)
    return max(cands, key=os.path.getsize)


def _route_from_points(pts, dt=15.0):
    """pts: (m,2) lon/lat at 15 s.  Returns x,y,v,th,a,w at 1 Hz (metres, m/s, rad) or None if the trip is unusable."""
    pts = np.asarray(pts, float)
    x = (pts[:, 0] - LON0) * R_EARTH_X
    y = (pts[:, 1] - LAT0) * R_EARTH_Y
    step = np.hypot(np.diff(x), np.diff(y))
    if step.max() > 40.0 * dt:                       # > 40 m/s average over one sample: GPS jump
        return None
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 800.0:  # barely moved
        return None
    tk = np.arange(len(x)) * dt
    t = np.arange(0.0, tk[-1] + 1e-9, 1.0)
    xs = CubicSpline(tk, x)(t); ys = CubicSpline(tk, y)(t)
    k = np.ones(5) / 5.0
    vx = np.convolve(np.gradient(xs), k, mode='same'); vy = np.convolve(np.gradient(ys), k, mode='same')
    v = np.minimum(np.hypot(vx, vy), 40.0)
    th = np.arctan2(vy, vx)
    # hold heading while (almost) stopped, then unwrap
    for i in range(1, len(th)):
        if v[i] < 1.0: th[i] = th[i - 1]
    th = np.unwrap(th)
    a = np.gradient(v)
    w = np.gradient(th)
    w[v < 1.0] = 0.0
    w = np.clip(w, -np.pi / 4, np.pi / 4)
    return xs, ys, v, th, a, w


def load_routes(csv_path, n_routes=400, min_s=600, max_s=1800, seed=0, nrows=None, radius_m=9000.0):
    df = pd.read_csv(csv_path, usecols=['MISSING_DATA', 'POLYLINE'], nrows=nrows, dtype=str)
    df = df[df['MISSING_DATA'].astype(str).str.lower() == 'false']
    min_pts = int(min_s / 15) + 1
    df = df[df['POLYLINE'].str.len() > min_pts * 18]          # cheap pre-filter before JSON parsing
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    routes = []
    for i in idx:
        try:
            pts = json.loads(df['POLYLINE'].iloc[i])
        except Exception:
            continue
        if len(pts) < min_pts: continue
        pts = pts[: int(max_s / 15) + 1]
        r = _route_from_points(pts)
        if r is None: continue
        if np.max(np.hypot(r[0], r[1])) > radius_m: continue
        routes.append(r)
        if len(routes) >= n_routes: break
    return routes


def to_world_frame(routes, margin=3000.0):
    """Shift all routes so the bounding box starts at `margin`; returns (routes, area)."""
    xmin = min(r[0].min() for r in routes); ymin = min(r[1].min() for r in routes)
    xmax = max(r[0].max() for r in routes); ymax = max(r[1].max() for r in routes)
    out = []
    for r in routes:
        out.append((r[0] - xmin + margin, r[1] - ymin + margin) + tuple(r[2:]))
    area = max(xmax - xmin, ymax - ymin) + 2 * margin
    return out, float(area)
