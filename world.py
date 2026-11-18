"""Synthetic urban GSM world and trace generator for the CATS-GSM study.

All quantities are SI (metres, seconds). Everything here is *model assumption*,
documented in the manuscript (Section 5) -- no field data are used.
"""
import numpy as np

AREA = 12000.0          # city side [m]
SPACING = 300.0         # road grid spacing [m]
NODES = int(AREA / SPACING)
TA_STEP = 553.5         # GSM timing-advance step [m]
ISD = 1000.0            # nominal inter-site distance [m]

SIG_GPS = 4.0           # open-sky GPS position noise [m]
SIG_GPS_CANYON = 20.0   # urban-canyon GPS noise [m]
P_FIX_CANYON = 0.45     # per-second fix availability inside canyon zones
SIG_ACC = 0.15          # accelerometer noise [m/s^2]
SIG_GYR = 0.01          # gyro noise [rad/s]


def make_world(seed=12345):
    rng = np.random.default_rng(seed)
    pts = []
    dy = ISD * np.sqrt(3) / 2
    y, row = -2 * ISD, 0
    while y < AREA + 2 * ISD:
        x = -2 * ISD + (ISD / 2 if row % 2 else 0.0)
        while x < AREA + 2 * ISD:
            pts.append((x + rng.uniform(-150, 150), y + rng.uniform(-150, 150)))
            x += ISD
        y += dy
        row += 1
    bts = np.array(pts)
    # urban-canyon zones (unknown to the algorithms): ~15 % of the area
    nz = 40
    zc = rng.uniform(0, AREA, size=(nz, 2))
    zr = rng.uniform(250, 500, size=nz)
    return dict(bts=bts, zc=zc, zr=zr)


WORLD = make_world()


def in_canyon(x, y):
    d = np.hypot(WORLD['zc'][:, 0] - x, WORLD['zc'][:, 1] - y)
    return bool(np.any(d < WORLD['zr']))


# ----------------------------------------------------------------- kinematics
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def drive(rng, start_node, n_steps, p_stop=0.25):
    """Random-walk driving on the grid. Returns arrays x,y,v,th,a,w (length n_steps)."""
    i, j = start_node
    d = int(rng.integers(0, 4))
    S = SPACING
    x = np.zeros(n_steps); y = np.zeros(n_steps); v = np.zeros(n_steps)
    th = np.zeros(n_steps); a = np.zeros(n_steps); w = np.zeros(n_steps)
    t = 0
    cx, cy = i * S, j * S
    cur_v = 0.0
    cur_th = d * np.pi / 2

    def target(i, j, d):
        dx, dy = DIRS[d]
        arterial = (j % 4 == 0) if dx != 0 else (i % 4 == 0)
        return 18.0 if arterial else 10.0

    def choose_dir(i, j, d):
        r = rng.random()
        nd = d if r < 0.6 else (d + 1) % 4 if r < 0.8 else (d + 3) % 4
        ni, nj = i + DIRS[nd][0], j + DIRS[nd][1]
        if not (0 <= ni <= NODES and 0 <= nj <= NODES):
            for k in (1, 3, 2, 0):
                nd = (d + k) % 4
                ni, nj = i + DIRS[nd][0], j + DIRS[nd][1]
                if 0 <= ni <= NODES and 0 <= nj <= NODES:
                    break
        return nd

    d = choose_dir(i, j, d)
    while t < n_steps:
        # start of a segment from node (i,j) in direction d
        dx, dy = DIRS[d]
        vt = target(i, j, d)
        nd = choose_dir(i + dx, j + dy, d)
        stop = rng.random() < p_stop
        turn = nd != d
        v_end = 0.0 if stop else (4.5 if turn else target(i + dx, j + dy, nd) * 0.9)
        tgt_th = d * np.pi / 2
        s = 0.0
        while s < S and t < n_steps:
            rem = S - s
            v_allow = np.sqrt(v_end ** 2 + 2 * 2.5 * max(rem, 0.0))
            v_des = min(vt, v_allow)
            nv = cur_v + np.clip(v_des - cur_v, -2.5, 1.5)
            nv = max(nv, 0.0)
            if nv < 0.5 and v_des > 1.0:
                nv = 0.5
            s += 0.5 * (cur_v + nv)
            # heading rotates towards segment heading over ~2 s
            dth = (tgt_th - cur_th + np.pi) % (2 * np.pi) - np.pi
            rot = np.clip(dth, -np.pi / 4, np.pi / 4)
            a[t] = nv - cur_v
            w[t] = rot
            cur_th += rot
            cur_v = nv
            x[t] = cx + dx * min(s, S) + 0.0
            y[t] = cy + dy * min(s, S)
            v[t] = cur_v
            th[t] = cur_th
            t += 1
        cx += dx * S; cy += dy * S; i += dx; j += dy
        if stop:
            dur = int(rng.integers(8, 40))
            for _ in range(dur):
                if t >= n_steps:
                    break
                x[t] = cx; y[t] = cy; v[t] = 0.0; th[t] = cur_th; a[t] = -cur_v if _ == 0 else 0
                cur_v = 0.0
                t += 1
        d = nd
    return x, y, v, th, a, w


# ----------------------------------------------------------------- radio
def cell_series(rng, x, y, v, dt=1.0):
    """Serving cell and timing advance along a trajectory.
    RSSI_i = -30 - 35 log10(d_i) + s_spatial_i + s_temporal_i ; idle reselection with 3 dB hysteresis."""
    bts = WORLD['bts']
    nb = len(bts)
    n = len(x)
    s_sp = rng.normal(0, 5.0, nb)
    s_tm = rng.normal(0, 2.5, nb)
    serving = np.zeros(n, dtype=np.int32)
    ta = np.zeros(n, dtype=np.int16)
    cur = -1
    px, py = x[0], y[0]
    for t in range(n):
        step = np.hypot(x[t] - px, y[t] - py)
        px, py = x[t], y[t]
        rho_s = np.exp(-step / 100.0)
        rho_t = np.exp(-dt / 120.0)
        s_sp = rho_s * s_sp + np.sqrt(1 - rho_s ** 2) * 5.0 * rng.standard_normal(nb)
        s_tm = rho_t * s_tm + np.sqrt(1 - rho_t ** 2) * 2.5 * rng.standard_normal(nb)
        d = np.hypot(bts[:, 0] - x[t], bts[:, 1] - y[t])
        rssi = -30.0 - 35.0 * np.log10(np.maximum(d, 10.0)) + s_sp + s_tm
        best = int(np.argmax(rssi))
        if cur < 0 or (best != cur and rssi[best] > rssi[cur] + 4.0):
            cur = best
        serving[t] = cur
        dc = d[cur]
        ta[t] = max(0, int(round(dc / TA_STEP + rng.normal(0, 0.35))))
    return serving, ta


# ----------------------------------------------------------------- traces
def _sensors(rng, x, y, v, th, a, w, dt=1.0):
    n = len(x)
    ba = rng.normal(0, 0.04)
    bw = rng.normal(0, 0.003)
    imu_a = a + ba + rng.normal(0, SIG_ACC, n)
    imu_w = w + bw + rng.normal(0, SIG_GYR, n)
    # GPS availability / noise
    gps_ok = np.ones(n, dtype=bool)
    sig = np.full(n, SIG_GPS)
    for t in range(n):
        if in_canyon(x[t], y[t]):
            gps_ok[t] = rng.random() < P_FIX_CANYON
            sig[t] = SIG_GPS_CANYON
    gx = x + rng.normal(0, 1, n) * sig
    gy = y + rng.normal(0, 1, n) * sig
    gv = np.maximum(v + rng.normal(0, 0.15, n), 0.0)
    gth = th + rng.normal(0, 1, n) * (0.04 + 0.5 / np.maximum(v, 0.3))
    return dict(imu_a=imu_a, imu_w=imu_w, gps_ok=gps_ok, gx=gx, gy=gy, gv=gv, gth=gth, ba=ba, bw=bw)


def driving_trace(seed, n=3600):
    rng = np.random.default_rng(seed)
    start = (int(rng.integers(5, NODES - 5)), int(rng.integers(5, NODES - 5)))
    x, y, v, th, a, w = drive(rng, start, n)
    sv, ta = cell_series(rng, x, y, v)
    tr = dict(x=x, y=y, v=v, th=th, a=a, w=w, serving=sv, ta=ta)
    tr.update(_sensors(rng, x, y, v, th, a, w))
    tr['n'] = n
    return tr


ACC_FA_RATE = 0.25 / 3600.0   # spurious motion-sensor triggers per second
ACC_PD = 0.90                 # prob. the motion sensor detects theft motion onset


def armed_trace(seed, n_pre, n_theft, attack, theft=True, acc_pd=ACC_PD):
    """Parked & armed for n_pre s (random location), then (optionally) theft driving for n_theft s."""
    rng = np.random.default_rng(seed)
    i = int(rng.integers(5, NODES - 5)); j = int(rng.integers(5, NODES - 5))
    px, py = i * SPACING, j * SPACING
    n = n_pre + (n_theft if theft else 0)
    x = np.full(n, px); y = np.full(n, py); v = np.zeros(n); th = np.zeros(n)
    a = np.zeros(n); w = np.zeros(n)
    if theft:
        xs, ys, vs, ths, as_, ws = drive(rng, (i, j), n_theft, p_stop=0.15)
        x[n_pre:] = xs; y[n_pre:] = ys; v[n_pre:] = vs; th[n_pre:] = ths
        a[n_pre:] = as_; w[n_pre:] = ws
    sv, ta = cell_series(rng, x, y, v)
    tr = dict(x=x, y=y, v=v, th=th, a=a, w=w, serving=sv, ta=ta)
    tr.update(_sensors(rng, x, y, v, th, a, w))
    t0 = n_pre if theft else -1
    gx, gy, gok = tr['gx'].copy(), tr['gy'].copy(), tr['gps_ok'].copy()
    if theft:
        if attack == 'jam':
            gok[t0 + int(rng.integers(0, 30)):] = False
        elif attack == 'replay':      # receiver spoofed to keep reporting the parked position
            gx[t0:] = px + rng.normal(0, SIG_GPS, n - t0)
            gy[t0:] = py + rng.normal(0, SIG_GPS, n - t0)
            tr['gv'] = tr['gv'].copy(); tr['gth'] = tr['gth'].copy()
            tr['gv'][t0:] = np.abs(rng.normal(0, 0.15, n - t0))     # spoofer also feeds a stationary velocity
            tr['gth'][t0:] = rng.uniform(-np.pi, np.pi, n - t0)
        elif attack == 'drift':       # slow carry-off: offset grows 2 m/s opposite to a random bearing
            ang = rng.uniform(0, 2 * np.pi)
            off = 2.0 * np.arange(n - t0)
            gx[t0:] += off * np.cos(ang)
            gy[t0:] += off * np.sin(ang)
    tr['gx'], tr['gy'], tr['gps_ok'] = gx, gy, gok
    # motion-sensor events
    fa_times = np.nonzero(rng.random(n) < ACC_FA_RATE)[0]
    tr['acc_fa'] = fa_times[fa_times < (t0 if theft else n)]
    tr['acc_hit'] = (t0 + int(rng.integers(1, 5))) if (theft and rng.random() < acc_pd) else -1
    tr['t0'] = t0
    tr['n'] = n
    tr['attack'] = attack
    tr['park'] = (px, py)
    return tr
