"""Baselines and the proposed CATS-GSM algorithm.  All algorithms consume pre-generated traces
(see world.py) and therefore see *identical* noise realisations."""
import numpy as np
from world import WORLD, TA_STEP

# ------------------------------------------------------------------ energy model (assumptions)
E_FIX = 0.60       # J, one GPS acquisition (hot start, ~5 s at 120 mW)
P_GPS_ON = 0.12    # W, receiver kept on between closely spaced attempts
E_ATT = 1.80       # J, GPRS attach / PDP-context set-up
E_MSG = 0.50       # J, one small GPRS packet inside an open context
CTX_T = 15.0       # s, context idle timeout
E_SMS = 1.20       # J, one SMS
P_BASE = 0.010     # W, MCU + camped modem (common to all schemes)
P_IMU = 0.001      # W, 1 Hz accelerometer+gyro sampling (CATS)
P_CELLQ = 0.002    # W, 1 Hz serving-cell/TA query (CATS, cell-only)
P_ACC = 0.0002     # W, wake-on-motion accelerometer (reactive baselines)
BATT_J = 3.7 * 2.0 * 3600.0   # 2000 mAh @ 3.7 V


class Meter:
    def __init__(self):
        self.e_gps = 0.0; self.e_tx = 0.0; self.n_fix = 0; self.n_fail = 0; self.n_tx = 0
        self.last_gps = -1e9; self.ctx_until = -1e9

    def fix(self, t, ok):
        cost = min(E_FIX, P_GPS_ON * (t - self.last_gps))
        self.e_gps += cost; self.last_gps = t
        if ok: self.n_fix += 1
        else: self.n_fail += 1
        return ok

    def tx(self, t, sms=False):
        if sms:
            self.e_tx += E_SMS
        else:
            self.e_tx += E_MSG + (E_ATT if t > self.ctx_until else 0.0)
            self.ctx_until = t + CTX_T
        self.n_tx += 1

    def total(self, dur, extra_power=0.0):
        return self.e_gps + self.e_tx + (P_BASE + extra_power) * dur


# ------------------------------------------------------------------ server estimators
def est_from_reports(reports, n, mode):
    """reports: list of (t,x,y,v,th). mode 'zoh' or 'cv'. Returns est arrays and first-report index."""
    ex = np.full(n, np.nan); ey = np.full(n, np.nan)
    if not reports:
        return ex, ey
    k = 0
    for t in range(n):
        while k + 1 < len(reports) and reports[k + 1][0] <= t:
            k += 1
        r = reports[k]
        if r[0] > t:
            continue
        if mode == 'zoh':
            ex[t], ey[t] = r[1], r[2]
        else:
            dt = t - r[0]
            ex[t] = r[1] + r[3] * np.cos(r[4]) * dt
            ey[t] = r[2] + r[3] * np.sin(r[4]) * dt
    return ex, ey


def track_error(tr, ex, ey, lo=0, hi=None):
    hi = tr['n'] if hi is None else hi
    d = np.hypot(ex[lo:hi] - tr['x'][lo:hi], ey[lo:hi] - tr['y'][lo:hi])
    d = d[~np.isnan(d)]
    return d


def pack(tr, m, ex, ey, extra_power):
    err = track_error(tr, ex, ey)
    dur = tr['n']
    E = m.total(dur, extra_power)
    return dict(mean=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())), p95=float(np.percentile(err, 95)),
                energy_J=float(E), power_mW=float(E / dur * 1e3), msgs=m.n_tx, fixes=m.n_fix + m.n_fail,
                life_days=float(BATT_J / (E / dur) / 86400.0))


# ------------------------------------------------------------------ D1 periodic GPS + report
def run_periodic(tr, T):
    n = tr['n']; m = Meter(); rep = []; nxt = 0
    for t in range(n):
        if t >= nxt:
            if m.fix(t, tr['gps_ok'][t]):
                rep.append((t, tr['gx'][t], tr['gy'][t], tr['gv'][t], tr['gth'][t])); m.tx(t); nxt = t + T
            else:
                nxt = t + 1
    ex, ey = est_from_reports(rep, n, 'zoh')
    return pack(tr, m, ex, ey, 0.0)


# ------------------------------------------------------------------ D2 distance-based reporting
def run_distance(tr, D, T=5):
    n = tr['n']; m = Meter(); rep = []; nxt = 0; last = None
    for t in range(n):
        if t >= nxt:
            if m.fix(t, tr['gps_ok'][t]):
                gx, gy = tr['gx'][t], tr['gy'][t]
                if last is None or np.hypot(gx - last[0], gy - last[1]) >= D:
                    rep.append((t, gx, gy, tr['gv'][t], tr['gth'][t])); m.tx(t); last = (gx, gy)
                nxt = t + T
            else:
                nxt = t + 1
    ex, ey = est_from_reports(rep, n, 'zoh')
    return pack(tr, m, ex, ey, 0.0)


# ------------------------------------------------------------------ D3 dead-reckoning, speed-adaptive sampling
def run_dr_threshold(tr, theta, kappa=0.3, tmax=60.0):
    n = tr['n']; m = Meter(); rep = []; nxt = 0
    for t in range(n):
        if t >= nxt:
            if m.fix(t, tr['gps_ok'][t]):
                gx, gy, gv, gth = tr['gx'][t], tr['gy'][t], tr['gv'][t], tr['gth'][t]
                send = True
                if rep:
                    r = rep[-1]; dt = t - r[0]
                    px = r[1] + r[3] * np.cos(r[4]) * dt; py = r[2] + r[3] * np.sin(r[4]) * dt
                    send = np.hypot(px - gx, py - gy) > theta
                if send:
                    rep.append((t, gx, gy, gv, gth)); m.tx(t)
                nxt = t + min(max(theta / (kappa * max(gv, 0.5)), 1.0), tmax)
            else:
                nxt = t + 1
    ex, ey = est_from_reports(rep, n, 'cv')
    return pack(tr, m, ex, ey, 0.0)


# ------------------------------------------------------------------ D4 cell-ID only
def run_cellid(tr):
    n = tr['n']; m = Meter(); bts = WORLD['bts']
    ex = np.zeros(n); ey = np.zeros(n); cur = tr['serving'][0]; m.tx(0)
    for t in range(n):
        if tr['serving'][t] != cur:
            cur = tr['serving'][t]; m.tx(t)
        ex[t], ey[t] = bts[cur]
    return pack(tr, m, ex, ey, P_CELLQ)


# ------------------------------------------------------------------ CATS driving mode
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_cats_drive(tr, eps, use_imu=True, use_trig=True, sched='learn'):
    n = tr['n']; m = Meter(); rep = []
    have = False; ex = ey = ev = eth = 0.0; ba = bw = 0.0; chat = 0.02
    tl = 0; nxt = 0; stationary = False; yaw_acc = 0.0; srv = None; last_rep_t = -99
    ia, iw = tr['imu_a'], tr['imu_w']
    for t in range(n):
        a_c = 0.0
        if have:
            if use_imu:
                a_c = ia[t] - ba; w_c = iw[t] - bw
                ev = max(0.0, ev + a_c); eth += w_c; yaw_acc += abs(w_c)
            ex += ev * np.cos(eth); ey += ev * np.sin(eth)
        do_fix = t >= nxt
        if have and use_imu and use_trig:
            if stationary and a_c > 0.4:
                do_fix = True
            if yaw_acc > 0.5 and nxt > t + 2:
                nxt = t + 2
        if do_fix:
            if m.fix(t, tr['gps_ok'][t]):
                gx, gy, gv, gth = tr['gx'][t], tr['gy'][t], tr['gv'][t], tr['gth'][t]
                if not have:
                    ex, ey, ev, eth = gx, gy, gv, gth; have = True
                else:
                    dt = max(t - tl, 1)
                    err = np.hypot(ex - gx, ey - gy)
                    chat = max(0.7 * chat + 0.3 * max(err - 8.0, 0.0) / dt ** 1.5, 1e-4)
                    if use_imu:
                        ba += 0.3 * np.clip((ev - gv) / dt, -0.1, 0.1)
                        if gv > 2 and ev > 2:
                            bw += 0.3 * np.clip(_wrap(eth - gth) / dt, -0.01, 0.01)
                    ex, ey = gx, gy
                    ev = (0.3 * ev + 0.7 * gv) if use_imu else gv
                    if gv > 2:
                        eth = gth
                stationary = gv < 0.3; yaw_acc = 0.0; tl = t
                if sched == 'learn':
                    delta = min(max((eps / chat) ** (1 / 1.5), 2.0), 300.0)
                else:
                    delta = min(max(eps / (0.3 * max(ev, 0.5)), 2.0), 60.0)
                if stationary: delta = min(delta, 60.0)
                nxt = t + delta
                send = srv is None
                if srv is not None:
                    dtp = t - srv[0]
                    px = srv[1] + srv[3] * np.cos(srv[4]) * dtp; py = srv[2] + srv[3] * np.sin(srv[4]) * dtp
                    send = np.hypot(px - gx, py - gy) > 0.7 * eps
                if send:
                    srv = (t, gx, gy, ev, eth); rep.append(srv); m.tx(t); last_rep_t = t
            else:
                nxt = t + 1
        elif have and use_imu and t - last_rep_t >= 2:
            dtp = t - srv[0]
            px = srv[1] + srv[3] * np.cos(srv[4]) * dtp; py = srv[2] + srv[3] * np.sin(srv[4]) * dtp
            if np.hypot(px - ex, py - ey) > eps:
                srv = (t, ex, ey, ev, eth); rep.append(srv); m.tx(t); last_rep_t = t
    ex_, ey_ = est_from_reports(rep, n, 'cv')
    return pack(tr, m, ex_, ey_, P_IMU if use_imu else 0.0)


# =====================================================================================
#                                     SECURITY (armed mode)
# =====================================================================================
class SecParams:
    verify_window = 120      # s
    confirm_dist = 100.0     # m, displacement that confirms theft
    cats_interval = 5        # s
    base_interval = 10       # s
    failsafe = 30            # s, baselines: no fix after trigger -> alarm
    cats_jam_wait = 20       # s
    imu_v_thr = 5.0          # m/s integrated speed that counts as IMU-motion
    chi_win = 8              # fixes in range-residual window
    chi_thr = 3.0            # mean z^2 threshold
    chi_sigma = 250.0        # m, TA range residual sigma
    ring_sigma_k = 2.5       # (|d_gps - TA*step|/step) threshold
    knear = 6


def _nearest_rank(px, py, cell):
    bts = WORLD['bts']
    d = np.hypot(bts[:, 0] - px, bts[:, 1] - py)
    return int(np.sum(d < d[cell]))     # 0 = nearest


def cell_novelty_events(tr, t_end=None):
    """Times at which the serving cell is new w.r.t. the 'home set' learnt in the first 300 s of arming."""
    sv = tr['serving']; n = tr['n'] if t_end is None else t_end
    H = set(int(c) for c in sv[:300])
    ev = []
    ch = np.nonzero(np.diff(sv[:n]) != 0)[0] + 1
    for t in ch:
        c = int(sv[t])
        if t >= 300 and c not in H:
            ev.append(int(t)); H.add(c)
    return ev


def cell_triggers(tr, t_end=None):
    """Cell-sentinel candidate triggers: 2 novel cells within 180 s."""
    ev = cell_novelty_events(tr, t_end)
    out = []
    for i in range(1, len(ev)):
        if ev[i] - ev[i - 1] <= 180:
            out.append(ev[i])
    return out, ev


def _verify(tr, tau, mode, m, P):
    """Verification window after a trigger at tau. Returns (alarm_time or None, reason)."""
    n = tr['n']; px, py = tr['park']; ok = tr['gps_ok']
    window = P.verify_window
    interval = P.cats_interval if mode.startswith('CATS') else P.base_interval
    ia = tr['imu_a']
    lo = max(0, tau - 300)
    ba_est = float(np.mean(ia[lo:max(lo + 1, tau - 5)])) if tau - lo > 10 else 0.0
    v_imu = 0.0; t_next = tau; any_ok = False; bad_imu = 0; bad_cell = 0
    last_ok = tau
    nov = cell_novelty_events(tr, min(n, tau + window + 1)) if mode in ('CATS', 'CATS-cell') else []
    for t in range(tau, min(n, tau + window + 1)):
        v_imu += ia[t] - ba_est
        if t >= t_next:
            success = m.fix(t, ok[t])
            if success:
                any_ok = True; last_ok = t; t_next = t + interval
                gx, gy, gv = tr['gx'][t], tr['gy'][t], tr['gv'][t]
                D = np.hypot(gx - px, gy - py)
                if mode in ('A3', 'CATS', 'CATS-imu'):
                    if abs(v_imu) > P.imu_v_thr and gv < 1.0:
                        bad_imu += 1
                    else:
                        bad_imu = 0
                    if bad_imu >= 2:
                        return t, 'spoof_imu'
                if mode in ('CATS', 'CATS-cell'):
                    c = tr['serving'][t]
                    bx, by = WORLD['bts'][c]
                    ring = abs(np.hypot(gx - bx, gy - by) - TA_STEP * tr['ta'][t]) / TA_STEP
                    rank = _nearest_rank(gx, gy, c)
                    if ring > P.ring_sigma_k or rank >= P.knear:
                        bad_cell += 1
                    else:
                        bad_cell = 0
                    if bad_cell >= 2:
                        return t, 'spoof_cell'
                if D > P.confirm_dist:
                    return t, 'theft'
            else:
                t_next = t + 1
                if mode in ('A2', 'A3') and (t - tau) >= P.failsafe and not any_ok:
                    return t, 'failsafe'
                if mode.startswith('CATS') and not any_ok and (t - tau) >= P.cats_jam_wait:
                    motion = (mode != 'CATS-cell' and abs(v_imu) > P.imu_v_thr) or any(tau - 60 <= e <= t for e in nov)
                    if motion:
                        return t, 'jam'
                if mode.startswith('CATS') and any_ok and (t - last_ok) >= P.cats_jam_wait:
                    motion = (mode != 'CATS-cell' and abs(v_imu) > P.imu_v_thr) or any(tau - 60 <= e <= t for e in nov)
                    if motion:
                        return t, 'jam'
    return None, 'cancel'


def armed_detect(tr, scheme, P=SecParams, rearm=False):
    """Run armed-mode logic. Returns dict(alarms=[(t,reason)], meter, extra_power, theft_alarm=(t,reason)|None)."""
    n = tr['n']; m = Meter(); alarms = []
    trig = list(tr['acc_fa'])
    if tr['acc_hit'] >= 0:
        trig.append(int(tr['acc_hit']))
    trig = sorted(trig)
    if scheme == 'A1':
        for t in trig:
            m.fix(t, True); m.tx(t, sms=True); alarms.append((t, 'acc'))
        return dict(alarms=alarms, meter=m, extra=P_ACC)
    triggers = [(t, 'acc') for t in trig]
    extra = P_ACC
    if scheme in ('CATS', 'CATS-cell'):
        ct, _ = cell_triggers(tr)
        triggers += [(t, 'cell') for t in ct]
        extra = P_IMU + P_CELLQ
    elif scheme == 'CATS-imu':
        extra = P_IMU
    triggers.sort()
    busy_until = -1
    for t, src in triggers:
        if t <= busy_until:
            continue
        at, reason = _verify(tr, t, scheme, m, P)
        busy_until = (at if at is not None else min(n, t + P.verify_window)) + 1
        if at is not None:
            alarms.append((at, reason)); m.tx(at, sms=True)
            if not rearm:
                break      # after an alarm the system leaves armed mode (owner is notified)
    return dict(alarms=alarms, meter=m, extra=extra)


# ------------------------------------------------------------------ post-alarm tracking
def ekf_track(meas, t_start, n, x0, y0, p0, qa=1.5):
    """Range-only EKF on TA rings. meas: list of (t, bx, by, ta)."""
    X = np.array([x0, y0, 0.0, 0.0]); Pm = np.diag([p0 ** 2, p0 ** 2, 8.0 ** 2, 8.0 ** 2])
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], float)
    Q = G @ G.T * qa ** 2
    out = np.full((n, 2), np.nan); k = 0
    for t in range(t_start, n):
        X = F @ X; Pm = F @ Pm @ F.T + Q
        while k < len(meas) and meas[k][0] <= t:
            _, bx, by, ta = meas[k]; k += 1
            dx, dy = X[0] - bx, X[1] - by; r = max(np.hypot(dx, dy), 50.0)
            Hm = np.array([[dx / r, dy / r, 0, 0]])
            robs = max(ta * TA_STEP, 140.0); sig = 140.0 if ta == 0 else 250.0
            S = float((Hm @ Pm @ Hm.T)[0, 0]) + sig ** 2
            K = (Pm @ Hm.T / S).ravel()
            X = X + K * (robs - r)
            Pm = (np.eye(4) - np.outer(K, Hm.ravel())) @ Pm
        out[t] = X[:2]
    return out


def track_after_alarm(tr, scheme, alarm, P=SecParams, period=30):
    """Server-side track over the theft window for a scheme. Returns est arrays (n,) and the time GPS was distrusted."""
    n = tr['n']; t0 = tr['t0']; px, py = tr['park']; bts = WORLD['bts']
    ex = np.full(n, np.nan); ey = np.full(n, np.nan)
    ex[t0:] = px; ey[t0:] = py
    if alarm is None:
        return ex, ey, None
    ta_, reason = alarm
    untrusted_at = None
    meas = []
    t = ta_; zs = []
    trusted = True
    if scheme.startswith('CATS') and reason in ('jam', 'spoof_imu', 'spoof_cell'):
        untrusted_at = ta_
    last_gps_ok = ta_
    gps_est = (px, py)
    rep = []
    while t < n:
        ok = tr['gps_ok'][t]
        if scheme.startswith('CATS') and untrusted_at is not None:
            meas.append((t, bts[tr['serving'][t]][0], bts[tr['serving'][t]][1], int(tr['ta'][t])))
            t += period; continue
        if ok:
            last_gps_ok = t
            gx, gy = tr['gx'][t], tr['gy'][t]
            rep.append((t, gx, gy))
            if scheme.startswith('CATS'):
                c = tr['serving'][t]; bx, by = bts[c]
                z = (np.hypot(gx - bx, gy - by) - TA_STEP * tr['ta'][t]) / P.chi_sigma
                zs.append(z * z)
                if len(zs) >= P.chi_win and np.mean(zs[-P.chi_win:]) > P.chi_thr:
                    untrusted_at = t
                meas.append((t, bx, by, int(tr['ta'][t])))
            t += period
        else:
            if scheme.startswith('CATS') and t - last_gps_ok >= 60:
                untrusted_at = t
            t += 1 if (not scheme.startswith('CATS')) or untrusted_at is None else period
    # build estimate
    k = 0; cur = (px, py)
    ptr = 0
    for tt in range(ta_, n):
        while ptr < len(rep) and rep[ptr][0] <= tt and (untrusted_at is None or rep[ptr][0] < untrusted_at):
            cur = (rep[ptr][1], rep[ptr][2]); ptr += 1
        ex[tt], ey[tt] = cur
    if scheme.startswith('CATS') and untrusted_at is not None:
        p0 = min(18.0 * max(untrusted_at - t0, 1) / 2.0, 3000.0)
        m2 = [mm for mm in meas if mm[0] >= untrusted_at - 60] if reason == 'theft' else meas
        out = ekf_track(m2, untrusted_at, n, px, py, max(p0, 300.0))
        ex[untrusted_at:] = out[untrusted_at:, 0]; ey[untrusted_at:] = out[untrusted_at:, 1]
    return ex, ey, untrusted_at
