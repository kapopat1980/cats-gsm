import sys, json, time, pickle
import numpy as np
from multiprocessing import Pool
import world, algos as A

DRIVE_SEEDS = list(range(1000, 1100))
ARMED_SEEDS = list(range(2000, 2150))
FA_SEEDS = list(range(9100, 9140))
ATTACKS = ['none', 'jam', 'replay', 'drift']
SCHEMES = ['A1', 'A2', 'A3', 'CATS-imu', 'CATS-cell', 'CATS']

EPS = (25, 35, 50, 70, 100, 140, 200, 280, 400, 600, 800)
DRIVE_CFG = ([('D1-periodic', 'T', T) for T in (10, 20, 30, 45, 60, 90, 120, 180, 300)] +
             [('D2-distance', 'D', D) for D in (100, 250, 500, 1000)] +
             [('D3-DR', 'theta', th) for th in EPS] +
             [('D3-IMU', 'eps', e) for e in EPS] +
             [('CATS', 'eps', e) for e in EPS] +
             [('CATS-noIMU', 'eps', e) for e in EPS] +
             [('D4-cellID', '-', 0)])


def run_cfg(tr, name, val):
    if name == 'D1-periodic': return A.run_periodic(tr, val)
    if name == 'D2-distance': return A.run_distance(tr, val)
    if name == 'D3-DR': return A.run_dr_threshold(tr, val)
    if name == 'D3-IMU': return A.run_cats_drive(tr, val, sched='speed')
    if name == 'CATS': return A.run_cats_drive(tr, val, sched='learn')
    if name == 'CATS-noIMU': return A.run_cats_drive(tr, val, use_imu=False)
    if name == 'D4-cellID': return A.run_cellid(tr)


def drive_job(seed):
    tr = world.driving_trace(seed)
    return {f'{nm}|{v}': run_cfg(tr, nm, v) for nm, _, v in DRIVE_CFG}


def armed_job(args):
    seed, atk, pd = args
    tr = world.armed_trace(seed, 900, 1200, atk, acc_pd=pd)
    out = {}
    d4 = None
    for sc in SCHEMES:
        d = A.armed_detect(tr, sc)
        th = [a for a in d['alarms'] if a[0] >= tr['t0']]
        alarm = th[0] if th else None
        ex, ey, untr = A.track_after_alarm(tr, sc, alarm)
        err = A.track_error(tr, ex, ey, tr['t0'], tr['n'])
        out[sc] = dict(delay=(alarm[0] - tr['t0']) if alarm else None, reason=alarm[1] if alarm else None,
                       mean=float(err.mean()), p95=float(np.percentile(err, 95)), untrusted=untr)
    bts = world.WORLD['bts']
    cx = bts[tr['serving'][tr['t0']:], 0]; cy = bts[tr['serving'][tr['t0']:], 1]
    e = np.hypot(cx - tr['x'][tr['t0']:], cy - tr['y'][tr['t0']:])
    out['D4-cellID'] = dict(delay=None, reason=None, mean=float(e.mean()), p95=float(np.percentile(e, 95)), untrusted=None)
    return out


def fa_job(seed):
    tr = world.armed_trace(seed, 86400, 0, 'none', theft=False)
    out = {}
    for sc in SCHEMES:
        d = A.armed_detect(tr, sc, rearm=True)
        out[sc] = dict(alarms=len(d['alarms']), energy_J=float(d['meter'].total(86400, d['extra'])),
                       fixes=d['meter'].n_fix + d['meter'].n_fail)
    return out


if __name__ == '__main__':
    t = time.time()
    res = {}
    with Pool(2) as p:
        res['drive'] = p.map(drive_job, DRIVE_SEEDS)
        print('drive done', time.time() - t, flush=True)
        jobs = [(s, atk, 0.9) for atk in ATTACKS for s in ARMED_SEEDS]
        r = p.map(armed_job, jobs, chunksize=5)
        res['armed'] = {atk: r[i * len(ARMED_SEEDS):(i + 1) * len(ARMED_SEEDS)] for i, atk in enumerate(ATTACKS)}
        print('armed done', time.time() - t, flush=True)
        res['fa'] = p.map(fa_job, FA_SEEDS)
        print('fa done', time.time() - t, flush=True)
        # sensitivity: motion-sensor detection probability (no-attack theft)
        sens = {}
        for pd in (0.7, 0.99):
            r = p.map(armed_job, [(s, 'none', pd) for s in ARMED_SEEDS], chunksize=5)
            sens[str(pd)] = r
        res['sens_pd'] = sens
    pickle.dump(res, open('results.pkl', 'wb'))
    print('all done', time.time() - t)
