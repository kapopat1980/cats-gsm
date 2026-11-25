"""CATS-GSM experiments on real Porto taxi routes (trace-driven).  Mirrors exp.py; results -> results.pkl for analysis.py"""
import sys, time, pickle, os
import numpy as np
import multiprocessing as mp
Pool = mp.get_context("fork").Pool
import world, algos as A
import realdata as R
from exp import DRIVE_CFG, SCHEMES, ATTACKS, run_cfg

CSV = os.environ.get('PORTO_CSV') or R.find_csv()
N_DRIVE = int(os.environ.get('N_DRIVE', 100))
N_ARMED = int(os.environ.get('N_ARMED', 150))
N_FA = int(os.environ.get('N_FA', 40))
NROWS = int(os.environ['NROWS']) if os.environ.get('NROWS') else None
NPROC = int(os.environ.get('NPROC', os.cpu_count() or 2))

print('reading', CSV, flush=True)
RAW = R.load_routes(CSV, n_routes=N_DRIVE + N_ARMED + N_FA, nrows=NROWS)
print('usable routes:', len(RAW), flush=True)
assert len(RAW) >= N_DRIVE + N_ARMED + N_FA, 'not enough usable trips; raise NROWS or lower the counts'
ROUTES, AREA = R.to_world_frame(RAW)
world.set_world(AREA)
print('world area %.1f km, %d base stations' % (AREA / 1000, len(world.WORLD['bts'])), flush=True)
DR, AR, FR = ROUTES[:N_DRIVE], ROUTES[N_DRIVE:N_DRIVE + N_ARMED], ROUTES[N_DRIVE + N_ARMED:]


def drive_job(i):
    tr = world.driving_trace(1000 + i, route=DR[i])
    return {f'{nm}|{v}': run_cfg(tr, nm, v) for nm, _, v in DRIVE_CFG}


def armed_job(args):
    i, atk, pd = args
    tr = world.armed_trace(2000 + i, 900, 1200, atk, acc_pd=pd, route=AR[i])
    out = {}
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


def fa_job(i):
    px, py = float(FR[i][0][0]), float(FR[i][1][0])
    tr = world.armed_trace(9100 + i, 86400, 0, 'none', theft=False, park=(px, py))
    out = {}
    for sc in SCHEMES:
        d = A.armed_detect(tr, sc, rearm=True)
        out[sc] = dict(alarms=len(d['alarms']), energy_J=float(d['meter'].total(86400, d['extra'])),
                       fixes=d['meter'].n_fix + d['meter'].n_fail)
    return out


if __name__ == '__main__':
    t = time.time(); res = {}
    with Pool(NPROC) as p:
        res['drive'] = p.map(drive_job, range(len(DR)))
        print('drive done %.0fs' % (time.time() - t), flush=True)
        jobs = [(i, atk, 0.9) for atk in ATTACKS for i in range(len(AR))]
        r = p.map(armed_job, jobs, chunksize=4)
        res['armed'] = {atk: r[k * len(AR):(k + 1) * len(AR)] for k, atk in enumerate(ATTACKS)}
        print('armed done %.0fs' % (time.time() - t), flush=True)
        res['fa'] = p.map(fa_job, range(len(FR)))
        print('false-alarm runs done %.0fs' % (time.time() - t), flush=True)
        sens = {}
        for pd_ in (0.7, 0.99):
            sens[str(pd_)] = p.map(armed_job, [(i, 'none', pd_) for i in range(len(AR))], chunksize=4)
        res['sens_pd'] = sens
    pickle.dump(res, open('results.pkl', 'wb'))
    print('all done %.0fs' % (time.time() - t))
