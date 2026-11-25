"""Re-run of the Porto driving comparison with a TUNED GPS-only baseline (D3-DR, kappa=0.3, cap 30 s; tuned on synthetic training seeds 7000-7029).
Uses exactly the same 100 driving routes as exp_real.py.  Output: tuned_driving.json"""
import os, json, time
import numpy as np, multiprocessing as mp
Pool = mp.get_context("fork").Pool
import world, algos as A
import realdata as R
from exp import EPS, run_cfg

CSV = os.environ.get('PORTO_CSV') or R.find_csv()
N_DRIVE = int(os.environ.get('N_DRIVE', 100)); N_ARMED = int(os.environ.get('N_ARMED', 150)); N_FA = int(os.environ.get('N_FA', 40))
NROWS = int(os.environ['NROWS']) if os.environ.get('NROWS') else None
NPROC = int(os.environ.get('NPROC', os.cpu_count() or 2))
RAW = R.load_routes(CSV, n_routes=N_DRIVE + N_ARMED + N_FA, nrows=NROWS)     # same call as exp_real.py -> same routes
assert len(RAW) >= N_DRIVE + N_ARMED + N_FA
ROUTES, AREA = R.to_world_frame(RAW); world.set_world(AREA); DR = ROUTES[:N_DRIVE]
VARIANTS = [('DR60', 0.3, 60.0), ('DR30', 0.3, 30.0), ('DR45', 0.3, 45.0), ('DR20', 0.3, 20.0)]

def job(i):
    tr = world.driving_trace(1000 + i, route=DR[i]); o = {}
    for e in EPS:
        for nm, k, t in VARIANTS:
            r = A.run_dr_threshold(tr, e, kappa=k, tmax=t); o[f'{nm}|{e}'] = (r['power_mW'], r['mean'])
        for nm in ('D3-IMU', 'CATS'):
            r = run_cfg(tr, nm, e); o[f'{nm}|{e}'] = (r['power_mW'], r['mean'])
    return o

def curve(res, nm, idx):
    P = np.array([np.mean([res[i][f'{nm}|{e}'][0] for i in idx]) for e in EPS]); E = np.array([np.mean([res[i][f'{nm}|{e}'][1] for i in idx]) for e in EPS])
    o = np.argsort(P); return P[o], E[o]
def at(nm, P, res, idx):
    P0, E0 = curve(res, nm, idx)
    return float(np.exp(np.interp(np.log(P), np.log(P0), np.log(E0)))) if P0.min() <= P <= P0.max() else float('nan')

if __name__ == '__main__':
    t0 = time.time()
    with Pool(NPROC) as p: res = p.map(job, range(len(DR)))
    print('runs done %.0fs' % (time.time() - t0), flush=True)
    rng = np.random.default_rng(0); out = {}
    for P in (60, 70, 80, 100):
        row = {nm: at(nm, P, res, range(len(DR))) for nm in ('DR60', 'DR30', 'DR45', 'DR20', 'D3-IMU', 'CATS')}
        row['CATS_vs'] = {}
        for base in ('DR60', 'DR30', 'D3-IMU'):
            bs = []
            for _ in range(300):
                idx = rng.integers(0, len(DR), len(DR)); c = at('CATS', P, res, idx); o = at(base, P, res, idx)
                if not (np.isnan(c) or np.isnan(o)): bs.append(100 * (1 - c / o))
            pt = 100 * (1 - row['CATS'] / row[base])
            row['CATS_vs'][base] = [pt, float(np.percentile(bs, 2.5)) if bs else None, float(np.percentile(bs, 97.5)) if bs else None]
        out[P] = row
        print(P, {k: (round(v, 1) if not isinstance(v, dict) else {a: [round(x, 1) if x is not None else None for x in b] for a, b in v.items()}) for k, v in row.items()}, flush=True)
    json.dump(out, open('tuned_driving.json', 'w'), indent=1)
