import json, numpy as np, world, algos as A
from exp import EPS
seeds = range(1000, 1040)
trs = [world.driving_trace(s) for s in seeds]
variants = {'baseline': {}, 'E_ATT=0.5J': {'E_ATT': 0.5}, 'E_ATT=3.6J': {'E_ATT': 3.6}, 'E_FIX=0.3J': {'E_FIX': 0.3}, 'E_FIX=1.2J': {'E_FIX': 1.2}}
base = dict(E_ATT=A.E_ATT, E_FIX=A.E_FIX)
res = {}
for nm, ch in variants.items():
    A.E_ATT = base['E_ATT']; A.E_FIX = base['E_FIX']
    for k, v in ch.items(): setattr(A, k, v)
    fam = {'D3-DR': lambda tr, e: A.run_dr_threshold(tr, e), 'D3-IMU': lambda tr, e: A.run_cats_drive(tr, e, sched='speed'),
           'CATS': lambda tr, e: A.run_cats_drive(tr, e)}
    cur = {}
    for f, fn in fam.items():
        P, E = [], []
        for e in EPS:
            r = [fn(tr, e) for tr in trs]
            P.append(np.mean([x['power_mW'] for x in r])); E.append(np.mean([x['mean'] for x in r]))
        o = np.argsort(P); cur[f] = (np.array(P)[o], np.array(E)[o])
    lo = max(cur[f][0].min() for f in cur); hi = min(cur[f][0].max() for f in cur)
    out = {}
    for frac, lab in ((0.25, 'low'), (0.5, 'mid'), (0.75, 'high')):
        Ps = float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))
        e = {f: float(np.exp(np.interp(np.log(Ps), np.log(cur[f][0]), np.log(cur[f][1])))) for f in cur}
        out[lab] = dict(P=Ps, **e, vs_DR=100 * (1 - e['CATS'] / e['D3-DR']), vs_IMU=100 * (1 - e['CATS'] / e['D3-IMU']))
    res[nm] = out
    print(nm, json.dumps({k: {a: round(b, 1) for a, b in v.items()} for k, v in out.items()}), flush=True)
json.dump(res, open('sens_energy.json', 'w'), indent=1)
