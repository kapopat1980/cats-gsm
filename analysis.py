import pickle, json, numpy as np
from scipy import stats
rng = np.random.default_rng(0)
res = pickle.load(open('results.pkl', 'rb'))
out = {}

# ---------------- driving
keys = list(res['drive'][0].keys())
N = len(res['drive'])
def arr(k, f): return np.array([r[k][f] for r in res['drive']])
fam = {}
for k in keys:
    nm, v = k.split('|'); fam.setdefault(nm, []).append(k)
rows = []
for k in keys:
    e = arr(k, 'mean'); p = arr(k, 'power_mW')
    bs = [rng.choice(e, N).mean() for _ in range(1000)]
    rows.append(dict(cfg=k, err=float(e.mean()), err_lo=float(np.percentile(bs, 2.5)), err_hi=float(np.percentile(bs, 97.5)),
                     p95=float(arr(k, 'p95').mean()), power=float(p.mean()), msgs=float(arr(k, 'msgs').mean()),
                     life=float(arr(k, 'life_days').mean())))
out['drive_rows'] = rows

def curve(name, idx):
    ks = fam[name]
    P = np.array([arr(k, 'power_mW')[idx].mean() for k in ks]); E = np.array([arr(k, 'mean')[idx].mean() for k in ks])
    o = np.argsort(P); return P[o], E[o]
def err_at(name, Pstar, idx):
    P, E = curve(name, idx)
    if len(P) < 2 or Pstar < P.min() or Pstar > P.max(): return np.nan
    return float(np.exp(np.interp(np.log(Pstar), np.log(P), np.log(E))))
levels = [50, 60, 80, 100]
match = {}
for Pstar in levels:
    for nm in ['D1-periodic', 'D2-distance', 'D3-DR', 'D3-IMU', 'CATS-noIMU', 'CATS']:
        vals = []
        for b in range(500):
            idx = rng.integers(0, N, N); vals.append(err_at(nm, Pstar, idx))
        point = err_at(nm, Pstar, np.arange(N))
        vv = np.array(vals); vv = vv[~np.isnan(vv)]
        match[f'{nm}@{Pstar}'] = dict(err=point, lo=float(np.percentile(vv, 2.5)) if len(vv) else None, hi=float(np.percentile(vv, 97.5)) if len(vv) else None)
out['matched'] = match
# paired improvement CATS vs others at matched power, bootstrap
imp = {}
for Pstar in levels:
    for base in ['D1-periodic', 'D2-distance', 'D3-DR', 'D3-IMU', 'CATS-noIMU']:
        r = []
        for b in range(500):
            idx = rng.integers(0, N, N)
            c, o = err_at('CATS', Pstar, idx), err_at(base, Pstar, idx)
            if not (np.isnan(c) or np.isnan(o)): r.append(100 * (1 - c / o))
        c0, o0 = err_at('CATS', Pstar, np.arange(N)), err_at(base, Pstar, np.arange(N))
        if r and not (np.isnan(c0) or np.isnan(o0)):
            imp[f'{base}@{Pstar}'] = dict(pct=100 * (1 - c0 / o0), lo=float(np.percentile(r, 2.5)), hi=float(np.percentile(r, 97.5)))
out['improvement_pct'] = imp

# ---------------- armed
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)
armed = {}
for atk, lst in res['armed'].items():
    armed[atk] = {}
    for sc in lst[0].keys():
        dl = [r[sc]['delay'] for r in lst]
        det300 = sum(1 for d in dl if d is not None and d <= 300)
        dv = [d for d in dl if d is not None]
        me = np.array([r[sc]['mean'] for r in lst]); p95 = np.array([r[sc]['p95'] for r in lst])
        armed[atk][sc] = dict(det300=det300, n=len(lst), det_ci=wilson(det300, len(lst)),
                              med_delay=float(np.median(dv)) if dv else None, p90_delay=float(np.percentile(dv, 90)) if dv else None,
                              track_mean=float(me.mean()), track_median=float(np.median(me)), track_p95=float(p95.mean()),
                              reasons={x: sum(1 for r in lst if r[sc]['reason'] == x) for x in set(r[sc]['reason'] for r in lst if r[sc]['reason'])})
out['armed'] = armed
tests = {}
for atk, lst in res['armed'].items():
    for base in ['A1', 'A2', 'A3', 'D4-cellID']:
        a = np.array([r['CATS']['mean'] for r in lst]); b = np.array([r[base]['mean'] for r in lst])
        try:
            w = stats.wilcoxon(a, b); tests[f'{atk}:CATS_vs_{base}'] = dict(p=float(w.pvalue), med_diff=float(np.median(a - b)))
        except Exception as e:
            tests[f'{atk}:CATS_vs_{base}'] = dict(p=None)
out['tests_track'] = tests
# detection paired McNemar (exact binomial) CATS vs A2/A3
mc = {}
for atk, lst in res['armed'].items():
    for base in ['A1', 'A2', 'A3']:
        b01 = sum(1 for r in lst if (r['CATS']['delay'] is not None and r['CATS']['delay'] <= 300) and not (r[base]['delay'] is not None and r[base]['delay'] <= 300))
        b10 = sum(1 for r in lst if not (r['CATS']['delay'] is not None and r['CATS']['delay'] <= 300) and (r[base]['delay'] is not None and r[base]['delay'] <= 300))
        p = stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
        mc[f'{atk}:CATS_vs_{base}'] = dict(cats_only=b01, base_only=b10, p=float(p))
out['mcnemar'] = mc
# FA
fa = {}
for sc in res['fa'][0].keys():
    al = np.array([r[sc]['alarms'] for r in res['fa']]); en = np.array([r[sc]['energy_J'] for r in res['fa']])
    fa[sc] = dict(alarms_per_day=float(al.mean()), traces_with_fa=int((al > 0).sum()), n=len(al), kJ_per_day=float(en.mean() / 1000),
                  life_days=float(26640 / (en.mean() / 86400) / 86400), fixes_per_day=float(np.mean([r[sc]['fixes'] for r in res['fa']])))
out['fa'] = fa
# sensitivity
sens = {}
for pd, lst in list(res['sens_pd'].items()) + [('0.9', res['armed']['none'])]:
    sens[pd] = {sc: dict(det300=sum(1 for r in lst if r[sc]['delay'] is not None and r[sc]['delay'] <= 300), n=len(lst),
                         med=float(np.median([r[sc]['delay'] for r in lst if r[sc]['delay'] is not None] or [np.nan])))
                for sc in ['A1', 'A2', 'A3', 'CATS']}
out['sens_pd'] = sens
json.dump(out, open('analysis.json', 'w'), indent=1, default=float)

# figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.figure(figsize=(6.4, 4.4))
sty = {'D1-periodic': ('s', '#888'), 'D2-distance': ('^', '#b07'), 'D3-DR': ('v', '#d80'), 'D3-IMU': ('D', '#07a'), 'CATS-noIMU': ('x', '#6a3'), 'CATS': ('o', '#c00'), 'D4-cellID': ('*', '#000')}
for nm, (mk, c) in sty.items():
    P, E = curve(nm, np.arange(N)); plt.plot(P, E, marker=mk, color=c, label=nm, lw=1.3, ms=6)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('Average tracker power (mW)'); plt.ylabel('Mean server tracking error (m)')
plt.grid(True, which='both', alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig('fig_pareto.png', dpi=200)
print(json.dumps({k: out[k] for k in ['matched', 'improvement_pct', 'fa', 'sens_pd']}, indent=1, default=float))
