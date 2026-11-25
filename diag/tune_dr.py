import numpy as np, pickle, multiprocessing as mp, itertools
import world, algos as A
from exp import run_cfg
from diag_resample import route_from_xy, EPS
KAP=(0.1,0.2,0.3,0.5,0.8); TMX=(30,60,120,300)
def tr_synth(seed): return world.driving_trace(seed)
def tr_res(seed):
    tr0=world.driving_trace(seed); rng=np.random.default_rng(seed+77)
    x=tr0['x']+rng.normal(0,5,tr0['n']); y=tr0['y']+rng.normal(0,5,tr0['n'])
    return world.driving_trace(seed,route=route_from_xy(x,y))
def job_train(seed):
    tr=tr_synth(seed); return {(k,t,e):(lambda r:(r['power_mW'],r['mean']))(A.run_dr_threshold(tr,e,kappa=k,tmax=t)) for k in KAP for t in TMX for e in EPS}
def job_test(a):
    seed,mode=a; tr=tr_synth(seed) if mode=='S' else tr_res(seed)
    out={(k,t,e):(lambda r:(r['power_mW'],r['mean']))(A.run_dr_threshold(tr,e,kappa=k,tmax=t)) for (k,t) in [(0.3,60),'BEST'] if False for e in EPS} if False else {}
    return tr
def curve(res,key):
    P=np.array([np.mean([r[(key[0],key[1],e)][0] for r in res]) for e in EPS]);E=np.array([np.mean([r[(key[0],key[1],e)][1] for r in res]) for e in EPS])
    o=np.argsort(P);return P[o],E[o]
def at(P0,E0,P):
    return float(np.exp(np.interp(np.log(P),np.log(P0),np.log(E0)))) if P0.min()<=P<=P0.max() else np.nan
if __name__=='__main__':
    ctx=mp.get_context('fork')
    with ctx.Pool(2) as p: tr=p.map(job_train,list(range(7000,7030)))
    best=None
    for k in KAP:
        for t in TMX:
            P0,E0=curve(tr,(k,t)); v=[at(P0,E0,P) for P in (70,80,100)]
            sc=np.nanmean(np.log(v)) if not np.isnan(v).any() else np.inf
            print(k,t,[round(x,1) for x in v],round(sc,3))
            if best is None or sc<best[0]: best=(sc,k,t)
    print('BEST',best); pickle.dump(best,open('best_dr.pkl','wb'))
