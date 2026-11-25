import numpy as np, pickle, multiprocessing as mp
import tune_dr as T, algos as A
from diag_resample import EPS
T.KAP=(0.2,0.3,0.4,0.5); T.TMX=(10,15,20,30,45)
def jt(seed):
    tr=T.tr_synth(seed); return {(k,t,e):(lambda r:(r['power_mW'],r['mean']))(A.run_dr_threshold(tr,e,kappa=k,tmax=t)) for k in T.KAP for t in T.TMX for e in EPS}
if __name__=='__main__':
    ctx=mp.get_context('fork')
    with ctx.Pool(2) as p: tr=p.map(jt,list(range(7000,7030)))
    best=None
    for k in T.KAP:
        for t in T.TMX:
            P0,E0=T.curve(tr,(k,t)); v=[T.at(P0,E0,P) for P in (70,80,100)]
            sc=np.nanmean(np.log(v)) if not np.isnan(v).any() else np.inf
            print(k,t,[round(x,1) for x in v],round(sc,3))
            if best is None or sc<best[0]: best=(sc,k,t)
    print('BEST',best); pickle.dump(best,open('best_dr.pkl','wb'))
