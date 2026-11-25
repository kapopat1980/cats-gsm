import numpy as np, pickle, multiprocessing as mp
import tune_dr as T, algos as A
from exp import run_cfg
from diag_resample import EPS
def j(a):
    seed,mode=a; tr=T.tr_synth(seed) if mode=='S' else T.tr_res(seed)
    d={}
    for e in EPS:
        for k,t in ((0.3,60),(0.3,30)):
            r=A.run_dr_threshold(tr,e,kappa=k,tmax=t); d[('DR',k,t,e)]=(r['power_mW'],r['mean'])
        for nm in ('D3-IMU','CATS'):
            r=run_cfg(tr,nm,e); d[(nm,0,0,e)]=(r['power_mW'],r['mean'])
    return d
def cur(res,key):
    P=np.array([np.mean([r[key+(e,)][0] for r in res]) for e in EPS]);E=np.array([np.mean([r[key+(e,)][1] for r in res]) for e in EPS]);o=np.argsort(P);return P[o],E[o]
if __name__=='__main__':
    ctx=mp.get_context('fork')
    for mode,seeds in (('S',range(1000,1100)),('R',range(1000,1100))):
        with ctx.Pool(2) as p: res=p.map(j,[(s,mode) for s in seeds])
        pickle.dump(res,open(f'eval_{mode}.pkl','wb'))
        print('mode',mode)
        for P in (70,80,100):
            v={n:T.at(*cur(res,k),P) for n,k in (('DR60',('DR',0.3,60)),('DR30',('DR',0.3,30)),('IMU',('D3-IMU',0,0)),('CATS',('CATS',0,0)))}
            print(P,{a:round(b,1) for a,b in v.items()},'CATS vs DR30 %.1f%%'%(100*(1-v['CATS']/v['DR30'])),'vs IMU %.1f%%'%(100*(1-v['CATS']/v['IMU'])))
