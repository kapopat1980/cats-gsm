"""Diagnostic: do Study-S gains survive 15 s sampling + spline interpolation (the Porto pipeline)?"""
import numpy as np, pickle, multiprocessing as mp
import world, algos as A
from scipy.interpolate import CubicSpline
from exp import run_cfg

EPS=(25,35,50,70,100,140,200,280,400,600,800)
CFG=[('D3-DR',e) for e in EPS]+[('D3-IMU',e) for e in EPS]+[('CATS',e) for e in EPS]+[('CATS-noIMU',e) for e in EPS]
N=100

def route_from_xy(x,y,dt=15):
    xs15=x[::dt]; ys15=y[::dt]; tk=np.arange(len(xs15))*float(dt)
    t=np.arange(0.0,tk[-1]+1e-9,1.0)
    xs=CubicSpline(tk,xs15)(t); ys=CubicSpline(tk,ys15)(t)
    k=np.ones(5)/5
    vx=np.convolve(np.gradient(xs),k,mode='same'); vy=np.convolve(np.gradient(ys),k,mode='same')
    v=np.minimum(np.hypot(vx,vy),40.0); th=np.arctan2(vy,vx)
    for i in range(1,len(th)):
        if v[i]<1.0: th[i]=th[i-1]
    th=np.unwrap(th); a=np.gradient(v); w=np.gradient(th); w[v<1.0]=0.0; w=np.clip(w,-np.pi/4,np.pi/4)
    return xs,ys,v,th,a,w

def job(i):
    seed=1000+i
    tr0=world.driving_trace(seed)
    # add 5 m GPS-like noise to the 15 s samples, as real taxi data carry it
    rng=np.random.default_rng(seed+77)
    x=tr0['x']+rng.normal(0,5,tr0['n']); y=tr0['y']+rng.normal(0,5,tr0['n'])
    route=route_from_xy(x,y)
    tr=world.driving_trace(seed,route=route)
    return {f'{n}|{e}':run_cfg(tr,n,e) for n,e in CFG}

def err_at(res,name,P,idx):
    ks=[f'{name}|{e}' for e in EPS]
    Pm=np.array([np.mean([res[i][k]['power_mW'] for i in idx]) for k in ks]); Em=np.array([np.mean([res[i][k]['mean'] for i in idx]) for k in ks])
    o=np.argsort(Pm); Pm,Em=Pm[o],Em[o]
    if P<Pm.min() or P>Pm.max(): return np.nan
    return float(np.exp(np.interp(np.log(P),np.log(Pm),np.log(Em))))

if __name__=='__main__':
    with mp.get_context('fork').Pool(2) as p: res=p.map(job,range(N))
    pickle.dump(res,open('diag_resample.pkl','wb'))
    rng=np.random.default_rng(0); out={}
    for P in (60,80,100):
        row={}
        for nm in ('D3-DR','D3-IMU','CATS-noIMU','CATS'): row[nm]=err_at(res,nm,P,range(N))
        imp={}
        for base in ('D3-DR','D3-IMU'):
            bs=[]
            for b in range(300):
                idx=rng.integers(0,N,N); c=err_at(res,'CATS',P,idx); o=err_at(res,base,P,idx)
                if not(np.isnan(c) or np.isnan(o)): bs.append(100*(1-c/o))
            imp[base]=(100*(1-row['CATS']/row[base]),np.percentile(bs,2.5),np.percentile(bs,97.5))
        out[P]=(row,imp)
        print(P,{k:round(v,1) for k,v in row.items()},{k:tuple(round(x,1) for x in v) for k,v in imp.items()})
