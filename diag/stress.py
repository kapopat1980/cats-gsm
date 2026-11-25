import numpy as np, pickle, time, multiprocessing as mp, json
import world, algos as A
SC={'base':{},'thief0.25':{'THIEF_F':0.25},'thief0.5':{'THIEF_F':0.5},'shadow3':{'SIG_SP':3.0,'SIG_TM':1.5},'shadow8':{'SIG_SP':8.0,'SIG_TM':4.0},'TAnoise0.7':{'TA_NOISE':0.7},'hyst2':{'HYST':2.0},'hyst6':{'HYST':6.0}}
DEF=dict(SIG_SP=5.0,SIG_TM=2.5,HYST=4.0,TA_NOISE=0.35,THIEF_F=1.0)
S=('A1','A2','A3','CATS')
def setp(name):
    for k,v in DEF.items(): setattr(world,k,v)
    for k,v in SC[name].items(): setattr(world,k,v)
def aj(a):
    name,seed,atk=a; setp(name); tr=world.armed_trace(seed,900,1200,atk); out={}
    for sc in S:
        d=A.armed_detect(tr,sc); th=[x for x in d['alarms'] if x[0]>=tr['t0']]
        out[sc]=(th[0][0]-tr['t0']) if th else None
    return out
def fj(a):
    name,seed=a; setp(name); tr=world.armed_trace(seed,86400,0,'none',theft=False)
    return {sc:len(A.armed_detect(tr,sc,rearm=True)['alarms']) for sc in S}
if __name__=='__main__':
    t=time.time(); res={}
    with mp.get_context('fork').Pool(2) as p:
        for name in SC:
            atks=('none','jam','replay','drift') if name!='thief0.5' or True else ()
            jobs=[(name,s,atk) for atk in atks for s in range(2000,2060)]
            r=p.map(aj,jobs,chunksize=5)
            res[name]={atk:r[i*60:(i+1)*60] for i,atk in enumerate(atks)}
            res[name]['fa']=p.map(fj,[(name,s) for s in range(9100,9112)])
            print(name,time.time()-t,flush=True)
            pickle.dump(res,open('stress.pkl','wb'))
