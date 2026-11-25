"""V5 panel probabilities. All record-level inputs and subprocess files stay private."""
from pathlib import Path
import argparse, ctypes, hashlib, json, os, shutil, subprocess, tempfile, time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from numpy.ctypeslib import ndpointer

ROOT=Path(__file__).resolve().parents[1]
PR=ROOT/'PRIVATE/PANEL'
RT=ROOT/'PRIVATE/PANEL_RUNTIME'
R='D:/Program Files/R/R-4.2.3/bin/Rscript.exe'
CXX='D:/Program Files/R/rtools42/x86_64-w64-mingw32.static.posix/bin/g++.exe'
ARMS={'H11_H01':('H11','H01'),'H10_H00':('H10','H00')}
SOURCES=['CHARLS','HRS','ELSA','SHARE']
NBOOT=300
ST={'A':1,'B':2,'D':3,'alive_unknown':99}
EDGES=('ab','ba','ad','bd')
DLL=None

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def native():
    global DLL
    if DLL is not None:
        return DLL
    code=Path(__file__).with_name('panel_likelihood.cpp')
    dll=RT/('panel_'+digest(code)[:12]+'.dll')
    if not dll.exists():
        with tempfile.TemporaryDirectory(prefix='ppc_v5_build_') as tmp:
            shutil.copyfile(code,Path(tmp)/'panel.cpp')
            env=dict(os.environ)
            env['PATH']=str(Path(CXX).parent)+os.pathsep+env['PATH']
            subprocess.run([CXX,'-O3','-std=c++17','-shared','-static-libgcc',
                            '-static-libstdc++','panel.cpp','-o','panel.dll'],cwd=tmp,env=env,check=True)
            stage=RT/f'panel-{os.getpid()}.staging'
            shutil.copyfile(Path(tmp)/'panel.dll',stage)
            try:
                stage.replace(dll)
            except PermissionError:
                if not dll.exists():
                    raise
    DLL=ctypes.CDLL(str(dll))
    d=ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    i=ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    DLL.ppc_panel_lik.argtypes=[ctypes.c_int,ctypes.c_int,d,i,d,i,d,d,d,d]
    DLL.ppc_panel_lik.restype=ctypes.c_int
    DLL.ppc_transition.argtypes=[ctypes.c_double,d,d,d]
    DLL.ppc_transition.restype=None
    return DLL

def read(stem):
    a=PR/f'{stem}.parquet'
    return pd.read_parquet(a) if a.exists() else pd.read_csv(PR/f'{stem}.csv',dtype={'person':str,'cluster':str,'wave':str})

def qmatrix(v):
    a,b,d,e=v
    return np.array([[-a-d,a,d],[b,-b-e,e],[0,0,0.]])

def probabilities(q,t):
    a,b,d,e=q.T
    u=a+d; v=b+e
    s=(u+v)*.5; x=(v-u)*.5
    z=np.sqrt(x*x+a*b)
    low=s+z
    high=(a*e+d*b+d*e)/low
    eh=np.exp(-high*t); el=np.exp(-low*t)
    f=eh*(-np.expm1(-2*z*t))/(2*z)
    A=(eh+el)*.5+f*x; B=f*a
    return np.column_stack((A,B,1-A-B))

def occupancy(q,t):
    # integrate exp of transient rate block via eigenspaces; numerically stable at small rates.
    a,b,d,e=q.T
    u=a+d;v=b+e;s=(u+v)*.5;x=(v-u)*.5
    z=np.sqrt(x*x+a*b); l=s+z; h=(a*e+d*b+d*e)/l
    ih=-np.expm1(-h*t)/h; il=-np.expm1(-l*t)/l
    m=(ih-il)/(2*z)
    A=(ih+il)*.5+m*x; B=m*a
    return np.column_stack((A,B,t-A-B))*12

def coarsen(val):
    return '<10' if val<10 else int(10*np.floor((val+5)/10))

def count_rows(rows, cols):
    a=pd.DataFrame(rows)
    for c in cols:
        if c in a:
            a[c]=a[c].map(lambda n: coarsen(n) if pd.notna(n) else '')
    return a

def output(name, source, comp, a):
    a.to_csv(ROOT/'TABLES'/f'PANEL_{name}_{source}_{comp}.csv',index=False,encoding='utf-8-sig')

def source_data(src):
    b=read(src+'_baseline'); o=read(src+'_observations')
    for df in [b,o]:
        df['person']=df.person.astype(str)
        df['wave']=df.wave.astype(str)
    b['country']=b.country.astype(str)
    b['cluster']=b.cluster.astype(str)
    o['time_years']=pd.to_numeric(o.time_years,errors='coerce')
    o['occasion']=pd.to_numeric(o.occasion,errors='coerce')
    assert b.person.is_unique, 'Earliest eligible baseline duplicated'
    assert b.age1.ge(45).all() and b.p0.isin([0,1,2,4]).all(), 'Index PPC/age malformed'
    assert b.history.isin(sum(ARMS.values(),())).all(), 'Index history malformed'
    assert o.person.isin(b.person).all(), 'Post-index record outside target'
    return b,o

def path_counts(b,o):
    rows=[]
    eligible=b.set_index('person')
    oo=o[o.occasion.isin([1,2])].copy()
    def slot(x):
        s=set(x)
        wanted=s & {'A','B','D'}
        return next(iter(wanted)) if len(wanted)==1 else ('no_opportunity' if s=={'no_opportunity'} else 'unknown')
    p=oo.groupby(['person','occasion'],sort=False).status.agg(slot).unstack('occasion')
    p=p.reindex(index=eligible.index,columns=[1,2]).fillna('no_opportunity')
    s=pd.Series('unclassified',index=p.index)
    ab=p[1].isin(['A','B'])&p[2].isin(['A','B'])
    s.loc[p[2].eq('no_opportunity')]='administrative_no_second'
    s.loc[ab]=p.loc[ab,1]+p.loc[ab,2]
    s.loc[p.eq('D').any(axis=1)]='death_by_either'
    exact=o[o.status.isin(['A','B']) & o.time_years.gt(0) & o.time_years.le(6) & o.occasion.gt(0)]
    bar=exact.sort_values(['person','time_years']).groupby('person',sort=False).status.agg(''.join).str.contains(r'B+A+B+')
    for h,g in eligible.groupby('history',sort=False):
        scan=s.reindex(g.index)
        for name in ['AA','AB','BA','BB','death_by_either','administrative_no_second','unclassified']:
            rows.append(dict(history=h,path=name,n=int(scan.eq(name).sum()),reference_n=len(g)))
        rows.append(dict(history=h,path='observed_B_A_B_any_three',n=int(bar.reindex(g.index,fill_value=False).sum()),reference_n=len(g)))
    return count_rows(rows,['n','reference_n'])

def fit_reference(b,src,comp,known=False):
    h1,h0=ARMS[comp]
    keys=['age1','sex1','education1','chronic_count1','cesd1','cog_z1','p0','wave','K1']
    a=b[b.history.isin([h1,h0])].dropna(subset=keys).copy()
    if known:
        a=a[a.prior_ppc.isin([0,1])].copy()
    a['kgrp']=a.K1.clip(upper=3)
    context=['wave','p0']
    if comp=='H11_H01': context.append('kgrp')
    filter0=a.groupby(context,sort=False).history.transform('nunique').eq(2)
    if src=='SHARE':
        filter0 &= a.groupby('country',sort=False).history.transform('nunique').eq(2)
    a=a.loc[filter0].reset_index(drop=True)
    a['h']=(a.history==h1).astype(float)
    return a

def design(a,src,comp,prior=False,compact=False,none=False):
    xs={}; gt={}
    def add(k,v,g):
        v=np.asarray(v,float)
        if len(v) and np.ptp(v)>0:
            xs[k]=v;gt[k]=g
    def factor(k,v,g):
        q=pd.get_dummies(v.astype(str),drop_first=True,dtype=float)
        for i,col in enumerate(q):
            add(f'{k}{i+1}',q[col],g)
    clock=[1]*4; live=[1,1,2,2]
    h=[1,2,3,3]
    if not none:
        add('hx',a.h,h)
        g=clock if compact else live
        add('age',(a.age1-65)/10,g)
        add('count',(a.chronic_count1-1).clip(lower=0),g)
        # Rescaling preserves the inherited continuous severity contrasts and improves conditioning.
        add('symptom',(a.cesd1-a.cesd1.mean())/max(float(a.cesd1.std()),1.),g)
        add('cognition',a.cog_z1-a.cog_z1.mean(),g)
        factor('pzero',a.p0,g)
        factor('sex',a.sex1,clock)
        if src=='HRS': add('edu',(a.education1-a.education1.mean())/4,clock)
        else: factor('edu',a.education1,clock)
        factor('idx',a.wave,clock)
        if src=='SHARE': factor('nation',a.country,clock)
        if comp=='H11_H01': factor('tasks',a.kgrp,clock)
        if prior: add('prior',a.prior_ppc,clock)
    p=4+sum(len(set(g)) for g in gt.values())
    X=np.zeros((len(a),4,p))
    X[:,range(4),range(4)]=1.
    rows=[{'name':'intercepts',**{e:0 for e in EDGES},**{'g_'+e:0 for e in EDGES}}]
    at=4
    for name,x in xs.items():
        z=np.array(gt[name])
        groups=sorted(set(z))
        for g in groups:
            X[:,z==g,at]=x[:,None]
            at+=1
        rows.append({'name':name,**{e:0 for e in EDGES},**{f'g_{e}':z[j] for j,e in enumerate(EDGES)}})
    return X,pd.DataFrame(xs,index=range(len(a))),pd.DataFrame(rows)

def sequences(a,raw):
    """Observed states or genuine known-living union; no synthetic visits."""
    idmap=pd.Series(np.arange(len(a)),index=a.person)
    r=raw[raw.person.isin(idmap.index) & raw.time_years.gt(0) & raw.time_years.le(6) & raw.status.isin(ST)].copy()
    r['pid']=r.person.map(idmap).astype(int)
    # Conflict handling retains observed living data, omitting incompatible death date.
    live=r.loc[r.status.ne('D')].groupby('pid').time_years.max()
    dtime=r.loc[r.status.eq('D')].groupby('pid').time_years.min()
    clash=live.index.intersection(dtime.index)
    invalid=clash[live.loc[clash].to_numpy()>=dtime.loc[clash].to_numpy()]
    r=r.loc[~(r.pid.isin(invalid)&r.status.eq('D'))]
    dup=r.groupby(['pid','time_years'],sort=False).status.transform('nunique').gt(1)
    r=r.loc[~dup].sort_values(['pid','time_years'],kind='stable').drop_duplicates(['pid','time_years'])
    next_d=r.loc[r.status.eq('D')].groupby('pid').time_years.min()
    da=r.pid.map(next_d)
    r=r.loc[da.isna()|r.time_years.le(da)].copy()
    r['state']=r.status.map(ST).astype(int)
    r['dt']=r.time_years-r.groupby('pid',sort=False).time_years.shift().fillna(0)
    r['origin_exact']=r.groupby('pid',sort=False).state.shift().fillna(1).astype(int)
    r['occ_prev']=r.groupby('pid',sort=False).occasion.shift().fillna(0.)
    r['h']=a.h.to_numpy()[r.pid]
    r['idx']=a.wave.to_numpy()[r.pid]
    totals=np.bincount(r.pid,minlength=len(a)).astype(np.int32)
    offsets=np.r_[0,np.cumsum(totals)].astype(np.int32)
    assert r.dt.gt(0).all(), 'Panel time must strictly increase'
    return r,offsets,len(invalid)

def likelihood(X,r,offset):
    lib=native()
    dt=r.dt.to_numpy(dtype=np.float64)
    states=r.state.to_numpy(dtype=np.int32)
    n,_,p=X.shape
    w0=np.ones(n)
    def fn(x,w=None,intervals=None):
        beta=np.ascontiguousarray(x,dtype=np.float64)
        g=np.zeros(p); z=np.zeros(1)
        status=lib.ppc_panel_lik(n,p,X,offset,dt if intervals is None else intervals,
                                states,w0 if w is None else w,beta,g,z)
        if status: return np.inf,np.zeros(p)
        return z[0],g
    return fn

def hessian(fn,b):
    eye=np.eye(len(b))*1e-4
    h=np.column_stack([(fn(b+v)[1]-fn(b-v)[1])/2e-4 for v in eye])
    return (h+h.T)/2

def optimize(fn,start,inv=None,w=None,full=True,method='BFGS'):
    # Direct maximum likelihood; all start conditions and caps are deterministic.
    opts={'gtol':1e-5,'maxiter':300 if full else 80}
    if inv is not None and method=='BFGS':
        opts['hess_inv0']=inv
    if method=='L-BFGS-B': opts['ftol']=1e-13
    try:
        fit=minimize(lambda b: fn(b,w),start,jac=True,method=method,options=opts)
    except (ValueError,FloatingPointError):
        return None
    return fit

def valid(fit):
    return fit is not None and np.isfinite(fit.fun) and np.isfinite(fit.x).all() and np.max(np.abs(fit.jac))<.02

def fit_pair(fn,b0):
    b1=b0.copy();b1[:4]+=np.log([.65,1.5,1.35,.75])
    fits=[optimize(fn,b,method='BFGS') for b in [b0,b1]]
    keep=[x for x in fits if valid(x)]
    if len(keep)!=2:
        return None,{},'two_prespecified_starts_did_not_converge'
    pick=min(keep,key=lambda z:z.fun)
    gap=abs(keep[0].fun-keep[1].fun)
    dpar=float(np.max(np.abs(keep[0].x-keep[1].x)))
    H=hessian(fn,pick.x)
    eig,V=np.linalg.eigh(H)
    diag=dict(nll=float(pick.fun),starts_nll_gap=float(gap),starts_max_coefficient_difference=dpar,
              min_hessian_eig=float(eig[0]),hessian_condition=float(eig[-1]/max(eig[0],1e-50)),
              max_score=float(np.max(np.abs(pick.jac))),n_params=len(pick.x))
    if gap>.002 or dpar>.12:
        return None,diag,'local_optima_disagreement'
    if eig[0]<.002 or eig[-1]/eig[0]>2e7 or np.max(np.abs(pick.x[:4]))>13 or np.max(np.abs(pick.x[4:]),initial=0.)>12:
        return None,diag,'poor_curvature_or_boundary'
    inv=(V/eig)@V.T
    return (pick.x,(inv+inv.T)/2),diag,'estimated'

def official(r,X,cols,plan,b,job,mode='fixed'):
    job.mkdir(parents=True,exist_ok=True)
    levels=cols.iloc[r.pid].reset_index(drop=True)
    d=pd.concat([r[['pid','time_years','state']].reset_index(drop=True).rename(columns={'time_years':'elapsed'}),levels],axis=1)
    has=np.array(sorted(r.pid.unique()))
    bk=pd.concat([pd.DataFrame(dict(pid=has,elapsed=0,state=1)),cols.iloc[has].reset_index(drop=True)],axis=1)
    pd.concat([bk,d]).sort_values(['pid','elapsed']).to_csv(job/'panel.csv',index=False)
    fp=plan.copy()
    fp[list(EDGES)]=fp[list(EDGES)].astype(float)
    fp.loc[0,list(EDGES)]=b[:4]
    for i,name in enumerate(cols,1):
        masks=cols[name].to_numpy()!=0.
        if not masks.any(): return {'official_status':'empty_covariate'}
        k=np.flatnonzero(masks)[0]
        # Compute just that covariate: all its columns are adjacent in the design.
        idx=4+sum(len(set(plan.loc[j,['g_'+e for e in EDGES]])) for j in range(1,i))
        z=np.array(plan.loc[i,['g_'+e for e in EDGES]],int)
        vals=b[idx+z-1]
        fp.loc[i,list(EDGES)]=vals
    fp.to_csv(job/'parameters.csv',index=False)
    ix=[int(has[0]),int(has[len(has)//2])]
    pd.concat([pd.Series([2,4],name='elapsed'),cols.iloc[ix].reset_index(drop=True)],axis=1).to_csv(job/'prediction.csv',index=False)
    args=[R,'--vanilla',str(Path(__file__).with_name('panel_verify_msm.R')),str(ROOT),str(job),mode]
    try:
        proc=subprocess.run(args,cwd=ROOT,capture_output=True,timeout=300)
        if proc.returncode!=0:
            return {'official_status':'official_subprocess_error'}
    except subprocess.TimeoutExpired:
        return {'official_status':'official_timeout'}
    ret=pd.read_csv(job/'official.csv').iloc[0].to_dict()
    dct={'official_status':ret['status'],'official_version':ret.get('version','')}
    if ret['status']!='official_ok':
        return dct
    dct.update(official_nll=ret['nll'],official_opt_convergence=ret['opt_convergence'])
    if mode=='fixed':
        observed=pd.read_csv(job/'official_prediction.csv',header=None).to_numpy()
        q=np.exp(X[ix]@b)
        want=np.vstack([np.r_[probabilities(q[j:j+1],float(t))[0],occupancy(q[j:j+1],4)[0]/12]
                        for j,t in enumerate([2,4])])
        dct['max_official_prediction_or_integral_absdiff']=float(np.max(np.abs(observed-want)))
    else:
        dct['unadjusted_rates']=[ret[f'rate_{e}'] for e in EDGES]
    return dct

def support_masks(a,r,src):
    masks={}
    rows=[]
    for T in (2,4):
        by=[]
        for wave in a.wave.unique():
            baseline=a.wave.eq(wave)&a.admin_horizon_years.ge(T)
            if src=='CHARLS' and T==4 and str(wave).strip().replace('.0','')=='2018':
                baseline &= False
            both=True
            wave_rows=[]
            for arm in (0,1):
                z=r.loc[r.pid.isin(a.index[baseline&a.h.eq(arm)])&r.state.isin([1,2])]
                hit=z.loc[z.time_years.between(T-1.5,T+1.5)]
                small=hit.pid.nunique()
                lo=bool(hit.time_years.le(T).any())
                hi=bool(hit.time_years.ge(T).any())
                both &= (small>=10 and lo and hi)
                wave_rows.append(dict(year=T,index_wave=wave,h=arm,eligible_n=int((baseline&a.h.eq(arm)).sum()),
                                 near_observed_living_n=small,has_lower_support=lo,has_upper_support=hi))
            for rr in wave_rows:
                rr['release']='shared_wave_support' if both else 'one_or_more_support_failures'
            rows.extend(wave_rows)
            if both: by.append(wave)
        masks[T]=(a.wave.isin(by)&a.admin_horizon_years.ge(T)).to_numpy()
    return masks,count_rows(rows,['eligible_n','near_observed_living_n'])

def bracket(r):
    rows=[]
    for h,z in r.groupby('h',sort=False):
        for p,q in [(1,2),(2,1),(1,3),(2,3),(99,3)]:
            ct=int((z.origin_exact.eq(p)&z.state.eq(q)).sum())
            rows.append(dict(h=int(h),start=p,end=q,observed_pairs=ct))
    return pd.DataFrame(rows,columns=['h','start','end','observed_pairs'])

def mortality_coverage(b,a,raw,r):
    d=raw.loc[raw.status.eq('D')].sort_values('time_years',na_position='last').drop_duplicates('person')
    d=d[d.time_years.isna()|d.time_years.between(0,6,inclusive='right')]
    if d.empty:
        return (0,0,np.nan),[np.nan]*2
    all_timed=int(d.time_years.notna().sum())
    source_pct=all_timed/len(d)
    pct=[]
    for h in [0,1]:
        m=d.person.isin(a.person[a.h.eq(h)])
        pct.append(int((r.h.eq(h)&r.state.eq(3)).sum())/int(m.sum()) if m.any() else np.nan)
    return (len(d),all_timed,source_pct),pct

def sufficient(br):
    def n(h,a,b):
        return br.loc[br.h.eq(h)&br.start.eq(a)&br.end.eq(b),'observed_pairs'].sum()
    return all(n(h,1,2)>=1 and n(h,2,1)>=1 and sum(n(h,s,3) for s in [1,2,99])>=2 for h in [0,1]) and sum(n(h,2,1) for h in [0,1])>=8

def estimates(b,X,cols,masks,w,months):
    info={}
    arm_i=list(cols.columns).index('hx')
    assert arm_i==0
    base=X@b
    hx=cols.hx.to_numpy()
    hh=np.array([b[4],b[5],b[6],b[6]])
    rates=[np.exp(base+(h-hx[:,None])*hh) for h in (0,1)]
    for T,mask in masks.items():
        if not np.any(mask) or w[mask].sum()<=0:
            continue
        for measure,g in [('probability',probabilities),('months',occupancy)]:
            if measure=='months' and (T!=4 or not months):
                continue
            rr=np.stack([np.average(g(q[mask],T),axis=0,weights=w[mask]) for q in rates])
            info[(T,measure)]=np.vstack([rr,rr[1]-rr[0]])
    return info

def cluster_boot(a,r,fn,fit,X,cols,masks,months,seed,job,mode):
    b,h_inv=fit
    rng=np.random.default_rng(seed)
    family=(a.country.astype(str)+' / '+a.cluster.astype(str)).factorize(sort=True)[0]
    group=pd.DataFrame(dict(f=family,c=a.country)).drop_duplicates()
    g=group.groupby('c').f.apply(np.array).tolist() if a.country.nunique()>1 else [group.f.to_numpy()]
    draws=[]; results=[]
    filename=job/'bootstrap_checkpoint.npz'
    if mode=='bootstrap' and filename.exists():
        # Resume by deterministic replicate index; no seed skipping by outcome.
        prev=np.load(filename,allow_pickle=False)
        draw_rows=prev['draw']
        if int(prev['seed'])!=seed:
            raise ValueError('Bootstrap checkpoint does not match locked seed')
        draws=draw_rows.tolist()
    for i in range(NBOOT):
        cl=np.concatenate([rng.choice(c,size=len(c),replace=True) for c in g])
        w=np.bincount(cl,minlength=int(np.max(family))+1)[family].astype(float)
        if i<len(draws): continue
        bb=optimize(fn,b,h_inv,w,full=False)
        if not valid(bb):
            bb=optimize(fn,b,None,w,full=False,method='L-BFGS-B')
        ok=valid(bb) and np.max(np.abs(bb.x[:4]))<13 and np.max(np.abs(bb.x[4:]),initial=0)<12
        draws.append(bb.x.tolist() if ok else [np.nan]*len(b))
        if (i+1)%10==0 or i+1==NBOOT:
            np.savez_compressed(filename,draw=np.asarray(draws),seed=seed)
    # Regenerate the same family weights to propagate estimation and target variability.
    rng=np.random.default_rng(seed)
    for b0 in np.asarray(draws):
        cl=np.concatenate([rng.choice(c,size=len(c),replace=True) for c in g])
        w=np.bincount(cl,minlength=int(np.max(family))+1)[family].astype(float)
        results.append(estimates(b0,X,cols,masks,w,months) if np.isfinite(b0).all() else {})
    return results

def model_calibration(a,r,X,b):
    sq=np.exp(X@b)
    rows=[]; out=[]
    for pid,g in r.groupby('pid',sort=False):
        pp=np.array([1.,0.,0.])
        for item in g.itertuples(index=False):
            lr=sq[pid]
            u=probabilities(lr[None,:],item.dt)[0]
            v=probabilities(lr[[1,0,3,2]][None,:],item.dt)[0][[1,0,2]]
            P=np.vstack([u,v,[0,0,1]])
            pred=pp@P
            if item.state in [1,2,3]:
                out.append(dict(pid=pid,h=int(item.h),index_wave=item.idx,starting_PPC=a.p0.iat[pid],
                                gap_band=int(min(3,item.dt//1.5)),state=item.state,
                                pred_A=pred[0],pred_B=pred[1],pred_D=pred[2]))
                pp=np.eye(3)[item.state-1]
            else:
                pp=np.r_[pred[:2]/pred[:2].sum(),0.]
    df=pd.DataFrame(out)
    worst=0.
    for keys,z in df.groupby(['h','index_wave','starting_PPC','gap_band'],sort=False):
        for s,name in [(1,'A'),(2,'B'),(3,'D')]:
            o=int(z.state.eq(s).sum())
            if len(z)>=100: worst=max(worst,abs(o/len(z)-z[f'pred_{name}'].mean()))
            rows.append(dict(h=keys[0],index_wave=keys[1],starting_PPC=int(keys[2]),gap_band=keys[3],state=name,
                             intervals=len(z),endpoints=o,
                             average_model_pct=round(100*z[f'pred_{name}'].mean(),2) if len(z)>=10 else np.nan))
    return count_rows(rows,['intervals','endpoints']),worst

def fallback(a,r,cols,src,comp,reason):
    """Discrete conditional next-known-living state; no replacement for status Q."""
    import statsmodels.api as sm
    rows=[]
    for s in (1,2):
        x=r[r.origin_exact.eq(s)&r.state.isin([1,2])&(r.occasion-r.occ_prev).eq(1.)].copy()
        if x.empty or x.groupby('h').state.nunique().reindex([0.,1.],fill_value=0).min()<2:
            rows.append(dict(origin='A' if s==1 else 'B',status='sparse_observed_pairs',reason=reason))
            continue
        main=cols.iloc[x.pid].reset_index(drop=True)
        main['gap']=((x.dt-x.dt.mean()).to_numpy())
        main=main.loc[:,main.nunique().gt(1)]
        M=sm.add_constant(main,has_constant='add').to_numpy(float)
        if 'hx' not in main or np.linalg.matrix_rank(M)<M.shape[1]:
            rows.append(dict(origin='A' if s==1 else 'B',status='rank_or_no_history',reason=reason))
            continue
        history_j=1+list(main).index('hx')
        y=x.state.eq(2).to_numpy(float)
        fg=(a.country.astype(str)+' / '+a.cluster.astype(str)).iloc[x.pid].to_numpy()
        try:
            fit=sm.GLM(y,M,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':fg},maxiter=200)
            b=np.array(fit.params)
            V=np.array(fit.cov_params())
            if not fit.converged or np.max(np.abs(b))>=15 or not np.isfinite(V).all():
                raise ValueError('separation or no convergence')
            lp=[M.copy(),M.copy()]
            for h in [0,1]: lp[h][:,history_j]=h
            from scipy.special import expit
            p=[expit(m@b) for m in lp]
            d=(p[1]-p[0]).mean()
            grad=np.mean(lp[1]*(p[1]*(1-p[1]))[:,None]-lp[0]*(p[0]*(1-p[0]))[:,None],axis=0)
            se=float(np.sqrt(grad@V@grad))
            if not np.isfinite(se): raise ValueError('covariance')
            rows.append(dict(origin='A' if s==1 else 'B',status='estimated_conditional_observed_alive',reason=reason,
                             intervals=len(y),people=x.pid.nunique(),reference_risk=p[0].mean(),index_risk=p[1].mean(),
                             rd=d,rd_lo=d-1.96*se,rd_hi=d+1.96*se))
        except (ValueError,np.linalg.LinAlgError):
            rows.append(dict(origin='A' if s==1 else 'B',status='failed_numerical_or_covariance',reason=reason))
    output('DISCRETE_FALLBACK',src,comp,count_rows(rows,['intervals','people']))

def infer(src,comp,b,raw,mode):
    t0=time.time()
    h1,h0=ARMS[comp]
    job=PR/f'{src}_{comp}_inference'
    job.mkdir(parents=True,exist_ok=True)
    a=fit_reference(b,src,comp)
    r,offset,conf=sequences(a,raw)
    cols0=['eligible_n','baseline_n','with_information_n','known_alive_partial_n','death_n','temporal_conflict_n']
    output('FLOW',src,comp,count_rows([dict(eligible_n=int(b.history.isin([h1,h0]).sum()),
          baseline_n=len(a),with_information_n=r.pid.nunique(),known_alive_partial_n=r.state.eq(99).sum(),
          death_n=r.state.eq(3).sum(),temporal_conflict_n=conf)],cols0))
    br=bracket(r)
    output('INFORMATION',src,comp,count_rows(br.to_dict('records'),['observed_pairs']))
    masks,support=support_masks(a,r,src)
    output('TIME_SUPPORT',src,comp,support)
    X,cols,plan=design(a,src,comp)
    diag={'mode':mode,'history_index':h1,'history_reference':h0,'contrast_unit':'person'}
    (death_n,timed_n,source_pct),pcts=mortality_coverage(b,a,raw,r)
    diag.update(sourced_death_n=coarsen(death_n),usable_dated_death_n=coarsen(timed_n),
        source_death_coverage=round(source_pct,3) if min(death_n,timed_n,death_n-timed_n)>=10 else ('complete' if death_n==timed_n and death_n>=10 else 'withheld'),
        observed_death_coverage_group_min=round(float(min(pcts)),2) if min(safe_per_h:=r.loc[r.state.eq(3)].groupby('h').pid.nunique().reindex([0,1],fill_value=0))>=10 else 'withheld')
    death_pass=source_pct>=.70 and all(x>=.65 for x in pcts)
    if not death_pass or not sufficient(br) or not len(r) or 'hx' not in cols:
        diag['status']='insufficient_death_ascertainment' if not death_pass else 'insufficient_direct_living_reversal_or_death_information'
        fallback(a,r,cols,src,comp,diag['status'])
        output('DIAGNOSTICS',src,comp,pd.DataFrame([diag]))
        return
    start=np.log([.1,.23,.018,.035])
    X0,cols0,plan0=design(a,src,comp,none=True)
    f0=likelihood(X0,r,offset)
    of=official(r,X0,cols0,plan0,start,job/'r_unadjusted','unadjusted')
    qa0=optimize(f0,start)
    diag.update({k if k=='official_version' else 'unadjusted_'+k:v for k,v in of.items() if k!='unadjusted_rates'})
    if of.get('official_status')=='official_ok' and valid(qa0):
        gap=abs(f0(np.log(of['unadjusted_rates']))[0]-of['official_nll'])
        diag['unadjusted_official_likelihood_absdiff']=gap
        if gap>1e-5:
            diag['status']='independent_panel_likelihood_not_reconciled'
            fallback(a,r,cols,src,comp,diag['status'])
            output('DIAGNOSTICS',src,comp,pd.DataFrame([diag]))
            return
        start=qa0.x
    elif not valid(qa0):
        diag['status']='unadjusted_panel_likelihood_failed'
        fallback(a,r,cols,src,comp,diag['status'])
        output('DIAGNOSTICS',src,comp,pd.DataFrame([diag]))
        return
    ans=None
    for compact in [False,True]:
        x,c,p=design(a,src,comp,compact=compact)
        fn=likelihood(x,r,offset)
        ans,dg,status=fit_pair(fn,np.r_[start,np.zeros(x.shape[2]-4)])
        diag.update(dg)
        if ans is not None: break
        diag['compact_trigger']=status
    diag['status']=status
    diag['compact_nuisance']=compact
    if ans is None:
        fallback(a,r,cols,src,comp,status)
        output('DIAGNOSTICS',src,comp,pd.DataFrame([diag]))
        return
    beta=ans[0]
    official_fit=official(r,x,c,p,beta,job/'r_adjusted','fixed')
    diag.update(official_fit)
    diag['adjusted_official_likelihood_absdiff']=abs(official_fit.get('official_nll',np.inf)-fn(beta)[0])
    if (diag['adjusted_official_likelihood_absdiff']>1e-5 or
        official_fit.get('max_official_prediction_or_integral_absdiff',np.inf)>1e-7):
        diag['status']='adjusted_official_verification_failed'
        fallback(a,r,cols,src,comp,diag['status'])
        output('DIAGNOSTICS',src,comp,pd.DataFrame([diag]))
        return
    checks,worst=model_calibration(a,r,x,beta)
    months=bool(masks[4].any() and r.state.eq(3).sum()>=20 and min(safe_per_h)>=5
                and source_pct>=.85 and min(pcts)>=.80 and abs(pcts[1]-pcts[0])<=.15 and worst<=.15)
    diag['max_living_or_death_calibration_gap_large_cell']=float(worst)
    est=estimates(beta,x,c,masks,np.ones(len(a)),months)
    diag['occupancy_supported']=months
    diag['finite_abs_probabilities']=bool(est)
    diag['timing_status']='target_available' if est else 'no_2_or_4_year_target_meets_support'
    np.savez_compressed(job/'point.npz',beta=beta,hinv=ans[1],compact=compact)
    output('CALIBRATION',src,comp,checks)
    # Publish all point estimates before the bootstrap; leave blank interval fields until valid.
    boot=[]
    for stage in (['point','bootstrap'] if mode=='bootstrap' else ['point']):
        if stage=='bootstrap':
            seed=12620+10*SOURCES.index(src)+list(ARMS).index(comp)
            boot=cluster_boot(a,r,fn,ans,x,c,masks,months,seed,job,stage)
        output_rows=[]
        for (T,measure),val in est.items():
            fellow=np.array([u[(T,measure)] for u in boot if (T,measure) in u])
            use=len(fellow)>=int(np.ceil(.85*NBOOT))
            lo,hi=(np.quantile(fellow,[.025,.975],axis=0) if use else (np.full((3,3),np.nan),)*2)
            for j,role in enumerate(['reference','index','index_minus_reference']):
                for k,st in enumerate(['A','B','D']):
                    output_rows.append(dict(year=T,measure=measure,role=role,state=st,estimate=val[j,k],
                         lo=lo[j,k],hi=hi[j,k],target_people=int(masks[T].sum()),bootstrap_attempted=len(boot),
                         bootstrap_success=len(fellow),inference='family_percentile' if use else 'interval_not_yet_valid'))
        output('STATE_ESTIMATES',src,comp,count_rows(output_rows,['target_people']))
        diag['bootstrap_attempted']=len(boot)
        diag['bootstrap_success']=sum(bool(z) for z in boot)
        diag['wall_seconds']=round(time.time()-t0,2)
        output('DIAGNOSTICS',src,comp,pd.DataFrame([diag]))
        print(f'{src} {comp}: {stage}; compatible horizons={len(est)}, family draws successful={diag["bootstrap_success"]}',flush=True)

def gate():
    k=json.loads((ROOT/'QA/PANEL_EXECUTION_GATE.json').read_text(encoding='utf-8'))
    if not k.get('share_original_primary_completed'):
        raise ValueError('SHARE original analysis gate is not open')
    if digest(ROOT/'PANEL_STATE_SPEC.md')!=k['panel_state_spec_sha256']:
        raise ValueError('Panel specification hash no longer matches')
    for name,value in k['panel_inputs_sha256'].items():
        if digest(PR/name)!=value:
            raise ValueError('A panel input differs from the locked pre-effect file')
    return True

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--source',choices=SOURCES,required=True)
    a.add_argument('--comparison',choices=ARMS)
    a.add_argument('--mode',choices=['audit','point','bootstrap'],default='audit')
    u=a.parse_args()
    b,obs=source_data(u.source)
    if u.mode!='audit': gate()
    counts=path_counts(b,obs)
    output('PLANNED_PATHS',u.source,'all',counts)
    if u.mode=='audit':
        for comp in ARMS:
            a0=fit_reference(b,u.source,comp)
            r,offset,conf=sequences(a0,obs)
            masks,support=support_masks(a0,r,u.source)
            output('PREFIT_TIME',u.source,comp,support)
            br=bracket(r)
            (death_n, timed_n, source_pct), flags=mortality_coverage(b,a0,obs,r)
            print(f'{u.source} {comp}; panel input n={len(a0)}, harmonized times={[k for k,v in masks.items() if v.any()]}; '
                  f'source dated-death coverage={source_pct:.3f}, both-arm coverage≥65%={all(a>=.65 for a in flags)}, '
                  f'direct-reversal/death structure passes={sufficient(br)}',flush=True)
        return
    for comp in [u.comparison] if u.comparison else ARMS:
        infer(u.source,comp,b,obs,u.mode)

if __name__=='__main__':
    main()
