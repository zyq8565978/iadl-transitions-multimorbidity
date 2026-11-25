"""Inherited one-stage outcome models; MI-Boot and final-functional Rubin pooling."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time,argparse,shutil
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
R=Path(__file__).resolve().parents[1];W=R.parent
sys.path.insert(0,str(W/'PPC_IADL_ZHOU_COVARIATE_QC_20261117/CODE'))
import run_qc as qc
old=qc.old
P=R/'PRIVATE';Q=R/'QA';T=R/'TABLES'

def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def input_hashes(mi):
    answer={}
    for source in old.SOURCES:
        p=P/f'{source}_mi_{mi:03d}.csv'
        with p.open('rb') as f:answer[p.name]=hashlib.file_digest(f,'sha256').hexdigest()
    return answer

def overlay(mi,scenario):
    frames=[]
    for s in old.SOURCES:
        n=pd.read_parquet(P/f'{s}_node_binding.parquet')
        v=pd.read_csv(P/f'{s}_mi_{mi:03d}.csv')
        assert len(n)==len(v)
        for name in ('smoking','alcohol','bmi'):
            observed=n[name].notna().to_numpy()
            np.testing.assert_allclose(v[name][observed],n[name][observed],atol=1e-10,rtol=1e-12)
        assert v.notna().all().all() and v.smoking.isin([0,1]).all() and v.alcohol.isin([0,1]).all()
        if scenario=='structural_BMI_minus2':
            mask=n.bmi.isna()&n.bmi_reason.eq('structurally_unavailable')
            v.loc[mask,'bmi']=(v.loc[mask,'bmi']-2).clip(lower=10)
        v['key']=n.key;frames.append(v)
    v=pd.concat(frames,ignore_index=True).set_index('key',verify_integrity=True)
    return v

def design(comp,mi,scenario):
    d=pd.read_parquet(P/f'{comp}_rows.parquet')
    f=np.load(P/f'{comp}_frozen_design.npz');x=f['x'];y=f['y'];src=f['src']
    a=overlay(mi,scenario).loc[d.key].reset_index(drop=True)
    a=a.rename(columns={v:v+'_t' for v in ('smoking','alcohol','bmi')})
    a['bmi_cat_t']=pd.cut(a.bmi_t,[-np.inf,18.5,25,30,np.inf],right=False,labels=False)
    extra,_=qc.extras(a,src,'t');xx=np.c_[x,extra]
    cols=qc.independent(xx);xx=xx[:,cols]
    assert np.isfinite(xx).all() and np.array_equal(d.y,y)
    return d,y,xx,src,cols

def run(mi,scenario,comp):
    stamp=f'{comp}_{scenario}_mi{mi:03d}'
    result=P/f'{stamp}_functional.npz'
    gate=json.loads((Q/'MI_DIAGNOSTIC_GATE.json').read_text());assert gate['allow_effect_analysis']
    assert hashlib.sha256((R/'ANALYSIS_LOCK.md').read_bytes()).hexdigest()==(R/'ANALYSIS_LOCK.sha256').read_text().split()[0]
    for name,digest in json.loads((Q/'ANALYSIS_CODE_HASHES.json').read_text()).items():
        assert hashlib.sha256((R/'CODE'/name).read_bytes()).hexdigest()==digest,('analysis_code_changed',name)
    fingerprints=input_hashes(mi)
    input_id=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
    sidecar=Q/f'{stamp}_complete.json'
    if result.exists():
        previous=json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if previous.get('imputed_input_sha256')==fingerprints:return 'CACHED '+stamp
        # A prespecified chain extension may update an already fitted MI set.
        # Preserve the preceding result, then refit the identical model; never
        # pair old coefficients or bootstrap draws with new completed covariates.
        prior_id=hashlib.sha256(result.read_bytes()).hexdigest()[:16]
        archive=P/'STAGE_HISTORY'/f'{stamp}_{prior_id}';archive.mkdir(parents=True,exist_ok=True)
        shutil.copy2(result,archive/result.name)
        if sidecar.exists():shutil.copy2(sidecar,archive/sidecar.name)
    tick=time.time();d,y,x,src,cols=design(comp,mi,scenario)
    if comp=='H11_vs_H01':
        beta,qa=qc.fit(x,y)
        center=old.flic_stat(x,y,src,beta,np.ones(len(y)))
        write(Q/f'{stamp}_point.json',{'center':center.tolist(),'diagnostics':qa,'n':len(d),'elapsed':time.time()-tick})
        plan,inv=qc.family_plan(d.family.to_numpy(str),d.stratum.to_numpy(str),61118501,499)
        cache=P/f'{stamp}_{input_id[:16]}_bootstrap.npz';draws=np.full((499,4),np.nan);start=0;fail=Counter()
        if cache.exists():
            z=np.load(cache);draws=z['draws'];start=int(z['done'])
            fp=Q/f'{stamp}_{input_id[:16]}_bootstrap.json'
            if fp.exists():fail.update(json.loads(fp.read_text()).get('failures',{}))
        for b in range(start,499):
            try:draws[b]=qc.boot_fit(x,y,src,plan[b,inv].astype(float),beta)
            except (ValueError,AssertionError,np.linalg.LinAlgError) as e:fail[str(e)]+=1
            if (b+1)%50==0 or b==498:
                np.savez_compressed(cache,draws=draws,done=b+1)
                write(Q/f'{stamp}_{input_id[:16]}_bootstrap.json',{'done':b+1,'failures':dict(fail),'elapsed':time.time()-tick})
        good=draws[np.isfinite(draws).all(1)];assert len(good)>=450,(stamp,'insufficient_valid_bootstrap',len(good))
        U=np.cov(good,rowvar=False,ddof=1)
        valid=len(good);method='Firth_source_FLIC_MI_Boot_499_family'
    else:
        center,inf,qa,beta=qc.mle_influence(d,y,x);U=inf.T@inf;valid=0;fail={};method='MLE_family_HC1'
    assert np.isfinite(center).all() and np.isfinite(U).all() and np.diag(U).min()>0
    np.testing.assert_allclose(center[1]-center[2],center[3],atol=1e-12)
    np.savez_compressed(result,center=center,U=U,beta=beta,columns=cols)
    write(Q/f'{stamp}_complete.json',{'comparison':comp,'scenario':scenario,'imputation':mi,'method':method,
          'n':len(d),'persons':int(d.person.nunique()),'families':int(d.family.nunique()),'events':int(y.sum()),
          'valid_bootstrap':valid,'failures':dict(fail),'diagnostics':qa,'elapsed':time.time()-tick,
          'imputed_input_sha256':fingerprints})
    return f'COMPLETE {stamp} {time.time()-tick:.1f}s'

def pool_values(q,u,dfcom):
    M=len(q);mean=q.mean(0);Wc=u.mean(0);Bc=np.cov(q,rowvar=False,ddof=1);total=Wc+(1+1/M)*Bc
    lam=(1+1/M)*np.diag(Bc)/np.diag(total)
    olddf=np.where(lam>1e-12,(M-1)/np.maximum(lam,1e-12)**2,np.inf)
    obsdf=(dfcom+1)/(dfcom+3)*dfcom*(1-lam)
    df=1/(1/olddf+1/obsdf)
    se=np.sqrt(np.diag(total));crit=tdist.ppf(.975,df)
    r=(1+1/M)*np.diag(Bc)/np.diag(Wc)
    fmi=(r+2/(df+3))/(r+1)
    mc=np.sqrt(np.diag(Bc)/M)
    return mean,Wc,Bc,total,df,se,mean-crit*se,mean+crit*se,fmi,mc

def pool(M=50):
    rows=[];mirows=[]
    for comp in old.COMPS:
        d=pd.read_parquet(P/f'{comp}_rows.parquet');dfcom=d.family.nunique()-1
        for scenario in ('primary','structural_BMI_minus2'):
            qq=[];uu=[]
            for mi in range(1,M+1):
                file=P/f'{comp}_{scenario}_mi{mi:03d}_functional.npz'
                if not file.exists():break
                audit=json.loads((Q/f'{comp}_{scenario}_mi{mi:03d}_complete.json').read_text())
                assert audit['imputed_input_sha256']==input_hashes(mi),'stale_completed_data_model'
                z=np.load(file);qq.append(z['center']);uu.append(z['U'])
                mirows.append({'comparison':comp,'scenario':scenario,'imputation':mi,
                    **{k:float(z['center'][j]) for j,k in enumerate(old.KEYS)},
                    **{k+'_within_se':float(np.sqrt(z['U'][j,j])) for j,k in enumerate(old.KEYS)}})
            if len(qq)!=M:continue
            mean,w,b,tot,df,se,lo,hi,fmi,mc=pool_values(np.array(qq),np.array(uu),dfcom)
            row={'comparison':comp,'scenario':scenario,'MI_count':M,'people':d.person.nunique(),'windows':len(d),
                 'families':d.family.nunique(),'events':int(d.y.sum()),**old.columns(mean,lo,hi,se)}
            for j,k in enumerate(old.KEYS):row.update({k+'_df':df[j],k+'_FMI':fmi[j],k+'_MCSE':mc[j],k+'_MCSE_over_SE':mc[j]/se[j]})
            row['p_two_sided']=2*tdist.sf(abs(mean[0]/se[0]),df[0]);rows.append(row)
            np.savez_compressed(Q/f'{comp}_{scenario}_pooled_covariance.npz',within=w,between=b,total=tot)
    pd.DataFrame(rows).to_csv(T/'MI_POOLED_RESULTS.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mirows).to_csv(T/'MI_FUNCTIONALS_ALL.csv',index=False,encoding='utf-8-sig')
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mi',type=int);ap.add_argument('--scenario',default='primary');ap.add_argument('--comparison',default='H11_vs_H01');ap.add_argument('--pool',action='store_true');aa=ap.parse_args()
    if aa.pool:print(json.dumps(pool(),indent=2,default=str))
    else:print(run(aa.mi,aa.scenario,aa.comparison),flush=True)
