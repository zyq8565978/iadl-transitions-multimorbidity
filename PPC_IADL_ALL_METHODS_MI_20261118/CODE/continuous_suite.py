"""Twelve frozen longitudinal models, MI baseline adjustment and functional pooling."""
from common import *
import argparse
from scipy.linalg import qr
co=load('allmi_continuous_models',S/'CONTINUOUS/run_continuous.py')

def independent(x,protected):
    u=np.linalg.qr(x[:,:protected],mode='reduced')[0]
    rest=x[:,protected:]-u@(u.T@x[:,protected:]);_,rr,piv=qr(rest,mode='economic',pivoting=True)
    tol=max(rest.shape)*np.finfo(float).eps*max(1,abs(rr).max());rank=(abs(rr.diagonal())>tol).sum()
    cols=np.r_[np.arange(protected),protected+np.sort(piv[:rank])]
    mapping=np.linalg.lstsq(x[:,cols],x,rcond=None)[0]
    assert np.max(abs(x[:,cols]@mapping-x))<1e-7
    return cols,mapping

def run(comp,domain,include):
    effect_gate();raw=dict(co.inputs());bounded=domain=='physical';ri=not include and not bounded
    shared=['hx','hx:post','hx:post_time2'] if include else ['hx','hx:time2']
    ds={s:co.initialize(s,raw[s],comp,domain,include) for s in SOURCES}
    x,parts,terms=co.make_blocks(ds,shared)
    d=pd.concat([ds[s][0].assign(source=s) for s in SOURCES],ignore_index=True)
    families=d.source+'|'+d.cluster.astype(str);people=d.source+'|'+d.person.astype(str);y=d.outcome.to_numpy(float)
    stem=f'CONTINUOUS_{comp}_{domain}_{"index_included" if include else "post_index"}'
    refs=None
    for lo,hi in [(2,4),(2.5,3.5),(3,4),(2,3)]:
        refs=co.references(ds,raw,domain,lo,hi)
        if refs:break
    prediction=[];refd=None
    if refs:
        refd=pd.concat([refs[s].assign(source=s) for s in SOURCES],ignore_index=True)
        for year in ([0] if include else [])+list(np.arange(lo,hi+.01,.25)):
            for h in (1,0):prediction.append((year,h,co.prediction_matrix(ds,refs,parts,shared,year,h)))
    for mi in range(1,51):
        file=R/f'PRIVATE/{stem}_mi{mi:03d}.npz';jp=R/f'QA/{stem}_mi{mi:03d}.json'
        if file.exists():continue
        tick=time.time();xx=np.c_[x,extras(d,mi,'index_wave')[0]];cols,mapping=independent(xx,len(shared))
        try:b,V,G,qa=co.estimate(xx[:,cols],y,families,people,bounded,ri)
        except (ValueError,AssertionError,np.linalg.LinAlgError) as exc:
            write(jp,dict(status='NOT_ESTIMABLE',mi=mi,reason=str(exc)));continue
        values=list(b[:len(shared)]);grads=list(np.eye(len(b))[:len(shared)]);labels=['coefficient_'+v for v in shared]
        scales=['logit_disease_fraction' if bounded else co.UNITS[domain]]*len(shared)
        pred={}
        ref_extra=extras(refd,mi,'index_wave')[0] if refs else None
        for year,h,base in prediction:
            nx=np.c_[base,ref_extra]
            assert np.max(abs(nx[:,cols]@mapping-nx))<1e-7
            nx=nx[:,cols];eta=nx@b
            if bounded:
                pp=expit(eta);q=float(7*pp.mean());g=7*(pp*(1-pp))@nx/len(nx)
                assert 0<q<7
            else:q=float(eta.mean());g=nx.mean(0)
            pred[(year,h)]=(q,g)
        if refs:
            years=sorted({k[0] for k in pred})
            for year in years:
                m1,m0=pred[year,1],pred[year,0]
                fs=[('mean_history',m1),('mean_reference',m0),('history_gap',co.difference(m1,m0))]
                if include and year>0:
                    c1=co.difference(m1,pred[0,1]);c0=co.difference(m0,pred[0,0])
                    fs.extend([('change_history',c1),('change_reference',c0),('difference_in_change',co.difference(c1,c0))])
                for label,(q,g) in fs:values.append(q);grads.append(g);labels.append(f'{label}@{year:g}y');scales.append(co.UNITS[domain])
            q,g=co.difference(co.difference(pred[hi,1],pred[hi,0]),co.difference(pred[lo,1],pred[lo,0]))
            values.append(q/(hi-lo));grads.append(g/(hi-lo));labels.append('average_change_in_history_gap_per_year');scales.append(co.UNITS[domain]+'/year')
        J=np.array(grads);U=J@V@J.T
        np.savez_compressed(file,q=np.array(values),U=U,beta=b,covariance=V)
        write(jp,dict(status='ESTIMATED',families=G,people=people.nunique(),windows=len(d),post_records=int(d.years.gt(0).sum()),
               comparison=comp,domain=domain,include_index=include,labels=labels,scales=scales,diagnostics=qa,
               curve_range=[lo,hi] if refs else None,reference_people=len(refd) if refs else None,seconds=time.time()-tick))
        if mi%5==0:print(stem,mi,flush=True)
    m=next((json.loads(j.read_text()) for j in (R/'QA').glob(stem+'_mi*.json') if 'labels' in json.loads(j.read_text())),None)
    meta=dict(module='continuous',comparison=comp,domain=domain,index_included=include)
    if m:pooled(stem,m['labels'],m['scales'],dict(**meta,curve_range=m['curve_range']))
    else:pd.DataFrame([dict(**meta,stem=stem,status='NOT_ESTIMABLE')]).to_csv(R/f'TABLES/{stem}.csv',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('comparison',choices=co.OLD['COMPARE']);a.add_argument('domain',choices=co.DOMAINS);a.add_argument('--include-index',action='store_true');v=a.parse_args();run(v.comparison,v.domain,v.include_index)
