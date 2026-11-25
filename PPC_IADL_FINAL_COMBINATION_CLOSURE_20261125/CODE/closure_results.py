"""Pool final functionals, independently reproduce scalar Rubin inference and apply the eight-OR Holm family."""
from binding import *
from scipy.stats import t as student
from scipy.special import expit
import math
T=ROOT/'TABLES';Q=ROOT/'QA';C=ROOT/'PRIVATE/dispatch'
def input_checkpoint(pl,m,rules):
    nm=pl['stem']+f'_mi{m:03}'
    for jf,nf in [(C/(nm+'.json'),C/(nm+'.npz')),(Q/(nm+'.json'),ROOT/'PRIVATE'/(nm+'.npz')),(ROOT/'PRIVATE/scheduler_aux'/(nm+'.json'),ROOT/'PRIVATE/scheduler_aux'/(nm+'.npz'))]:
        if jf.exists():
            z=json.loads(jf.read_text())
            if nf.exists():
                with np.load(nf) as f:return z,f['q'].copy(),f['U'].copy()
            if z['status']=='NON_ESTIMABLE':return z,None,None
    if pl['engine']=='MLE':
        jj=C/(nm+'_tryMLE.json');ff=C/(nm+'_tryMLE.npz')
        if jj.exists() and ff.exists():
            with np.load(ff) as f:return dict(status='ESTIMATED',diagnostics=json.loads(jj.read_text()),implementation='original family HC1'),f['q'].copy(),f['U'].copy()
    if pl['stem'] in rules and rules[pl['stem']]['bootstrap_variance_mathematically_impossible']:
        pr=rules[pl['stem']];assert pr['planned']==199 and pr['max_possible_valid']<180
        jj=C/(nm+'_point.json');ff=C/(nm+'_point.npz')
        if jj.exists() and ff.exists():
            aux=C/(nm+'_redraws.json');b=json.loads(aux.read_text()) if aux.exists() else {}
            with np.load(ff) as f:return dict(status='POINT_ONLY_STRUCTURAL_BOOTSTRAP_ROOT_GATE',bootstrap_attempted=b.get('completed',0),root_plans_all_checked=199,
                max_possible_valid=int(pr['max_possible_valid']),diagnostics=json.loads(jj.read_text())['diagnostics'],implementation='Original full-data fit; exact ceiling on fixed-plan success'),f['q'].copy(),np.full((4,4),np.nan)
    return None
def independent(q,u,dfcom):
    a=sum(map(float,q))/len(q);w=sum(map(float,u))/len(q)
    b=sum((float(v)-a)**2 for v in q)/(len(q)-1);vb=(1+1/len(q))*b
    total=w+vb;lam=vb/total;observed=(dfcom+1)/(dfcom+3)*dfcom*(1-lam)
    deg=1/(1/observed+(lam**2/(len(q)-1) if lam>1e-12 else 0.))
    se=math.sqrt(total);half=se*student.isf(.025,deg)
    return a,se,deg,a-half,a+half
def combine(plans=None):
    plans=plans or json.loads((Q/'EXECUTION_PLAN.json').read_text())['plan_by_spec']
    assert len(plans)==8
    out=[];checks=[];runs=[];data=[];rules=pd.read_csv(Q/'FIXED_PLAN_FLIC_ROOT_GATE.csv').set_index('stem').to_dict('index')
    for pl in plans:
        stem=pl['stem'];initial=pl.copy();initial.pop('MLE_failure',None);initial['role']='Result-informed secondary outcome decomposition'
        if pl['engine'].startswith('NO_'):
            out.append(initial|dict(status='NON_ESTIMABLE_'+pl['engine'],MI_completed=0));continue
        qs=[];us=[];accuracy=[]
        for m in range(1,51):
            value=input_checkpoint(pl,m,rules);assert value is not None,stem+f'_mi{m:03} has not finished'
            z,x,u=value;runs.append(dict(stem=stem,mi=m,status=z['status'],bootstrap_valid=z.get('bootstrap_valid'),bootstrap_planned=(199 if pl['engine']=='FIRTH' else None),bootstrap_attempted=z.get('bootstrap_attempted'),root_plans_checked=z.get('root_plans_all_checked'),max_possible_valid=z.get('max_possible_valid'),
                implementation=z.get('implementation'),reason=z.get('reason'),adjusted_score=z.get('diagnostics',{}).get('adjusted_score_max_scaled')))
            if x is None:continue
            assert x.shape==(4,) and u.shape==(4,4) and np.isfinite(x).all()
            assert 0<=x[1]<=1 and 0<=x[2]<=1 and abs(x[3]-x[1]+x[2])<1e-10
            if np.isfinite(u).all():
                assert np.allclose(u,u.T,atol=1e-12) and np.linalg.eigvalsh(u).min()>-1e-9 and abs(u[3,3]-(u[1,1]+u[2,2]-2*u[1,2]))<1e-9
            qs.append(x);us.append(u)
            row=dict(stem=stem,mi=m,log_OR=x[0],prob_history=x[1],prob_reference=x[2],paired_RD=x[3])
            row.update({f'U_{j}{l}':u[j,l] for j in range(4) for l in range(j,4)});data.append(row)
        q,u=np.asarray(qs),np.asarray(us);n=len(q);nvar=sum(np.isfinite(ui).all() for ui in us)
        initial.update(MI_completed=n,valid_joint_covariances=nvar)
        if n!=50:
            out.append(initial|dict(status='NOT_POOLED_INCOMPLETE_VALID_MI'));continue
        fun=['OR','risk_history','risk_reference','risk_difference'];scale=['ratio','probability','probability','probability_difference']
        if nvar<50:
            for j in range(4):out.append(initial|dict(status='MI_POINT_ONLY_NO_VALID_POOLED_VARIANCE',functional=fun[j],scale=scale[j],estimate=np.exp(q[:,0].mean()) if j==0 else q[:,j].mean()))
            continue
        mu,w,b,total,df,se,lo,hi,fmi,mc=base.pool_values(q,u,pl['families']-1)
        for j in range(4):
            cc=independent(q[:,j],u[:,j,j],pl['families']-1)
            a=np.array([mu[j],se[j],df[j],lo[j],hi[j]])
            np.testing.assert_allclose(cc,a,rtol=1e-10,atol=1e-11)
            checks.append(dict(stem=stem,functional=fun[j],MI=n,max_absolute_discrepancy=max(abs(a-np.array(cc))),covariance_RD_identity=True))
            tr=np.exp if j==0 else float
            out.append(initial|dict(functional=fun[j],scale=scale[j],estimate=tr(mu[j]),lower=tr(lo[j]),upper=tr(hi[j]),pooled_analysis_scale=mu[j],SE_analysis_scale=se[j],within_variance=w[j,j],between_variance=b[j,j],total_variance=total[j,j],df=df[j],FMI=fmi[j],MCSE=mc[j],MCSE_over_SE=mc[j]/se[j],p_two_sided=(2*student.sf(abs(mu[j]/se[j]),df[j]) if j in [0,3] else None),status='ESTIMATED'))
    result=pd.DataFrame(out);pvals=[]
    for pl in plans:
        rr=result[(result.stem==pl['stem']) & (result.get('functional')=='OR')];p=rr.p_two_sided.iloc[0] if len(rr) and 'p_two_sided' in rr else np.nan
        pvals.append(float(p) if np.isfinite(p) else 1.)
    order=np.argsort(pvals,kind='stable');adjusted=np.empty(8);adjusted[order]=np.minimum(1,np.maximum.accumulate((8-np.arange(8))*np.array(pvals)[order]))
    for i,pl in enumerate(plans):
        if 'functional' in result:result.loc[(result.stem==pl['stem'])&(result.functional=='OR')&(result.status=='ESTIMATED'),'p_Holm_8_OR']=adjusted[i]
    for name,frame in [('PPC_COMBINATION_MI_RESULTS',result),('PPC_COMBINATION_MI_DIAGNOSTICS',pd.DataFrame(runs)),('PPC_COMBINATION_FINAL_FUNCTIONALS_PER_MI',pd.DataFrame(data)),('PPC_COMBINATION_INDEPENDENT_POOL_CHECK',pd.DataFrame(checks))]:
        frame.to_csv(T/(name+'.csv'),index=False,encoding='utf-8-sig')
    dump(Q/'COMBINATION_GATE.json',dict(all_eight_accounted=True,number_planned=8,estimated_OR=int(((result.get('functional')=='OR')&(result.status=='ESTIMATED')).sum()),rubin_scalar_checks=len(checks),Holm_family=8,Holm_missing_input_only=1,no_successful_subset_pooling=True))
    print('Closed and checked eight specifications.',flush=True)
def status():
    plans=json.loads((Q/'EXECUTION_PLAN.json').read_text())['plan_by_spec']
    rules=pd.read_csv(Q/'FIXED_PLAN_FLIC_ROOT_GATE.csv').set_index('stem').to_dict('index')
    data=[]
    for pl in plans:
        if pl['engine'].startswith('NO_'):data.append([pl['stem'],pl['engine'],0,0]);continue
        n=sum(input_checkpoint(pl,m,rules) is not None for m in range(1,51));data.append([pl['stem'],pl['engine'],n,50])
    print(pd.DataFrame(data,columns=['specification','estimator','resolved_MI','required_MI']).to_string(index=False),flush=True)
    return sum(q[2] for q in data)==350
if __name__=='__main__':
    if '--status' in sys.argv:status()
    else:combine()
