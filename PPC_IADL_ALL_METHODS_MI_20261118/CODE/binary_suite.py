"""Original PPC and every existing ancillary risk set, covariate MI added."""
from common import *
import argparse
anc=load('allmi_ancillary',S/'ANCILLARY/run_ancillary.py')

def prepare(comp,spec):
    if spec in ('primary','basic_adjustment','prior_covariates','structural_BMI_minus2'):
        d,y,x,src=old.load(comp,'M_basic' if spec=='basic_adjustment' else 'M_severity');d=d.copy()
    else:
        d,x,src,*_=anc.prepare(comp,spec);y=d.y.to_numpy(float)
    d['prior_wave']=d.window.str.split('->').str[0].astype(int)
    return d,y,x,src

def run(comp,spec):
    effect_gate();d,y,x,src=prepare(comp,spec);stem=f'BINARY_{comp}_{spec}'
    history=R/f'QA/{stem}_engine.json'
    wave='prior_wave' if spec=='prior_covariates' else 'wave'
    if history.exists():engine=json.loads(history.read_text())['engine']
    else:
        # Preflight the same estimator in all MI sets; a numerical fallback is
        # applied to the entire specification, never selected by effect direction.
        engine='FIRTH' if comp=='H11_vs_H01' and spec in ('primary','prior_covariates','structural_BMI_minus2') else 'MLE'
        failures=[]
        if engine=='MLE':
            for mi in range(1,51):
                xx=np.c_[x,extras(d,mi,wave,stress=spec=='structural_BMI_minus2')[0]];xx=xx[:,qc.independent(xx)]
                try:qc.mle_influence(d,y,xx)
                except (ValueError,AssertionError,np.linalg.LinAlgError) as exc:
                    failures.append({'mi':mi,'reason':str(exc)});engine='FIRTH';break
        write(history,dict(engine=engine,numerical_fallback=failures))
    for mi in range(1,51):
        xx=np.c_[x,extras(d,mi,wave,stress=spec=='structural_BMI_minus2')[0]]
        try:binary(d,y,xx,src,stem,mi,engine,B=499 if spec in ('primary','structural_BMI_minus2') and comp=='H11_vs_H01' else 199)
        except (ValueError,AssertionError,np.linalg.LinAlgError) as exc:
            write(R/f'QA/{stem}_mi{mi:03d}.json',dict(status='NOT_ESTIMABLE',mi=mi,engine=engine,reason=str(exc)))
        if mi%5==0:print(stem,mi,'of 50',flush=True)
    out=pooled(stem,['OR','risk_history','risk_reference','risk_difference'],['log','probability','probability','probability_difference'],dict(comparison=comp,specification=spec,engine=engine))
    print('COMPLETE',stem,'pooled rows',len(out),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--comparison',choices=COMPS,required=True);a.add_argument('--spec',choices=['primary','basic_adjustment','prior_covariates','structural_BMI_minus2',*anc.SPECS],required=True);v=a.parse_args();run(v.comparison,v.spec)
