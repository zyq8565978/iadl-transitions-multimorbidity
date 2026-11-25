"""No effect fitting: verify every old model's baseline/history MI node exists."""
from common import *
from binary_suite import prepare,anc

def main():
    bound=set();histories={};next_histories={}
    for s in SOURCES:
        n=pd.read_parquet(R/f'PRIVATE/{s}_node_binding.parquet');assert n.key.is_unique
        bound.update(n.key)
        histories.update(n.set_index('key').frozen_history.to_dict())
        next_histories.update(n.set_index('key').next_index_history.to_dict())
    def check_history(d,wave='wave'):
        got=keys(d,wave).map(histories).to_numpy(str)
        assert np.array_equal(got,d.history.to_numpy(str)),('history_auxiliary_binding',wave)
    def check_prior_context(d):
        got=keys(d,'prior_wave').map(next_histories).to_numpy(str)
        assert np.array_equal(got,d.history.to_numpy(str)),'previous_covariate_node_history_context'
    checks=[]
    for comp in COMPS:
        for spec in ('primary','basic_adjustment','prior_covariates',*anc.SPECS):
            d,y,x,src=prepare(comp,spec)
            check_history(d)
            check_prior_context(d)
            for wave in ('wave','prior_wave'):
                q=keys(d,wave);assert set(q)<=bound,(comp,spec,wave,'unbound nodes')
            assert np.array_equal(y,d.y) and len(d)==len(x)
            checks.append(dict(module='binary',comparison=comp,spec=spec,rows=len(d),people=d.person.nunique(),pass_binding=True))
    life=load('binding_life',S/'LIFESTYLE/run_lifestyle.py');aa,dd=life.prep(False)
    for mode,allgroups in [('A2',aa),('D',dd)]:
        for (comp,spec),groups in allgroups.items():
            for s,obj in groups.items():
                d=(obj if mode=='D' else obj[1]).copy();d['source']=s;d['prior_wave']=d.window.str.split('->').str[0].astype(int)
                check_history(d)
                check_prior_context(d)
                assert set(keys(d,'prior_wave'))<=bound,(mode,comp,spec,s)
                checks.append(dict(module=mode,comparison=comp,spec=spec,source=s,rows=len(d),people=d.person.nunique(),pass_binding=True))
                if mode=='A2':
                    d=obj[0].copy();d['source']=s;d['prior_wave']=d.window.str.split('->').str[0].astype(int)
                    check_history(d)
                    check_prior_context(d)
                    assert set(keys(d,'prior_wave'))<=bound,('A2_broad',comp,spec,s)
                    checks.append(dict(module='A2_broad',comparison=comp,spec=spec,source=s,rows=len(d),people=d.person.nunique(),pass_binding=True))
    co=load('binding_continuous',S/'CONTINUOUS/run_continuous.py');raw=dict(co.inputs())
    for comp in co.OLD['COMPARE']:
        for domain in co.DOMAINS:
            for include in (False,True):
                for s in SOURCES:
                    d,_,_=co.initialize(s,raw[s],comp,domain,include);d=d.assign(source=s)
                    check_history(d,'index_wave')
                    assert set(keys(d,'index_wave'))<=bound,('continuous',s,comp,domain,include)
                    checks.append(dict(module='continuous',comparison=comp,domain=domain,include_index=include,source=s,
                        rows=len(d),people=d.person.nunique(),pass_binding=True))
    sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117/CODE'))
    import association
    d,_=association.read_locked(W/'PPC_IADL_ZHOU_ADAPTATION_20261117')
    for comp in COMPS:
        z,_=association.cohort_overlap(d,tuple(comp.split('_vs_')))
        check_history(z,'index_wave')
        assert set(keys(z,'index_wave'))<=bound,('trajectory',comp)
        checks.append(dict(module='trajectory',comparison=comp,rows=len(z),people=z.person.nunique(),pass_binding=True))
    rv=load('binding_reversible',S/'REVERSIBLE/run.py')
    native_panel_spec_sha256=rv.identity()
    for comp in COMPS:
        for item in rv.source_data(comp.replace('_vs_','_'),False):
            d=item['a'].assign(source=item['source'])
            check_history(d)
            assert set(keys(d))<=bound,('reversible',item['source'],comp)
            checks.append(dict(module='reversible',comparison=comp,source=item['source'],rows=len(d),people=d.person.nunique(),pass_binding=True))
    ob=load('binding_observation',S/'OBSERVATION/run_observation.py')
    for s in SOURCES:
        raw=ob.S['load']() if s=='SHARE' else ob.O['linked'](s)
        for comp in COMPS:
            d=(ob.S['select_target'](raw,comp) if s=='SHARE' else ob.O['obs_prepare'](raw,s,comp)).assign(source=s)
            check_history(d)
            assert set(keys(d))<=bound,('observation',s,comp)
            checks.append(dict(module='observation',comparison=comp,source=s,rows=len(d),people=d.person.nunique(),pass_binding=True))
    for p,h in json.loads((R/'QA/HISTORICAL_HASHES.json').read_text()).items():assert digest(Path(p))==h
    write(R/'QA/PREFLIGHT_BINDINGS.json',dict(pass_all=True,checks=checks,no_exposure_or_outcome_imputed=True,
        native_panel_spec_sha256=native_panel_spec_sha256))
    print('PASS: baseline and previous-wave mappings;',len(checks),'checks; old hashes unchanged')

if __name__=='__main__':main()
