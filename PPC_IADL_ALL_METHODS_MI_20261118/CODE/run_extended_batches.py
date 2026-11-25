"""Two bounded local workers, input-bound immutable snapshots, real completion."""
from inventory import R,W
from prepare_extended import sha
from pathlib import Path
import json,sys,subprocess,time,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
RS='D:/Program Files/R/R-4.2.3/bin/Rscript.exe'

def launch(source,mode,batch,iterations=10):
    stem=f'{source}_{mode}_{batch}_r4'
    status=R/f'QA/{stem}_status.json'
    if status.exists() and json.loads(status.read_text())['iterations']>=(iterations if mode=='full' else 5):return
    script_id=sha(R/'CODE/run_extended_mice.R')[:12]
    snap=R/f'PRIVATE/{stem}_it{iterations}_{script_id}_runner.R'
    if not snap.exists():shutil.copy2(R/'CODE/run_extended_mice.R',snap)
    with (R/f'QA/{stem}_it{iterations}.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([RS,'--vanilla',str(snap),source,mode,str(batch),str(iterations)],cwd=W,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if p.returncode:raise RuntimeError(stem+' exited '+str(p.returncode))

def source_validation(s):
    launch(s,'validation',1)
    d=pd.read_csv(R/f'TABLES/{s}_validation_1_r4_validation.csv')
    a=d.groupby('variable').bias.mean()
    ok=abs(a['bmi'])<=1 and abs(a['smoking'])<=.10 and abs(a['alcohol'])<=.10
    (R/f'QA/{s}_validation_gate.json').write_text(json.dumps({'pass':bool(ok),'mean_bias':a.to_dict()},indent=2))
    if not ok:raise RuntimeError(s+' held-out imputation calibration flag')
    print('VALIDATION PASS',s,a.to_dict(),flush=True)
    return s

def main():
    inputs=json.loads((R/'QA/INPUT_HASHES.json').read_text())
    for p,h in inputs.items():assert sha(Path(p))==h
    lock={'time_started':time.time(),'input_hashes':inputs,'amendment_sha256':sha(R/'ANALYSIS_AMENDMENT.md'),
          'legacy_imputer_sha256':sha(W/'PPC_IADL_LONGITUDINAL_MI_20261117/CODE/run_mice.R'),'workers':2}
    lp=R/'QA/MI_EXECUTION_LOCK.json'
    if lp.exists():assert json.loads(lp.read_text())['input_hashes']==inputs
    else:lp.write_text(json.dumps(lock,indent=2))
    tasks=[];done=[];t=time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures={ex.submit(source_validation,s):('validation',s,1) for s in ('CHARLS','HRS','ELSA','SHARE')}
        while futures:
            f=next(as_completed(futures));kind,s,b=futures.pop(f);f.result()
            done.append([kind,s,b]);print('DONE',kind,s,b,round(time.time()-t,1),flush=True)
            (R/'QA/MI_PROGRESS.json').write_text(json.dumps({'completed':done,'elapsed_seconds':time.time()-t,'running_or_queued':len(futures)},indent=2))
            if kind=='validation':
                for bb in range(1,11):futures[ex.submit(launch,s,'full',bb)]=('full',s,bb)
    print('ALL 40 PRODUCTION BATCHES COMPLETED',flush=True)

if __name__=='__main__':main()
