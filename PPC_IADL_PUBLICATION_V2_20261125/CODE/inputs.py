from pathlib import Path
import hashlib,json,sys,zipfile
R=Path(__file__).resolve().parents[1];W=R.parent
V=W/'PPC_IADL_PUBLICATION_FINAL_20261125';C=W/'PPC_IADL_FINAL_COMBINATION_CLOSURE_20261125';M=W/'PPC_IADL_ALL_METHODS_MI_20261118';Z=W/'PPC_IADL_ZHOU_ADAPTATION_20261117'
A=R/'AGGREGATES';Q=R/'QA';S=R/'SUBMISSION';G=R/'FIGURES';REP=R/'REPORT';REND=R/'RENDER_QA'
for path in (A,Q,S,G,REP,REND):path.mkdir(exist_ok=True)
def rdj(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,data):p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cohorts=['CHARLS','HRS','ELSA','SHARE'];hs=['H00','H10','H01','H11'];cc=['H11_vs_H01','H10_vs_H00'];dm=['physical','psychological','cognitive']
hist=['No difficulties at either visit','No longer reporting difficulties','Newly reported difficulties','Difficulties reported at both visits']
contr=[hist[3]+' vs '+hist[2].lower(), hist[1]+' vs '+hist[0].lower()]
short=['Both visits vs newly reported','No longer reporting vs neither visit']
classes=['Low physical burden with favourable psychological and cognitive health','Increasing physical burden with relatively preserved psychological and cognitive health','Lower cognitive performance with further decline','Higher physical burden with increasing depressive symptoms']
def authority():
    ledger={}; manifests={}
    for project,name,stamp in [(V,'SAFE_DELIVERY_PUBLICATION_FINAL_20261125.zip','QA/FINAL_RECEIPT.json'),(C,'SAFE_DELIVERY_FINAL_COMBINATION_20261125.zip','QA/FINAL_RELEASE_RECEIPT.json'),(M,'SAFE_DELIVERY_ALL_METHODS_MI_20261118.zip','QA/RELEASE_RECEIPT.json')]:
        file=project/name;v=sha(file);assert v==rdj(project/stamp)['sha256']
        with zipfile.ZipFile(file) as z:
            m=json.loads(z.read('RELEASE_MANIFEST.json')); m=m['files'] if isinstance(m,dict) else {k['path']:k for k in m}
            assert len(z.namelist())==len(m)+1 and z.testzip() is None
            for path,x in m.items():assert hashlib.sha256(z.read(path)).hexdigest()==x['sha256']
            manifests[project]={x['sha256'] for x in m.values()}
        ledger[name]=v
    signed=[]
    for project,files in [(V,[V/'AGGREGATES/publication_tables.json',V/'AGGREGATES/source_MI_functionals.csv',V/'QA/cognitive_algorithm_check.csv',V/'AGGREGATES/Figure2_coordinates.csv']),(C,[C/'TABLES'/n for n in ['FROZEN_SUBMITTED_FUNCTIONALS.csv','TABLE_2_FULL_PRECISION.csv','PPC_COMBINATION_MI_RESULTS.csv']]),(M,list((M/'TABLES').glob('CONTINUOUS_*.csv')))]:
        for f in files:
            assert sha(f) in manifests[project],str(f.name);signed.append({'path':f.relative_to(W).as_posix(),'sha256':sha(f)})
    assert len(list((M/'TABLES').glob('CONTINUOUS_*.csv')))==12
    put(Q/'authority.json',{'archives':ledger,'exact_inputs':signed,'new_effect_fits':0,'source_QA':'reuse previously executed and signed four cohorts x two comparisons x 50 imputations; no meta-analysis or new estimate run'})
