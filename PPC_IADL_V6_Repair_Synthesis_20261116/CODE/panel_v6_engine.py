"""Fast likelihood and independent generic-matrix validation (no real data on --selftest)."""
from functools import lru_cache
from pathlib import Path
import argparse, ctypes, hashlib, json, os, shutil, subprocess, tempfile

for option in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[option]='1'
import numpy as np
import pandas as pd
from scipy.linalg import expm
from numpy.ctypeslib import ndpointer

R=Path(__file__).resolve().parents[1]
V5=R.parent/'PPC_IADL_SHARE_V5_20261116'
RT=R/'PRIVATE/PANEL_V6_RUNTIME'
RS=Path('D:/Program Files/R/R-4.2.3/bin/Rscript.exe')
CXX=Path('D:/Program Files/R/rtools42/x86_64-w64-mingw32.static.posix/bin/g++.exe')
R_LIB=V5/'PRIVATE/PANEL_RUNTIME/library'

@lru_cache
def native():
    RT.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).with_name('panel_v6_likelihood.cpp')
    dll=RT/('panel_v6_'+hashlib.sha256(code.read_bytes()).hexdigest()[:12]+'.dll')
    if not dll.exists():
        with tempfile.TemporaryDirectory(prefix='ppc_v6_build_') as tmp:
            staging=Path(tmp)/'panel.dll'
            env=dict(os.environ)
            env['PATH']=str(CXX.parent)+os.pathsep+env['PATH']
            shutil.copyfile(code,Path(tmp)/'panel.cpp')
            subprocess.run([str(CXX),'-O3','-std=c++17','-shared','-static-libgcc',
                            '-static-libstdc++','panel.cpp','-o','panel.dll'],
                           cwd=tmp,env=env,check=True,capture_output=True)
            target=dll.with_suffix(f'.{os.getpid()}.staging')
            shutil.copyfile(staging,target)
            try: target.replace(dll)
            except PermissionError:
                if not dll.exists(): raise
    lib=ctypes.CDLL(str(dll))
    d=ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    i=ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    lib.ppc_panel_v6_lik.argtypes=[ctypes.c_int,ctypes.c_int,d,i,d,i,d,d,d,d]
    lib.ppc_panel_v6_lik.restype=ctypes.c_int
    return lib

def likelihood(X,r,off):
    lib=native()
    X=np.ascontiguousarray(X,dtype=np.float64)
    dt=r.dt.to_numpy(dtype=np.float64)
    obs=r.state.to_numpy(dtype=np.int32)
    off=np.ascontiguousarray(off,dtype=np.int32)
    n,_,p=X.shape
    w0=np.ones(n)
    def evaluate(b,w=None):
        g=np.zeros(p); val=np.zeros(1)
        status=lib.ppc_panel_v6_lik(n,p,X,off,dt,obs,w0 if w is None else w,
                                   np.ascontiguousarray(b,dtype=np.float64),g,val)
        return (float(val[0]),g) if not status else (np.inf,np.zeros(p))
    return evaluate

def qmat(eta):
    a,b,d,e=np.exp(eta)
    return np.array([[-a-d,a,d],[b,-b-e,e],[0,0,0.]])

def generic(eta,obs):
    """Direct generic full 3-state matrix engine; last component may be a density."""
    Q=qmat(eta)
    alpha=np.array([1.,0.,0.])
    for dt,state in obs:
        alpha=alpha@expm(float(dt)*Q)
        if state==4: return float(alpha@Q[:,2])
        keep=([0] if state==1 else [1] if state==2 else [2] if state==3 else [0,1])
        alpha[np.setdiff1d(np.arange(3),keep)]=0
    return float(alpha.sum())

def rcheck(eta,obs,folder):
    """Official, fixed-parameter msm log-likelihood for a single observation sequence."""
    folder.mkdir(parents=True,exist_ok=True)
    z=np.vstack(([0.,1,1],np.array([[d,3 if s==4 else s,3 if s==4 else 1] for d,s in obs])))
    z[:,0]=np.cumsum(z[:,0])
    pd.DataFrame(z,columns=['elapsed','state','type']).to_csv(folder/'panel_fixture.csv',index=False)
    pd.DataFrame(np.exp(np.asarray(eta).reshape(1,4)),columns=['ab','ba','ad','bd']).to_csv(folder/'q.csv',index=False)
    args=[str(RS),'--vanilla',str(Path(__file__).with_name('panel_v6_verify.R')),str(R_LIB),str(folder)]
    child=subprocess.run(args,cwd=R,capture_output=True,timeout=90)
    if child.returncode: raise ValueError('Official msm verification returned an error')
    d=pd.read_csv(folder/'official.csv').iloc[0]
    return float(d.nll),str(d.version)

def test():
    theta=np.log([.27,.42,.075,.16])
    S={
      'regular_panel':((1.1,2),(.9,1),(1.7,2)),
      'month_interval':((.7,2),(1.,99),(1/12,3)),
      'year_interval':((1.4,99),(1.,3)),
      'wide_confirmation':((3.1,3),),
      'exact_event':((.7,2),(.6,99),(1.2,4)),
      'alive_union_censoring':((1.1,2),(2.2,99)),
      'unknown_no_observation':(),
    }
    N=len(S); X=np.tile(np.eye(4)[None,:,:],(N,1,1))
    base=lambda obs:pd.DataFrame(obs,columns=['dt','state'])
    Q=qmat(theta)
    annual=generic(theta,S['year_interval'])
    assert abs(annual-(expm(2.4*Q)[0,2]-expm(1.4*Q)[0,2]))<1e-13
    monthly=generic(theta,S['month_interval'])
    assert abs(monthly-expm(.7*Q)[0,1]*(expm((1+1/12)*Q)[1,2]-expm(Q)[1,2]))<1e-13
    exact=generic(theta,S['exact_event'])
    assert abs(exact-expm(.7*Q)[0,1]*(expm(1.8*Q)@Q)[1,2])<1e-13
    alive=generic(theta,S['alive_union_censoring'])
    assert abs(alive-expm(1.1*Q)[0,1]*expm(2.2*Q)[1,:2].sum())<1e-13
    lengths=np.r_[0,np.cumsum([len(v) for v in S.values()])].astype(np.int32)
    fast=likelihood(X,base([r for v in S.values() for r in v]),lengths)
    f=lambda b: -sum(np.log(generic(b,x)) for x in S.values())
    value,grad=fast(theta)
    h=np.eye(4)*1e-5
    finite=np.array([(f(theta+z)-f(theta-z))/2e-5 for z in h])
    official={}
    for k in ('month_interval','exact_event','alive_union_censoring'):
        ll,version=rcheck(theta,S[k],RT/'synthetic'/k)
        official[k]=abs(ll+np.log(generic(theta,S[k])))
    report=dict(status='verified',fixtures=list(S),synthetic_only=True,msm=version,
                generic_abs_error=abs(value-f(theta)),max_analytic_gradient_error=float(np.max(abs(grad-finite))),
                official_fixture_abs_errors=official)
    assert report['generic_abs_error']<1e-10 and report['max_analytic_gradient_error']<1e-7
    assert max(official.values())<1e-8
    (R/'QA').mkdir(parents=True,exist_ok=True)
    (R/'QA/PANEL_V6_SYNTHETIC_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--selftest',action='store_true',required=True)
    p.parse_args()
    test()
