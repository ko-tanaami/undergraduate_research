"""Compare planar-film kernels to independent author implementation (MIT)."""
import importlib.util
import json
import hashlib
from pathlib import Path
import numpy as np
from shkuratov_model import film_rt

path=Path(__file__).resolve().parent.parent/'references'/'reference_tmm'/'tmm_core.py'
spec=importlib.util.spec_from_file_location('reference_tmm_core',path)
tmm=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmm)
rng=np.random.default_rng(4711)
max_error=0.
cases=[]
for i in range(500):
    n0=rng.uniform(1,2.5)
    nf=rng.uniform(.5,3)+1j*rng.uniform(0,1)
    n1=rng.uniform(.5,3)+1j*(0 if i%2==0 else rng.uniform(0,1))
    theta=rng.uniform(0,1.55)
    d=10**rng.uniform(-3,0)
    w=rng.uniform(.5,5)
    reference=[tmm.coh_tmm(pol,[n0,nf,n1],[np.inf,d,np.inf],theta,w) for pol in ('s','p')]
    expected=np.array([sum(x[key] for x in reference)/2 for key in ('R','T')])
    actual=np.array(film_rt(n0,nf,n1,theta,d,w))
    error=float(np.max(abs(expected-actual)))
    max_error=max(max_error,error)
    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-12)
    if i<5:
        cases.append({'n0':n0,'film':[nf.real,nf.imag],'n1':[n1.real,n1.imag],'theta':theta,'d':d,'w':w,'expected':expected.tolist()})
result={'source':'https://github.com/sbyrnes321/tmm','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':500,'seed':4711,'max_abs_difference':max_error,'scope':'Nonabsorbing incident medium, planar coherent single layer only','regression_cases':cases}
Path(__file__).with_name('tmm_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='regression_cases'},indent=2))
