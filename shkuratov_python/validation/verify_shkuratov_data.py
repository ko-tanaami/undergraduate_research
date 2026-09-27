"""Local integration checks. Writes validation results and a standalone SVG."""
import json
from pathlib import Path
import numpy as np
from shkuratov_model import load_opcon,Component,reflectance

root=Path('C:/Users/2017j/Downloads/opcon')
examples=Path(__file__).resolve().parent.parent/'examples'
results=[]
for code,limits in [('Gser',(5.1,12)),('opx',(.5,2.4)),('cpx',(.5,2.4)),('olv',(.5,2.4))]:
    data=load_opcon(root,code)
    w=np.linspace(*limits,101)
    c=[Component(data,1,30)]
    r=reflectance(w,c,porosity=.3,order=256)
    fine=reflectance(w,c,porosity=.3,order=512)
    np.testing.assert_allclose(r,fine,rtol=2e-5,atol=1e-9)
    assert np.all(np.isfinite(r)) and np.all((r>=0)&(r<=1))
    results.append({'code':code,'points':len(w),'range':limits,'min':float(r.min()),'max':float(r.max()),'quadrature_max_abs_difference':float(np.max(abs(r-fine)))})
mixed=[Component(load_opcon(root,'opx'),.7,30),Component(load_opcon(root,'olv'),.3,50)]
w=np.linspace(.5,2.4,101)
r=reflectance(w,mixed,.3)
np.savetxt(examples/'shkuratov_mixture.csv',np.c_[w,r],delimiter=',',header='wavelength_um,model_reflectance',comments='')
np.testing.assert_allclose(r,reflectance(w,mixed,.3,order=512),rtol=2e-5,atol=1e-9)
results.append({'code':'opx/olv 0.7/0.3 model weights','points':len(w),'min':float(r.min()),'max':float(r.max())})
Path(__file__).with_name('shkuratov_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
d=np.loadtxt(examples/'shkuratov_serpentine.csv',delimiter=',',skiprows=1)
x,y=d.T
points=' '.join(f'{70+(a-x.min())/(x.max()-x.min())*680:.2f},{360-b/.1*300:.2f}' for a,b in zip(x,y))
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="430" viewBox="0 0 800 430">
<rect width="800" height="430" fill="white"/>
<g font-family="sans-serif" font-size="14" fill="#222">
<text x="70" y="28">Shkuratov model: Gser, diameter 30 um, porosity 0.3</text>
<text x="70" y="50" font-size="12">Model calculation; not an observational fit or validated geometric albedo</text>
<path d="M70 60V360H750" fill="none" stroke="#222"/>
<text x="32" y="65">0.10</text><text x="45" y="365">0</text>
<text x="60" y="385">5.1</text><text x="725" y="385">12.0</text>
<text x="330" y="415">Wavelength (um)</text>
<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="2"/>
</g></svg>'''
(examples/'shkuratov_serpentine.svg').write_text(svg,encoding='utf-8')
print(json.dumps(results,indent=2))
