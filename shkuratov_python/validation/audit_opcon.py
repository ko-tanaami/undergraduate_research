"""Read-only inventory of every supplied file and every original GUI material."""
import hashlib
import json
import re
from pathlib import Path
from collections import Counter
from shkuratov_model import load_opcon

root=Path('C:/Users/2017j/Downloads/opcon')
source=(Path(__file__).resolve().parent.parent/'references'/'shkur_wid.pro').read_text()
block=source.split("codearr=['H'",1)[1].split('mixarr=indgen',1)[0]
codes=['H']+re.findall(r"'([^']*)'",block)
codes += [f'amorph_{t}K' for t in (15,25,40,50,60,80,100,120)]
codes += [f'crys_{t}K' for t in (20,30,40,50,60,70,80,90,100,110,120,130,140,150)]
inventory=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():
        continue
    raw=p.read_bytes()
    inventory.append({'file':str(p.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
materials=[]
for code in sorted(set(codes)):
    try:
        c=load_opcon(root,code)
        materials.append({'code':code,'status':'loaded_and_basic_validation_passed','documented_gaps':c.gaps,'rows':len(c.wavelength),'wavelength_min':float(c.wavelength[0]),'wavelength_max':float(c.wavelength[-1]),'n_min':float(c.n.min()),'n_max':float(c.n.max()),'k_min':float(c.k.min()),'k_max':float(c.k.max()),'zero_k':int((c.k==0).sum())})
    except (ValueError,OSError) as e:
        reason=str(e)
        if isinstance(e,FileNotFoundError) or 'filename case mismatch' in reason:
            category='source_file_unavailable'
        elif 'Duplicate wavelengths' in reason:
            category='conflicting_duplicate_wavelength'
        elif 'n must be finite and positive' in reason:
            category='nonphysical_refractive_index'
        else:
            category='other'
        materials.append({'code':code,'status':'requires_review','issue_category':category,'reason':reason})
result={'note':'Basic validation is not verification of units, provenance, or physical accuracy. No source files changed.', 'files':inventory,'materials':materials,
        'summary':{'status':dict(Counter(m['status'] for m in materials)),
                   'issue_category':dict(Counter(m['issue_category'] for m in materials if m['status']=='requires_review'))}}
Path(__file__).with_name('opcon_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('files',len(inventory),'materials',len(materials),dict(Counter(m['status'] for m in materials)))
for m in materials:
    if m['status']=='requires_review':
        print(m['code'],m['reason'])
