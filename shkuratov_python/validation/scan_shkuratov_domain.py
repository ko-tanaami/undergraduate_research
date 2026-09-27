"""Synthetic numerical-domain scan; passing does not validate ray optics."""
import csv
import json
from pathlib import Path
from collections import Counter
from shkuratov_model import OpticalConstants, Component, reflectance, angular_integrals

rows=[]
for n in (.5,1.,1.3,1.5,2.,3.):
    for k in (0.,1e-6,1e-3,.01,.1,1.,10.):
        rb,rf,te,ri,ti=angular_integrals(n,k,1.)
        c=OpticalConstants([.9,1.1],[n,n],[k,k])
        for d in (.01,.1,1.,10.,100.):
            row={'n':n,'k':k,'diameter_over_wavelength':d,'internal_R_plus_T':ri+ti,'geometric_optics_caution':d<=1}
            try:
                r=reflectance([1.],[Component(c,1,d)])
                row.update(status='finite_physical_closure',reflectance=float(r[0]),reason='')
            except ValueError as e:
                row.update(status='rejected',reflectance='',reason=str(e))
            rows.append(row)
with Path(__file__).with_name('shkuratov_domain_scan.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(dict(Counter(r['status'] for r in rows))))
print('Rejected with d/lambda>1:',sum(r['status']=='rejected' and not r['geometric_optics_caution'] for r in rows))
