"""Numerical smoke scan across every readable original GUI material.

This checks finite bounded output under an explicitly artificial path length.
It does not establish material provenance or physical applicability.
"""
import json
from pathlib import Path
import numpy as np
from shkuratov_model import load_opcon, Component, reflectance


def main():
    root=Path('C:/Users/2017j/Downloads/opcon')
    audit=json.loads(Path(__file__).with_name('opcon_audit.json').read_text(encoding='utf-8'))
    results=[]
    for material in audit['materials']:
        if material['status']!='loaded_and_basic_validation_passed':
            continue
        code=material['code']
        try:
            constants=load_opcon(root,code)
            ids=np.unique(np.linspace(0,len(constants.wavelength)-1,5,dtype=int))
            wavelength=constants.wavelength[ids]
            # Deliberately well above every wavelength; a numerical test only.
            path_length=100*float(wavelength[-1])
            spectrum=reflectance(wavelength,[Component(constants,1,path_length)],porosity=.3,order=64)
            if not np.all(np.isfinite(spectrum)) or np.any((spectrum<0)|(spectrum>1)):
                raise ValueError('nonfinite or out-of-range reflectance')
            results.append({'code':code,'status':'passed','min':float(spectrum.min()),
                            'max':float(spectrum.max())})
        except (ValueError,ArithmeticError,OverflowError) as error:
            results.append({'code':code,'status':'failed','reason':str(error)})
    output={'method':'Five source wavelengths, S=100 times largest tested wavelength, porosity=0.3, 64 angle nodes; numerical domain check only',
            'passed':sum(x['status']=='passed' for x in results),
            'failed':sum(x['status']=='failed' for x in results),'results':results}
    Path(__file__).with_name('shkuratov_material_scan.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(output['passed'],'passed,',output['failed'],'failed')


if __name__=='__main__':
    main()
