"""Audit local JWST observation tables against available optical constants.

This is a coverage and provenance check, not a mineral identification or fit.
"""
import json
from pathlib import Path
import numpy as np
from shkuratov_model import (load_observations, load_opcon, load_mooney1985,
                             load_constants, load_optool, load_refractiveindex_yml,
                             numeric_table)


ROOT=Path(__file__).resolve().parent.parent
OBS=ROOT/'data'/'jwst_driss'
OPCON=Path.home()/'Downloads'/'opcon'
RAW=ROOT/'data'/'optical_constants_full'


def coverage(wavelength, constants):
    available=(wavelength>=constants.wavelength[0])&(wavelength<=constants.wavelength[-1])
    for left,right in constants.gaps:
        available &= ~((wavelength>left)&(wavelength<right))
    return int(available.sum())


def main():
    observations={}
    for path in sorted(OBS.glob('*.txt')):
        data=load_observations(path)
        observations[path.name]={
            'rows':len(data),'source_columns':numeric_table(path,None).shape[1],
            'wavelength_range_um':[float(data[0,0]),float(data[-1,0])],
            'median_uncertainty':float(np.median(data[:,2]))}
    sources={}
    if OPCON.is_dir():
        for code in ('K','M','Gser','Gcro'):
            constants=load_opcon(OPCON,code)
            sources[code]={'description':{'K':'legacy kerogen','M':'legacy Murchison',
                                          'Gser':'Glotch generic serpentine',
                                          'Gcro':'Glotch cronstedtite'}[code],
                           'wavelength_range_um':[float(constants.wavelength[0]),float(constants.wavelength[-1])],
                           'gaps_um':[list(pair) for pair in constants.gaps],
                           'coverage_rows':{name:coverage(load_observations(OBS/name)[:,0],constants)
                                            for name in observations}}
        path=OPCON/'serp_chlor_mooney85'
        if path.exists():
            table=numeric_table(path,5)
            anchors={4000:[1.525,2.68,1.479,2.97],2000:[1.489,3.78,1.427,4.45]}
            for wavenumber,values in anchors.items():
                row=table[table[:,0]==wavenumber]
                if len(row)!=1 or not np.array_equal(row[0,1:],values):
                    raise ValueError(f'Mooney 1985 published table anchor differs at {wavenumber} cm^-1')
            for mineral in ('chlorite','serpentine'):
                constants=load_mooney1985(path,mineral)
                sources['Mooney1985_'+mineral]={
                    'wavelength_range_um':[float(constants.wavelength[0]),float(constants.wavelength[-1])],
                    'rows':len(constants.wavelength),
                    'gaps_um':[list(pair) for pair in constants.gaps],
                    'coverage_rows':{name:coverage(load_observations(OBS/name)[:,0],constants)
                                     for name in observations}}
    public_tables={
        'Querry1985_magnetite':('magnetite_Querry1985.yml',load_refractiveindex_yml),
        'composite_serpentine_proxy':('mg_serpentine_proxy_composite.lnk',load_optool),
        'Khare1984_Titan_tholin':('organics_tholin_khare1984.lnk',load_optool),
        'Roush2021_calcite':('roush2021_calcite.csv',lambda path:load_constants(path,columns=(0,2,3))),
        'Roush2021_dolomite':('roush2021_dolomite.csv',lambda path:load_constants(path,columns=(0,2,3))),
        'Roush2021_magnesite':('roush2021_magnesite.csv',lambda path:load_constants(path,columns=(0,2,3))),
        'Querry1987_dolomite_o':('dolomite_querry1987_o.yml',load_refractiveindex_yml),
        'Querry1987_dolomite_e':('dolomite_querry1987_e.yml',load_refractiveindex_yml),
    }
    for key,(filename,reader) in public_tables.items():
        path=RAW/filename
        if path.exists():
            constants=reader(path)
            sources[key]={'wavelength_range_um':[float(constants.wavelength[0]),float(constants.wavelength[-1])],
                          'rows':len(constants.wavelength),
                          'coverage_rows':{name:coverage(load_observations(OBS/name)[:,0],constants)
                                           for name in observations}}
    report={'observations':observations,'available_optical_constants':sources,
            'interpretation':'Coverage only; no fit, mineral detection, or independent numerical verification of full optical-constant tables.'}
    output=ROOT/'validation'/'jwst_shkuratov_input_audit.json'
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
