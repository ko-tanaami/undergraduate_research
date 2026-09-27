"""Audit the user-supplied optical-constants ZIP against public raw tables.

This script reads data only. It never executes code distributed in the ZIP.
"""
from __future__ import annotations
import csv
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from shkuratov_model import load_opcon, load_constants


ROOT=Path(__file__).resolve().parent.parent
PACKAGE=Path.home()/'Downloads'/'optical_constants_package.zip'
RAW=ROOT/'data'/'optical_constants_full'


def archive_csv(archive, name):
    lines=[line for line in archive.read(name).decode('utf-8-sig').splitlines()
           if line.strip() and not line.lstrip().startswith('#')]
    return list(csv.DictReader(io.StringIO('\n'.join(lines))))


def lnk_rows(path):
    numeric=[line.split() for line in path.read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    count=int(numeric[0][0])
    if len(numeric)!=count+1 or len(numeric[0])!=2 or any(len(row)!=3 for row in numeric[1:]):
        raise ValueError(f'Invalid optool lnk row count: {path}')
    return np.asarray(numeric[1:],float)


def yml_nk_rows(path):
    content=path.read_text(encoding='utf-8')
    if 'type: tabulated nk' not in content:
        raise ValueError(f'Not a tabulated n,k file: {path}')
    rows=[]
    for line in content.splitlines():
        if re.fullmatch(r'\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?(?:\s+[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?){2}\s*',line):
            rows.append([float(value) for value in line.split()])
    return np.asarray(rows,float)


def validate_nk(rows, label):
    if rows.ndim!=2 or rows.shape[1]!=3 or len(rows)<2 or not np.all(np.isfinite(rows)) or np.any(rows[:,0]<=0) or np.any(rows[:,1]<=0) or np.any(rows[:,2]<0):
        raise ValueError(f'Invalid wavelength,n,k rows in {label}')
    sorted_rows=rows[np.argsort(rows[:,0])]
    if np.any(np.diff(sorted_rows[:,0])<=0):
        raise ValueError(f'Duplicate wavelengths in {label}')
    return {'rows':len(rows),'wavelength_range_um':[float(sorted_rows[0,0]),float(sorted_rows[-1,0])],
            'out_of_order_pairs':int(np.sum(np.diff(rows[:,0])<0)),
            'n_range':[float(rows[:,1].min()),float(rows[:,1].max())],
            'k_range':[float(rows[:,2].min()),float(rows[:,2].max())]}


def compare_preview(preview, original):
    original=original[np.argsort(original[:,0])]
    if len(preview)>len(original):
        return {'matched_rows':0,'rows':len(preview)}
    indices=np.searchsorted(original[:,0],preview[:,0])
    valid=indices<len(original)
    matched=np.zeros(len(preview),bool)
    matched[valid]=np.all(np.isclose(original[indices[valid]],preview[valid],rtol=0,atol=1e-5),axis=1)
    return {'matched_rows':int(matched.sum()),'rows':len(preview),
            'all_match_original_at_reported_precision':bool(np.all(matched))}


def main():
    names={
        'magnetite':('magnetite_Querry1985_preview.csv','magnetite_Querry1985.yml',yml_nk_rows),
        'mg_serpentine_proxy':('Mg_serpentine_proxy_composite_preview.csv','mg_serpentine_proxy_composite.lnk',lnk_rows),
        'titan_tholin':('organics_tholin_Khare1984_full.csv','organics_tholin_khare1984.lnk',lnk_rows),
        'dolomite_o':('dolomite_Querry1987_o_preview.csv','dolomite_querry1987_o.yml',yml_nk_rows),
        'dolomite_e':('dolomite_Querry1987_e_preview.csv','dolomite_querry1987_e.yml',yml_nk_rows),
    }
    report={}
    with zipfile.ZipFile(PACKAGE) as archive:
        for key,(preview_file,raw_file,reader) in names.items():
            source=reader(RAW/raw_file)
            preview=np.array([[float(row[k]) for k in ('wavelength_um','n','k')]
                              for row in archive_csv(archive,preview_file)])
            report[key]={'raw':validate_nk(source,raw_file),
                         'package':validate_nk(preview,preview_file),
                         'comparison':compare_preview(preview,source)}
        calcite=archive_csv(archive,'calcite_Ghosh1999_n.csv')
        report['calcite_Ghosh1999']={'rows':len(calcite),'has_k':any(row['k'].strip() for row in calcite),
                                     'wavelength_range_um':[float(calcite[0]['wavelength_um']),float(calcite[-1]['wavelength_um'])]}
        report['fe_serpentine']={'has_spectral_nk':False}
        proxy_preview=np.array([[float(row[k]) for k in ('wavelength_um','n','k')]
                                for row in archive_csv(archive,'Mg_serpentine_proxy_composite_preview.csv')])
        proxy_full=lnk_rows(RAW/'mg_serpentine_proxy_composite.lnk')
        report['mg_serpentine_proxy']['at_2_7_um']={
            'full_n_k':[float(np.interp(2.7,proxy_full[:,0],proxy_full[:,i])) for i in (1,2)],
            'preview_interpolated_n_k':[float(np.interp(2.7,proxy_preview[:,0],proxy_preview[:,i])) for i in (1,2)]}
    # The previously supplied opcon already contains a full Titan-tholin table.
    try:
        existing=load_opcon(Path.home()/'Downloads'/'opcon','T')
        tholin=lnk_rows(RAW/'organics_tholin_khare1984.lnk')
        report['titan_tholin']['matches_legacy_opcon_T']=bool(
            np.array_equal(existing.wavelength,tholin[:,0]) and
            np.array_equal(existing.n,tholin[:,1]) and
            np.array_equal(existing.k,tholin[:,2]))
        legacy=np.c_[existing.wavelength,existing.n,existing.k]
        report['titan_tholin']['legacy_opcon_T_difference']={
            'rows_with_different_values':int(np.sum(np.any(legacy!=tholin,axis=1))),
            'max_absolute_k_difference':float(np.max(np.abs(legacy[:,2]-tholin[:,2])))}
    except (FileNotFoundError,ValueError):
        report['titan_tholin']['matches_legacy_opcon_T']=None
    for mineral in ('calcite','dolomite','magnesite'):
        path=RAW/f'roush2021_{mineral}.csv'
        original=np.loadtxt(path,delimiter=',',skiprows=1)
        constants=load_constants(path,columns=(0,2,3))
        report[f'PDS_Roush2021_{mineral}']={
            'rows':len(original),
            'wavelength_range_um':[float(constants.wavelength[0]),float(constants.wavelength[-1])],
            'has_n_and_k':True,
            'strictly_increasing_wavelength':bool(np.all(np.diff(original[:,0])>0)),
            'max_wavenumber_rounding_difference_cm_inverse':float(
                np.max(np.abs(original[:,1]-1e4/original[:,0])))}
    output=ROOT/'validation'/'optical_package_audit.json'
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
