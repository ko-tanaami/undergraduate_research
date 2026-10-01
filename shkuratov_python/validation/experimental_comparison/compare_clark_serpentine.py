"""Fixed-parameter comparison of legacy Clark HS318 and spliced proxy n,k."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from validation.local_data import opcon_directory
from shkuratov_model import Component, load_opcon, load_optool, reflectance
from validation.experimental_comparison.compare_usgs_spectra import read_usgs, band


def metrics(w, observed, modeled):
    residual = modeled - observed
    norm_obs = observed / np.interp(1., w, observed)
    norm_mod = modeled / np.interp(1., w, modeled)
    return {
        'rmse': float(np.sqrt(np.mean(residual**2))),
        'mae': float(np.mean(abs(residual))),
        'mean_signed_error': float(residual.mean()),
        'pearson_correlation': float(np.corrcoef(observed, modeled)[0, 1]),
        'normalized_at_1um_rmse': float(np.sqrt(np.mean((norm_mod-norm_obs)**2))),
        'reflectance_at_1um': float(np.interp(1., w, modeled)),
        'band_2p3um': band(w, modeled, 2.20, 2.48),
    }


def plot(curves, path):
    image = Image.new('RGB', (1500, 1000), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 22)
        small = ImageFont.truetype('DejaVuSans.ttf', 19)
    except OSError:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=19)
    colors = [(25, 65, 140), (185, 70, 35), (20, 130, 90)]
    for i, (title, w, series, ylabel) in enumerate(curves):
        x0, y0 = 95 + (i % 2)*740, 90 + (i//2)*450
        x1, y1 = x0+600, y0+320
        high = float(np.concatenate(series).max())*1.08
        xp = lambda x: x0+(x-.5)/2*(x1-x0)
        yp = lambda y: y1-y/high*(y1-y0)
        draw.text((x0, y0-65), title, font=font, fill='black')
        draw.text((x0, y0-35), ylabel, font=small, fill='black')
        for tick in (.5, 1., 1.5, 2., 2.5):
            x = xp(tick)
            draw.line((x,y0,x,y1), fill='#dddddd')
            draw.text((x-16,y1+8), str(tick), font=small, fill='black')
        for tick in np.linspace(0, high, 5):
            y = yp(tick)
            draw.line((x0,y,x1,y), fill='#dddddd')
            draw.text((x0-66,y-10), f'{tick:.2f}', font=small, fill='black')
        for values, color in zip(series, colors):
            # Break the line at the documented Clark gap.
            breaks = np.r_[0, np.flatnonzero(np.diff(w)>.012)+1, len(w)]
            for a,b in zip(breaks[:-1],breaks[1:]):
                if b-a>1:
                    draw.line([(xp(x),yp(y)) for x,y in zip(w[a:b],values[a:b])], fill=color,width=3)
        draw.rectangle((x0,y0,x1,y1), outline='black', width=2)
        draw.text((x0+195,y1+38), 'Wavelength (um)', font=small, fill='black')
    for i,label in enumerate(('USGS measured', 'Spliced proxy', 'Clark HS318 n,k')):
        x = 95+i*440
        draw.line((x,960,x+50,960),fill=colors[i],width=4)
        draw.text((x+60,947),label,font=font,fill='black')
    image.save(path)


def main():
    out = HERE/'clark_serpentine_comparison'
    out.mkdir(exist_ok=True)
    source = opcon_directory() / 'Sz_nk'
    clark = load_opcon(source.parent, 'Sz')
    proxy = load_optool(ROOT/'data/optical_constants_full/mg_serpentine_proxy_composite.lnk')
    report = {'settings': {'range_um':[.5,2.5], 'S_um':30., 'porosity':.3,
                          'angular_nodes':64, 'fit_performed':False,
                          'excluded_gaps_um':clark.gaps},
              'clark_original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'samples':{}}
    curves = []
    for sample, filename in (('HS8','serpentine_hs8.6201.asc'),
                             ('HS318','serpentine_hs318.6189.asc')):
        data = read_usgs(HERE/filename)
        mask = (data[:,0]>=.5)&(data[:,0]<=2.5)
        for left,right in clark.gaps:
            mask &= ~((data[:,0]>left)&(data[:,0]<right))
        w, obs, sigma = data[mask].T
        assert len(w)>300 and np.all(obs>=0)
        outputs = {}
        for name, constants in (('proxy',proxy),('clark',clark)):
            r = reflectance(w,[Component(constants,1.,30.)],porosity=.3,order=64)
            fine = reflectance(w,[Component(constants,1.,30.)],porosity=.3,order=128)
            assert np.all(np.isfinite(r)) and np.all((r>=0)&(r<=1))
            outputs[name] = r
            entry = metrics(w,obs,r)
            entry['order64_vs128_max_abs_difference'] = float(np.max(abs(r-fine)))
            report['samples'].setdefault(sample,{})[name] = entry
        series = [obs,outputs['proxy'],outputs['clark']]
        normalized = [r/np.interp(1.,w,r) for r in series]
        report['samples'][sample].update({
            'points':len(w), 'actual_range_um':[float(w[0]),float(w[-1])],
            'observed_at_1um':float(np.interp(1.,w,obs)),
            'observed_band_2p3um':band(w,obs,2.20,2.48),
            'rmse_reduction_percent':100*(1-report['samples'][sample]['clark']['rmse']/
                                        report['samples'][sample]['proxy']['rmse']),
        })
        np.savetxt(out/f'{sample.lower()}_comparison.csv',
                   np.column_stack((w,*series,sigma,*normalized)),delimiter=',',
                   header='wavelength_um,usgs_reflectance,proxy_shkuratov_reflectance,clark_shkuratov_reflectance,usgs_stddev,usgs_normalized_1um,proxy_normalized_1um,clark_normalized_1um',comments='')
        curves.extend([(sample+' - reflectance',w,series,'Reflectance R(lambda)'),
                       (sample+' - normalized shape',w,normalized,'R(lambda) / R(1 um)')])
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plot(curves,out/'comparison.png')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
