"""Corrected numerical port of shkur_wid.pro. See SHKURATOV_README.md.

Forward calculations require NumPy; least-squares fitting also requires SciPy.
Wavelengths and lengths are in micrometres.
Coherent coatings and dilute inclusions are explicit experimental options.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from functools import lru_cache
import numpy as np


def positive(x, name, zero=False):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x < 0 if zero else x <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return x


@lru_cache(maxsize=1)
def _gauss8():
    return np.polynomial.legendre.leggauss(8)


def gaussian_grid(center, sigma, knots):
    """Resolve source breakpoints before integrating a truncated Gaussian."""
    low, high = center-4*sigma, center+4*sigma
    knots = np.asarray(knots)
    boundaries = np.unique(np.r_[np.linspace(low, high, 17), knots[(knots>low)&(knots<high)]])
    x, weights = _gauss8()
    left, right = boundaries[:-1], boundaries[1:]
    half = (right-left)/2
    grid = (left[:,None]+half[:,None]*(x+1)).ravel()
    integration_weights = (half[:,None]*weights).ravel()
    kernel = np.exp(-.5*((grid-center)/sigma)**2)*integration_weights
    return grid, kernel/kernel.sum()


@dataclass
class OpticalConstants:
    wavelength: np.ndarray
    n: np.ndarray
    k: np.ndarray
    gaps: tuple[tuple[float, float], ...] = ()
    interpolation: str = 'linear'

    def __post_init__(self):
        self.wavelength = positive(self.wavelength, 'wavelength')
        self.n = positive(self.n, 'n')
        self.k = positive(self.k, 'k', True)
        if self.wavelength.ndim != 1 or len(self.wavelength) < 2 or not (self.wavelength.shape == self.n.shape == self.k.shape):
            raise ValueError('Need matching 1D arrays with at least two rows')
        order = np.argsort(self.wavelength)
        self.wavelength, self.n, self.k = (x[order] for x in (self.wavelength, self.n, self.k))
        if np.any(np.diff(self.wavelength) <= 0):
            raise ValueError('Duplicate wavelengths: resolve explicitly in the source data')
        if self.interpolation not in ('linear', 'log_wavelength_log_k'):
            raise ValueError('Unknown optical-constant interpolation')
        if self.interpolation == 'log_wavelength_log_k' and np.any(self.k <= 0):
            raise ValueError('Logarithmic k interpolation requires positive k at every row')

    def _interpolate(self, w):
        if self.interpolation == 'linear':
            return np.interp(w, self.wavelength, self.n), np.interp(w, self.wavelength, self.k)
        log_w = np.log(w)
        log_source = np.log(self.wavelength)
        return np.interp(log_w, log_source, self.n), np.exp(np.interp(log_w, log_source, np.log(self.k)))

    def sample(self, wavelength, fwhm=None):
        w = positive(wavelength, 'requested wavelength')
        if w.ndim != 1 or len(w) == 0 or np.any(np.diff(w) <= 0):
            raise ValueError('Requested wavelengths must be strictly increasing')
        if w[0] < self.wavelength[0] or w[-1] > self.wavelength[-1]:
            raise ValueError('Requested range exceeds measured optical constants; extrapolation disabled')
        for left,right in self.gaps:
            if np.any((w>left)&(w<right)):
                raise ValueError(f'Requested wavelength crosses a documented missing-data interval ({left}, {right})')
        if fwhm is None:
            return self._interpolate(w)
        widths = np.broadcast_to(positive(fwhm, 'FWHM'), w.shape)
        sigma = widths / np.sqrt(8*np.log(2))
        for left,right in self.gaps:
            if np.any((w+4*sigma>left)&(w-4*sigma<right)):
                raise ValueError('Gaussian window crosses a documented missing-data interval')
        if np.any(w-4*sigma < self.wavelength[0]) or np.any(w+4*sigma > self.wavelength[-1]):
            raise ValueError('Gaussian window extends beyond measured constants')
        out = []
        for center, sig in zip(w, sigma):
            grid, weight = gaussian_grid(center, sig, self.wavelength)
            n_grid, k_grid = self._interpolate(grid)
            out.append([np.sum(x*weight) for x in (n_grid, k_grid)])
        return np.asarray(out).T


def numeric_table(path, columns):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig', errors='replace').splitlines(), 1):
        line = line.split(';')[0].strip()
        if not line or line.startswith('#'):
            continue
        words = line.replace(',', ' ').split()
        try:
            float(words[0].replace('D', 'E').replace('d', 'e'))
        except ValueError:
            if rows:
                raise ValueError(f'{path}:{line_no}: unexpected nonnumeric row')
            continue
        expected = len(rows[0]) if columns is None and rows else columns
        if expected is not None and len(words) != expected:
            raise ValueError(f'{path}:{line_no}: expected {expected} columns')
        rows.append([float(s.replace('D', 'E').replace('d', 'e')) for s in words])
    if len(rows) < 2:
        raise ValueError(f'{path}: insufficient data')
    return np.asarray(rows)


def load_observations(path, columns=(0, 1, 2), wavelength_range=None):
    """Read wavelength, reflectance and 1-sigma error from a numeric table.

    Columns are zero-based; extra columns are retained in the source file but
    are not silently interpreted as independent measurements.
    """
    table = numeric_table(path, None)
    if len(columns) != 3 or any(not isinstance(i, int) or i < 0 for i in columns) or len(set(columns)) != 3:
        raise ValueError('Observation columns must be three distinct zero-based indices')
    if max(columns) >= table.shape[1]:
        raise ValueError('Observation column index exceeds table width')
    selected = table[:, columns]
    if wavelength_range is not None:
        if len(wavelength_range) != 2:
            raise ValueError('Observation wavelength_range needs two endpoints')
        low, high = map(float, wavelength_range)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError('Invalid observation wavelength_range')
        selected = selected[(selected[:, 0] >= low) & (selected[:, 0] <= high)]
    if len(selected) < 2 or not np.all(np.isfinite(selected)) or np.any(selected[:, 0] <= 0) or np.any(np.diff(selected[:, 0]) <= 0) or np.any(selected[:, 2] <= 0):
        raise ValueError('Observations need at least two finite, increasing wavelengths and positive uncertainties')
    return selected


def load_constants(path, interpolation='linear', columns=(0,1,2)):
    table=numeric_table(path,None)
    if len(columns)!=3 or any(not isinstance(i,int) or i<0 for i in columns) or len(set(columns))!=3 or max(columns)>=table.shape[1]:
        raise ValueError('Optical constant columns must be three distinct valid zero-based indices')
    return OpticalConstants(*table[:,columns].T,interpolation=interpolation)


def load_optool(path):
    """Read an optool .lnk file; first numeric row is count and density."""
    rows=[]
    for line in Path(path).read_text(encoding='utf-8-sig',errors='replace').splitlines():
        stripped=line.strip()
        if stripped and not stripped.startswith('#'):
            rows.append(stripped.split())
    if not rows or len(rows[0])!=2 or len(rows)<3:
        raise ValueError('Invalid optool header')
    count=float(rows[0][0])
    density=float(rows[0][1])
    if not count.is_integer() or count!=len(rows)-1 or not np.isfinite(density) or density<=0:
        raise ValueError('Optool row count or density is invalid')
    if any(len(row)!=3 for row in rows[1:]):
        raise ValueError('Optool rows must contain wavelength,n,k')
    return OpticalConstants(*np.asarray(rows[1:],float).T)


def load_refractiveindex_yml(path):
    """Read a refractiveindex.info YAML file containing one tabulated n,k block."""
    content=Path(path).read_text(encoding='utf-8-sig')
    if content.count('type: tabulated nk')!=1:
        raise ValueError('Expected one tabulated n,k block in refractiveindex.info YAML')
    number=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'
    row_pattern=re.compile(rf'^\s*({number})\s+({number})\s+({number})\s*$')
    rows=[]
    for line in content.splitlines():
        match=row_pattern.fullmatch(line)
        if match:
            rows.append([float(value) for value in match.groups()])
    if len(rows)<2:
        raise ValueError('Insufficient tabulated n,k rows in YAML')
    return OpticalConstants(*np.asarray(rows).T)


def load_mooney1985(path, mineral):
    """Read the five-column Mooney & Knacke table (cm^-1, n, 1000k, n, 1000k).

    Zero entries in the legacy transcription are missing data, not physical
    optical constants. Adjacent measured rows bordering such entries define
    a gap that cannot be crossed by interpolation.
    """
    if mineral not in ('chlorite', 'serpentine'):
        raise ValueError('Mooney 1985 mineral must be chlorite or serpentine')
    table=numeric_table(path,5)
    # Verified directly against Table II in the user-supplied 1985 paper.
    # The group-supplied transcription has two wavenumber typos; leave the
    # source file untouched and correct only these exact row fingerprints.
    for wrong,correct,values in (
        (3430.,3440.,[1.537,27.,1.512,12.7]),
        (2299.,2290.,[1.517,2.75,0.,0.])):
        matches=np.flatnonzero((table[:,0]==wrong)&np.all(table[:,1:]==values,axis=1))
        if len(matches)==1:
            table[matches[0],0]=correct
    wavenumber=table[:,0]
    if np.any(~np.isfinite(wavenumber)) or np.any(wavenumber<=0) or np.any(np.diff(wavenumber)>=0):
        raise ValueError('Mooney 1985 wavenumbers must be positive and strictly decreasing')
    n_col,k_col=(1,2) if mineral=='chlorite' else (3,4)
    n,k=table[:,n_col],table[:,k_col]
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(k)) or np.any(n<0) or np.any(k<0):
        raise ValueError('Mooney 1985 optical constants must be finite and nonnegative')
    valid=(n>0)&(k>0)
    indices=np.flatnonzero(valid)
    if len(indices)<2:
        raise ValueError('Insufficient Mooney 1985 data')
    wavelength=1e4/wavenumber
    gaps=tuple((float(wavelength[a]),float(wavelength[b]))
               for a,b in zip(indices[:-1],indices[1:])
               if b>a+1 or wavenumber[a]-wavenumber[b]>10.0001)
    return OpticalConstants(wavelength[valid],n[valid],k[valid]*1e-3,gaps)


def load_opcon(directory, code, mg=75):
    """Original material naming/units; Mg formulas corrected from file headers."""
    root = Path(directory)
    def table(path, columns):
        # IDL material codes are case-sensitive; Windows filenames are not.
        # Never let material 'a' silently read ammonia file 'A_k'.
        if path.exists() and path.name not in {p.name for p in path.parent.iterdir()}:
            raise ValueError(f'Material filename case mismatch: {path.name}; refusing a different material')
        return numeric_table(path, columns)
    lucey = {'cpx': ('cpx_kfunc', 1.726, .00082, 65, 95),
             'opx': ('opx_kfunc', 1.768, .00118, 40, 91),
             'olv': ('ol_kfunc', 1.827, .00192, 0, 93)}
    if code in lucey:
        filename, a, b, low, high = lucey[code]
        if not np.isfinite(mg) or not low <= mg <= high:
            raise ValueError(f'Mg# for {code} must lie in [{low}, {high}]')
        w, c, slope = table(root/filename, 3).T
        return OpticalConstants(w, np.full_like(w, a-b*mg), c+mg*slope)
    bob = {'co': '12co_1-5um_opcon', 'ch4': '12ch4_1-5um_opcon',
           'h2o': 'h2o_1-5um_opcon', 'm90': 'ch3oh_90k_opcon', 'm120': 'ch3oh_120k_opcon'}
    if code in bob:
        w, alpha, n = table(root/bob[code], 3).T
        return OpticalConstants(w, n, alpha*w*1e-4/(4*np.pi))
    path = root/('trit_opcon' if code == 'Tr' else code+'_nk')
    if code.startswith(('amorph_', 'crys_')):
        path = root/'mastrapa_h2o'/(code+'.txt')
    if path.exists():
        w, n, k = table(path, 3).T
        grund = {'n2-37','n2-38','n2-41','h2o-40'} | {f'ch4-{v}' for v in (20,30,40,50,60,70,80,90,93)}
        if code in {'Si','Sk','Ss','Sx','Sz'}:
            valid = k > -1e30
            # Header documents -1.23e34 as deleted points. Keep their gaps;
            # never interpolate through an unmeasured interval.
            order=np.argsort(w)
            w,n,k,valid=(a[order] for a in (w,n,k,valid))
            if not np.all(np.diff(w)>0):
                raise ValueError('Duplicate wavelengths in Clark data')
            ids=np.flatnonzero(valid)
            if len(ids)<2:
                raise ValueError('Insufficient valid Clark data')
            gaps=tuple((float(w[a]),float(w[b])) for a,b in zip(ids[:-1],ids[1:]) if b>a+1)
            w,n,k=w[valid],n[valid],k[valid]
            k = k*w*1e-4/(4*np.pi)
            return OpticalConstants(w,n,k,gaps)
        elif code in grund:
            k = k*w*1e-4/(4*np.pi)
        return OpticalConstants(w,n,k)
    wn,n = table(root/(code+'_n'), 2).T
    wk,k = table(root/(code+'_k'), 2).T
    if code == 'asphalt':
        wn, wk = 1e4/wn, 1e4/wk
    # Some legacy tables include a lambda=0 placeholder outside every
    # physically usable wavelength. It cannot enter the common grid.
    if np.any(wk<0):
        raise ValueError('Negative wavelength in k table')
    wk,k=wk[wk>0],k[wk>0]
    if len(wk)<2:
        raise ValueError('Insufficient positive wavelengths in k table')
    def ordered(w, v):
        order=np.argsort(w)
        w,v=w[order],v[order]
        positive(w,'wavelength')
        unique,first,count=np.unique(w,return_index=True,return_counts=True)
        for start,length in zip(first[count>1],count[count>1]):
            if np.any(v[start:start+length]!=v[start]):
                raise ValueError('Conflicting values at duplicate wavelengths')
        w,v=unique,v[first]
        return w,v
    wn,n=ordered(wn,n)
    wk,k=ordered(wk,k)
    positive(n,'n')
    positive(k,'k',True)
    low,high=max(wn[0],wk[0]),min(wn[-1],wk[-1])
    grid=np.unique(np.r_[wn[(wn>=low)&(wn<=high)],wk[(wk>=low)&(wk<=high)]])
    return OpticalConstants(grid,np.interp(grid,wn,n),np.interp(grid,wk,k))


def dielectric(n, k):
    return (positive(n,'n')+1j*positive(k,'k',True))**2


def inclusion_k(n, k, ni, ki, fraction, *, experimental=False):
    """Shkuratov (1999) Eq. 15; dilute, fine inclusions only.

    For an absorbing host this does not equal the general complex
    Maxwell-Garnett first-order increment and is not separately validated.
    """
    if not experimental:
        raise ValueError('Dilute-inclusion approximation requires experimental=True; no universal concentration limit is established')
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('Inclusion fraction must lie in [0,1)')
    eps = dielectric(ni,ki)/dielectric(n,k)
    out = np.asarray(k)+1.5*fraction*np.asarray(n)*np.imag((eps-1)/(eps+2))
    return positive(out,'effective k',True)


def interface_rt(n0, n1, theta):
    """Fresnel fluxes. Absorbing incident medium retains the original ray convention."""
    c0=np.cos(theta)
    if n0 == n1:
        return np.zeros_like(c0),np.ones_like(c0)
    c1=np.sqrt(1-(n0*np.sin(theta)/n1)**2+0j)
    z=n1*c1
    if np.imag(n0)==0:
        c1=np.where((z.imag < -1e-14) | ((abs(z.imag)<1e-14)&(z.real<0)), -c1,c1)
    rs=(n0*c0-n1*c1)/(n0*c0+n1*c1)
    rp=(n1*c0-n0*c1)/(n1*c0+n0*c1)
    ts=2*n0*c0/(n0*c0+n1*c1)
    tp=2*n0*c0/(n1*c0+n0*c1)
    r=(abs(rs)**2+abs(rp)**2)/2
    t=(abs(ts)**2*np.real(n1*c1)/np.real(n0*c0)+abs(tp)**2*np.real(n1*np.conj(c1))/np.real(n0*np.conj(c0)))/2
    return r,t


def film_rt(n0, film, n1, theta, thickness, wavelength):
    """Coherent planar film; incident medium must be nonabsorbing."""
    for index in (n0,film,n1):
        if not np.isfinite(index) or np.real(index)<=0 or np.imag(index)<0:
            raise ValueError('Film refractive indices must be finite, n>0 and k>=0')
    theta=np.asarray(theta,float)
    if np.any(~np.isfinite(theta)) or np.any(theta<0) or np.any(theta>=np.pi/2):
        raise ValueError('Incidence angle must lie in [0,pi/2)')
    if np.any(np.imag(n0)!=0):
        raise ValueError('Absorbing incident core for coating is not validated; use an uncoated model')
    positive(thickness,'thickness',True)
    positive(wavelength,'wavelength')
    if thickness == 0:
        return interface_rt(n0,n1,theta)
    if film==n1 and np.imag(film)==0:
        return interface_rt(n0,n1,theta)
    c0=np.cos(theta)
    c2=np.sqrt(1-(n0*np.sin(theta)/film)**2+0j)
    c3=np.sqrt(1-(n0*np.sin(theta)/n1)**2+0j)
    phase=np.exp(2j*np.pi*film*c2*thickness/wavelength)
    results=[]
    for pol in ('s','p'):
        def pair(a,b,ca,cb):
            if pol=='s':
                return (a*ca-b*cb)/(a*ca+b*cb), 2*a*ca/(a*ca+b*cb)
            return (b*ca-a*cb)/(b*ca+a*cb),2*a*ca/(b*ca+a*cb)
        r12,t12=pair(n0,film,c0,c2)
        r23,t23=pair(film,n1,c2,c3)
        den=1+r12*r23*phase**2
        with np.errstate(invalid='ignore',divide='ignore'):
            r=(r12+r23*phase**2)/den
            t=t12*t23*phase/den
        flux=np.real(n1*(c3 if pol=='s' else np.conj(c3)))/np.real(n0*c0)
        R,T=abs(r)**2,abs(t)**2*flux
        # At zero normal wavevector in the film, amplitude recursion is 0/0.
        # Characteristic matrix has a finite sin(delta)/kz limit instead.
        special=abs(c2)<1e-6
        if np.any(special):
            z=film*c2
            length=2*np.pi*thickness/wavelength
            delta=length*z
            sin_over_z=length*np.sinc(delta/np.pi)
            a=np.cos(delta)
            if pol=='s':
                eta0=n0*c0
                etas=n1*c3
                b=-1j*sin_over_z
                cc=-1j*z*z*sin_over_z
            else:
                eta0=n0/c0
                etas=n1/c3
                b=-1j*z*z*sin_over_z/(film*film)
                cc=-1j*film*film*sin_over_z
            denom=eta0*a+eta0*etas*b+cc+etas*a
            rr=(eta0*a+eta0*etas*b-cc-etas*a)/denom
            tt=2*eta0/denom
            R=np.where(special,abs(rr)**2,R)
            T=np.where(special,abs(tt)**2*np.real(etas)/np.real(eta0),T)
        results.append((R,T))
    return tuple((results[0][j]+results[1][j])/2 for j in (0,1))


@lru_cache(maxsize=16)
def nodes(order):
    if order < 16:
        raise ValueError('At least 16 quadrature nodes required')
    return np.polynomial.legendre.leggauss(order)


def angular_integrals(n,k,wavelength,coat=None,thickness=0,order=256,interface_mode='idl_complex'):
    n=float(positive(n,'n'))
    k=float(positive(k,'k',True))
    wavelength=float(positive(wavelength,'wavelength'))
    if interface_mode not in ('idl_complex','paper_real_n'):
        raise ValueError('interface_mode must be idl_complex or paper_real_n')
    if interface_mode=='paper_real_n' and n<1:
        raise ValueError('paper_real_n requires n>=1; the paper uses a critical angle arcsin(1/n)')
    if interface_mode=='paper_real_n' and coat is not None and thickness>0:
        raise ValueError('paper_real_n applies only to uncoated particles')
    m=n+1j*(k if interface_mode=='idl_complex' else 0.)
    x,weight=nodes(order)
    def integrate(a,b,internal=False):
        g=(x+1)*(b-a)/2+a
        if coat is None or thickness==0:
            r,t=interface_rt(m if internal else 1.,1. if internal else m,g)
        else:
            r,t=film_rt(m if internal else 1.,coat,1. if internal else m,g,thickness,wavelength)
        wg=weight*(b-a)*np.sin(g)*np.cos(g)
        return np.sum(wg*r),np.sum(wg*t)
    rb,tb=integrate(0,np.pi/4)
    rf,tf=integrate(np.pi/4,np.pi/2)
    # Split near the critical transition, also for weakly absorbing media.
    if n>1:
        critical=np.arcsin(1/n)
        a=integrate(0,critical,True)
        b=integrate(critical,np.pi/2,True)
        ri,ti=a[0]+b[0],a[1]+b[1]
    else:
        ri,ti=integrate(0,np.pi/2,True)
    if coat is None or thickness==0:
        # Shkuratov et al. (1999), Eq. (7a): at a bare interface each
        # encounter either reflects or transmits. The absorbing-incident
        # Poynting ratio is not a particle-transport probability.
        return rb,rf,1-rb-rf,ri,1-ri
    return rb,rf,tb+tf,ri,ti


@dataclass
class Component:
    constants: OpticalConstants
    weight: float
    diameter: float
    coating: OpticalConstants | None = None
    thickness: float = 0
    inclusion: OpticalConstants | None = None
    inclusion_fraction: float = 0


def reflectance(wavelength, components, porosity=0., order=256, experimental=False, fwhm=None, interface_mode='idl_complex', instrument_fwhm=None):
    w=positive(wavelength,'wavelength')
    if instrument_fwhm is not None:
        if fwhm is not None:
            raise ValueError('Index smoothing and instrument convolution cannot be combined')
        if w.ndim != 1 or len(w) == 0 or np.any(np.diff(w) <= 0):
            raise ValueError('Requested wavelengths must be strictly increasing')
        widths=np.broadcast_to(positive(instrument_fwhm,'instrument FWHM'),w.shape)
        result=[]
        for center,width in zip(w,widths):
            sigma=width/np.sqrt(8*np.log(2))
            nodes=[]
            for component in components:
                for constants in (component.constants,component.inclusion,component.coating):
                    if constants is not None:
                        nodes.append(constants.wavelength)
            grid, weight=gaussian_grid(center,sigma,np.concatenate(nodes) if nodes else np.array([]))
            spectrum=reflectance(grid,components,porosity,order,experimental,None,interface_mode)
            result.append(np.sum(spectrum*weight))
        return np.asarray(result)
    if not components:
        raise ValueError('At least one component is required')
    weights=positive([c.weight for c in components],'weights',True)
    if not np.isclose(weights.sum(),1,rtol=0,atol=1e-10):
        raise ValueError('Weights must sum to one')
    if not np.isfinite(porosity) or not 0<=porosity<1:
        raise ValueError('Porosity must lie in [0,1)')
    if interface_mode not in ('idl_complex','paper_real_n'):
        raise ValueError('interface_mode must be idl_complex or paper_real_n')
    back=np.zeros_like(w)
    forward=np.zeros_like(w)
    for c in components:
        positive(c.diameter,'diameter')
        positive(c.thickness,'thickness',True)
        if np.any(c.diameter<=w):
            raise ValueError('Effective path length must exceed every requested wavelength; geometric optics requires it to be much greater')
        if c.thickness>=c.diameter:
            raise ValueError('Coating thickness must be smaller than diameter')
        if c.coating is None and c.thickness!=0:
            raise ValueError('Coating constants are required')
        if c.inclusion is None and c.inclusion_fraction!=0:
            raise ValueError('Inclusion constants are required')
        n,k=c.constants.sample(w,fwhm)
        if c.inclusion is not None:
            ni,ki=c.inclusion.sample(w,fwhm)
            k=inclusion_k(n,k,ni,ki,c.inclusion_fraction,experimental=experimental)
        coat=None
        if c.coating is not None and c.thickness>0:
            if not experimental:
                raise ValueError('Coherent coating extension requires experimental=True')
            nc,kc=c.coating.sample(w,fwhm)
            coat=nc+1j*kc
        for j,lam in enumerate(w):
            rb,rf,te,ri,ti=angular_integrals(n[j],k[j],lam,None if coat is None else coat[j],c.thickness,order,interface_mode)
            attenuation=np.exp(-4*np.pi*k[j]*c.diameter/lam)
            denominator=1-ri*attenuation
            if denominator<=0:
                raise ValueError('Singular internal reflection series')
            term=.5*te*ti*ri*attenuation**2/denominator
            back[j]+=c.weight*(rb+term)
            forward[j]+=c.weight*(rf+te*ti*attenuation+term)
    return albedo_from_indicatrix(back,forward,porosity)


def albedo_from_indicatrix(back, forward, porosity=0.):
    """Shkuratov closure, without cancellation at porosity approaching one.

    A=(1+pb**2-pf**2)/(2*pb); factor out q=1-porosity BEFORE
    evaluating the numerator. The q=0 material-free case remains excluded.
    """
    b,f=np.broadcast_arrays(positive(back,'backward fraction',True),positive(forward,'forward fraction',True))
    if not np.isfinite(porosity) or not 0<=porosity<1:
        raise ValueError('Porosity must lie in [0,1)')
    # rb/rf represent fractions of scattered energy, not arbitrary coefficients.
    if np.any(b+f>1+1e-9):
        raise ValueError('Scattered energy exceeds incident energy; interface approximation invalid')
    result=np.zeros_like(b)
    active=b>0
    loss_forward=1-f[active]
    q=1-porosity
    # Compute 1/A directly to avoid overflow for extremely small b.
    numerator=2*loss_forward+q*(b[active]-loss_forward)*(b[active]+loss_forward)
    inv_a=2*b[active]/numerator
    if np.any(~np.isfinite(inv_a)) or np.any(inv_a<0) or np.any(inv_a>1+1e-9):
        raise ValueError('Nonphysical scattering closure')
    inv_a=np.minimum(inv_a,1)
    result[active]=inv_a/(1+np.sqrt((1-inv_a)*(1+inv_a)))
    return result


def chi_square(observed, predicted, error, fitted_parameters=None):
    y,p,e=(np.asarray(x,float) for x in (observed,predicted,error))
    if y.ndim!=1 or not y.shape==p.shape==e.shape:
        raise ValueError('Data shapes differ')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError('Nonfinite spectrum')
    positive(e,'uncertainties')
    chi=float(np.sum(((y-p)/e)**2))
    if fitted_parameters is None:
        return chi,None
    if not isinstance(fitted_parameters,int) or not 0<=fitted_parameters<len(y):
        raise ValueError('Invalid degrees of freedom')
    return chi,chi/(len(y)-fitted_parameters)


def fit_reflectance(wavelength, observed, error, components, parameter_specs, *,
                    fit_weights=False, porosity=0., order=256, experimental=False,
                    fwhm=None, interface_mode='idl_complex', instrument_fwhm=None,
                    max_nfev=200, normalize_at=None):
    """Bounded weighted least squares, with a simplex for mixture weights.

    Each parameter spec names ``porosity`` or ``diameter`` plus a component
    index for the latter. The solver is local; a successful stop is not a
    uniqueness or physical-validity certificate.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError('Fitting requires SciPy; install scipy in the Python environment') from exc
    w=positive(wavelength,'wavelength')
    y=np.asarray(observed,float)
    e=positive(error,'uncertainties')
    if w.ndim!=1 or y.shape!=w.shape or e.shape!=w.shape or len(w)<2 or np.any(np.diff(w)<=0) or not np.all(np.isfinite(y)):
        raise ValueError('Fitting requires matching finite 1D observations at increasing wavelengths')
    if normalize_at is not None:
        normalize_at=float(positive(normalize_at,'normalization wavelength'))
        if not w[0]<=normalize_at<=w[-1]:
            raise ValueError('Normalization wavelength must lie within observations')
        model_w=np.union1d(w,[normalize_at])
        ref_index=int(np.searchsorted(model_w,normalize_at))
    else:
        model_w=w
    if not components or not isinstance(parameter_specs,list):
        raise ValueError('Provide components and a list of fit parameters')
    if not isinstance(max_nfev,int) or max_nfev<=0:
        raise ValueError('max_nfev must be a positive integer')
    if fit_weights and len(components)<2:
        raise ValueError('Fitting mixture weights requires at least two components')
    base_weights=positive([c.weight for c in components],'weights',True)
    if not np.isclose(base_weights.sum(),1,atol=1e-10,rtol=0):
        raise ValueError('Weights must sum to one')
    if fit_weights and np.any(base_weights<=0):
        raise ValueError('Weight fitting requires positive starting weights')
    names=[]
    x0=[]
    lower=[]
    upper=[]
    for spec in parameter_specs:
        if not isinstance(spec,dict) or spec.get('name') not in ('porosity','diameter'):
            raise ValueError('Fit parameter must name porosity or diameter')
        name=spec['name']
        if name=='porosity':
            if 'component' in spec:
                raise ValueError('Porosity has no component index')
            key='porosity'
            initial=porosity
            physical_low,physical_high=0.,np.nextafter(1.,0.)
        else:
            index=spec.get('component')
            if not isinstance(index,int) or not 0<=index<len(components):
                raise ValueError('Invalid fitted component index')
            key=f'components.{index}.diameter'
            initial=components[index].diameter
            physical_low=np.nextafter(float(model_w[-1]),np.inf)
            physical_low=max(physical_low,np.nextafter(float(components[index].thickness),np.inf))
            physical_high=np.inf
        if key in names:
            raise ValueError(f'Duplicate fit parameter: {key}')
        bounds=spec.get('bounds',[physical_low,physical_high])
        if not isinstance(bounds,(list,tuple)) or len(bounds)!=2:
            raise ValueError('Parameter bounds must contain lower and upper values')
        lo,hi=(float(value) for value in bounds)
        if np.isnan(lo) or np.isnan(hi) or lo<physical_low or hi>physical_high or lo>=hi:
            raise ValueError(f'Invalid bounds for {key}')
        if not np.isfinite(initial) or not lo<=initial<=hi:
            raise ValueError(f'Initial {key} lies outside fitted bounds')
        names.append(key)
        x0.append(initial)
        lower.append(lo)
        upper.append(hi)
    count_named=len(names)
    if fit_weights:
        ratios=np.log(base_weights[:-1]/base_weights[-1])
        x0.extend(ratios)
        lower.extend([-np.inf]*len(ratios))
        upper.extend([np.inf]*len(ratios))
        names.extend(f'weight_logratio_{i}_to_{len(components)-1}' for i in range(len(ratios)))
    if not names or len(names)>=len(w):
        raise ValueError('Fit requires at least one free parameter and more observations than parameters')

    def evaluate(parameters):
        model_components=[replace(component) for component in components]
        model_porosity=porosity
        for key,value in zip(names[:count_named],parameters[:count_named]):
            if key=='porosity':
                model_porosity=float(value)
            else:
                index=int(key.split('.')[1])
                model_components[index].diameter=float(value)
        if fit_weights:
            logits=np.r_[parameters[count_named:],0.]
            probabilities=np.exp(logits-logits.max())
            probabilities/=probabilities.sum()
            for component,weight in zip(model_components,probabilities):
                component.weight=float(weight)
        model=reflectance(model_w,model_components,model_porosity,order,experimental,fwhm,
                          interface_mode,instrument_fwhm)
        if normalize_at is not None:
            reference=model[ref_index]
            if not np.isfinite(reference) or reference<=0:
                raise ValueError('Cannot normalize a nonpositive model reflectance')
            model=model/reference
            model=np.delete(model,ref_index) if model_w.size>w.size else model
        return model,model_components,model_porosity

    def residual(parameters):
        model,_,_=evaluate(parameters)
        return (model-y)/e

    solution=least_squares(residual,np.asarray(x0,float),bounds=(lower,upper),
                           method='trf',loss='linear',max_nfev=max_nfev,x_scale='jac')
    if not solution.success:
        raise RuntimeError(f'Least-squares fit did not converge: {solution.message}')
    model,fitted_components,fitted_porosity=evaluate(solution.x)
    chi,reduced=chi_square(y,model,e,len(names))
    # Solver finite differences can label identical components as independent
    # because roundoff is amplified by their ~sqrt(eps) step. Diagnose with
    # a wider finite difference; this is only a local numerical rank check.
    diagnostic_jac=np.empty((len(w),len(names)))
    for j,value in enumerate(solution.x):
        step=1e-4*max(1.,abs(value))
        lo=max(lower[j],value-step)
        hi=min(upper[j],value+step)
        left=solution.x.copy()
        right=solution.x.copy()
        left[j]=lo
        right[j]=hi
        diagnostic_jac[:,j]=(residual(right)-residual(left))/(hi-lo)
    singular=np.linalg.svd(diagnostic_jac,compute_uv=False)
    rank=int(np.sum(singular>max(1e-8,singular[0]*max(diagnostic_jac.shape)*np.finfo(float).eps*100))) if len(singular) else 0
    report={'method':'scipy.optimize.least_squares:trf', 'success':bool(solution.success),
            'message':str(solution.message),'free_parameters':len(names),
            'parameter_names':names,'optimizer_values':[float(v) for v in solution.x],
            'porosity':float(fitted_porosity),
            'weights':[float(c.weight) for c in fitted_components],
            'diameters_um':[float(c.diameter) for c in fitted_components],
            'chi_square':chi,'reduced_chi_square':reduced,'jacobian_rank':rank,
            'jacobian_full_rank':rank==len(names),
            'normalization_wavelength_um':normalize_at,
            'nfev':int(solution.nfev)}
    return model,fitted_components,fitted_porosity,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path,help='JSON configuration; relative paths use its directory')
    parser.add_argument('--output',type=Path,default=Path('shkuratov_result.csv'))
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    base=args.config.resolve().parent
    def constants(spec):
        if 'path' in spec:
            return load_constants(base/spec['path'],spec.get('interpolation','linear'),
                                  spec.get('columns',(0,1,2)))
        if 'optool_path' in spec:
            if 'interpolation' in spec or 'columns' in spec:
                raise ValueError('Optool tables use source-specific three-column data')
            return load_optool(base/spec['optool_path'])
        if 'refractiveindex_yml_path' in spec:
            if 'interpolation' in spec or 'columns' in spec:
                raise ValueError('Refractiveindex.info YAML uses source-specific tabulation')
            return load_refractiveindex_yml(base/spec['refractiveindex_yml_path'])
        if 'mooney1985_path' in spec:
            if 'interpolation' in spec:
                raise ValueError('Mooney 1985 table uses source-specific linear interpolation')
            return load_mooney1985(base/spec['mooney1985_path'],spec['mineral'])
        if 'interpolation' in spec:
            raise ValueError('Interpolation override requires a three-column path; legacy opcon modes are source-specific')
        return load_opcon(base/spec['opcon_dir'],spec['code'],spec.get('mg',75))
    comps=[]
    for spec in cfg['components']:
        comps.append(Component(constants(spec),spec['weight'],spec['diameter'],
            constants(spec['coating']) if 'coating' in spec else None,spec.get('thickness',0),
            constants(spec['inclusion']) if 'inclusion' in spec else None,spec.get('inclusion_fraction',0)))
    observation=None
    if 'observations' in cfg:
        obs_spec=cfg['observations']
        if isinstance(obs_spec,str):
            observation=load_observations(base/obs_spec)
        elif isinstance(obs_spec,dict):
            observation=load_observations(base/obs_spec['path'],
                                          obs_spec.get('columns',(0,1,2)),
                                          obs_spec.get('wavelength_range'))
        else:
            raise ValueError('observations must be a path or an object')
        w=observation[:,0]
    else:
        w=np.linspace(*cfg['wavelength_range'],cfg.get('points',201))
    interface_mode=cfg.get('interface_mode','idl_complex')
    fit_report=None
    if 'fit' in cfg:
        if observation is None:
            raise ValueError('Fitting requires an observations file')
        fit=cfg['fit']
        if not isinstance(fit,dict):
            raise ValueError('fit must be an object')
        r,comps,fitted_porosity,fit_report=fit_reflectance(
            w,observation[:,1],observation[:,2],comps,fit.get('parameters',[]),
            fit_weights=fit.get('weights',False),porosity=cfg.get('porosity',0),
            order=cfg.get('quadrature_order',256),experimental=cfg.get('experimental',False),
            fwhm=cfg.get('fwhm'),interface_mode=interface_mode,
            instrument_fwhm=cfg.get('instrument_fwhm'),max_nfev=fit.get('max_nfev',200),
            normalize_at=fit.get('normalize_at'))
    else:
        fitted_porosity=cfg.get('porosity',0)
        r=reflectance(w,comps,fitted_porosity,cfg.get('quadrature_order',256),cfg.get('experimental',False),cfg.get('fwhm'),interface_mode,cfg.get('instrument_fwhm'))
    normalized=fit_report is not None and fit_report['normalization_wavelength_um'] is not None
    value_name='normalized_model_reflectance' if normalized else 'model_reflectance'
    np.savetxt(args.output,np.c_[w,r],delimiter=',',header=f'wavelength_um,{value_name}',comments='')
    quantity=('Shkuratov model reflectance normalized at the specified wavelength' if normalized
              else 'Shkuratov model reflectance; geometric-albedo equivalence unverified')
    metadata={'config':cfg,'numpy':np.__version__,'quantity':quantity,'idl_comparison':False,
              'bare_interface_transport':'Te=1-Re; Ti=1-Ri (Shkuratov et al. 1999 Eq. 7a)',
              'interface_mode':interface_mode,
              'spectral_smoothing':'reflectance_convolution' if 'instrument_fwhm' in cfg else ('optical_constants' if 'fwhm' in cfg else 'none'),
              'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if fit_report is not None:
        metadata['fit']=fit_report
    def provenance(constants):
        if constants is None:
            return None
        samples=np.c_[constants.wavelength,constants.n,constants.k].astype('<f8')
        gaps=[list(pair) for pair in constants.gaps]
        digest=hashlib.sha256(samples.tobytes()+json.dumps(gaps,separators=(',',':')).encode()).hexdigest()
        return {'numeric_constants_and_gaps_sha256':digest,'interpolation':constants.interpolation,'rows':len(constants.wavelength),
                'wavelength_range_um':[float(constants.wavelength[0]),float(constants.wavelength[-1])],
                'documented_missing_intervals':gaps}
    metadata['source_optical_constants']=[{'core':provenance(c.constants),'coating':provenance(c.coating),
                                           'inclusion':provenance(c.inclusion)} for c in comps]
    if observation is not None:
        parameter_count=fit_report['free_parameters'] if fit_report is not None else cfg.get('fitted_parameters')
        chi,reduced=chi_square(observation[:,1],r,observation[:,2],parameter_count)
        metadata.update(chi_square=chi,reduced_chi_square=reduced)
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'{args.output}: {len(w)} points, reflectance [{r.min():.8g}, {r.max():.8g}]')


if __name__=='__main__':
    main()
