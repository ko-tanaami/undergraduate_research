"""Compare the current Fresnel kernel with the public Vernazza IDL approximation.

This is a comparison of two model choices, not a claim that either is ground truth.
"""
import csv
from pathlib import Path
import numpy as np

from shkuratov_model import angular_integrals, albedo_from_indicatrix


def vernazza_coefficients(n):
    re = ((n - 1) / (n + 1)) ** 2 + .05
    ri = 1.04 - 1 / n**2
    rb = (.28*n - .2)*re
    return rb, re-rb, 1-re, ri, (1-re)/n**2


def conserving_approximation(n):
    rb,rf,te,_,ti=vernazza_coefficients(n)
    return rb,rf,te,1-ti,ti


def model_value(coefficients, k, diameter, wavelength, porosity=.3):
    rb, rf, te, ri, ti = coefficients
    attenuation = np.exp(-4*np.pi*k*diameter/wavelength)
    term = .5*te*ti*ri*attenuation**2/(1-ri*attenuation)
    return float(albedo_from_indicatrix(rb+term, rf+te*ti*attenuation+term, porosity))


def main():
    cases = [(n,k) for n in (1.3,1.5,1.7,2.) for k in (0.,.001,.01,.1,1.)]
    output=Path(__file__).with_name('shkuratov_kernel_comparison.csv')
    with output.open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(('n','k','Ri_plus_Ti_fresnel','Ri_plus_Ti_ver_n_only','reflectance_fresnel','reflectance_ver_n_only','reflectance_conserving_approx','difference','status'))
        for n,k in cases:
            fresnel=angular_integrals(n,k,1.)
            empirical=vernazza_coefficients(n)
            conserving=conserving_approximation(n)
            actual=model_value(fresnel,k,10.,1.)
            conserved=model_value(conserving,k,10.,1.)
            try:
                legacy=model_value(empirical,k,10.,1.)
                difference=actual-legacy
                status='valid'
            except ValueError as error:
                legacy=difference=float('nan')
                status=str(error)
            writer.writerow((n,k,fresnel[3]+fresnel[4],empirical[3]+empirical[4],actual,legacy,conserved,difference,status))
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
