"""Quantify the paper real-n and original IDL complex-index interface choices."""
import csv
from pathlib import Path
import numpy as np
from validation.local_data import opcon_directory
from shkuratov_model import Component, load_opcon, reflectance


def main():
    root=opcon_directory()
    output=Path(__file__).with_name('shkuratov_interface_modes.csv')
    cases=(('Gser',5.1,12.),('opx',.5,2.4),('cpx',.5,2.4),('olv',.5,2.4))
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(('code','wavelength_um','n','k','idl_complex','paper_real_n','absolute_difference'))
        for code,low,high in cases:
            constants=load_opcon(root,code)
            wavelengths=np.linspace(low,high,101)
            n,k=constants.sample(wavelengths)
            components=[Component(constants,1,30)]
            idl=reflectance(wavelengths,components,.3,interface_mode='idl_complex')
            paper=reflectance(wavelengths,components,.3,interface_mode='paper_real_n')
            for row in zip(wavelengths,n,k,idl,paper,abs(idl-paper)):
                writer.writerow((code,*row))
            index=int(np.argmax(abs(idl-paper)))
            print(f'{code}: max absolute difference {abs(idl-paper)[index]:.8g} at {wavelengths[index]:.5g} um')
    print(output)


if __name__=='__main__':
    main()
