import unittest
import tempfile
import importlib.util
import json
from math import erf, sqrt, pi
from pathlib import Path
import numpy as np
from shkuratov_model import (
    Component,
    OpticalConstants,
    albedo_from_indicatrix,
    angular_integrals,
    chi_square,
    dielectric,
    film_rt,
    fit_reflectance,
    inclusion_k,
    interface_rt,
    load_constants,
    load_mooney1985,
    load_observations,
    load_opcon,
    load_optool,
    load_refractiveindex_yml,
    main,
    numeric_table,
    reflectance,
)


class Tests(unittest.TestCase):
    def test_four_column_pds_and_optool_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pds=root/'carbonate.csv'
            pds.write_text('micrometers,inverse cm,real index,imaginary index\n'
                           '1,10000,1.5,.01\n2,5000,1.6,.02\n')
            constants=load_constants(pds,columns=[0,2,3])
            np.testing.assert_allclose(constants.n,[1.5,1.6])
            np.testing.assert_allclose(constants.k,[.01,.02])
            with self.assertRaisesRegex(ValueError,'three distinct'):
                load_constants(pds,columns=[0,2,2])
            optool=root/'mineral.lnk'
            optool.write_text('# optical constants\n2 2.5\n1 1.5 .01\n2 1.6 .02\n')
            np.testing.assert_allclose(load_optool(optool).k,[.01,.02])
            optool.write_text('3 2.5\n1 1.5 .01\n2 1.6 .02\n')
            with self.assertRaisesRegex(ValueError,'row count'):
                load_optool(optool)

    def test_refractiveindex_yml_tabulated_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mineral.yml'
            path.write_text('DATA:\n  - type: tabulated nk\n    data: |\n'
                            '      2 1.6 .02\n      1 1.5 .01\n')
            constants=load_refractiveindex_yml(path)
            np.testing.assert_allclose(constants.wavelength,[1,2])
            np.testing.assert_allclose(constants.k,[.01,.02])
            path.write_text('DATA:\n  - type: formula 2\n    wavelength_range: 1 2\n')
            with self.assertRaisesRegex(ValueError,'tabulated'):
                load_refractiveindex_yml(path)

    def test_mooney1985_units_and_missing_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mooney.txt'
            path.write_text('4000 1.525 2.68 1.479 2.97\n'
                            '3000 1.5 0 0 0\n'
                            '2000 1.489 3.78 1.427 4.45\n')
            chlorite=load_mooney1985(path,'chlorite')
            serpentine=load_mooney1985(path,'serpentine')
            np.testing.assert_allclose(chlorite.wavelength,[2.5,5.])
            np.testing.assert_allclose(chlorite.k,[.00268,.00378])
            np.testing.assert_allclose(serpentine.n,[1.479,1.427])
            self.assertEqual(chlorite.gaps,((2.5,5.),))
            with self.assertRaisesRegex(ValueError,'missing-data interval'):
                chlorite.sample([3.])
            with self.assertRaisesRegex(ValueError,'mineral'):
                load_mooney1985(path,'magnetite')

    def test_mooney1985_verified_transcription_corrections(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mooney.txt'
            path.write_text('3450 1.531 25.3 1.512 12.9\n'
                            '3430 1.537 27.0 1.512 12.7\n'
                            '3430 1.544 27.7 1.511 12.8\n'
                            '3420 1.549 28.5 1.509 12.4\n'
                            '2299 1.517 2.75 0 0\n'
                            '2280 1.516 2.76 0 0\n')
            constants=load_mooney1985(path,'chlorite')
            self.assertAlmostEqual(constants.wavelength[1],1e4/3440)
            self.assertAlmostEqual(constants.wavelength[-2],1e4/2290)
            self.assertTrue(any(left<4<right for left,right in constants.gaps))

    def test_jwst_observation_tables(self):
        root=Path(__file__).parent/'data'/'jwst_driss'
        expected={'brokoff.txt':(880,5,.7025,5.0975),
                  'eulalia.txt':(3517,5,.970318,5.09944),
                  'hibbs.txt':(880,5,.7025,5.0975),
                  'horns.txt':(880,5,.7025,5.0975),
                  'Klio.txt':(3554,3,.970318,5.099445),
                  'polana_nirspec.txt':(2576,3,1.67113,5.249805)}
        for name,(rows,width,first,last) in expected.items():
            with self.subTest(name=name):
                path=root/name
                self.assertEqual(numeric_table(path,None).shape,(rows,width))
                observation=load_observations(path,wavelength_range=[2.5,3.0])
                self.assertGreater(len(observation),1)
                self.assertAlmostEqual(numeric_table(path,None)[0,0],first)
                self.assertAlmostEqual(numeric_table(path,None)[-1,0],last)
                self.assertTrue(np.all((observation[:,0]>=2.5)&(observation[:,0]<=3.0)))

    def test_observation_column_selection_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'obs.txt'
            path.write_text('wl rf err extra\n1 .5 .01 7\n2 .6 .02 8\n')
            np.testing.assert_allclose(load_observations(path,columns=[0,1,2]),[[1,.5,.01],[2,.6,.02]])
            with self.assertRaisesRegex(ValueError,'column index'):
                load_observations(path,columns=[0,1,4])
            with self.assertRaisesRegex(ValueError,'distinct'):
                load_observations(path,columns=[0,1,1])
            path.write_text('1 .5 0 7\n2 .6 .02 8\n')
            with self.assertRaisesRegex(ValueError,'positive uncertainties'):
                load_observations(path)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy is required for fitting')
    def test_normalized_model_fit(self):
        w=np.array([1.1,1.4,1.7,2.1,2.4,2.8])
        constants=OpticalConstants(np.array([1.,3.]),np.full(2,1.5),np.array([.001,.02]))
        target=reflectance(np.union1d(w,[2.6]),[Component(constants,1,27.)],order=32)
        reference=target[np.searchsorted(np.union1d(w,[2.6]),2.6)]
        observed=np.delete(target/reference,np.searchsorted(np.union1d(w,[2.6]),2.6))
        model,components,_,report=fit_reflectance(
            w,observed,np.full(len(w),.01),[Component(constants,1,18.)],
            [{'name':'diameter','component':0,'bounds':[3.,100.]}],order=32,normalize_at=2.6)
        self.assertAlmostEqual(components[0].diameter,27.,places=4)
        self.assertEqual(report['normalization_wavelength_um'],2.6)
        np.testing.assert_allclose(model,observed,atol=1e-8)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy is required for fitting')
    def test_fitting_cli_five_column_normalized_observations(self):
        import json
        import sys
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from shkuratov_model import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'optics.txt').write_text('1 1.5 .001\n3 1.5 .02\n')
            constants=load_constants(root/'optics.txt')
            w=np.array([1.1,1.4,1.7,2.1,2.4,2.8])
            grid=np.union1d(w,[2.6])
            spectrum=reflectance(grid,[Component(constants,1,27.)],order=32)
            observed=np.delete(spectrum/spectrum[np.searchsorted(grid,2.6)],np.searchsorted(grid,2.6))
            np.savetxt(root/'observed.txt',np.c_[w,np.zeros(len(w)),observed,
                                                  np.full(len(w),.01),np.ones(len(w))])
            config={'observations':{'path':'observed.txt','columns':[0,2,3]},'quadrature_order':32,
                    'components':[{'path':'optics.txt','weight':1,'diameter':18}],
                    'fit':{'parameters':[{'name':'diameter','component':0,'bounds':[3,100]}],
                           'normalize_at':2.6}}
            path=root/'config.json'
            path.write_text(json.dumps(config))
            output=root/'result.csv'
            with patch.object(sys,'argv',['shkuratov_model.py',str(path),'--output',str(output)]),redirect_stdout(StringIO()):
                main()
            metadata=json.loads(output.with_suffix('.json').read_text())
            self.assertAlmostEqual(metadata['fit']['diameters_um'][0],27.,places=4)
            self.assertIn('normalized_model_reflectance',output.read_text().splitlines()[0])
            np.testing.assert_allclose(np.loadtxt(output,delimiter=',',skiprows=1)[:,1],observed,atol=1e-8)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy is required for fitting')
    def test_weighted_least_squares_recovers_path_length(self):
        w=np.linspace(1.1,2.8,9)
        constants=OpticalConstants(np.array([1.,3.]),np.array([1.5,1.5]),np.array([.001,.008]))
        observed=reflectance(w,[Component(constants,1,27.)],order=32)
        error=np.linspace(.005,.02,len(w))
        model,components,porosity,report=fit_reflectance(
            w,observed,error,[Component(constants,1,15.)],
            [{'name':'diameter','component':0,'bounds':[3.,100.]}],order=32)
        self.assertTrue(report['success'])
        self.assertAlmostEqual(components[0].diameter,27.,places=5)
        self.assertAlmostEqual(report['chi_square'],0.,places=8)
        self.assertEqual(report['free_parameters'],1)
        np.testing.assert_allclose(model,observed,atol=1e-8)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy is required for fitting')
    def test_weight_fit_preserves_simplex(self):
        w=np.linspace(1.1,2.8,9)
        clear=OpticalConstants(np.array([1.,3.]),np.array([1.5,1.5]),np.array([.001,.001]))
        dark=OpticalConstants(np.array([1.,3.]),np.array([1.5,1.5]),np.array([.02,.02]))
        observed=reflectance(w,[Component(clear,.3,30.),Component(dark,.7,30.)],order=32)
        fitted,components,_,report=fit_reflectance(
            w,observed,np.full(len(w),.01),
            [Component(clear,.5,30.),Component(dark,.5,30.)],[],fit_weights=True,order=32)
        self.assertAlmostEqual(components[0].weight,.3,places=5)
        self.assertAlmostEqual(sum(c.weight for c in components),1.,places=14)
        self.assertEqual(report['free_parameters'],1)
        np.testing.assert_allclose(fitted,observed,atol=1e-8)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy is required for fitting')
    def test_joint_fit_recovers_identifiable_mixture(self):
        w=np.linspace(1.05,2.95,20)
        a=OpticalConstants(np.array([1.,2.,3.]),np.full(3,1.5),np.array([.001,.005,.008]))
        b=OpticalConstants(np.array([1.,2.,3.]),np.full(3,1.7),np.array([.01,.002,.005]))
        observed=reflectance(w,[Component(a,.4,25.),Component(b,.6,40.)],porosity=.2,order=32)
        _,components,porosity,report=fit_reflectance(
            w,observed,np.full(len(w),.01),[Component(a,.5,20.),Component(b,.5,35.)],
            [{'name':'diameter','component':0,'bounds':[3,100]},
             {'name':'diameter','component':1,'bounds':[3,100]},
             {'name':'porosity','bounds':[0,.8]}],fit_weights=True,porosity=.3,order=32)
        self.assertEqual(report['jacobian_rank'],4)
        np.testing.assert_allclose([components[0].diameter,components[1].diameter,
                                    porosity,components[0].weight],[25,40,.2,.4],rtol=1e-8)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy is required for fitting')
    def test_identical_materials_report_unidentifiable_weights(self):
        w=np.linspace(1.1,2.8,7)
        constants=OpticalConstants(np.array([1.,3.]),np.full(2,1.5),np.full(2,.002))
        components=[Component(constants,.5,30.),Component(constants,.5,30.)]
        observed=reflectance(w,components,order=32)
        _,_,_,report=fit_reflectance(w,observed,np.full(len(w),.01),components,[],
                                     fit_weights=True,order=32)
        self.assertFalse(report['jacobian_full_rank'])

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy is required for fitting')
    def test_fitting_cli_writes_fitted_spectrum_and_dof(self):
        import json
        import sys
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from shkuratov_model import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'optics.txt').write_text('1 1.5 .001\n3 1.5 .008\n')
            constants=load_constants(root/'optics.txt')
            w=np.linspace(1.1,2.8,8)
            observed=reflectance(w,[Component(constants,1,27.)],order=32)
            np.savetxt(root/'observed.txt',np.c_[w,observed,np.full(len(w),.01)])
            config={'observations':'observed.txt','quadrature_order':32,
                    'components':[{'path':'optics.txt','weight':1,'diameter':15}],
                    'fit':{'parameters':[{'name':'diameter','component':0,'bounds':[3,100]}]}}
            path=root/'config.json'
            path.write_text(json.dumps(config))
            output=root/'result.csv'
            with patch.object(sys,'argv',['shkuratov_model.py',str(path),'--output',str(output)]),redirect_stdout(StringIO()):
                main()
            metadata=json.loads(output.with_suffix('.json').read_text())
            self.assertAlmostEqual(metadata['fit']['diameters_um'][0],27.,places=5)
            self.assertAlmostEqual(metadata['chi_square'],0.,places=8)
            self.assertAlmostEqual(metadata['reduced_chi_square'],0.,places=8)
            self.assertEqual(metadata['fit']['free_parameters'],1)
            np.testing.assert_allclose(np.loadtxt(output,delimiter=',',skiprows=1)[:,1],observed,atol=1e-8)

    def test_index_smoothing_resolves_narrow_source_peak(self):
        wavelength=np.array([1.,1.49995,1.5,1.50005,2.])
        constants=OpticalConstants(wavelength,np.full(5,1.5),np.array([0.,0.,1.,0.,0.]))
        sigma=.1/np.sqrt(8*np.log(2))
        expected=.00005/(sigma*sqrt(2*pi)*erf(4/sqrt(2)))
        n,k=constants.sample(np.array([1.5]),.1)
        self.assertAlmostEqual(n[0],1.5,places=13)
        self.assertAlmostEqual(k[0],expected,delta=1e-9)

    def test_instrument_convolution_of_reflectance(self):
        constants=OpticalConstants(np.array([1.,1.5,2.]),np.array([1.4,1.4,1.4]),
                                   np.array([1e-6,.05,1e-6]))
        component=Component(constants,1,30)
        center=1.5
        width=.1
        sigma=width/np.sqrt(8*np.log(2))
        grid=np.linspace(center-4*sigma,center+4*sigma,2049)
        weight=np.exp(-.5*((grid-center)/sigma)**2)
        expected=np.trapezoid(reflectance(grid,[component],order=32)*weight,grid)/np.trapezoid(weight,grid)
        actual=reflectance([center],[component],order=32,instrument_fwhm=width)[0]
        self.assertAlmostEqual(actual,expected,delta=1e-6)
        with self.assertRaisesRegex(ValueError,'cannot be combined'):
            reflectance([center],[component],order=32,fwhm=width,instrument_fwhm=width)
        with self.assertRaisesRegex(ValueError,'exceeds measured'):
            reflectance([1.02],[component],order=32,instrument_fwhm=width)

    def test_source_specified_log_interpolation(self):
        c=OpticalConstants(np.array([1.,4.]),np.array([1.2,1.6]),np.array([1e-6,1e-2]),
                           interpolation='log_wavelength_log_k')
        n,k=c.sample(np.array([2.]))
        np.testing.assert_allclose([n[0],k[0]],[1.4,1e-4])
        n_smooth,k_smooth=c.sample(np.array([2.]),.1)
        self.assertGreater(k_smooth[0],0)
        self.assertAlmostEqual(n_smooth[0],n[0],delta=.002)
        self.assertAlmostEqual(k_smooth[0],k[0],delta=.00001)
        with self.assertRaisesRegex(ValueError,'positive k'):
            OpticalConstants(np.array([1.,4.]),np.array([1.2,1.6]),np.array([0.,1e-2]),
                             interpolation='log_wavelength_log_k')

    def test_output_provenance_covers_all_optical_inputs(self):
        import json
        import sys
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from shkuratov_model import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,n,k in [('core',1.5,0.),('coat',1.2,.01),('inclusion',1.7,0.)]:
                (root/f'{name}.txt').write_text(f'1 {n} {k}\n3 {n} {k}\n')
            config={'wavelength_range':[1.5,2.], 'points':2,'experimental':True,
                    'components':[{'path':'core.txt','weight':1,'diameter':30,
                                   'coating':{'path':'coat.txt'},'thickness':.01,
                                   'inclusion':{'path':'inclusion.txt'},'inclusion_fraction':0}]}
            path=root/'config.json'
            path.write_text(json.dumps(config))
            output=root/'spectrum.csv'
            with patch.object(sys,'argv',['shkuratov_model.py',str(path),'--output',str(output)]),redirect_stdout(StringIO()):
                main()
            metadata=json.loads(output.with_suffix('.json').read_text())
            for role in ('core','coating','inclusion'):
                self.assertEqual(metadata['source_optical_constants'][0][role]['rows'],2)
                self.assertEqual(len(metadata['source_optical_constants'][0][role]['numeric_constants_and_gaps_sha256']),64)

    def test_split_tables_zero_placeholder_and_identical_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'A_n').write_text('1 1.4\n2 1.4\n3 1.4\n')
            (root/'A_k').write_text('0 .5\n1 .01\n2 .02\n2 .02\n3 .03\n')
            result=load_opcon(root,'A')
            np.testing.assert_allclose(result.wavelength,[1,2,3])
            np.testing.assert_allclose(result.k,[.01,.02,.03])
            (root/'A_k').write_text('0 .5\n1 .01\n2 .02\n2 .025\n3 .03\n')
            with self.assertRaisesRegex(ValueError,'Conflicting values'):
                load_opcon(root,'A')

    def test_paper_real_n_interface_mode(self):
        n,k,lam=1.5,.2,1.
        np.testing.assert_allclose(angular_integrals(n,k,lam,interface_mode='paper_real_n'),angular_integrals(n,0,lam))
        constants=self.constants(n=n,k=k)
        dark=reflectance([1.5],[Component(constants,1,30)],interface_mode='paper_real_n')
        clear=reflectance([1.5],[Component(self.constants(n=n,k=0),1,30)],interface_mode='paper_real_n')
        self.assertLess(float(dark[0]),float(clear[0]))
        with self.assertRaises(ValueError):
            angular_integrals(n,k,lam,interface_mode='unknown')
        with self.assertRaisesRegex(ValueError,'requires n>=1'):
            angular_integrals(.9,k,lam,interface_mode='paper_real_n')
        with self.assertRaisesRegex(ValueError,'uncoated'):
            angular_integrals(n,k,lam,coat=2.,thickness=.1,interface_mode='paper_real_n')
        with self.assertRaises(ValueError):
            reflectance([1.5],[Component(constants,1,30)],interface_mode='unknown')

    def test_bare_interface_transport_probabilities_conserve(self):
        for n in (1.3,1.5,1.7,2.):
            for k in (0.,.001,.1,1.):
                rb,rf,te,ri,ti=angular_integrals(n,k,1.)
                self.assertAlmostEqual(rb+rf+te,1,places=13)
                self.assertAlmostEqual(ri+ti,1,places=13)
                self.assertTrue(0<=min(rb,rf,te,ri,ti))
                if k==0:
                    self.assertAlmostEqual(ti,te/n**2,places=6)

    def test_public_idl_coefficients_are_not_conservative(self):
        from validation.compare_shkuratov_kernels import (vernazza_coefficients,
                                                conserving_approximation,
                                                model_value)
        empirical=vernazza_coefficients(1.7)
        conserving=conserving_approximation(1.7)
        self.assertLess(empirical[3]+empirical[4],1)
        self.assertAlmostEqual(conserving[3]+conserving[4],1)
        self.assertLess(model_value(empirical,0,10,1),.95)
        self.assertAlmostEqual(model_value(conserving,0,10,1),1,places=6)

    def test_film_critical_layer_limit(self):
        g=np.arcsin(1/1.5)
        r,t=film_rt(1.5,1.,1.5,g,.1,2.)
        self.assertTrue(np.isfinite(r+t))
        self.assertAlmostEqual(r+t,1)
        left=np.array(film_rt(1.5,1.,1.5,g-1e-7,.1,2.))
        right=np.array(film_rt(1.5,1.,1.5,g+1e-7,.1,2.))
        np.testing.assert_allclose([r,t],(left+right)/2,atol=1e-9)
        np.testing.assert_allclose(film_rt(1.5,1.,1.,g,.1,2.),[1,0],atol=1e-12)

    def test_film_invalid_optics(self):
        for nf in (0,-1,1-.1j,complex('nan')):
            with self.assertRaises(ValueError):
                film_rt(1,nf,1.5,.1,.1,2.)

    def test_clark_missing_intervals(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'Si_nk'
            p.write_text('deleted points = -1.23e34\n1 1.5 -1.23e34\n2 1.5 1\n3 1.5 2\n4 1.5 -1.23e34\n5 1.5 3\n6 1.5 4\n')
            c=load_opcon(d,'Si')
            self.assertEqual(c.gaps,((3.,5.),))
            c.sample([2.5,5.5])
            np.testing.assert_allclose(c.k[0],2e-4/(4*np.pi))
            for w in ([1.5],[3.5],[4.5]):
                with self.assertRaises(ValueError):
                    c.sample(w)
            with self.assertRaises(ValueError):
                c.sample([3.],.1)

    def test_closure_by_layer_adding(self):
        # Independent finite-stack multiple-reflection construction.
        for b,f,p in [(.1,.6,0),(.1,.6,.3),(.2,.3,.8)]:
            rb=(1-p)*b
            rf=(1-p)*f+p
            stacked=0.
            for _ in range(2000):
                stacked=rb+rf*rf*stacked/(1-rb*stacked)
            self.assertAlmostEqual(stacked,float(albedo_from_indicatrix(b,f,p)),places=13)

    def test_opcon_case_sensitive_materials(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'A_n').write_text('1 1.4\n2 1.4\n')
            (root/'A_k').write_text('1 .01\n2 .01\n')
            self.assertAlmostEqual(load_opcon(root,'A').n[0],1.4)
            with self.assertRaises((ValueError,FileNotFoundError)):
                load_opcon(root,'a')

    def test_closure_extreme_porosity_decimal(self):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec=80
            for p in (0.,.3,1-1e-12,np.nextafter(1.,0.)):
                q=1-Decimal(float(p))
                b=Decimal('.1')
                f=Decimal('.6')
                pb=q*b
                pf=q*f+1-q
                a=(1+pb*pb-pf*pf)/(2*pb)
                expected=float(1/(a+(a*a-1).sqrt()))
                self.assertAlmostEqual(float(albedo_from_indicatrix(.1,.6,p)),expected,places=14)

    def test_closure_tiny_backscatter(self):
        for b in (1e-100,1e-250,1e-310):
            value=float(albedo_from_indicatrix(b,0))
            self.assertTrue(np.isfinite(value))
            self.assertAlmostEqual(value/b,1,places=12)

    def test_closure_invalid_energy(self):
        with self.assertRaises(ValueError):
            albedo_from_indicatrix(.8,.8)
        with self.assertRaises(ValueError):
            albedo_from_indicatrix(0,1.1)

    def constants(self,n=1.5,k=.01):
        return OpticalConstants(np.array([1.,2.,3.]),np.full(3,n),np.full(3,k))

    def test_dielectric(self):
        self.assertAlmostEqual(dielectric(1.5,.1),2.24+.3j)

    def test_inclusion_identical(self):
        self.assertAlmostEqual(inclusion_k(1.5,.1,1.5,.1,.01,experimental=True),.1)

    def test_inclusion_matches_dilute_effective_medium_for_clear_host(self):
        # Independent Maxwell-Garnett calculation in the f -> 0 limit.
        n,ni,ki,f=1.5,2.,.2,1e-6
        em=n*n
        ei=(ni+1j*ki)**2
        ee=em*(ei+2*em+2*f*(ei-em))/(ei+2*em-f*(ei-em))
        expected=np.sqrt(ee).imag
        actual=inclusion_k(n,0,ni,ki,f,experimental=True)
        self.assertAlmostEqual(actual/expected,1.,delta=2e-6)

    def test_inclusion_zero(self):
        self.assertAlmostEqual(inclusion_k(1.5,.1,2,.2,0,experimental=True),.1)

    def test_inclusion_opt_in(self):
        with self.assertRaises(ValueError):
            inclusion_k(1.5,0,2,.1,.01)

    def test_normal_fresnel(self):
        r,t=interface_rt(1,1.5,0)
        self.assertAlmostEqual(r,.04)
        self.assertAlmostEqual(t,.96)

    def test_total_internal_reflection(self):
        r,t=interface_rt(1.5,1,np.pi/4)
        self.assertAlmostEqual(r,1)
        self.assertAlmostEqual(t,0)

    def test_lossless_flux(self):
        for theta in np.linspace(0,1.5,15):
            for n0,n1 in [(1,1.5),(1.5,1)]:
                r,t=interface_rt(n0,n1,theta)
                self.assertAlmostEqual(r+t,1)

    def test_absorbing_exit_flux(self):
        for theta in (.1,.8,1.4):
            r,t=interface_rt(1,1.5+.2j,theta)
            self.assertAlmostEqual(r+t,1)

    def test_film_zero(self):
        for theta in (.1,.8,1.4):
            np.testing.assert_allclose(film_rt(1.5,2,1,theta,0,2),interface_rt(1.5,1,theta))

    def test_film_lossless(self):
        for theta in (.1,.8,1.4):
            r,t=film_rt(1,1.3,1.5,theta,.1,2)
            self.assertAlmostEqual(r+t,1)

    def test_film_quarter_wave(self):
        n=np.sqrt(1.5)
        r,t=film_rt(1,n,1.5,0,1/(4*n),1)
        self.assertLess(r,1e-25)
        self.assertAlmostEqual(t,1)

    def test_film_absorption(self):
        r,t=film_rt(1,1.3+.2j,1.5,.6,.1,2)
        self.assertTrue(0<r+t<1)

    def test_film_absorbing_incident_rejected(self):
        with self.assertRaises(ValueError):
            film_rt(1.5+.1j,2,1,.4,.1,2)

    def test_zero_preserved(self):
        _,k=self.constants(k=0).sample(np.array([1.5,2.5]))
        np.testing.assert_array_equal(k,0)

    def test_bad_constants(self):
        for w,n,k in [([1,1],[1,1],[0,0]),([1,2],[0,1],[0,0]),([1,2],[1,1],[-1,0])]:
            with self.assertRaises(ValueError):
                OpticalConstants(w,n,k)

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            self.constants().sample([.5,2])

    def test_smoothing_constant(self):
        n,k=self.constants().sample([1.8,2.2],.1)
        np.testing.assert_allclose(n,1.5)
        np.testing.assert_allclose(k,.01)

    def test_nonabsorbing_closure(self):
        r=reflectance([1.5,2.], [Component(self.constants(k=0),1,30)])
        np.testing.assert_allclose(r,1,atol=1e-6)

    def test_no_contrast(self):
        r=reflectance([1.5,2.], [Component(self.constants(n=1,k=0),1,30)])
        np.testing.assert_allclose(r,0,atol=1e-12)

    def test_permutation_and_split(self):
        a=Component(self.constants(),.3,30)
        b=Component(self.constants(k=.1),.7,10)
        w=[1.5,2.]
        np.testing.assert_allclose(reflectance(w,[a,b]),reflectance(w,[b,a]))
        np.testing.assert_allclose(reflectance(w,[Component(a.constants,1,30)]),reflectance(w,[Component(a.constants,.4,30),Component(a.constants,.6,30)]))

    def test_absorption_trend(self):
        w=[1.5,2.]
        a=reflectance(w,[Component(self.constants(),1,10)])
        b=reflectance(w,[Component(self.constants(),1,100)])
        self.assertTrue(np.all(a>b))

    def test_geometric_optics_lower_bound(self):
        for path_length in (1.,2.):
            with self.assertRaisesRegex(ValueError,'Effective path length'):
                reflectance([1.5,2.],[Component(self.constants(),1,path_length)])

    def test_convergence(self):
        c=[Component(self.constants(),1,30)]
        np.testing.assert_allclose(reflectance([1.5,2.],c,order=256),reflectance([1.5,2.],c,order=512),rtol=1e-6)

    def test_input_validation(self):
        for cs,p in [([Component(self.constants(),1,30)],1),([Component(self.constants(),-1,30)],0),([Component(self.constants(),1,-1)],0)]:
            with self.assertRaises(ValueError):
                reflectance([1.5,2.],cs,p)

    def test_chi(self):
        self.assertEqual(chi_square([1,2],[1,1],[1,1]),(1,None))
        self.assertEqual(chi_square([1,2],[1,1],[1,1],1),(1,1))
        with self.assertRaises(ValueError):
            chi_square([1],[1],[0])
        with self.assertRaises(ValueError):
            chi_square([1],[1],[1],1)

    def test_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'nk.txt'
            p.write_text('; header\nwavelength n k\n1 1.5 0\n2 1.5 0.1\n')
            self.assertEqual(load_constants(p).k[0],0)

    def test_coated_model_zero(self):
        a=self.constants(k=0)
        b=self.constants(n=2,k=.1)
        np.testing.assert_allclose(reflectance([1.5,2.],[Component(a,1,30)]),reflectance([1.5,2.],[Component(a,1,30,b,0)]))

    def test_coated_model_absorption(self):
        a=self.constants(k=0)
        b=self.constants(n=2,k=.1)
        r=reflectance([1.5,2.],[Component(a,1,30,b,.1)],experimental=True)
        self.assertTrue(np.all((r>=0)&(r<=1)))

    def test_original_uncoated_formula_reference(self):
        # Independent transcription of original 301-338 and 608-645.
        # k=0 removes the ambiguous absorbing-incident flux convention.
        x,wt=np.polynomial.legendre.leggauss(512)
        def integ(m,lo,hi):
            g=lo+(x+1)*(hi-lo)/2
            c=np.cos(g)
            cp=np.sqrt(1-np.sin(g)**2/complex(m)**2)
            r=(abs((c-m*cp)/(c+m*cp))**2+abs((m*c-cp)/(m*c+cp))**2)/2
            t=np.real(cp*m/c)*(abs(2*c/(m*c+cp))**2+abs(2*c/(c+m*cp))**2)/2
            weights=wt*(hi-lo)*np.sin(g)*c
            return np.sum(weights*r),np.sum(weights*t)
        rb,tb=integ(1.5,0,np.pi/4)
        rf,tf=integ(1.5,np.pi/4,np.pi/2)
        critical=np.arcsin(1/1.5)
        aa=integ(1/1.5,0,critical)
        bb=integ(1/1.5,critical,np.pi/2)
        expected=[rb,rf,tb+tf,aa[0]+bb[0],aa[1]+bb[1]]
        np.testing.assert_allclose(angular_integrals(1.5,0,2,order=256),expected,rtol=2e-7,atol=1e-9)


if __name__=='__main__':
    unittest.main(verbosity=2)
