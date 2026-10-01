"""Exercise portable optical-data lookup with synthetic, non-research data."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shkuratov_model import load_opcon
from validation.local_data import opcon_directory


class LocalDataTests(unittest.TestCase):
    def test_default_is_independent_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.dict(os.environ, {'OPCON_DIR': ''}):
                    self.assertEqual(
                        opcon_directory(),
                        Path(__file__).resolve().parent / 'data' / 'opcon',
                    )
            finally:
                os.chdir(previous)

    def test_override_loads_optical_data_with_spaces_in_path(self):
        with tempfile.TemporaryDirectory(prefix='optical data ') as directory:
            root = Path(directory)
            (root / 'Gser_nk').write_text('5 1.5 0.01\n12 1.6 0.02\n')
            with patch.dict(os.environ, {'OPCON_DIR': str(root)}):
                constants = load_opcon(opcon_directory(), 'Gser')
            np.testing.assert_array_equal(constants.wavelength, [5, 12])
            np.testing.assert_allclose(constants.n, [1.5, 1.6])
            np.testing.assert_allclose(constants.k, [0.01, 0.02])
