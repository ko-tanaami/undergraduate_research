"""Archive local candidate tables without changing active model inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shkuratov_model import load_opcon


def main():
    source = Path('C:/Users/2017j/Downloads/opcon')
    target = Path(__file__).resolve().parents[1] / 'data/serpentine_candidates'
    target.mkdir(exist_ok=True)
    records = {}
    for code in ('Sz', 'Gcro'):
        original = source / (code + '_nk')
        shutil.copy2(original, target / original.name)
        constants = load_opcon(source, code)
        values = np.column_stack((constants.wavelength, constants.n, constants.k))
        assert np.all(np.isfinite(values))
        assert np.all(np.diff(values[:, 0]) > 0)
        assert np.all(values[:, 1] > 0) and np.all(values[:, 2] >= 0)
        filename = {'Sz': 'serpentine_hs318_clark2002_nk.csv',
                    'Gcro': 'cronstedtite_glotch_legacy_nk.csv'}[code]
        np.savetxt(target / filename, values, delimiter=',',
                   header='wavelength_um,n,k', comments='', fmt='%.17g')
        restored = np.loadtxt(target / filename, delimiter=',', skiprows=1)
        assert np.array_equal(restored, values)
        records[code] = {
            'original_path': str(original),
            'sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
            'csv': filename, 'rows': len(values),
            'range_um': [float(values[0, 0]), float(values[-1, 0])],
            'gaps_um': constants.gaps,
            'status': 'candidate; original publication table not verified',
        }
    shutil.copy2(source / 'Materials_List.txt', target / 'Materials_List_original.txt')
    (target / 'candidate_audit.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
