"""Locate user-supplied optical constants without machine-specific paths."""
import os
from pathlib import Path


def opcon_directory():
    """Use OPCON_DIR when set, otherwise shkuratov_python/data/opcon.

    Relative overrides are resolved from the caller's working directory.
    The default is independent of that working directory.
    """
    configured = os.environ.get('OPCON_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / 'data' / 'opcon'
