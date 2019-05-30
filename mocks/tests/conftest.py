import os
import pytest

import numpy as np

from astropy.io import fits

from mocks.utils import load_yaml_config
from mocks import mocks


@pytest.fixture(scope='module')
def sim_data_dir():
    data_dir = os.path.normpath('sim_data')
    return data_dir


@pytest.fixture
def galaxy_sim_data(sim_data_dir):
    fits_path = os.path.join(sim_data_dir, 'cl19.fits')
    galaxy_sim = fits.open(fits_path)
    yield galaxy_sim[0].data
    galaxy_sim.close()


@pytest.fixture
def particle_positions_3D():
    return (np.arange(1, 16, dtype=np.float).reshape(5, 3))


@pytest.fixture
def mass_weights():
    return np.arange(1, 6, dtype=np.float)


@pytest.fixture
def max_outlier_fraction():
    return 0.02


@pytest.fixture
def n_sigma():
    return 3


@pytest.fixture
def custom_cosmology_data():
    custom_data = dict()
    custom_data['hubble_constant'] = 70.
    custom_data['T_CMB0'] = 2.5
    return custom_data


@pytest.fixture(scope='module')
def configuration():
    return load_yaml_config('config_example.yaml')


@pytest.fixture
def gadget_data_path(sim_data_dir):
    gadget_path = os.path.join(sim_data_dir, 'test_g2_snap')
    return gadget_path


@pytest.fixture
def config():
    return mocks.parse_config()
