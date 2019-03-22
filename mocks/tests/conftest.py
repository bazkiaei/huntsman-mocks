import os
import pytest

from astropy.io import fits


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
def max_outlier_fraction():
    return 0.02


@pytest.fixture
def n_sigma():
    return 3
