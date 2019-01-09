import os
import pytest

from astropy.io import fits
import astropy.units as u


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
def config():
    mock_image_input = dict()
    mock_image_input['galaxy_coordinates'] = '14h40m56.435s -60d53m48.3s'
    mock_image_input['observation_time'] = '2018-04-12T08:00'
    mock_image_input['imager_filter'] = 'g'
    mock_image_input['pixel_scale'] = 3.52270697 * u.arcsec / u.pix
    mock_image_input['total_mag'] = 9.1 * u.ABmag
    return mock_image_input
