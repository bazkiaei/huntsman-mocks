import pytest

import numpy as np

import astropy.units as u
from astropy.nddata import CCDData, NDData

from gunagala import imager

from mocks import mocks


@pytest.fixture(scope='function')
def pixel_scale_value():
    return 3.5 * u.arcsecond / u.pixel


@pytest.fixture(scope='module')
def huntsman_sbig_dark_imager():
    imagers = imager.create_imagers()
    return imagers['canon_sbig_dark']


@pytest.fixture
def coordinates_string():
    return '14h40m56.435s -60d53m48.3s'


@pytest.fixture
def pixelated_psf_data(huntsman_sbig_dark_imager):
    psf_data = huntsman_sbig_dark_imager.psf.pixellated(size=(5, 5),
                                                        offsets=(0, 0))
    return psf_data


def test_prepare_mocks():
    mocks_config, data = mocks.prepare_mocks()
    assert mocks_config['galaxy_coordinates'] == '14h40m56.435s -60d53m48.3s'
    assert mocks_config['observation_time'] == '2018-04-12T08:00'
    assert mocks_config['imager_filter'] == 'g'
    assert mocks_config['pixel_scale'].to(u.arcsec / u.pixel).value ==\
        pytest.approx(3.5227069721693924,
                      rel=1e-12)
    assert mocks_config['total_mag'].to(u.ABmag).value == pytest.approx(
        9.10466504537635,
        rel=1e-12)


def test_create_mock_galaxy_noiseless_image(huntsman_sbig_dark_imager):
    config, galaxy_sim_data = mocks.prepare_mocks()
    noiseless_image = \
        mocks.create_mock_galaxy_noiseless_image(config,
                                                 galaxy_sim_data,
                                                 huntsman_sbig_dark_imager)
    assert isinstance(noiseless_image, CCDData)
    assert noiseless_image.data.shape == (3326, 2504)
    assert noiseless_image.data.min() == pytest.approx(1.274705830156097,
                                                       rel=1e-12)
    assert noiseless_image.data.max() == pytest.approx(61.909652519566656,
                                                       rel=1e-12)
    assert np.mean(noiseless_image) == pytest.approx(1.2776798645535397,
                                                     rel=1e-12)
    assert np.median(noiseless_image) == pytest.approx(1.274705830156097,
                                                       rel=1e-12)
    assert noiseless_image.data.sum() == pytest.approx(10640906.326680703,
                                                       rel=1e-8)
    assert np.std(noiseless_image) == pytest.approx(0.11920044545602064,
                                                    rel=1e-12)
    assert noiseless_image.unit == "electron / (pix s)"
    assert noiseless_image.uncertainty is None
    assert noiseless_image.flags is None


def test_convolve_image_psf(galaxy_sim_data,
                            pixelated_psf_data):
    convolved = mocks.convolve_image_psf(galaxy_sim_data,
                                         pixelated_psf_data)
    assert isinstance(convolved, CCDData)
    assert convolved.shape == (300, 300)
    assert convolved.data.min() == pytest.approx(0.0,
                                                 rel=1e-6)
    assert convolved.data.max() == pytest.approx(629.6088528303596,
                                                 rel=1e-12)
    assert convolved.data.sum() == pytest.approx(185750.07999405114,
                                                 rel=1e-11)
    assert np.mean(convolved.data) == pytest.approx(2.063889777711679,
                                                    rel=1e-12)
    assert np.median(convolved.data) == pytest.approx(0.4029667249588263,
                                                      rel=1e-12)
    assert np.std(convolved.data) == pytest.approx(10.19965960180967,
                                                   rel=1e-12)


def test_mock_image_stack(galaxy_sim_data,
                          huntsman_sbig_dark_imager):
    stacked = mocks.mock_image_stack(galaxy_sim_data,
                                     huntsman_sbig_dark_imager,
                                     n_exposures=100,
                                     exptime=50 * u.s)
    assert isinstance(stacked, CCDData)
    assert stacked.shape == galaxy_sim_data.shape
    assert stacked.data.max() == pytest.approx(65535.0, rel=1.)
    assert stacked.data.min() == pytest.approx(1093., rel=4.)
    assert np.mean(stacked) == pytest.approx(1384.2, rel=1.)
    assert np.median(stacked) == pytest.approx(1109.3, rel=1.)
    assert np.std(stacked) == pytest.approx(1559.8, rel=1.)
    assert stacked.data[33, 48] == pytest.approx(1374., rel=6.)
    assert stacked.data[130, 160] == pytest.approx(1647., rel=6.)
    assert stacked.data[235, 203] == pytest.approx(1240., rel=7.)


def test_mock_image_stack_with_convolve(galaxy_sim_data,
                                        pixelated_psf_data,
                                        huntsman_sbig_dark_imager):
    convolved = mocks.convolve_image_psf(galaxy_sim_data,
                                         pixelated_psf_data,
                                         convolution_boundary='extend')
    stacked = mocks.mock_image_stack(convolved,
                                     huntsman_sbig_dark_imager,
                                     n_exposures=100,
                                     exptime=50 * u.s)
    assert isinstance(stacked, CCDData)
    assert stacked.shape == convolved.shape
    assert stacked.data.max() == pytest.approx(65535.0, rel=1.)
    assert stacked.data.min() == pytest.approx(1095., rel=3.)
    assert np.mean(stacked) == pytest.approx(1384.6, rel=1.)
    assert np.median(stacked) == pytest.approx(1163.3, rel=1.)
    assert np.std(stacked) == pytest.approx(1365.7, rel=1.)
    assert stacked.data[33, 48] == pytest.approx(1321., rel=6.)
    assert stacked.data[130, 160] == pytest.approx(1674., rel=6.)
    assert stacked.data[235, 203] == pytest.approx(1216., rel=6.)


def test_compute_redshift():
    z = mocks.compute_redshift(10)
    assert z == pytest.approx(0.0023080874949477728,
                              rel=1e-12)
    z1 = mocks.compute_redshift(1e4 * u.kpc)
    assert z == z1


def test_compute_pixel_scale():
    pixel_scale = mocks.compute_pixel_scale(0.0023080874949477728,
                                            sim_pc_pixel=170)
    assert isinstance(pixel_scale, u.quantity.Quantity)
    assert pixel_scale.to(u.arcsec / u.pixel).value == pytest.approx(3.522,
                                                                     1e-3)
    assert pixel_scale.unit == 'arcsec / pix'


def test_compute_total_mass(galaxy_sim_data):
    total_mass = mocks.compute_total_mass(galaxy_sim_data,
                                          9.6031355)
    assert total_mass == pytest.approx(126212972737.866,
                                       1e-3)


def test_compute_apparent_ABmag():
    apparent_ABmag = mocks.compute_apparent_ABmag(0.0023080874949477728,
                                                  126212972737.866,
                                                  5,
                                                  5.11)
    assert apparent_ABmag.to(u.ABmag).value == pytest.approx(9.10466504537635,
                                                             rel=1e-12)


def test_mock_image_stack_input_units(galaxy_sim_data,
                                      huntsman_sbig_dark_imager):
    data = CCDData(galaxy_sim_data, unit="electron / (pixel * h)")
    stacked = mocks.mock_image_stack(data,
                                     huntsman_sbig_dark_imager)
    assert isinstance(stacked, CCDData)
    assert stacked.unit == "adu"


def test_mock_image_stack_input_NDData(galaxy_sim_data,
                                       huntsman_sbig_dark_imager):
    data = NDData(galaxy_sim_data)
    stacked = mocks.mock_image_stack(data,
                                     huntsman_sbig_dark_imager)
    assert isinstance(stacked, CCDData)
    assert stacked.unit == "adu"


def test_mock_image_stack_NDData_units(galaxy_sim_data,
                                       huntsman_sbig_dark_imager):
    data = NDData(galaxy_sim_data, unit='meter / pixel')
    with pytest.raises(u.UnitsError):
        stacked = mocks.mock_image_stack(data,
                                         huntsman_sbig_dark_imager)


def test_mock_image_stack_input_quantity(galaxy_sim_data,
                                         huntsman_sbig_dark_imager):
    data = galaxy_sim_data * u.electron / (u.pixel * u.s)
    stacked = mocks.mock_image_stack(data,
                                     huntsman_sbig_dark_imager)
    assert isinstance(stacked, CCDData)
    assert stacked.unit == "adu"


def test_mock_image_stack_quantity_units(galaxy_sim_data,
                                         huntsman_sbig_dark_imager):
    data = galaxy_sim_data * u.meter / u.pixel
    with pytest.raises(u.UnitsError):
        stacked = mocks.mock_image_stack(data,
                                         huntsman_sbig_dark_imager)


def test_mock_image_stack_compatible_units(galaxy_sim_data,
                                           huntsman_sbig_dark_imager,
                                           N_SIGMA,
                                           MAX_OUTLIER_FRACTION):
    data1 = galaxy_sim_data * u.electron / (u.pixel * u.second)
    data2 = galaxy_sim_data * 3600 * u.electron / (u.pixel * u.hour)
    stacked1 = mocks.mock_image_stack(data1, huntsman_sbig_dark_imager)
    stacked2 = mocks.mock_image_stack(data2, huntsman_sbig_dark_imager)
    assert np.mean(stacked1) == pytest.approx(np.mean(stacked2), rel=2)
    assert np.median(stacked1) == pytest.approx(np.median(stacked2), rel=2)
    difference_image = stacked1.subtract(stacked2)
    pixels_different = abs(difference_image.data) > N_SIGMA * np.std(
        difference_image)
    assert np.count_nonzero(pixels_different) <\
        MAX_OUTLIER_FRACTION * stacked1.size
