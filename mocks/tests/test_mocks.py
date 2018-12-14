import os

import pytest

import numpy as np

import astropy.units as u
from astropy.nddata import CCDData
from astropy.io import fits

from gunagala import imager

from mocks.mocks import compute_pixel_scale
from mocks.mocks import create_mock_galaxy_noiseless_image
from mocks.mocks import convolve_image_psf
from mocks.mocks import mock_image_stack


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


def test_compute_pixel_scale():
    pixel_scale = compute_pixel_scale(distance=10.,
                                      sim_pc_pixel=170)
    assert isinstance(pixel_scale, u.quantity.Quantity)
    assert pixel_scale * u.pixel / u.arcsec == pytest.approx(3.522, 1e-3)


def test_create_mock_galaxy_noiseless_image(galaxy_sim_data,
                                            huntsman_sbig_dark_imager,
                                            pixel_scale_value,
                                            coordinates_string):
    noiseless_image = create_mock_galaxy_noiseless_image(galaxy_sim_data,
                                                         huntsman_sbig_dark_imager,
                                                         pixel_scale_value,
                                                         coordinates_string,
                                                         total_mag=9.1)
    assert isinstance(noiseless_image, CCDData)
    assert noiseless_image.data.shape == (3326, 2504)
    assert noiseless_image.data.min() == pytest.approx(1.275, 1e-3)
    assert noiseless_image.data.max() == pytest.approx(62.192, 1e-3)
    assert np.mean(noiseless_image) == pytest.approx(1.278, 1e-3)
    assert np.median(noiseless_image) == pytest.approx(1.2745, 1e-3)
    assert noiseless_image.data.sum() == pytest.approx(10641012.8)
    assert np.std(noiseless_image) == pytest.approx(0.120, 1e-2)
    assert noiseless_image.unit == "electron / (pix s)"
    assert noiseless_image.uncertainty is None
    assert noiseless_image.flags is None


def test_convolve_image_psf(galaxy_sim_data,
                            pixelated_psf_data):
    convolved = convolve_image_psf(galaxy_sim_data,
                                   pixelated_psf_data)
    assert isinstance(convolved, CCDData)
    assert convolved.shape == (300, 300)
    assert convolved.data.min() == 0.0
    assert convolved.data.max() == pytest.approx(629.60885)
    assert convolved.data.sum() == pytest.approx(185750.08)
    assert np.mean(convolved.data) == pytest.approx(2.0638898)
    assert np.median(convolved.data) == pytest.approx(0.4029667)
    assert np.std(convolved.data) == pytest.approx(10.1996596)


def test_mock_image_stack(galaxy_sim_data,
                          huntsman_sbig_dark_imager):
    stacked = mock_image_stack(galaxy_sim_data,
                               huntsman_sbig_dark_imager,
                               n_exposures=100,
                               exptime=50 * u.s)
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == galaxy_sim_data.shape
    assert stacked.max() == pytest.approx(65535.0, 1.)
    assert stacked.min() == pytest.approx(1093., 4.)
    assert np.mean(stacked) == pytest.approx(1384.2, 1.)
    assert np.median(stacked) == pytest.approx(1109.3, 1.)
    assert np.std(stacked) == pytest.approx(1559.8, 1.)
    assert stacked[33, 48] == pytest.approx(1374., 6.)
    assert stacked[130, 160] == pytest.approx(1647., 6.)
    assert stacked[235, 203] == pytest.approx(1240., 7.)


def test_mock_image_stack_with_convolve(galaxy_sim_data,
                                        pixelated_psf_data,
                                        huntsman_sbig_dark_imager):
    convolved = convolve_image_psf(galaxy_sim_data,
                                   pixelated_psf_data,
                                   convolution_boundary='extend')
    stacked = mock_image_stack(convolved.data,
                               huntsman_sbig_dark_imager,
                               n_exposures=100,
                               exptime=50 * u.s)
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == convolved.shape
    assert stacked.max() == pytest.approx(65535.0, 1.)
    assert stacked.min() == pytest.approx(1095., 3.)
    assert np.mean(stacked) == pytest.approx(1384.6, 1.)
    assert np.median(stacked) == pytest.approx(1163.3, 1.)
    assert np.std(stacked) == pytest.approx(1365.7, 1.)
    assert stacked[33, 48] == pytest.approx(1321., 6.)
    assert stacked[130, 160] == pytest.approx(1674., 6.)
    assert stacked[235, 203] == pytest.approx(1216., 6.)
