import pytest

import numpy as np

import astropy.units as u
from astropy.nddata import CCDData, NDData
from astropy import cosmology

from gunagala import imager

import pynbody

import logging

import logbook

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


def test_create_logger():
    logger = mocks.create_logger()
    assert isinstance(logger, logging.Logger)


def test_create_logbook():
    logger = mocks.create_logbook()
    assert isinstance(logger, logbook.base.Logger)


def test_crop_simulation_data(particle_positions_3D,
                              mass_weights):
    pos, mass = mocks.crop_simulation_data(particle_positions_3D,
                                           mass_weights,
                                           z_range=[29, 42] * u.Mpc,
                                           y_range=[29, 42] * u.Mpc,
                                           x_range=[0, 71.] * u.Mpc)
    assert pos.shape == (5, 3)
    assert mass.shape == (5,)
    assert pos.max() == 41.5
    assert pos.min() == 29.5
    assert mass.max() == 5.
    assert mass.min() == 1.


def test_load_configuration(configuration):
    assert configuration['data_path'] == 'sim_data/cl19.fits'
    assert configuration['sim_data_path'] == 'sim_data/test_g2_snap'
    assert configuration['galaxy_distance'] == 10.
    assert isinstance(configuration, dict)


def test_parse_config():
    config = mocks.parse_config()
    assert isinstance(config, dict)
    assert config['galaxy_coordinates'] == '14h40m56.435s -60d53m48.3s'
    assert config['observation_time'] == '2018-04-12T08:00'
    assert config['data_path'] == 'sim_data/cl19.fits'
    assert config['sim_data_path'] == 'sim_data/test_g2_snap'
    assert config['imager_filter'] == 'g'
    assert config['imager'] == 'canon_sbig_dark'
    assert config['mass_to_light_ratio']['g'].value == 5
    assert config['mass_to_light_ratio']['g'].unit == u.M_sun / u.L_sun
    assert config['mass_to_light_ratio']['r'].value == 5
    assert config['mass_to_light_ratio']['r'].unit == u.M_sun / u.L_sun
    assert config['abs_mag_sun']['g'].value == 5.11
    assert config['abs_mag_sun']['g'].unit == u.M_sun / u.L_sun
    assert config['abs_mag_sun']['r'].value == 4.68
    assert config['abs_mag_sun']['r'].unit == u.M_sun / u.L_sun
    assert config['abs_mag_sun']['i'].value == 4.53
    assert config['abs_mag_sun']['i'].unit == u.M_sun / u.L_sun
    assert config['galaxy_distance'].value == 10.
    assert config['galaxy_distance'].unit == u.Mpc
    assert config['target_galaxy_comoving_depth'].value == 1.
    assert config['target_galaxy_comoving_depth'].unit == u.Mpc
    assert config['viewing_axis'] == 'z'
    assert config['sim_pc_pixel'].value == 170.
    assert config['sim_pc_pixel'].unit == u.pc / u.pixel
    assert config['particle_baryonic_mass_sim'] == 9.6031355
    assert config['hubble_constant'].value == 69.3
    assert config['hubble_constant'].unit == u.km / (u.Mpc * u.s)
    assert config['Omega_0'] == 0.286
    assert config['Omega_b'] == 0.04
    assert config['Omega_Lambda'] == 0.714
    assert config['T_CMB0'].value == 2.725
    assert config['T_CMB0'].unit == u.K
    assert config['Neff'] == 3.04
    assert config['m_nu'][2].value == 0.
    assert config['m_nu'].unit == u.eV


def test_parse_config_kwargs():
    config = mocks.parse_config(
        galaxy_coordinates='00h00m00.0s 0d00m0.0s',
        observation_time='2019-04-12T08:00',
        data_path='sim_data/cl20.fits',
        sim_data_path='sim_data/test_g5_snap',
        imager_filter='r',
        imager='canon',
        mass_to_light_ratio={'g': 5.5, 'r': 5.5},
        abs_mag_sun={'g': 5., 'r': 4., 'i': 4.5},
        galaxy_distance=8,
        target_galaxy_comoving_depth=2,
        viewing_axis='x',
        sim_pc_pixel=120,
        particle_baryonic_mass_sim=8,
        hubble_constant=71,
        Omega_0=0.3,
        Omega_b=0.05,
        Omega_Lambda=0.7,
        T_CMB0=2.8,
        Neff=3.02,
        m_nu=[1, 1, 1])
    assert isinstance(config, dict)
    assert config['galaxy_coordinates'] == '00h00m00.0s 0d00m0.0s'
    assert config['observation_time'] == '2019-04-12T08:00'
    assert config['data_path'] == 'sim_data/cl20.fits'
    assert config['sim_data_path'] == 'sim_data/test_g5_snap'
    assert config['imager_filter'] == 'r'
    assert config['imager'] == 'canon'
    assert config['mass_to_light_ratio']['g'].value == 5.5
    assert config['mass_to_light_ratio']['g'].unit == u.M_sun / u.L_sun
    assert config['mass_to_light_ratio']['r'].value == 5.5
    assert config['mass_to_light_ratio']['r'].unit == u.M_sun / u.L_sun
    assert config['abs_mag_sun']['g'].value == 5.
    assert config['abs_mag_sun']['g'].unit == u.M_sun / u.L_sun
    assert config['abs_mag_sun']['r'].value == 4.
    assert config['abs_mag_sun']['r'].unit == u.M_sun / u.L_sun
    assert config['abs_mag_sun']['i'].value == 4.5
    assert config['abs_mag_sun']['i'].unit == u.M_sun / u.L_sun
    assert config['galaxy_distance'].value == 8.
    assert config['galaxy_distance'].unit == u.Mpc
    assert config['target_galaxy_comoving_depth'].value == 2.
    assert config['target_galaxy_comoving_depth'].unit == u.Mpc
    assert config['viewing_axis'] == 'x'
    assert config['sim_pc_pixel'].value == 120.
    assert config['sim_pc_pixel'].unit == u.pc / u.pixel
    assert config['particle_baryonic_mass_sim'] == 8.
    assert config['hubble_constant'].value == 71.
    assert config['hubble_constant'].unit == u.km / (u.Mpc * u.s)
    assert config['Omega_0'] == 0.3
    assert config['Omega_b'] == 0.05
    assert config['Omega_Lambda'] == 0.7
    assert config['T_CMB0'].value == 2.8
    assert config['T_CMB0'].unit == u.K
    assert config['Neff'] == 3.02
    assert config['m_nu'].value.all() == 1
    assert config['m_nu'].unit == u.eV


def test_read_gadget(config):
    pos, mass, info = mocks.read_gadget(config)
    assert isinstance(pos, u.Quantity)
    assert isinstance(mass, pynbody.array.SimArray)
    assert isinstance(info, pynbody.simdict.SimDict)
    assert pos.unit == 'Mpc'
    assert mass.units == 'Msol'
    assert pos.shape == (39, 3)
    assert mass.shape == (39,)
    assert info['h'] == 0.71
    assert pos.dtype == np.float64
    assert mass.dtype == np.float64


# def test_prepare_mocks(configuration):
#     mocks_config, data = mocks.prepare_mocks(configuration)
#     assert mocks_config['galaxy_coordinates'] == '14h40m56.435s -60d53m48.3s'
#     assert mocks_config['observation_time'] == '2018-04-12T08:00'
#     assert mocks_config['imager_filter'] == 'g'
#     assert mocks_config['pixel_scale'].to(u.arcsec / u.pixel).value ==\
#         pytest.approx(3.506664448419332,
#                       rel=1e-12)
#     assert mocks_config['total_mag'].to(u.ABmag).value == pytest.approx(
#         9.12330433833332,
#         rel=1e-12)


# def test_create_mock_galaxy_noiseless_image(huntsman_sbig_dark_imager,
#                                             configuration):
#     config, galaxy_sim_data = mocks.prepare_mocks(configuration)
#     noiseless_image = \
#         mocks.create_mock_galaxy_noiseless_image(config,
#                                                  galaxy_sim_data,
#                                                  huntsman_sbig_dark_imager)
#     assert isinstance(noiseless_image, CCDData)
#     assert noiseless_image.data.shape == (3326, 2504)
#     assert noiseless_image.data.min() == pytest.approx(1.2747058301560972,
#                                                        rel=1e-12)
#     assert noiseless_image.data.max() == pytest.approx(61.057191738173756,
#                                                        rel=1e-12)
#     assert np.mean(noiseless_image) == pytest.approx(1.2776290071962388,
#                                                      rel=1e-12)
#     assert np.median(noiseless_image) == pytest.approx(1.2747058301560972,
#                                                        rel=1e-12)
#     assert noiseless_image.data.sum() == pytest.approx(10640482.771148466,
#                                                        rel=1e-8)
#     assert np.std(noiseless_image) == pytest.approx(0.11768233792504638,
#                                                     rel=1e-12)
#     assert noiseless_image.unit == "electron / (pix s)"
#     assert noiseless_image.uncertainty is None
#     assert noiseless_image.flags is None


def test_scale_light_by_distance(particle_positions_3D,
                                 mass_weights,
                                 configuration):
    positions, luminosity =\
        mocks.scale_light_by_distance(particle_positions_3D,
                                      mass_weights,
                                      configuration)
    assert type(positions) == np.ndarray
    assert type(luminosity) == u.Quantity
    assert positions.shape == (5, 3)
    assert luminosity.shape == (5,)
    assert positions.max() == pytest.approx(22.000885907655896, rel=1e-12)
    assert luminosity.value.max() == pytest.approx(1.18340324,
                                                   1e-6)


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


def test_compute_apparent_mag(mass_weights,
                              config):
    apparent_ABmag = mocks.compute_apparent_mag(0.0023080874949477728,
                                                mass_weights,
                                                config)
    assert apparent_ABmag.to(u.ABmag).value ==\
        pytest.approx(32.169771876760834,
                      rel=1e-12)


def test_other_axes():
    y, x = mocks.other_axes('z')
    assert y == 1 and x == 2
    z, x = mocks.other_axes('y')
    assert z == 0 and x == 2
    z, y = mocks.other_axes('x')
    assert z == 0 and y == 1
    y, x = mocks.other_axes('Z')
    assert y == 1 and x == 2
    z, x = mocks.other_axes('Y')
    assert z == 0 and x == 2
    z, y = mocks.other_axes('X')
    assert z == 0 and y == 1
    with pytest.raises(KeyError):
        z, x = mocks.other_axes(1)
    with pytest.raises(KeyError):
        z, x = mocks.other_axes("wrong_string")


def test_cut_data(particle_positions_3D,
                  mass_weights):
    data, weights = mocks.cut_data(particle_positions_3D,
                                   mass_weights,
                                   [2, 11],
                                   [7, 16],
                                   [0, 10])
    assert data.shape == (1, 3)
    assert data[0, 0] == 7
    assert data[0, 1] == 8
    assert data[0, 2] == 9
    assert weights == 3


def test_compute_correction_length():
    correction_length = mocks.compute_correction_length(10, 25)
    assert correction_length.unit == 'Mpc'
    assert correction_length.value == pytest.approx(-15.023027612837932,
                                                    rel=1e-12)


def test_compute_depth_distance_to_observer(particle_positions_3D,
                                            mass_weights):
    data, weights =\
        mocks.compute_depth_distance_to_observer(particle_positions_3D,
                                                 mass_weights,
                                                 -5)
    assert data.shape == (3, 3)
    assert weights.shape == (3,)
    assert data[0, 0] == 2
    assert np.median(data) == 9


def test_convert_to_lumonsity_distance(particle_positions_3D):
    data = mocks.convert_to_lumonsity_distance(particle_positions_3D,
                                               'z')
    assert type(data) == u.Quantity
    assert data.shape == particle_positions_3D.shape
    assert data[0, 0].value == 1.000231238130914
    assert np.median(data[:, 0]).value == 7.011334201935506
    assert np.mean(data[:, 0]).value == 7.015500264430392


def test_project_3D_to_2D(particle_positions_3D,
                          mass_weights):
    projected_data, x_bins, y_bins =\
        mocks.project_3D_to_2D(particle_positions_3D,
                               mass_weights,
                               bin_size=1. * u.kpc,
                               x_range=[0, 15] * u.kpc,
                               y_range=[0, 15] * u.kpc)
    assert projected_data.shape == (15, 15)
    assert projected_data[14, 14] == 5
    assert np.median(projected_data) == 0
    assert x_bins.shape == (16,) and y_bins.shape == (16,)
    assert np.median(x_bins.value) == 7.5 and np.median(y_bins.value) == 7.5


def test_AxisNumber():
    assert mocks.AxisNumber.x == 2
    assert mocks.AxisNumber['z'] == 0


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
                                           n_sigma,
                                           max_outlier_fraction):
    data1 = galaxy_sim_data * u.electron / (u.pixel * u.second)
    data2 = galaxy_sim_data * 3600 * u.electron / (u.pixel * u.hour)
    stacked1 = mocks.mock_image_stack(data1, huntsman_sbig_dark_imager)
    stacked2 = mocks.mock_image_stack(data2, huntsman_sbig_dark_imager)
    assert np.mean(stacked1) == pytest.approx(np.mean(stacked2), rel=2)
    assert np.median(stacked1) == pytest.approx(np.median(stacked2), rel=2)
    difference_image = stacked1.subtract(stacked2)
    pixels_different = abs(difference_image.data) > n_sigma * np.std(
        difference_image)
    assert np.count_nonzero(pixels_different) <\
        max_outlier_fraction * stacked1.size


def test_create_cosmology(custom_cosmology_data):
    cosmo = mocks.create_cosmology(custom_cosmology_data)
    assert type(cosmo) == cosmology.core.FlatLambdaCDM
    assert cosmo.H0.value == 70.
    assert cosmo.Tcmb0.value == 2.5
    assert cosmo.Om0 == 0.2865


def test_convert_mass_to_light(config,
                               mass_weights):
    light = mocks.convert_mass_to_light(config,
                                        mass_weights)
    assert isinstance(light, np.ndarray)
    assert light.shape == (5,)
    assert light.unit == u.L_sun
    assert light[0].to(u.L_sun).value == .2
    assert light.sum().to(u.L_sun).value == 3.0


def test_convert_pynbody_to_desired_distance(config,
                                             gadget_pos_data):
    data = mocks.convert_pynbody_to_desired_distance(config,
                                                     gadget_pos_data)
    assert data.shape == (39, 3)
    assert isinstance(data, u.Quantity)
    assert data.unit == 'Mpc'
    assert data.max().value == pytest.approx(1036.84033203125,
                                             rel=1e-8)
    assert data.min().value == pytest.approx(10.83410930633545,
                                             rel=1e-8)
