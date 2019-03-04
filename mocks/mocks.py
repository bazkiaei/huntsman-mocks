import os

import numpy as np

import time

import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import Distance
from astropy.convolution import convolve
from astropy.nddata import CCDData
from astropy.io import fits

import ccdproc

import gunagala
from gunagala.utils import ensure_unit


def prepare_mocks(observation_time='2018-04-12T08:00',
                  galaxy_coordinates='14h40m56.435s -60d53m48.3s',
                  filename='input'):
    """
    Creates a dictionary containing configuration data and a numpy.array of
    the simulation data.

    Parameters
    ----------
    observation_time : str, optional
        The date and tim of observation.
        Format: 'yyyy-mm-ddThh:mm'
    galaxy_coordinates : str, optional
        Coordinates of the object.
    filename : str, optional
        The name of the yaml file that contains initial information.

    Returns
    -------
    mock_image_input: dict
    galaxy_sim_data_raw: numpy.array
        Description
    """
    path = os.path.join(os.path.abspath('huntsman-mocks'),
                        'mocks/data/{}.yaml'.format(filename))

    input_info = gg.config.load_config(path)

    mock_image_input = dict()

    mock_image_input['galaxy_coordinates'] = galaxy_coordinates

    mock_image_input['observation_time'] = observation_time

    mock_image_input['imager_filter'] = input_info['imager_filter']

    # Computing the pixel scale.
    sim_pc_pixel = gg.utils.ensure_unit(input_info['sim_pc_pixel'],
                                        u.parsec / u.pixel)
    dis = Distance(input_info['distance'] * u.Mpc)
    z = dis.compute_z(cosmo)
    angular_pc = cosmo.kpc_proper_per_arcmin(z).to(u.parsec / u.arcsec)
    mock_image_input['pixel_scale'] = sim_pc_pixel / angular_pc

    # Reading the data.
    sim_data_path = input_info['data_path']
    galaxy_sim_data_raw = fits.open(sim_data_path)[0].data

    # Computing the total mass of the galaxy:
    p_mass = input_info['particle_baryonic_mass_sim'] * 1e5
    H = input_info['hubble_constant']
    galaxy_mass = np.sum(galaxy_sim_data_raw) * p_mass * H

    band = input_info['imager_filter']
    # Mass to light ration of demanded band (filter).
    M_TO_L = input_info['mass_to_light_ratio']
    m_to_l = M_TO_L[band]
    # AB mag http://mips.as.arizona.edu/~cnaw/sun.html
    M_sun = 5.11 * u.ABmag
    # Total luminosity of the simulated galaxy.
    total_lum_sim = galaxy_mass / m_to_l
    # Total absolute ABmag of the simulated galaxy in the demanded band.
    absolute_ABmag_sim = (-2.5 * np.log10(total_lum_sim) *
                          u.ABmag + M_sun)
    # Distance modulus at redshift `z`.
    mM = cosmo.distmod(z)
    # Total apparent ABmag of the simulated galaxy in the demanded band.
    mock_image_input['total_mag'] = absolute_ABmag_sim + mM

    return mock_image_input, galaxy_sim_data_raw


def create_mock_galaxy_noiseless_image(config,
                                       galaxy_sim_data_raw,
                                       imager,
                                       oversampling=10):
    """
    This function produces a noiseless image using gunagala psf mudule.

    Parameters
    ----------
    config : dict
        a dictionary consist of configuration data for the function
    galaxy_sim_data_raw : numpy.ndarray
        The raw simulation data.
    imager : gunagala.imager.Imager
        Imager instance from gunagala.
    oversampling : int, optional
        Oversampling factor used when shifting & resampling the PSF.

    Returns
    -------
    astropy.nddata.ccddata.CCDData
        Mock simulation data ndarray.
     Note, this function is a quick mock generator using gunagala's PSF
     infrastructure to drop in simulation data as "stars" into an image.
     To derive the PSF, it is assumed that the galaxy is in the centre.

    Deleted Parameters
    ------------------
    sim_arcsec_pixel : astropy.units.Quantity
        Pixel scale (angle/pixel) of psf_data.
    imager_filter : str, optional
        Optical filter name.
    total_mag : float, optional
        Total apparent ABmag of the simulated object in determined band.
        TODO: Should be defined by another function?
    """

    # restoring configuration data.
    sim_arcsec_pixel = config['pixel_scale']
    galaxy_coordinates = config['galaxy_coordinates']
    observation_time = config['observation_time']
    imager_filter = config['imager_filter']
    total_mag = config['total_mag']

    sim_arcsec_pixel = ensure_unit(sim_arcsec_pixel,
                                   u.arcsec / u.pixel)

    galaxy_centre = ((galaxy_sim_data_raw.shape[0] / 2) - .5,
                     (galaxy_sim_data_raw.shape[1] / 2) - .5)

    galaxy_psf = gunagala.psf.PixellatedPSF(galaxy_sim_data_raw,
                                            psf_sampling=sim_arcsec_pixel,
                                            oversampling=oversampling,
                                            psf_centre=galaxy_centre)
    galaxy = gunagala.imager.Imager(optic=imager.optic,
                                    camera=imager.camera,
                                    filters=imager.filters,
                                    psf=galaxy_psf,
                                    sky=imager.sky)

    mock_image_array = galaxy.make_noiseless_image(centre=galaxy_coordinates,
                                                   obs_time=observation_time,
                                                   filter_name=imager_filter,
                                                   stars=[(galaxy_coordinates,
                                                           total_mag.value)])

    return mock_image_array


def convolve_image_psf(input_data,
                       psf_data,
                       convolution_boundary=None,
                       mask_copied_to_output=True):
    """
    Convolves an image with input PSF.

    Parameters
    ----------
    input_data : CCDData, CCDData-like, numpy.ndarray or similar
        Image data that is going to be processed.
    psf_data : numpy.ndarray
        Imager's psf which should be provided by user.
    convolution_boundary : None, optional
        It indicates how to handle boundaries.
    mask_copied_to_output : bool, optional
        Determines that the mask of the input data should be copied to the
        output data or not.

    Returns
    -------
    numpy.ndarray
        Convolved data of the simulated object.
    This function convert an image to a convolved image with input PSF that is
    provided by user. It uses the PSF as kernel for astropy.convolve, which
    only accepts kernels with odd shapes. In addition, the function normalises
    the PSF.

    """
    # check the input data to be CCDData type and convert to it if it is not.
    try:
        # This will work for CCDData or CCDData-like, which have units.
        convolved = CCDData(input_data)

    except ValueError:
        # Try again with manually set units.
        # This will work for numpy.array or similar.
        convolved = CCDData(input_data, unit="electron / (pixel * second)")

    convolved.data = convolve(convolved,
                              psf_data,
                              boundary=convolution_boundary,
                              normalize_kernel=True)

    # Deleting the mask of the input data from the output data if user asks.
    if not mask_copied_to_output:
        convolved.mask = np.ma.nomask

    return convolved


def mock_image_stack(input_image,
                     imager,
                     n_exposures=100,
                     exptime=500 * u.s):
    """
    Creating a sequence of simulated images and stacking them.

    This function takes a noiseless input image (with units of
    photo-electrons/sec/pixel) and creates a series of simulated images with a
    given exposure time, then stacks them into a single output image.

    Parameters
    ----------
    input_image : astropy.nddata.ccddata.CCDData, CCDData-like, numpy.ndarray
    or similar
        The input image which is going to be used to create quasi-real images.
    imager : gunagala.imager.Imager
        Imager instance from gunagala.
    n_exposures : int, optional
        Number of exposures of the imager.
    exptime : TYPE, optional
        The exposures time.

    Returns
    -------
    astropy.nddata.CCDData

    Notes
    -----
        This function is using the `gunagala.Imager.make_image_real` method to
        create simulated images. Check the gunagala package for further
        information.
    """
    # measuring the time for stacking images.
    start_time = time.time()

    # check the input data to be CCDData type and convert to it if it is not.
    try:
        # This will work for CCDData or CCDData-like, which have units.
        input_image = CCDData(input_image)
    except ValueError:
        # Try again with manually set units.
        # This will work for numpy.array or similar.
        input_image = CCDData(input_image, unit="electron / (pixel * second)")

    real_images = [imager.make_image_real(input_image, exptime)
                   for i in range(n_exposures)]

    real_images = ccdproc.Combiner(real_images)

    stacked_image = real_images.average_combine()

    # reporting how long the stacking took.
    print("Stacking ", n_exposures,
          " images took", time.time() - start_time, "to run")

    return stacked_image


def compute_pixel_scale(distance=10.,
                        sim_pc_pixel=170):
    """
    This function produces pixel scales for the main functions
    of mocks.py

    Parameters
    ----------
    distance : float, optional
        The assumed distance between the telescope and the simulated galaxy
        in Mpc units.
    sim_pc_pixel : astropy.units, optional
        The resolution of the simulation in parsec/pixel units.

    Returns
    -------
    float
        Pixel scale that will be used by other function of mocks.py.
    """
    sim_pc_pixel = ensure_unit(sim_pc_pixel, u.parsec / u.pixel)
    d = Distance(distance * u.Mpc)
    z = d.compute_z(cosmo)
    angular_pc = cosmo.kpc_proper_per_arcmin(z).to(u.parsec / u.arcsec)
    image_arcsec_pixel = sim_pc_pixel / angular_pc

    return image_arcsec_pixel
