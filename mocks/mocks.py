import os

import yaml

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
                  config_filename='config_example',
                  folder_name='config_directory'):
    """
    Creates a dictionary containing configuration data and a numpy.array of
    the simulation data.

    Parameters
    ----------
    observation_time : astropy.time.Time or str, optional
        The date and time of the observation, default 2018-04-12T08:00.
    galaxy_coordinates : astropy.coordinates.SkyCoord, optional
        Coordinates of the object, default 14h40m56.435s -60d53m48.3s.
    config_filename : str, optional
        The name of the yaml file  that contains initial information, default
        config_example.yaml.
    folder_name : str, optional
        The name of the folder that the yaml file is in it.
        The folder should be in the huntsman-mocks package, default
        config_directory.

    Returns
    -------
    mock_image_input: dict
    galaxy_sim_data_raw : numpy.array
        Description
    """
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../'))
    config_file_path = os.path.join(path,
                                    folder_name,
                                    '{}.yaml'.format(config_filename))

    input_info = dict()
    with open(config_file_path, 'r') as f:
        c = yaml.load(f.read())
        input_info.update(c)

    z = compute_redshift(input_info['physical_distance'])

    # Computing the pixel scale.
    input_info['pixel_scale'] = compute_pixel_scale(z,
                                                    input_info['sim_pc_\
pixel'])

    # Reading the data.
    sim_data_path = input_info['data_path']
    galaxy_sim_data_raw = fits.open(sim_data_path)[0].data
    # Computing the total mass of the galaxy:
    galaxy_mass = compute_total_mass(galaxy_sim_data_raw,
                                     input_info['particle_baryonic_mass_sim'],
                                     mass_factor=1e5,
                                     H=input_info['hubble_constant'])

    band = input_info['imager_filter']
    # Mass to light ratio of demanded band (filter).
    M_TO_L = input_info['mass_to_light_ratio']
    m_to_l = M_TO_L[band]
    # Total apparent ABmag of the simulated galaxy in the demanded band.
    total_apparent_ABmag_sim = compute_apparent_ABmag(z,
                                                      galaxy_mass,
                                                      m_to_l)

    input_info['total_mag'] = total_apparent_ABmag_sim

    return input_info, galaxy_sim_data_raw


def create_mock_galaxy_noiseless_image(config,
                                       galaxy_sim_data_raw,
                                       imager,
                                       oversampling=10):
    """
    This function produces a noiseless image using gunagala psf module.

    Parameters
    ----------
    config : dict
        a dictionary consist of configuration data for the function
    galaxy_sim_data_raw : numpy.ndarray
        The raw simulation data.
    imager : gunagala.imager.Imager
        Imager instance from gunagala.
    oversampling : int, optional
        Oversampling factor used when shifting & re-sampling the PSF.

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
    only accepts kernels with odd shapes. In addition, the function normalizes
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


def compute_redshift(distance):
    """
    This function computes redshift for the provided distance.

    Parameters
    ----------
    distance : float
        The provided distance.

    Returns
    -------
    float
        The computed redshift.
    """
    distance = ensure_unit(distance, u.Mpc)
    d = Distance(distance)
    z = d.compute_z(cosmo)
    return z


def compute_pixel_scale(z,
                        sim_pc_pixel=170):
    """
    This function produces pixel scales for the main functions
    of mocks.py

    Parameters
    ----------
    z : float
        The redshift of the target galaxy at the demanded distance.
    sim_pc_pixel : astropy.units, optional
        The resolution of the simulation in parsec/pixel units.

    Returns
    -------
    float
        Pixel scale that will be used by other function of mocks.py.
    """
    sim_pc_pixel = ensure_unit(sim_pc_pixel, u.parsec / u.pixel)
    proper_parsec_per_arcsec =\
        cosmo.kpc_proper_per_arcmin(z).to(u.parsec / u.arcsec)
    image_arcsec_pixel = sim_pc_pixel / proper_parsec_per_arcsec

    return image_arcsec_pixel


def compute_total_mass(galaxy_sim_data_raw,
                       particle_mass_sim,
                       mass_factor=1e5,
                       H=0.705):
    """
    With respect to input data, this function computes the total mass of the
    favored object.

    Parameters
    ----------
    galaxy_sim_data_raw : numpy.ndarray
        A 2D array, representing the number particles in each pixel.
    particle_mass_sim : float
        The mass of one particle. The simulation information should provide
        it.
    mass_factor : float, optional
        The factor that determines the  real mass of each particle, default
        1e5.
    H : float, optional
        Hubble constant. The simulation information should provides it,
        default 0.705.

    Returns
    -------
    float
        The total mass of the target galaxy and all masses in its environment.
    """
    p_mass = particle_mass_sim * mass_factor
    galaxy_mass = np.sum(galaxy_sim_data_raw) * p_mass * H
    return galaxy_mass


def compute_apparent_ABmag(z,
                           galaxy_mass,
                           mass_to_light):
    """
    This function computes the apparent magnitude of the target galaxy and its
    environment with respect to the demanded distance.

    Parameters
    ----------
    z : float
        The redshift of the target galaxy at the demanded distance.
    galaxy_mass : float
        The total mass of the target galaxy and all masses in its environment
        in solar mass units.
    mass_to_light : float
        The mass to light ratio of the target.

    Returns
    -------
    float
        The apparent magnitude of the target and its environment.
    """
    # AB mag http://mips.as.arizona.edu/~cnaw/sun.html
    abs_mag_sun = 5.11
    # Total luminosity of the simulated galaxy.
    total_lum_sim = galaxy_mass / mass_to_light
    # Total absolute ABmag of the simulated galaxy in the demanded band.
    absolute_ABmag = (abs_mag_sun - 2.5 * np.log10(total_lum_sim)) * u.ABmag
    # Distance modulus at redshift `z`.
    distance_modulus = cosmo.distmod(z)

    apparent_mag = absolute_ABmag + distance_modulus
    return apparent_mag
