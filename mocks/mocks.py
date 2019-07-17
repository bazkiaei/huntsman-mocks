import numpy as np

import time

import enum

import sys

import astropy.units as u
from astropy.cosmology import WMAP9
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import Distance
from astropy.convolution import convolve
from astropy.nddata import CCDData
from astropy import cosmology
from astropy.io import fits

import ccdproc

import logging

import logbook

import pynbody

import gunagala
from gunagala.utils import ensure_unit

from mocks.utils import load_yaml_config
from mocks import utils


def crop_simulation_data(particle_pos,
                         particle_mass,
                         z_range=[35.69, 35.80] * u.Mpc,
                         y_range=[25.78, 25.89] * u.Mpc,
                         x_range=[0, 71.] * u.Mpc):
    """
    Does some steps to prepare the data for the progress. This function will
    not remain in its current shape and likely will be removed.

    Parameters
    ----------
    particle_pos : numpy.ndarray
        The 3D position of particles, simulations output.
    particle_mass : numpy.ndarray
        The mass of particles from simulations.
    z_range : astropy.units.Quantity, optional
        The range of the data in the z direction.
    y_range : astropy.units.Quantity, optional
        The range of the data in the y direction.
    x_range : astropy.units.Quantity, optional
        The range of the data in the x direction.

    Returns
    -------
    numpy.ndarray
        The prepared data for the main functions of the code.
    """
    pos, mass = utils.select_low_mass_star_particles(particle_pos,
                                                     particle_mass)
    pos = utils.position_data(pos)
    pos, mass = cut_data(pos,
                         mass,
                         z_range,
                         y_range,
                         x_range)
    return pos, mass


def init_mocks(config):
    """
    Creates some initials for the code. This function will be converted to
    the __init__ when the code uses class.

    Parameters
    ----------
    config : dict
        A dictionary of configuration items.

    Returns
    -------
    config : dict
        A dictionary of configuration items.
    huntsman : gunagala.imager.Imager
        Imager instance from gunagala.
    psf_data : numpy.ndarray
        Imager's psf data.
    cosmo : astropy.cosmology.core.FlatLambdaCDM
        The main cosmology that will be used by the code.
    redshift : numpy.float
        The redshift of the galaxy computed with respect to the distance of
        the observer to the galaxy.
    """
    cosmo = create_cosmology(config)
    redshift = compute_redshift(config['galaxy_distance'],
                                cosmo)
    config['pixel_scale'] = compute_pixel_scale(redshift,
                                                cosmo,
                                                config['sim_pc_pixel'])
    imagers = gunagala.imager.create_imagers()
    huntsman = imagers[config['imager']]
    psf_data = huntsman.psf.pixellated()
    return config, huntsman, psf_data, cosmo, redshift


def create_logger():
    """
    Creates logger based on logging module.

    Returns
    -------
    logging.Logger
        The logger to use for logging system.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    return logger


def create_logbook():
    """
    Creates logger based on logbook module.

    Returns
    -------
    logbook.base.Logger
        The logger to use for logging system.
    """
    logbook.StreamHandler(sys.stdout).push_application()
    return logbook.Logger('logbook')


def parse_config(config_file='config_example.yaml',
                 verbose=True,
                 **kwargs):
    """
    Parses the configuration.

    Parameters
    ----------
    config_file : str, optional
        The path to the configuration file or its name if it is in the folder
        `config_directory` of `huntsman_mocks`.
    **kwargs
        The user can override the value of a quantity from the configuration
        file and give another value to that quantity by kwargs.

    Returns
    -------
    dict
        A dictionary that contains configuration.

    Notes
    -----
        This function loads the configuration file with
        `utils.load_yaml_config`, then goes through all the parameters and
        applies any keyword arguments that override the configuration file
        values, gets the remaining values from the configuration file, adds
        default values for any parameters that are missing, and does any
        required unit conversions. This function returns the fully populated,
        validated configuration dictionary for all the other functions to use.
    """
    logger = create_logbook()

    logger.info('Parsing configuration')
    config = load_yaml_config(config_file)
    # Coordinates of the target galaxy.
    config['galaxy_coordinates'] =\
        kwargs.get('galaxy_coordinates',
                   config.get('galaxy_coordinates',
                              '14h40m56.435s -60d53m48.3s'))
    # The time of the observation.
    config['observation_time'] = kwargs.get('observation_time',
                                            config['observation_time'])
    # The path to the data.
    # This will be removed since we will not use fits files as input.
    config['data_path'] = kwargs.get('data_path',
                                     config['data_path'])
    # The path to the simulation data.
    config['sim_data_path'] = kwargs.get('sim_data_path',
                                         config['sim_data_path'])
    config['imager_filter'] = kwargs.get('imager_filter',
                                         config['imager_filter'])
    config['imager'] = kwargs.get('imager',
                                  config['imager'])
    config['mass_to_light_ratio'] = kwargs.get('mass_to_light_ratio',
                                               config['mass_to_light_ratio'])
    for i in config['mass_to_light_ratio']:
        config['mass_to_light_ratio'][i] = ensure_unit(
            config['mass_to_light_ratio'][i],
            u.M_sun / u.L_sun)
    config['abs_mag_sun'] = kwargs.get('abs_mag_sun',
                                       config['abs_mag_sun'])
    for i in config['abs_mag_sun']:
        config['abs_mag_sun'][i] = ensure_unit(
            config['abs_mag_sun'][i],
            u.M_sun / u.L_sun)
    # The distance between the observer and the target galaxy. This is
    # luminosity distance.
    config['galaxy_distance'] = kwargs.get('galaxy_distance',
                                           config['galaxy_distance'])
    config['galaxy_distance'] = ensure_unit(config['galaxy_distance'], u.Mpc)
    # The position of the target galaxy along the projection direction in the
    # simulation box. Since it is coming from simulations, it is co-moving
    # distance in Mpc.
    config['target_galaxy_comoving_depth'] =\
        kwargs.get('target_galaxy_comoving_depth',
                   config['target_galaxy_comoving_depth'])
    config['target_galaxy_comoving_depth'] = ensure_unit(
        config['target_galaxy_comoving_depth'], u.Mpc)
    config['viewing_axis'] = kwargs.get('viewing_axis',
                                        config['viewing_axis'])
    # The resolution of the simulation(parsec / pixel).
    config['sim_pc_pixel'] = kwargs.get('sim_pc_pixel',
                                        config['sim_pc_pixel'])
    config['sim_pc_pixel'] = ensure_unit(config['sim_pc_pixel'],
                                         u.pc / u.pixel)
    config['particle_baryonic_mass_sim'] =\
        kwargs.get('particle_baryonic_mass_sim',
                   config['particle_baryonic_mass_sim'])
    # To Do: The Hubble constant in the simulations is h, which is H / 100, it
    # should be checked again that what constant is used in where and make
    # proper conversion where it applies.
    config['hubble_constant'] = kwargs.get('hubble_constant',
                                           config['hubble_constant'])
    config['hubble_constant'] = ensure_unit(config['hubble_constant'],
                                            u.km / (u.Mpc * u.s))
    if config['hubble_constant'].value < 1:
        config['hubble_constant'] = config['hubble_constant'] * 100
        config['hubble_constant'] = ensure_unit(config['hubble_constant'],
                                                u.km / (u.Mpc * u.s))
    # Density of matter(Dark + baryonic)
    config['Omega_0'] = kwargs.get('Omega_0',
                                   config['Omega_0'])
    # Density of baryonic matter.
    config['Omega_b'] = kwargs.get('Omega_b',
                                   config['Omega_b'])
    # Dark Energy density
    config['Omega_Lambda'] = 1 - config['Omega_0']
    # Temperature of the CMB at z=0.
    config['T_CMB0'] = kwargs.get('T_CMB0',
                                  config['T_CMB0'])
    config['T_CMB0'] = ensure_unit(config['T_CMB0'],
                                   u.K)
    # Effective number of Neutrino species.
    config['Neff'] = kwargs.get('Neff',
                                config['Neff'])
    # Mass of each neutrino species.
    config['m_nu'] = kwargs.get('m_nu',
                                config['m_nu'])
    config['m_nu'] = ensure_unit(config['m_nu'],
                                 u.eV)
    # Redshift
    config['redshift'] = kwargs.get('redshift',
                                    config['redshift'])

    logger.info('Parsing configuration completed')
    return config


def read_gadget(config):
    """
    Reads the raw output of simulations based on gadget code.

    Parameters
    ----------
    config : dict
        A dictionary of configuration items.

    Returns
    -------
    desired_pos : astropy.units.Quantity
        The positions of the simulation particles (only stars at the moment).
    particle_value : pynbody.array.SimArray
        The mass of the simulation particles (only stars at the moment).
    sim_properties : pynbody.simdict.SimDict
        Contains cosmological information of the simulation.
    """
    sim_data = pynbody.load(config['sim_data_path'])
    particle_pos = sim_data.stars['pos']
    # Converting the distances in viewing axis to co-moving distances and
    # distances in viewing plane to proper distances.
    desired_pos = convert_pynbody_to_desired_distance(config,
                                                      particle_pos)
    desired_pos = desired_pos.astype(np.float64)
    particle_value = sim_data.stars['mass'].in_units('Msol')
    particle_value = particle_value.astype(np.float64)
    sim_properties = sim_data.properties
    return desired_pos, particle_value, sim_properties


def scale_light_by_distance(particle_positions,
                            particle_values,
                            config):
    """
    Projecting 3D data to 2D data with respect to the depth of particles.

    Parameters
    ----------
    particle_positions : numpy.ndarray
        The 3D position of particles, simulations output.
    particle_values : numpy.ndarray
        The luminosity of particles from simulations.
    config : dict
        A dictionary of configuration items.

    Returns
    -------
    numpy.ndarray
        Projected 2D data.

    Notes
    -----
        To consider the depth of particles' positions for their contribution
        in the total luminosity, this function computes each particle's
        distance to the observer first. To do that, user should provide the
        position of the target galaxy along the viewing axis (projection axis)
        which is called `target_galaxy_comoving_depth` and the requested
        distance between the observer and the target galaxy which is called
        `physical_distance` here in this function. The difference between
        these two, which we call it `correction_length` here in this function,
        will add to the other particles positions to compute their distances
        to the observer. During this process, the function considers the
        difference between co-moving and luminosity distances.
        Then, it scales the contribution of each particle in the luminosity
        whit respect to its distance to the observer.
        At the moment, for the particles at the requested distance to the
        observer (`physical_distance`), the scale factor is one, for the
        particles closer to the observer it is more than one and for the
        farther particles it is less than one.
        Assumed that the distances between particles along other axes than the
        viewing axis, which are co-moving distances, are small enough.
        Therefore, the function does not convert those positions to luminosity
        distance.

    """
    # Computing the length value that should be added to the positions of
    # particles in the viewing axis (projection axis) as correction.
    cosmo = create_cosmology(config)
    comoving_correction_length = \
        compute_correction_length(config['galaxy_distance'],
                                  config['target_galaxy_comoving_depth'],
                                  cosmo)
    # Computing the distance of all particles to the observer.
    # The particles positions in GNESIS Simulations are in Mpc units.
    particle_positions, particle_values =\
        compute_depth_distance_to_observer(
            particle_positions,
            particle_values,
            comoving_correction_length,
            viewing_axis=config['viewing_axis'])
    # Converting co-moving distances to luminosity distances.
    particle_positions = convert_to_lumonsity_distance(particle_positions,
                                                       config['viewing_axis'],
                                                       cosmo)
    # Applying corrections to the mass weights of particles.
    particle_values =\
        particle_values / (particle_positions[:, AxisNumber[
            config['viewing_axis']]] / config['galaxy_distance']) ** 2

    return particle_positions.value, particle_values


# def prepare_mocks(config):
#     """
#     Creates a dictionary containing configuration data and a numpy.array of
#     the simulation data.

#     Parameters
#     ----------
#     config : dict
#         A dictionary of configuration items.

#     Returns
#     -------
#     mock_image_input: dict
#     galaxy_sim_data_raw : numpy.array
#         Prepared information for creating mock images and the simulation data.

#     Deleted Parameters
#     ------------------
#     config_location : str, optional
#         The name (location) of the yaml file that contains initial
#         information, default `config_example.yaml`.
#     """
#     # Creating the cosmology
#     cosmo = create_cosmology(config)

#     z = compute_redshift(config['galaxy_distance'],
#                          cosmo)

#     # Computing the pixel scale.
#     config['pixel_scale'] = compute_pixel_scale(z,
#                                                 cosmo,
#                                                 config['sim_pc_\
# pixel'])
#     Reading the data.
#     with fits.open(config['data_path']) as sim_fits:
#         galaxy_sim_data_raw = sim_fits[0].data
#     # Computing the total mass of the galaxy:
#     galaxy_mass = compute_total_mass(galaxy_sim_data_raw,
#                                      config['particle_baryonic_mass_sim'],
#                                      mass_factor=1e5,
#                                      H=config['hubble_constant'])

#     band = config['imager_filter']
#     # Mass to light ratio for demanded band (filter).
#     mass_to_light = config['mass_to_light_ratio'][band]
#     # Absolute magnitude of the Sun for the demanded band.
#     abs_mag_sun = config['abs_mag_sun'][band]
#     # Total apparent ABmag of the simulated galaxy in the demanded band.
#     total_apparent_ABmag_sim = compute_apparent_ABmag(z,
#                                                       galaxy_mass,
#                                                       mass_to_light,
#                                                       abs_mag_sun,
#                                                       cosmo)

#     config['total_mag'] = total_apparent_ABmag_sim

#     return config, galaxy_sim_data_raw


def create_mock_galaxy_noiseless_image(config,
                                       galaxy_sim_data_raw,
                                       imager,
                                       oversampling=10):
    """
    This function produces a noiseless image using gunagala psf module.

    Parameters
    ----------
    config : dict
        The first output of the `mocks.prepare_mocks` function, a dictionary
        consist of configuration data for the function.
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

    Raises
    ------
    u.UnitsError
        If the units of the input_image is not compatible this erroe will
        raise.
    """
    logger = create_logbook()

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
        logger.info('The input image data converted to CCDData.')

    try:
        logger.info('Creating noisy images started.')
        real_images = [imager.make_image_real(input_image, exptime)
                       for i in range(n_exposures)]
        logger.info('Creating noisy images finished.')
    except u.UnitConversionError as error:
        message = "Input data units must be e/pixel/s or compatible. Got unit\
conversion error: {}".format(error)
        raise u.UnitsError(message)

    logger.info('Combining noisy images.')
    real_images = ccdproc.Combiner(real_images)

    logger.info('Averaging combined images.')
    stacked_image = real_images.average_combine()
    logger.info('The stacked image is ready.')

    # reporting how long the stacking took.
    print("Stacking ", n_exposures,
          " images took", time.time() - start_time, "to run")

    return stacked_image


def compute_redshift(distance,
                     cosmo=WMAP9):
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
                        cosmo=WMAP9,
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


def compute_apparent_mag(z,
                         sim_particle_luminosities,
                         config,
                         mag_units=u.ABmag,
                         cosmo=WMAP9):
    """
    This function computes the apparent magnitude of the target galaxy and its
    environment with respect to a given distance.

    Parameters
    ----------
    z : float
        The redshift of the target galaxy at the demanded distance.
    sim_particle_luminosities : astropy.units.Quantity
        The light of the simulation particles.
    config : dic
        A dictionary of configuration items.
    cosmo : astropy.cosmology.core.FlatLambdaCDM, optional
        The main cosmology, default WMAP9.

    Returns
    -------
    float
        The apparent magnitude of the target and its environment.

    """
    sim_particle_luminosities = ensure_unit(sim_particle_luminosities, u.L_sun)
    # Total luminosity of the simulated galaxy in solar luminosities,
    # integrated over the demanded filter band.
    total_lum_sim = sim_particle_luminosities.sum()
    # Total absolute ABmag of the simulated galaxy in the demanded band.
    absolute_mag =\
        (config['abs_mag_sun'][config['imager_filter']].value -
            2.5 * np.log10(total_lum_sim.value)) * mag_units
    # Distance modulus at redshift `z`.
    distance_modulus = cosmo.distmod(z)

    apparent_mag = absolute_mag + distance_modulus
    return apparent_mag


def other_axes(axis_name):
    """
    Accepts the name of an axis and returns the other two axes.

    Parameters
    ----------
    axis_name : string
        The name of an axis. Usually the viewing axis which will be used in
        `scale_light_by_distance` function.

    Returns
    -------
    tuple
        Two other axes than the input in sequential order with respect to the
        Numpy axis ordering.

    Notes
    -----
        The main use of this function is in 'scale_light_by_distance' where
        the user provides the projection axis as `viewing_axis` and wants to
        project the data to a 2D plate. This function makes it possible to
        keep the sequential order of the 2D plate axes.

    input           outputs
    -----------------------
    'z'             1, 2    for (y, x)
    'y'             0, 2    for (z, x)
    'x'             0, 1    for (z, y)
    """
    axes = {AxisNumber.z, AxisNumber.y, AxisNumber.x}
    axes.remove(AxisNumber[axis_name])
    return axes


def cut_data(particle_positions,
             particle_values,
             z_range=[24.8, 25.2] * u.Mpc,
             y_range=[24.8, 25.2] * u.Mpc,
             x_range=[24.8, 25.2] * u.Mpc):
    """
    This function cut the data with respect to the requested borders and it
    keeps the weights of the particles inside the borders.

    Parameters
    ----------
    particle_positions : numpy.ndarray
        The 3D position of particles.
    particle_values : numpy.ndarray
        The values of the particles (could be the luminosity weight of them).
    z_range : list, optional
        The desired range in the z direction (in Mpc).
    y_range : list, optional
        The desired range in the y direction (in Mpc).
    x_range : list, optional
        The desired range in the x direction (in Mpc).

    Returns
    -------
    particle_positions: numpy.ndarray
    particle_values: numpy.ndarray
        The cut data and the mass wights components corresponding to the cut
        data are outputs of this function.
    """
    z_range = ensure_unit(z_range, u.Mpc)
    y_range = ensure_unit(y_range, u.Mpc)
    x_range = ensure_unit(x_range, u.Mpc)
    for i, minmax in enumerate([z_range, y_range, x_range]):
        low_limit = minmax[0].value
        upper_limit = minmax[1].value
        selection = \
            np.where(np.logical_and((low_limit < particle_positions[:, i]),
                                    (particle_positions[:, i] < upper_limit)))
        particle_positions = particle_positions[selection]
        particle_values = particle_values[selection]
    return particle_positions, particle_values


def compute_correction_length(distance,
                              target_galaxy_comoving_depth,
                              cosmo=WMAP9):
    """
    This function computes the correction that should be added to the position
    of the particles along the projection axis.

    Parameters
    ----------
    distance : astropy.units.Quantity
        Distance between the observer and the target galaxy. The user should
        provide it and it is luminosity distance.
    target_galaxy_comoving_depth : astropy.units.Quantity
        The position of the target galaxy along the projection direction in
        the simulation box.

    Returns
    -------
    astropy.units.Quantity
        The value that should be added to the positions of particles as
        correction.

    Notes
    -----
        Whit respect to the position of the target galaxy and the requested
        distance between it and the observer, this function computes the
        correction that should be added to the position of all particles such
        that the target galaxy places on the requested distance to the
        observer.
        The user should notice that the requested distance is luminosity
        distance while the correction length and the position of the target
        galaxy (`target_galaxy_comoving_depth`), which this function computes,
        are co-moving distances.
    """
    distance = ensure_unit(distance, u.Mpc)
    # Converting the distance between observer and the target galaxy, from
    # luminosity distance to co-moving distance.
    obs_dis = Distance(distance)
    z = obs_dis.compute_z(cosmo)
    obs_comoving_dis = cosmo.comoving_distance(z)
    # Finding correction value for all positions.
    target_galaxy_comoving_depth = ensure_unit(target_galaxy_comoving_depth,
                                               u.Mpc)
    correction_length = obs_comoving_dis - target_galaxy_comoving_depth
    return correction_length


def compute_depth_distance_to_observer(particle_positions,
                                       particle_values,
                                       correction,
                                       viewing_axis='z'):
    """
    This function computes the depth distance of all particles to the observer
    with respect to the correction length. It will trim the data if the
    particle is at a distance less than 9.1e-5 Mpc away

    Parameters
    ----------
    particle_positions : numpy.ndarray
        The 3D position of particles.
    particle_values : numpy.ndarray
        The values of the particles (could be the luminosity weight of them).
    correction : astropy.units.Quantity
        The value that should be added to the positions of particles as
        correction.
    viewing_axis : str, optional
        The name of the viewing axis (projection axis).

    Returns
    -------
    particle_positions: numpy.ndarray
    particle_values: astropy.units.Quantity
        The data and the mass wights components corresponding to the data are
        outputs of this function.

    Notes
    -----
        This function applies the correction length to the position of
        particles through the viewing axis. It trims the data by discarding
        positions less than 9.1e-5 Mpc.
        The function `scale_light_by_distance` gets the main use of this
        function and the input and output array (`data`) which represent
        particle positions, are co-moving distances.
    """
    correction = ensure_unit(correction, u.Mpc)
    # Computing the distance of all particles to the observer.
    # The particles positions in GNESIS Simulations are in Mpc units.
    particle_positions[:, AxisNumber[viewing_axis]]\
        = particle_positions[:, AxisNumber[viewing_axis]] + correction.value
    # Removing unaccepted elements.
    if particle_positions[particle_positions[:, AxisNumber[viewing_axis]] <=
                          9.1e-5].any():
        wanted_ind = \
            np.where(particle_positions[:, AxisNumber[viewing_axis]] > 9.1e-5)
        particle_values = particle_values[wanted_ind]
        particle_positions = \
            particle_positions[
                particle_positions[:, AxisNumber[viewing_axis]] > 9.1e-5]
    return particle_positions, particle_values


def convert_to_lumonsity_distance(particle_positions,
                                  viewing_axis,
                                  cosmo=WMAP9):
    """
    This function converts the co-moving distance to the luminosity distance.

    Parameters
    ----------
    particle_positions : numpy.ndarray
        The 3D position of particles.
    viewing_axis : str
        The name of the viewing axis (projection axis).

    Returns
    -------
    astropy.units.Quantity
        The data with converted to luminosity distances along viewing axis.
    """
    particle_positions = ensure_unit(particle_positions, u.Mpc)
    zmin = cosmology.z_at_value(cosmo.comoving_distance,
                                particle_positions[:, AxisNumber[
                                    viewing_axis]].min())
    zmax = cosmology.z_at_value(
        cosmo.comoving_distance,
        particle_positions[:, AxisNumber[viewing_axis]].max())
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
    Dgrid = cosmo.comoving_distance(zgrid)
    zvals = np.interp(
        particle_positions[:, AxisNumber[viewing_axis]].value,
        Dgrid.value, zgrid)
    particle_positions[:, AxisNumber[viewing_axis]] \
        = cosmo.luminosity_distance(zvals)
    return particle_positions


def project_3D_to_2D(particle_positions,
                     particle_values,
                     bin_size=100. * u.pc,
                     x_range=[0, 200.] * u.kpc,
                     y_range=[0, 200.] * u.kpc,
                     viewing_axis='z'):
    """
    This function projects 3D data to 2D along the requested axis with respect
    to the light of particles.

    Parameters
    ----------
    particle_positions : numpy.ndarray
        The 3D position of particles.
    particle_values : numpy.ndarray
        The luminosity of particles.
    bin_size : astropy.units.Quantity, optional
        The pixel size of the output.
    x_range : astropy.units.Quantity, optional
        The desired range in the projected 2D plane for the x direction.
    y_range : astropy.units.Quantity, optional
        The desired range in the projected 2D plane for the y direction.
    viewing_axis : str, optional
        The name of the viewing axis (projection axis).

    Returns
    -------
    data_2D: numpy.ndarray
    x and y: astropy.units.Quantity
        The projected 2D array with respect to the luminosity of each particle
        and the the bin edges along the x and y dimensions.
    """
    y_axis, x_axis = other_axes(viewing_axis)
    bin_size = ensure_unit(bin_size, u.kpc)
    x_range = ensure_unit(x_range, u.kpc)
    y_range = ensure_unit(y_range, u.kpc)
    steps = bin_size.value
    yedges = np.arange(y_range[0].value, y_range[1].value + (steps / 2), steps)
    xedges = np.arange(x_range[0].value, x_range[1].value + (steps / 2), steps)

    particle_positions_2D, y, x = np.histogram2d(particle_positions[:, y_axis],
                                                 particle_positions[:, x_axis],
                                                 bins=(yedges, xedges),
                                                 weights=particle_values)
    return particle_positions_2D, x * u.kpc, y * u.kpc


def create_cosmology(custom_cosmology_parameters,
                     default_cosmology=WMAP9):
    """
    Customizes the cosmology parameters.

    Parameters
    ----------
    custom_cosmology_parameters : dict
        Contains preferred cosmology parameters.
    default_cosmology : astropy.cosmology.core.FlatLambdaCDM, optional
        The main cosmology, default WMAP9.

    Returns
    -------
    astropy.cosmology.core.FlatLambdaCDM
        Customized cosmology.

    Notes
    -----
        This function creates a flat Lambda CDM cosmology using the
        hubble_constant, Omega_0, T_CMB0, N_eff, mass_nu and Omega_b0
        parameters from custom_cosmology_parameters and for parameters
        that aren't in custom_cosmology_parameters, from default_cosmology.
    """
    H0 = custom_cosmology_parameters.get('hubble_constant',
                                         default_cosmology.H0)
    H0 = ensure_unit(H0, u.km / (u.s * u.Mpc))
    Om0 = custom_cosmology_parameters.get('Omega_0', default_cosmology.Om0)
    Tcmb0 = custom_cosmology_parameters.get('T_CMB0', default_cosmology.Tcmb0)
    Neff = custom_cosmology_parameters.get('N_eff', default_cosmology.Neff)
    m_nu = custom_cosmology_parameters.get('mass_nu', default_cosmology.m_nu)
    Ob0 = custom_cosmology_parameters.get('Omega_b0', default_cosmology.Ob0)
    cosmology = FlatLambdaCDM(H0,
                              Om0,
                              Tcmb0,
                              Neff,
                              m_nu,
                              Ob0)
    return cosmology


def convert_mass_to_light(config,
                          mass,
                          band='g'):
    """
    Converts the mass of the galaxy to light.

    Parameters
    ----------
    config : dic
        A dictionary of configuration items.
    mass : pynbody.array.SimArray
        The mass of the simulation particles.
    band : str, optional
        The filter of the observation.

    Returns
    -------
    astropy.units.Quantity
        The light value of the simulation particles.
    """
    mass = ensure_unit(mass, u.M_sun)
    mass = mass / config['mass_to_light_ratio'][band]
    return mass


def convert_pynbody_to_desired_distance(configuration,
                                        positions):
    """
    Converts the positions to co-moving distance (in viewing axis) and proper
    distance in vertical direction of that.

    Parameters
    ----------
    configuration : dict
        A dictionary of configuration items.
    positions : pynbody.array.SimArray
        The positions of the particles.

    Returns
    -------
    astropy.units.Quantity
        The positions of particles in the desired format.

    Notes
    -----
        Distances between simulations particles to the observer should be
        luminosity distances while the transparent distances which the
        observer measures should be proper distances. Pynbody provides data in
        units of 'Mpc a / h'. To convert the data to co-moving distance, this
        function converts it to the units of 'Mpc' for the viewing axis
        direction. For the other two directions, the vertical plane, this
        function converts the distances to the units of 'Mpc a' which is the
        proper distance.
    """
    desired_pos = np.zeros(positions.shape)
    for i in range(3):
        # Viewing plane directions:
        if AxisNumber[configuration['viewing_axis']] != i:
            desired_pos[:, i] = positions[:, i].in_units('Mpc a')
        # Viewing axis direction:
        if AxisNumber[configuration['viewing_axis']] == i:
            desired_pos[:, i] = positions[:, i].in_units('Mpc')
    del(positions)
    desired_pos = ensure_unit(desired_pos, u.Mpc)
    return desired_pos


class AxisNumber(enum.IntEnum):

    """
    This class relates axes' to integers.

    Attributes
    ----------
    x : int
        X axis representative.
    X : int
        X axis representative.
    y : int
        Y axis representative.
    Y : int
        Y axis representative.
    z : int
        Z axis representative.
    Z : int
        Z axis representative.

    Notes
    -----
    We prefer to use strings for representing the axes, but in some
    circumstances it is easier to use integers instead of strings. This class
    provides the ability to relates an integer to the axes' names with respect
    to the Numpy axis ordering.
    """

    x = 2
    y = 1
    z = 0
    X = 2
    Y = 1
    Z = 0
