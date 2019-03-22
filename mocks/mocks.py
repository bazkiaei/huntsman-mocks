import numpy as np

import time

import enum

import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import Distance
from astropy.convolution import convolve
from astropy.nddata import CCDData
from astropy import cosmology

import ccdproc

import gunagala
from gunagala.utils import ensure_unit


def scale_light_by_distance(particle_positions,
                            particle_values,
                            physical_distance=10 * u.Mpc,
                            target_galaxy_comoving_depth=1 * u.Mpc,
                            viewing_axis='z'):
    """
    Projecting 3D data to 2D data with respect to the depth of particles.

    Parameters
    ----------
    particle_positions : numpy.ndarray
        The 3D position of particles, simulations output.
    particle_values : numpy.ndarray
        The luminosity of particles from simulations.
    physical_distance : astropy.units.Quantity, optional
        Distance between the observer and the target galaxy. It should be
        luminosity (physical) distance. Any units are accepted.
    target_galaxy_comoving_depth : astropy.units.Quantity, optional
        The position of the target galaxy along the projection direction in
        the simulation box. Since it is coming from simulations, it is
        co-moving distance.
    viewing_axis : string, optional
        The projection direction. The user should choose among 'x', 'y', 'z',
        'X', 'Y', 'Z'.

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
    physical_distance = ensure_unit(physical_distance, u.Mpc)
    comoving_correction_length = \
        compute_correction_length(physical_distance,
                                  target_galaxy_comoving_depth)
    # Computing the distance of all particles to the observer.
    # The particles positions in GNESIS Simulations are in Mpc units.
    particle_positions, particle_values =\
        compute_depth_distance_to_observer(
            particle_positions,
            particle_values,
            comoving_correction_length,
            viewing_axis=viewing_axis)
    # Converting co-moving distances to luminosity distances.
    particle_positions = convert_to_lumonsity_distance(particle_positions,
                                                       viewing_axis)
    # Applying corrections to the mass weights of particles.
    particle_values =\
        particle_values / (particle_positions[:, AxisNumber[
                           viewing_axis]] / physical_distance) ** 2

    return particle_positions.value, particle_values


def create_mock_galaxy_noiseless_image(galaxy_sim_data_raw,
                                       imager,
                                       sim_arcsec_pixel,
                                       galaxy_coordinates,
                                       observation_time='2018-04-12T08:00',
                                       imager_filter='g',
                                       oversampling=10,
                                       total_mag=9.105):
    """
    This function produces a noiseless image using gunagala psf module.

    Parameters
    ----------
    galaxy_sim_data_raw : numpy.ndarray
        The raw simulation data.
    imager : gunagala.imager.Imager
        Imager instance from gunagala.
    sim_arcsec_pixel : astropy.units.Quantity
        Pixel scale (angle/pixel) of psf_data.
    imager_filter : str, optional
        Optical filter name.
    total_mag : float, optional
        Total apparent ABmag of the simulated object in determined band.
        TODO: Should be defined by another function?

    Returns
    -------
    astropy.nddata.ccddata.CCDData
        Mock simulation data ndarray.
     Note, this function is a quick mock generator using gunagala's PSF
     infrastructure to drop in simulation data as "stars" into an image.
     To derive the PSF, it is assumed that the galaxy is in the centre.
        """

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
                                                           total_mag)])

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

    try:
        real_images = [imager.make_image_real(input_image, exptime)
                       for i in range(n_exposures)]
    except u.UnitConversionError as error:
        message = "Input data units must be e/pixel/s or compatible. Got unit\
conversion error: {}".format(error)
        raise u.UnitsError(message)

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

    Raises
    ------
    ValueError
        The input (viewing_axis) should be one of 'x', 'y', 'z', 'X', 'Y',
        'Z', otherwise the error will be raised.

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
                              target_galaxy_comoving_depth):
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
                                  viewing_axis):
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
