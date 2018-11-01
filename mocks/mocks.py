import numpy as np

import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import Distance
from astropy.convolution import convolve

import gunagala as gg


def create_mock_galaxy_noiseless_image(galaxy_sim_data_raw,
                                       imager,
                                       sim_arcsec_pixel,
                                       galaxy_coordinates,
                                       observation_time='2018-04-12T08:00',
                                       imager_filter='g',
                                       total_mag=9.):
    """
    This function produces a noiseless image using gunagala psf mudule.

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

    sim_arcsec_pixel = gg.utils.ensure_unit(sim_arcsec_pixel,
                                            u.arcsec / u.pixel)
    galaxy_centre = (galaxy_sim_data_raw.shape[0] / 2,
                     galaxy_sim_data_raw.shape[1] / 2)

    galaxy_psf = gg.psf.PixellatedPSF(galaxy_sim_data_raw,
                                      psf_sampling=sim_arcsec_pixel,
                                      oversampling=10,
                                      psf_centre=galaxy_centre)
    galaxy = gg.imager.Imager(optic=imager.optic,
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
                       image_arcsec_pixel,
                       oversampling=10,
                       convolution_boundary="None"):
    """
    Convolves an image with input PSF.

    Parameters
    ----------
    input_data : numpy.ndarray
        Raw data of simulation that is going to be processed.
    psf_data : numpy.ndarray
        Imager's psf which should be provided by user.
    image_arcsec_pixel : astropy.units.quantity.Quantity
        Input image resolution (arcsec/pixel).
    convolution_boundary="None" : bool, optional
        Determines if user need to use extended boundary for convolution.

    Returns
    -------
    numpy.ndarray
        Convolved data of the simulated object.
    This function convert an image to a convolved image with input PSF that is
    provided by user. It uses the PSF to produce psf data which is the input to
    astropy.convolve as its kernel. The shape of the kernel should be odd and
    convolve_image_psf makes its shape odd if it is not.
    """

    # Making sure pixel scale has the right unit.
    image_arcsec_pixel = gg.utils.ensure_unit(image_arcsec_pixel,
                                              u.arcsec / u.pixel)
    # Defining centre for the galaxy_psf.
    psf_shape = psf_data.shape
    psf_centre = (int(psf_shape[0] / 2), int(psf_shape[1] / 2))

    # Producing psf data to be used as a cernel for `convolve()`.
    image_psf = gg.psf.PixellatedPSF(psf_data,
                                     psf_sampling=image_arcsec_pixel,
                                     oversampling=oversampling,
                                     psf_centre=psf_centre)

    kernel = image_psf._psf_data
    shape = kernel.shape
    col_len = shape[0]

    # Making the kernel's shape odd numbers if they are not already.
    if shape[0] / 2 == int(shape[0] / 2):
        add_row = np.zeros((1, shape[0]))
        add_row[0, :] = kernel[shape[0] - 1, :]
        kernel = np.append(kernel, add_row, axis=0)
        col_len += 1

    if shape[1] / 2 == int(shape[1] / 2):
        add_col = np.zeros((col_len, 1))
        add_col[:, 0] = kernel[:, shape[1] - 1]
        kernel = np.append(kernel, add_col, axis=1)

    convolved = convolve(input_data, kernel, boundary=convolution_boundary)

    return convolved


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
    sim_pc_pixel = gg.utils.ensure_unit(sim_pc_pixel, u.parsec / u.pixel)
    d = Distance(distance * u.Mpc)
    z = d.compute_z(cosmo)
    angular_pc = cosmo.kpc_proper_per_arcmin(z).to(u.parsec / u.arcsec)
    image_arcsec_pixel = sim_pc_pixel / angular_pc

    return image_arcsec_pixel
