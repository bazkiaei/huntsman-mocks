import numpy as np

import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import Distance

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


def mock_image_stack(galaxy,
                     imager,
                     n_exposures=100,
                     exptime=500 * u.s):

    coadd = imager.make_image_real(galaxy, exptime).data

    for i in range(n_exposures):
        coadd = coadd + imager.make_image_real(galaxy, exptime).data
    coadd = coadd / n_exposures

    return coadd


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
    sim_arcsec_pixel = sim_pc_pixel / angular_pc

    return sim_arcsec_pixel
