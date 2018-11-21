import os

import numpy as np

import gunagala as gg

import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import Distance


def mocks_input(input_information='input',
			path='/Users/amir.ebadati-bazkiaei/huntsman-mocks/mocks/data'):

	path = os.path.join(path, input_information)

	input_information = gg.config.load_config(path)

	mock_image_input = dict()

	mock_image_input['galaxy_coordinates'] = input_information['galaxy_coordinates']

	mock_image_input['observation_time'] = input_information['observation_time']

	mock_image_input['imager_filter'] = input_information['imager_filter']

	# Compution of the pixel scale.
	sim_pc_pixel = gg.utils.ensure_unit(input_information['sim_pc_pixel'],
										u.parsec / u.pixel)
	d = Distance(input_information['distance'] * u.Mpc)
	z = d.compute_z(cosmo)
	angular_pc = cosmo.kpc_proper_per_arcmin(z).to(u.parsec / u.arcsec)
	mock_image_input['pixel_scale'] = sim_pc_pixel / angular_pc

	return mock_image_input
