import os

import yaml

import numpy as np


def load_yaml_config(config_location):
    """
    Loads configuration information
    Parameters
    ----------
    config_location : string
        The path to the configuration file. If the configuration file is
        located in the `config_directory` folder, then its name is enough
        for the function to load the information.

    Returns
    -------
    dict
        A dictionary of config items.

    Raises
    ------
    OSError
        If specified configuration file does not exist or is not accessible
        this error will raise.
    yaml.YAMLError
        If the code can not read the configuration file as a yaml file, this
        error will raise.
    """

    if os.path.isfile(config_location):
        # Got a direct path to the file, either relative or absolute,
        # use as is.
        config_path = config_location
    else:
        # Not a direct path to the file, try adding default config location to
        # the path
        config_path = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                    '../../config_directory',
                                                    config_location))
    with open(config_path) as config_file:
        config = yaml.load(config_file)
    return config


def select_low_mass_star_particles(particle_pos,
                                   particle_mass,
                                   mass_limit=1e8):
    """
    Removes the heavy particles which represent Dark Matter.

    Parameters
    ----------
    particle_pos : numpy.ndarray
        The 3D position of particles, simulations output.
    particle_mass : numpy.ndarray
        The mass of particles from simulations.
    mass_limit : float, optional
        The higher mass limit for the particles.

    Returns
    -------
    numpy.ndarray
        The selected particles with masses less than the mass limit.
    """
    selection = np.where(particle_mass < mass_limit)
    desired_pos = particle_pos[selection]
    desired_mass = particle_mass[selection]
    return desired_pos, desired_mass


def position_data(pos,
                  box_size=71.,
                  centre=None):
    """
    Put the data somewhere close to where user ask, center of the simulation
    box is its default.

    Parameters
    ----------
    pos : numpy.ndarray
        The 3D position of particles, simulations output.
    box_size : float, optional
        The size of the simulation box.
    centre : None, optional
        The position that the user want to put the data.

    Returns
    -------
    numpy.ndarray
        The corrected position of particles.
    """
    corrected_pos = np.zeros(pos.shape)
    if not centre:
        centre = ((box_size / 2), (box_size / 2))
    for i in range(3):
        med = np.median(pos[:, i])
        mod = (box_size / 2) - med
        corrected_pos[:, i] = (pos[:, i] + mod) % box_size
    return corrected_pos


def find_dense_area(particle_pos,
                    particle_mass,
                    box_size=71.,
                    bin_size=.01):
    """
    Finds the area that the density of mass is the most.

    Parameters
    ----------
    particle_pos : numpy.ndarray
        The 3D position of particles, simulations output.
    particle_mass : numpy.ndarray
        The mass of particles from simulations.
    box_size : float, optional
        The size of the simulation box.
    bin_size : float, optional
        The size of bins that the function will look at to find the most dense
        area.

    Returns
    -------
    numpy.float
        The position of the most dense area in the simulation box.
    """
    box_size = box_size + bin_size / 2
    dges = np.arange(0., box_size, bin_size)
    z_axis, z = np.histogram(particle_pos[:, 0],
                             bins=dges,
                             weights=particle_mass)
    y_axis, y = np.histogram(particle_pos[:, 1],
                             bins=dges,
                             weights=particle_mass)
    x_axis, x = np.histogram(particle_pos[:, 2],
                             bins=dges,
                             weights=particle_mass)
    return np.argmax(z_axis) * bin_size, np.argmax(y_axis) * bin_size,\
        np.argmax(x_axis) * bin_size
