import os

import yaml

import numpy as np

import pytest

from mocks import utils, mocks


def test_load_yaml_local_config():
    config = utils.load_yaml_config('config_example.yaml')
    assert config['imager_filter'] == 'g'
    assert config['sim_pc_pixel'] == 170


def test_load_yaml_nonlocal_config():
    config_file_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            '../data/nonlocal_config.yaml'))
    config = utils.load_yaml_config(config_file_path)
    assert config['imager_filter'] == 'g'
    assert config['sim_pc_pixel'] == 170


def test_load_non_existent_file():
    config_file_path = '/home/fake_user/fake_config.yaml'
    with pytest.raises(OSError):
        config = utils.load_yaml_config(config_file_path)


def test_load_corrupted_file():
    config_file_path = '../mocks/tests/data/corrupted_file.yaml'
    with pytest.raises(yaml.YAMLError):
        config = utils.load_yaml_config(config_file_path)


def test_select_low_mass_star_particles(config):
    pos, mass, info = mocks.read_gadget(config)
    pos, mass = utils.select_low_mass_star_particles(pos,
                                                     mass)
    assert pos.shape == (0, 3)
    assert mass.shape == (0,)


def test_position_data(particle_positions_3D):
    pos = utils.position_data(particle_positions_3D,
                              box_size=20)
    assert pos.max() == 16.
    assert pos.min() == 4.
    assert np.median(pos) == 10.
    assert pos.shape == (5, 3)


def test_find_dense_area(particle_positions_3D,
                         mass_weights):
    dense_area_pos = utils.find_dense_area(particle_positions_3D,
                                           mass_weights,
                                           box_size=20,
                                           bin_size=.1)
    assert dense_area_pos == (13., 14., 15.)
