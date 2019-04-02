import os

import yaml

import pytest

from mocks import mocks


def test_load_yaml_local_config():
    config = mocks.load_yaml_config('config_example.yaml')
    assert config['imager_filter'] == 'g'
    assert config['sim_pc_pixel'] == 170


def test_load_yaml_nonlocal_config():
    config_file_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            '../data/nonlocal_config.yaml'))
    config = mocks.load_yaml_config(config_file_path)
    assert config['imager_filter'] == 'g'
    assert config['sim_pc_pixel'] == 170


def test_load_non_existent_file():
    config_file_path = '/home/fake_user/fake_config.yaml'
    with pytest.raises(OSError):
        config = mocks.load_yaml_config(config_file_path)


def test_load_corrupted_file():
    config_file_path = '../mocks/tests/data/corrupted_file.yaml'
    with pytest.raises(yaml.YAMLError):
        config = mocks.load_yaml_config(config_file_path)
