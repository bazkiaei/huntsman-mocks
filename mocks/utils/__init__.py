import os

import yaml


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
