import yaml
from typing import Union


def read_yaml(path: str) -> Union[dict, list]:
    """Reads .yaml file and extracts data from it

    Args:
        path (str): path to .yaml file

    Returns:
        dict: content of yaml file
    """
    with open(path, 'r') as f:
        config = yaml.full_load(f)
    return config