import yaml
import pathlib
from typing import Dict, Any


def load_yaml_config(path: pathlib.Path):
    """Load a YAML config as an hashmap (Dictionary) structure."""
    with open(path, "r") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return config_dict


def dump_config(config: Dict[str, Any], save_path: pathlib.Path):
    yaml_file_path = save_path / "save_config.yaml"
    with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(config, yaml_file, sort_keys=False, allow_unicode=True)
