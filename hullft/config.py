"""YAML config loading and dot-path CLI overrides."""

import os
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config_to_dir(
    config_dict: dict, dest_dir: str, filename: str = "config_used.yaml"
) -> None:
    if not config_dict:
        return
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    with open(dest_path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved config to {dest_path}")


def _set_by_dot_path(config_dict: dict, path: str, value: object) -> None:
    parts = path.split(".")
    target = config_dict
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def apply_config_overrides(config_dict: dict, overrides: list[str]) -> None:
    """
    Apply CLI-friendly overrides such as 'selection.n_select=20'.
    """

    if not overrides:
        return

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Must be KEY=VALUE.")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid override '{override}': key is empty.")
        try:
            parsed = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            parsed = raw_value
        _set_by_dot_path(config_dict, key, parsed)
