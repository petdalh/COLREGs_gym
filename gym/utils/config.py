import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_config(config, default_path="configuration/config.yaml"):
    if config is None:
        return load_config(default_path)
    if isinstance(config, dict):
        return config
    return load_config(config)
