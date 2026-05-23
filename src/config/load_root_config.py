from hydra import compose, initialize
from hydra.utils import instantiate


from config.configuration_objects import RootConfig

def load_config() -> RootConfig:
    with initialize(version_base=None, config_path="../../config"):
        hydra_cfg = compose(config_name="config")

        cfg: RootConfig  = instantiate(hydra_cfg)
    return cfg