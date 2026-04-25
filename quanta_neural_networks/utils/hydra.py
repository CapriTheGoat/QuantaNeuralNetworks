from pathlib import Path
from typing import List

import git
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf


def get_original_cwd() -> str:
    """
    :return: the original working directory the Hydra application was launched from
    """
    if not HydraConfig.initialized():
        return str(Path.cwd())
    ret = HydraConfig.get().runtime.cwd
    assert ret is not None and isinstance(ret, str)
    return ret


def print_and_save_cfg(cfg, config_path_ll: List[str] = []):
    """
    Print and save a hydra config
    :param cfg:
    :return:
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        # Save git hash
        cfg.git_sha = sha
    except git.exc.InvalidGitRepositoryError:
        logger.info("No git repo found")

    yaml_data: str = OmegaConf.to_yaml(cfg)
    print(yaml_data)

    config_path_ll = ["config.yaml", *config_path_ll]
    # Dump to file
    for config_path in config_path_ll:
        logger.info(f"Saving train config to {Path(config_path).resolve()}")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
