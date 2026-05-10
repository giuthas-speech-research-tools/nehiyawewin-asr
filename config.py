"""
Configuration parser for Cree Whisper ASR training.

This module intercepts command line arguments to load the
appropriate YAML configuration file, exposing the settings
as attributes of a SimpleNamespace for dot-notation access.
"""

import argparse
import yaml
import os
from types import SimpleNamespace


def load_config() -> SimpleNamespace:
    """
    Parse the CLI for a config file and load its YAML contents.

    Returns
    -------
    SimpleNamespace
        An object containing configuration keys as attributes.

    Raises
    ------
    FileNotFoundError
        If the specified YAML configuration file does not exist.
    """
    parser = argparse.ArgumentParser(
        description="Load YAML configuration for Whisper ASR."
    )
    # Default to the quick test so you don't accidentally start a 10-hour run
    parser.add_argument(
        "--config",
        type=str,
        default="configs/desktop_test.yaml",
        help="Path to the YAML configuration file."
    )

    # parse_known_args allows Hugging Face to still
    # use its own CLI args if needed
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    # Derive final model dir automatically to keep YAML clean
    config_dict["final_model_dir"] = os.path.join(
        config_dict["output_dir"], "final"
    )
    config_dict["test_results_tsv"] = os.path.join(
        config_dict["output_dir"], "test_results.tsv"
    )

    return SimpleNamespace(**config_dict)


# Instantiate the config globally so other modules can simply
# import `config` from config
config = load_config()
