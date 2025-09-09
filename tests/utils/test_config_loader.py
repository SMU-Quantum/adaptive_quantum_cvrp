# tests/utils/test_config_loader.py

import pytest
import yaml
from pathlib import Path

from src.adaptive_quantum_cvrp.utils.config_loader import load_config

def test_load_valid_config(tmp_path: Path):
    """Tests loading a correctly formatted YAML file."""
    config_content = """
    experiment:
      name: "test_run"
      type: "classical"
    data:
      path: "/data/instance.vrp"
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    config = load_config(config_file)
    assert config["experiment"]["name"] == "test_run"
    assert config["data"]["path"] == "/data/instance.vrp"

def test_load_missing_file():
    """Tests that a FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("non_existent_file.yaml"))

def test_load_invalid_yaml(tmp_path: Path):
    """Tests that a YAMLError is raised for a malformed file."""
    invalid_content = "experiment: name: 'test_run'" # Invalid indentation
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(invalid_content)

    with pytest.raises(yaml.YAMLError):
        load_config(config_file)