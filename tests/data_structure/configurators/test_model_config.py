# coding: utf-8

import pytest
from pydantic import ValidationError

from sigma_chan_network.data_structure.configrators import ModelConfig


class TestModelConfig:
    """Testing model config"""

    def test_init(self, mock_model_config):
        """Testing basic model configuration"""
        config = ModelConfig(**mock_model_config)

        assert isinstance(config, ModelConfig)
        assert isinstance(config.input_dim, int)
        assert config.input_dim == mock_model_config["input_dim"]
        assert config.input_dim_x == config.input_dim
        assert config.input_dim_y == config.input_dim

    def test_different_input_dim(self, mock_model_config_different_dims):
        """Testing different input dimension model configuration"""
        config = ModelConfig(**mock_model_config_different_dims)

        assert isinstance(config, ModelConfig)
        assert isinstance(config.input_dim, dict)

        assert config.input_dim_x == mock_model_config_different_dims["input_dim"]["x"]
        assert config.input_dim_y == mock_model_config_different_dims["input_dim"]["y"]

    @pytest.mark.parametrize("num_classes", [0, -1])
    def test_negative_value_input(self, mock_model_config, num_classes):
        """Testing validators"""
        config_dict = mock_model_config
        print("config dict: ", config_dict)
        config_dict["n_classes"] = num_classes

        with pytest.raises(ValidationError):
            config = ModelConfig(**config_dict)
