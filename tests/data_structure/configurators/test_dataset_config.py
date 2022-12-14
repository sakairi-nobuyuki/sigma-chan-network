# coding: utf-8

import pytest

from sigma_chan_network.data_structure.configrators import DatasetConfig


class TestDatasetConfig:
    """Test of dataset config"""

    def test_init(self, mock_dataset_config):
        """Test of basic config"""

        config = DatasetConfig(**mock_dataset_config)

        assert isinstance(config, DatasetConfig)
