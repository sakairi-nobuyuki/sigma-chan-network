# coding: utf-8

from sigma_chan_network.data_structure.configrators import StorageConfig


class TestStorageConfig:
    """Testing storage configuration"""

    def test_init(self):
        """Test initialization"""
        config = StorageConfig(**{})

        assert isinstance(config, StorageConfig)
