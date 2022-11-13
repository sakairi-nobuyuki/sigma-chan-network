# coding: utf-8

from sigma_chan_network.data_structure.configrators import StorageConfig
import pytest

@pytest.fixture
def mock_minio_config():
    return StorageConfig(**{})