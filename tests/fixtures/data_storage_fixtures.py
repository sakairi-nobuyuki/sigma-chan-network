# coding: utf-8

import pytest

from sigma_chan_network.data_structure.configrators import StorageConfig


@pytest.fixture
def mock_minio_config():
    return StorageConfig(**{})
