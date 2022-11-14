# coding: utf-8

import pytest

from sigma_chan_network.data_structure.configrators import StorageConfig


@pytest.fixture(scope="function")
def mock_minio_config():
    return StorageConfig(**{})
