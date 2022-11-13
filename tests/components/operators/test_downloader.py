# coding: utf-8

import os
import pytest

from sigma_chan_network.components.operators import download_data
from sigma_chan_network.data_structure.configrators import TrainConfig
from sigma_chan_network.io import S3Storage

@pytest.mark.skip
def test_downloader(mock_train_config):
    """Testing downloader"""
    print(mock_train_config)
    config = TrainConfig(**mock_train_config)
    s3 = S3Storage(config.cloud_storage)
    download_data(s3, config.local_storage)

    for file_path in s3.blob:
        local_file_path = os.path.join(config.local_storage.dir_name, file_path)
        os.remove(local_file_path)
