# coding: utf-8

import pytest

from sigma_chan_network.data_structure.configrators import StorageConfig
from sigma_chan_network.io import S3Storage



class TestS3Storage:
    """Testing S3 storage"""

    def test_init_minio(self, mock_minio_config: StorageConfig) -> None:
        """Test initializing minio"""
        assert isinstance(mock_minio_config, StorageConfig)
        print(mock_minio_config)
        mock_minio_config.endpoint_url = "http://localhost:9000"
        s3 = S3Storage(mock_minio_config)
        assert isinstance(s3, S3Storage)
        assert isinstance(s3.config, StorageConfig)
        assert len(s3.blob) > 0
