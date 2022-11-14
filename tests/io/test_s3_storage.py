# coding: utf-8

import os

import pytest

from sigma_chan_network.data_structure.configrators import StorageConfig
from sigma_chan_network.io import S3Storage


@pytest.mark.skip
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

    def test_upload_file(self, mock_minio_config: StorageConfig) -> None:
        """Testing the file upload"""
        content = "hoge"
        file_path = "data/hoge.test.dat"
        print("local file created")
        with open(file_path, "w") as f_out:
            f_out.write(content)
        print("initialize the storage")
        mock_minio_config.endpoint_url = "http://localhost:9000"
        s3 = S3Storage(mock_minio_config)
        print("storage instance", s3)
        assert isinstance(s3, S3Storage)
        print("check if the local file exists")
        assert os.path.exists(file_path)
        print("file transfer", file_path, file_path)
        s3.upload_file(file_path, file_path)
        mock_minio_config.dir_name = ""


def test_file_path():
    file_path = "getter-robo/hoge/piyo/fuga.dat"
    model_file_name = str(os.path.join(*file_path.split("/")[-3:]))
    print(model_file_name)
