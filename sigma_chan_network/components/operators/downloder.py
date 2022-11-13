# coding: utf-8

import os
from sigma_chan_network.io import S3Storage
from sigma_chan_network.data_structure.configrators import StorageConfig

def download_data(s3: S3Storage, local_storage_config: StorageConfig) -> bool:
    """Download data from cloud storage to the local storage

    In the storage: 
        url: s3://bucket-name/storage-dir-name/
        bucket_name: bucket-name
        dir_name: storage-dir-name/
        file_path: following-file-name
    Local file path: local-dir-name/following-file-name

    Args:
        s3 (S3Storage): An instance of S3Storage to download data
        local_storage_config (StorageConfig): Local storage config

    Returns:
        bool: True if succeeded, else False.
    """
    
    ### create local file path
    for file_path in s3.blob:
        local_file_path = os.path.join(local_storage_config.dir_name, file_path)
        dir_path = os.path.dirname(local_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        s3.download_file(file_path, local_file_path)