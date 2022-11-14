# coding: utf-8

from typing import List

import boto3

from sigma_chan_network.data_structure.configrators import StorageConfig
from sigma_chan_network.io import AbstractStorage


class S3Storage(AbstractStorage):
    """Abstract class of storage

    Args:
        metaclass (_type_, optional): Abstract class. Defaults to ABCMeta.
    """

    def __init__(self, config: StorageConfig) -> None:
        """Constructor

        Args:
            config (StorageConfig): Configuration dataclass of storages by pydantic
        """
        print("Initializing the S3 instance")
        super().__init__(config)
        if self.config.type != "s3":
            raise NotImplementedError(f"{type} is not implemented.")
        print(">> initialize the resource")
        self.s3_resource = boto3.resource(
            service_name="s3",
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_id,
            aws_secret_access_key=self.config.access_key,
            region_name=self.config.region_name,
        )
        print(">> set bucket name")
        self.bucket_name = self.config.bucket_name
        print(">> initialize the bucket")
        self.bucket = self.s3_resource.Bucket(self.bucket_name)
        print(">> get blob")
        self.blob = self.get_blob()

    def save_data(self, data: bytes, file_name: str) -> bool:
        """Save data

        Args:
            data (bytes): Data to save
            file_name (str): Destination path

        Returns:
            bool: If succeeded True, else False
        """
        s3_object = self.s3_resource.Object(self.bucket_name, file_name)
        if s3_object.put(Body=data):
            return True
        else:
            return False

    def load_data(self, file_name: str) -> bytes:
        """Load data

        Args:
            file_name (str): Data file path in the Storage

        Returns:
            bytes: Data
        """
        pass

    def download_file(self, file_path_storage: str, file_path_local: str) -> bool:
        """Dowunloading a file from a storage.

        Args:
            file_path_storage (str): File path in the storage.
            file_path_local (str): File path in local.

        Returns:
            bool: True for success, else False.
        """
        self.bucket.download_file(file_path_storage, file_path_local)

    def upload_file(self, file_path_local: str, file_path_storage: str) -> bool:
        """Uploading a file from a storage.

        Args:
            file_path_local (str): File path in local.
            file_path_storage (str): File path in the storage.

        Returns:
            bool: True for success, else False.
        """
        self.bucket.upload_file(file_path_local, file_path_storage)

    def get_blob(self) -> List[str]:
        """Get file list in the storage

        Returns:
            List[str]: File path list
        """

        blob_files = self.bucket.objects.filter(Prefix=self.config.dir_name)

        return [file_obj.key for file_obj in blob_files]
