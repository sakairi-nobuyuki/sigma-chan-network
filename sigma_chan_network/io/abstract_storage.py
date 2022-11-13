# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import List

from sigma_chan_network.data_structure.configrators import StorageConfig


class AbstractStorage(metaclass=ABCMeta):
    """Abstract class of storage

    Args:
        metaclass (_type_, optional): Abstract class. Defaults to ABCMeta.
    """

    @abstractmethod
    def __init__(self, config: StorageConfig) -> None:
        """Constructor

        Args:
            config (StorageConfig): Configuration dataclass of storages by pydantic
        """
        self.config = config

    @abstractmethod
    def save_data(self, data: bytes, file_name: str) -> bool:
        """Save data

        Args:
            data (bytes): Data to save
            file_name (str): Destination path

        Returns:
            bool: If succeeded True, else False
        """
        pass

    @abstractmethod
    def load_data(self, file_name: str) -> bytes:
        """Load data

        Args:
            file_name (str): Data file path in the Storage

        Returns:
            bytes: Data
        """
        pass

    @abstractmethod
    def download_file(self, file_path_storage: str, file_path_local: str) -> bool:
        """Dowunloading a file from a storage.

        Args:
            file_path_storage (str): File path in the storage.
            file_path_local (str): File path in local.

        Returns:
            bool: True for success, else False.
        """
        pass

    @abstractmethod
    def upload_file(self, file_path_local: str, file_path_storage: str) -> bool:
        """Uploading a file from a storage.

        Args:
            file_path_local (str): File path in local.
            file_path_storage (str): File path in the storage.

        Returns:
            bool: True for success, else False.
        """
        pass

    @abstractmethod
    def get_blob(self) -> List[str]:
        """Get file list in the storage

        Returns:
            List[str]: File path list
        """
        pass
