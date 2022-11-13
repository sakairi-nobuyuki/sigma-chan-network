# coding: utf-8

from sigma_chan_network.data_structure.configrators import StorageConfig, ModelConfig, DatasetConfig
from pydantic import BaseModel

class TrainConfig(BaseModel):
    """Train config"""
    cloud_storage: StorageConfig
    local_storage: StorageConfig
    model: ModelConfig
    dataset: DatasetConfig
