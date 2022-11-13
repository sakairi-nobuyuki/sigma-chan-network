# coding: utf-8

from pydantic import BaseModel

from sigma_chan_network.data_structure.configrators import (
    DatasetConfig,
    ModelConfig,
    StorageConfig,
)


class TrainConfig(BaseModel):
    """Train config"""

    n_epoch: int = 50
    cloud_storage: StorageConfig
    local_storage: StorageConfig
    model: ModelConfig
    dataset: DatasetConfig
